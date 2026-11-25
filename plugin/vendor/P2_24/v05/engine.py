"""Prediction dataflow: no annotations, oracle prototypes or event times imported."""
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from p1.baseline import run_v
from p1.core import Parameters, pair_metrics, quality
from p1.io import command, dump, read_video
from p1.photometric import brightness_metrics
from p14_scene_prototypes.features import histogram, sift_pair
from validation_p13_20261116.engine import local_states

PROJECT = Path(__file__).resolve().parents[1]
CFG = json.loads((Path(__file__).parent / 'protocol.json').read_text())
FEATURE_CFG = json.loads((PROJECT / 'p14_scene_prototypes/protocol.json').read_text())


def public(frame):
    return {k: v for k, v in frame.items() if k not in ('gray', 'rgb')}


def select_expression(indices):
    terms = [f'eq(n\\,{i})' for i in indices]
    def balance(a):
        if len(a) == 1:
            return a[0]
        m = len(a) // 2
        return '(' + balance(a[:m]) + '+' + balance(a[m:]) + ')'
    return balance(terms)


def load_source(path, out, fps=5, window=None, limit=240):
    # Existing PTS reader performs a fresh sequential source decode on every call.
    frames, source = read_video(path, fps=fps, width=CFG['width'], max_frames=5000)
    if window is not None:
        frames = [f for f in frames if window[0] <= f['t'] < window[1]]
    if not frames:
        raise ValueError('No source frames in requested window')
    if len(frames) > limit:
        # Bound downstream pair work without silently truncating source time.
        ids = sorted(set(np.linspace(0, len(frames)-1, limit).round().astype(int).tolist()))
        frames = [frames[i] for i in ids]
        source['sampling_downgrade'] = f'uniformly bounded to {limit} candidates'
    h, w = frames[0]['gray'].shape
    with tempfile.TemporaryDirectory(prefix='v05_rgb_') as temp:
        script = Path(temp) / 'select.txt'
        script.write_text(f"select='{select_expression([f['source_frame'] for f in frames])}',scale={w}:{h},format=rgb24")
        raw = command(['ffmpeg', '-v', 'error', '-noautorotate', '-i', path, '-/filter:v', script,
                       '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    if len(raw) != len(frames)*h*w*3:
        raise ValueError('RGB/source-PTS join mismatch')
    rgb = np.frombuffer(raw, np.uint8).reshape(-1,h,w,3)
    (out / 'candidates').mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        f.update(sample=i, rgb=rgb[i].copy(), **quality(f['gray']), time_base=source['time_base'])
        image = out / 'candidates' / f'{i:04d}.png'
        Image.fromarray(f['rgb']).save(image)
        f['image'] = str(image.resolve())
    dump(out/'frames.json', {'source': source, 'window': window, 'frames': [public(f) for f in frames]})
    return frames, source


def compare(frames, out):
    n = len(frames)
    if n*(n-1)//2 > CFG['max_pairs']:
        raise ValueError('global comparison budget exceeded')
    sift = cv2.SIFT_create(nfeatures=FEATURE_CFG['sift']['nfeatures'])
    desc = [sift.detectAndCompute(f['gray'], None) for f in frames]
    hist = [histogram(f['rgb'], FEATURE_CFG['histogram']) for f in frames]
    hsv = np.zeros((n,n)); gray = np.zeros((n,n)); verified = np.eye(n,dtype=bool)
    support = np.eye(n,dtype=bool)
    pairfile = out/'pair_evidence.jsonl'
    failures = []
    with pairfile.open('w',encoding='utf-8') as handle:
        for i in range(n):
            for j in range(i+1,n):
                hd = math.sqrt(max(0.,1-min(float(np.sqrt(hist[i]*hist[j]).sum()),1.)))
                gd = float(np.abs(frames[i]['gray'].astype(float)-frames[j]['gray']).mean()/255)
                try:
                    s = sift_pair(desc[i],desc[j],frames[i]['gray'].shape,frames[j]['gray'].shape,FEATURE_CFG['sift'],f'{i}:{j}')
                except Exception as exc:
                    s = {'verified':False,'reason':str(exc),'distance':1.}
                    failures.append({'pair':[i,j],'error':str(exc)})
                if len(s.get('inlier_pairs',[]))>16:
                    s['inlier_coordinates_recorded']=16
                    s['inlier_coordinates_note']='first16 deterministic correspondences; all counts, geometry and coverage use full set; reproduce with frozen sift_pair for all coordinates'
                    s['inlier_pairs']=s['inlier_pairs'][:16]
                ok = (s['verified'] and hd <= CFG['hsv_cap']) or (hd <= CFG['tight_hsv'] and gd <= CFG['tight_gray'])
                hsv[i,j]=hsv[j,i]=hd; gray[i,j]=gray[j,i]=gd
                verified[i,j]=verified[j,i]=s['verified']; support[i,j]=support[j,i]=ok
                row={'id':f'pair:{i}:{j}','samples':[i,j],'hsv_distance':hd,'gray_mad':gd,'sift':s,'supported':bool(ok)}
                handle.write(json.dumps(row,allow_nan=False)+'\n')
    np.savez_compressed(out/'similarity_matrices.npz',hsv=hsv,gray=gray,sift_verified=verified,support=support)
    dump(out/'comparison_status.json',{'pairs':n*(n-1)//2,'sift_pair_failures':failures})
    return hsv,gray,verified,support


def group_and_assign(frames, matrices):
    hsv,gray,verified,support = matrices
    remaining = {i for i,f in enumerate(frames) if f['sharpness']>=CFG['minimum_sharpness']}
    raw=[]; groups=[]
    while remaining:
        seed=min(remaining,key=lambda i:(-sum(bool(support[i,j]) for j in remaining),i))
        members=[seed]
        for i in sorted(remaining-{seed},key=lambda i:(hsv[seed,i],-frames[i]['sharpness'],i)):
            if all(support[i,j] for j in members):
                members.append(i)
        remaining.difference_update(members)
        members.sort()
        accepted=len(members)>=CFG['minimum_group_members'] and frames[members[-1]]['t']-frames[members[0]]['t']>=CFG['minimum_group_span_s']
        prot=[]
        if accepted:
            prot=[min(members,key=lambda i:(sum(hsv[i,j] for j in members),-frames[i]['sharpness'],i))]
            while len(prot)<min(CFG['max_prototypes'],len(members)-1):
                far=max((i for i in members if i not in prot),key=lambda i:(min(hsv[i,j] for j in prot),frames[i]['sharpness'],-i))
                if min(hsv[far,j] for j in prot)<=CFG['prototype_coverage_stop']:
                    break
                prot.append(far)
        row={'raw_group':len(raw),'members':members,'accepted':bool(accepted),'prototype_samples':prot,
             'reason':'size_span_and_complete_link' if accepted else 'insufficient_members_or_span',
             'maximum_pair_hsv':max((float(hsv[i,j]) for i in members for j in members),default=0.)}
        raw.append(row)
        if accepted:
            groups.append(row)
    groups.sort(key=lambda g:min(g['members']))
    for k,g in enumerate(groups):
        g['scene_id']=f'S{k+1:02d}'
        g['prototype_frames']=[{**public(frames[i]),'selection_basis':'HSV medoid then farthest coverage; sharpness tie-break'} for i in g['prototype_samples']]
        g['independent_holdout_samples']=[i for i in g['members'] if i not in g['prototype_samples'] and any(support[i,j] for j in g['prototype_samples'])]
        g['semantic_identity']='unknown'
    rows=[]
    for i,f in enumerate(frames):
        scores=[]
        for g in groups:
            refs=[j for j in g['prototype_samples'] if j!=i]
            valid=[j for j in refs if support[i,j]]
            best=min(valid or refs,key=lambda j:(hsv[i,j],j)) if refs else None
            scores.append({'scene_id':g['scene_id'],'distance':float(hsv[i,best]) if best is not None else 1.,
                           'absolute_supported':bool(valid),'prototype_sample':best,
                           'sift_supported':any(verified[i,j] and support[i,j] for j in refs),
                           'evidence_refs':[f'pair:{min(i,j)}:{max(i,j)}' for j in refs],
                           'prototype_self_excluded':i in g['prototype_samples']})
        scores.sort(key=lambda s:(not s['absolute_supported'],s['distance'],s['scene_id']))
        candidates=[s['scene_id'] for s in scores if s['absolute_supported']]
        margin=None; label=None; reason='no_absolute_support'
        if scores and candidates:
            best=scores[0]
            second=next((s for s in scores[1:] if s['absolute_supported']),None)
            margin=(second['distance']-best['distance'])/(second['distance']+best['distance']+1e-9) if second else 1.
            both_geometry=sum(s['sift_supported'] for s in scores)>1
            if margin>=CFG['relative_margin'] and not both_geometry:
                label=best['scene_id']; reason='supported_unique_prototype_evidence'
            else:
                reason='multiple_supported_groups'
        if f['sharpness']<CFG['minimum_sharpness']:
            label=None; reason='low_sharpness_or_texture'
        rows.append({'sample':i,'t':f['t'],'source_frame':f['source_frame'],'pts':f['pts'],'scene_id':label,
                     'candidate_scene_ids':candidates,'assignment_state':'supported' if label else 'unknown',
                     'reason':reason,'relative_margin':margin,'scores':scores,
                     'prototype_in_any_group':any(i in g['prototype_samples'] for g in groups),
                     'quality':quality(f['gray'])})
    return {'version':1,'groups':groups,'raw_groups':raw,'assignments':rows,'physical_identity':'unknown'}


def motion_diagnostics(frames):
    p=Parameters(); metrics=[{'sample':0,'t':frames[0]['t'],'raw_mad':0.,'gap_s':0.,'motion_supported':False}]
    for i in range(1,len(frames)):
        a,b=frames[i-1],frames[i]
        m=pair_metrics(a['gray'],b['gray'],p)
        m.update(sample=i,from_t=a['t'],t=b['t'],gap_s=b['t']-a['t'])
        m['brightness']=brightness_metrics(a['gray'],b['gray'],m,p)
        m['edit_type']='unknown'
        if m['brightness']['supported']:
            m['edit_type']='photometric_explanation'
        elif m['motion_supported']:
            m['edit_type']='continuous_motion_support'
        metrics.append(m)
    states=local_states(frames,metrics,frames,p,{'support_window_seconds':1.,'minimum_valid_window_fraction':.6}) if len(frames)>1 else []
    return {'pair_metrics':metrics,'local_support_series':states,'interpretation':'motion and quality independent of visual ID; no physical scene/edit classifier'}


def intervals(frames,source,ids,motion):
    rows=ids['assignments']; runs=[]
    for r in rows:
        if not r['scene_id']:
            continue
        if runs and runs[-1]['scene_id']==r['scene_id'] and r['t']-runs[-1]['end']<=CFG['maximum_run_gap_s']:
            runs[-1]['samples'].append(r['sample']); runs[-1]['end']=r['t']
        else:
            runs.append({'scene_id':r['scene_id'],'start':r['t'],'end':r['t'],'samples':[r['sample']]})
    runs=[r for r in runs if len(r['samples'])>=CFG['minimum_run_samples']]
    for i,r in enumerate(runs):
        r['occurrence_id']=f'I{i+1:03d}'
    boundaries=[]
    for a,b in zip(runs,runs[1:]):
        if a['scene_id']==b['scene_id']:
            continue
        boundaries.append({'left_scene_id':a['scene_id'],'right_scene_id':b['scene_id'],
                           'left_occurrence':a['occurrence_id'],'right_occurrence':b['occurrence_id'],
                           'left_sample':a['samples'][-1],'right_sample':b['samples'][0],
                           'bracket':[a['end'],b['start']],'midpoint':(a['end']+b['start'])/2})
    edges=[source['start']]+[b['midpoint'] for b in boundaries]+[source['end']]
    labels=([boundaries[0]['left_scene_id']]+[b['right_scene_id'] for b in boundaries]) if boundaries else [runs[0]['scene_id'] if runs else None]
    coarse=[{'segment_id':f'C{i+1:03d}','start':a,'end':b,'scene_id':labels[i],'state':'provisional' if labels[i] else 'unknown'} for i,(a,b) in enumerate(zip(edges,edges[1:]))]
    diagnostics=[]
    for k,b in enumerate(boundaries):
        links=motion['pair_metrics'][b['left_sample']+1:b['right_sample']+1]
        withdrawn=bool(links) and all(m.get('brightness',{}).get('supported',False) for m in links)
        curves=[]
        for r in rows[b['left_sample']:b['right_sample']+1]:
            scores={s['scene_id']:s for s in r['scores']}
            a=scores[b['left_scene_id']]; c=scores[b['right_scene_id']]
            curves.append({'sample':r['sample'],'t':r['t'],'d_A':a['distance'],'d_B':c['distance'],
                           'S':(a['distance']-c['distance'])/(a['distance']+c['distance']+1e-9),
                           'absolute_support_A':a['absolute_supported'],'absolute_support_B':c['absolute_supported']})
        diagnostics.append({**b,'boundary_id':f'B{k+1:03d}','revision':'withdraw_photometric_boundary_keep_identity_ambiguous' if withdrawn else 'expand_midpoint_to_evidence_bracket',
                            'transition_range':None if withdrawn else b['bracket'],'curves':curves,
                            'motion_metric_samples':[m['sample'] for m in links],
                            'stop_reason':'single_pass_budget_no_recursive_refinement'})
    # Atomic partition from observed sample cells + revised transition brackets.
    cell_edges=[source['start']]+[(a['t']+b['t'])/2 for a,b in zip(rows,rows[1:])]+[source['end']]
    edges=sorted(set(cell_edges+[v for d in diagnostics if d['transition_range'] for v in d['transition_range']]))
    refined=[]
    for lo,hi in zip(edges,edges[1:]):
        if hi<=lo: continue
        mid=(lo+hi)/2
        idx=min(range(len(rows)),key=lambda i:abs(rows[i]['t']-mid))
        r=rows[idx]; active=[d for d in diagnostics if d['transition_range'] and d['bracket'][0]<=mid<d['bracket'][1]]
        state='transition' if active else ('supported' if r['scene_id'] else 'unknown')
        label=None if active else r['scene_id']
        key=(state,label,tuple(d['boundary_id'] for d in active))
        if refined and refined[-1]['_key']==key:
            refined[-1]['end']=hi; refined[-1]['evidence_samples']=sorted(set(refined[-1]['evidence_samples']+[idx]))
        else:
            refined.append({'start':lo,'end':hi,'scene_id':label,'state':state,'boundary_refs':[d['boundary_id'] for d in active],
                            'evidence_samples':[idx],'_key':key})
    for i,s in enumerate(refined):
        s.pop('_key'); s['segment_id']=f'R{i+1:03d}'
    return {'segments':coarse,'boundaries':boundaries,'occurrences':runs}, {'segments':refined,'diagnostics':diagnostics,'version':2,'identity_revision':'IDs unchanged; boundaries and time assignments revised; group re-clustering not implemented'}


def sheet(frames,indices,path):
    if not indices: return
    w=280; h=round(frames[0]['rgb'].shape[0]*w/frames[0]['rgb'].shape[1]); cols=4
    canvas=Image.new('RGB',(cols*w,math.ceil(len(indices)/cols)*(h+32)),'white'); draw=ImageDraw.Draw(canvas)
    for p,i in enumerate(indices):
        x=p%cols*w; y=p//cols*(h+32)
        canvas.paste(Image.fromarray(frames[i]['rgb']).resize((w,h)),(x,y))
        draw.text((x+2,y+h+2),f"s={i} f={frames[i]['source_frame']} t={frames[i]['t']:.6f}",fill='black')
    canvas.save(path)


def choose(frames,segments,budget=8):
    baseline=run_v(frames)
    v_order=sorted(range(len(frames)),key=lambda i:(-baseline['diff_series'][i],frames[i]['t']))
    eligible={i for i,f in enumerate(frames) if f['sharpness']>=CFG['minimum_sharpness']}
    picks=[]; reasons={}
    for s in sorted(segments,key=lambda s:(-(s['end']-s['start']),s['start'])):
        pool=[i for i in v_order if i in eligible and s['start']<=frames[i]['t']<s['end']]
        if pool and len(picks)<budget and pool[0] not in picks:
            picks.append(pool[0]); reasons[pool[0]]=f"interval_coverage:{s['segment_id']}:{s['state']}"
    # Frozen V NMS and tie logic supply the remaining ranking, then uniform time fill.
    for i in baseline['budgets'][str(budget)]['sample_indices']+v_order:
        if len(picks)>=min(budget,len(eligible)): break
        if i in eligible and i not in picks:
            picks.append(i); reasons[i]='frozen_V_fill_or_rank_fallback'
    picks.sort()
    uniform=sorted(set(np.linspace(0,len(frames)-1,min(budget,len(frames))).round().astype(int).tolist()))
    return picks,reasons,baseline,uniform


def export_native(source,frames,indices,directory):
    if not indices: return []
    directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v05_native_') as temp:
        script=Path(temp)/'select.txt'
        script.write_text(f"select='{select_expression([frames[i]['source_frame'] for i in indices])}'")
        command(['ffmpeg','-v','error','-noautorotate','-i',source['path'],'-/filter:v',script,'-fps_mode','passthrough',str(directory/'frame_%04d.png')])
    images=sorted(directory.glob('frame_*.png'))
    if len(images)!=len(indices): raise ValueError('Native export count mismatch')
    return [{**public(frames[i]),'native_image':str(p.resolve()),'native_image_sha256':__import__('hashlib').sha256(p.read_bytes()).hexdigest()} for i,p in zip(indices,images)]


def deliver(frames,source,refined,out,budget=8):
    picks,reasons,baseline,uniform=choose(frames,refined['segments'],budget)
    dump(out/'baseline_v_original_config.json',baseline)
    baseids=baseline['budgets'][str(budget)]['sample_indices']
    exports=export_native(source,frames,sorted(set(picks+baseids+uniform)),out/'native_frames')
    by_sample={r['sample']:r for r in exports}
    rows=[{**by_sample[i],'selection_reason':reasons[i]} for i in picks]
    for name,idx in [('integrated',picks),('baseline_v',baseids),('baseline_uniform',uniform)]:
        sheet(frames,idx,out/f'{name}_contact.png')
    missing=[s for s in refined['segments'] if not any(s['start']<=frames[i]['t']<s['end'] for i in picks)]
    result={'budget':budget,'actual_count':len(rows),'frames':rows,'unrepresented_intervals':missing,
            'quality_excluded_samples':[i for i,f in enumerate(frames) if f['sharpness']<CFG['minimum_sharpness']],
            'controls':{'same_pool':True,'same_budget':budget,'v':[by_sample[i] for i in baseids],'uniform':[by_sample[i] for i in uniform]},
            'limitation':'coverage+V heuristic; no semantic/relationship optimization; native PNG pixels, no autorotate'}
    dump(out/'keyframes.json',result)
    dump(out/'baseline_uniform.json',{'frames':result['controls']['uniform'],'segmentation':'none','budget':budget})
    return result


def run_case(path,out):
    out.mkdir(parents=True,exist_ok=False)
    failures=[]
    frames,source=load_source(path,out,CFG['sample_fps'],limit=CFG['max_samples'])
    try:
        matrices=compare(frames,out)
        ids=group_and_assign(frames,matrices)
    except Exception as exc:
        failures.append({'module':'global_identity','error':str(exc),'substitute':'all unknown; V/uniform downstream'})
        ids={'version':1,'groups':[],'raw_groups':[],'assignments':[{'sample':i,'t':f['t'],'scene_id':None,'scores':[],'reason':'identity_failure'} for i,f in enumerate(frames)]}
    dump(out/'scene_ids.json',ids)
    try:
        motion=motion_diagnostics(frames)
    except Exception as exc:
        failures.append({'module':'motion','error':str(exc),'substitute':'unknown edit type; retain identity curves'})
        motion={'pair_metrics':[],'local_support_series':[]}
    dump(out/'motion_quality.json',motion)
    try:
        coarse,refined=intervals(frames,source,ids,motion)
    except Exception as exc:
        failures.append({'module':'intervals','error':str(exc),'substitute':'whole source unknown'})
        s={'start':source['start'],'end':source['end'],'state':'unknown','scene_id':None,'segment_id':'fallback'}
        coarse={'segments':[s],'boundaries':[],'occurrences':[]}; refined={'segments':[s],'diagnostics':[]}
    dump(out/'segments_coarse.json',coarse); dump(out/'segments_refined.json',refined)
    selected=deliver(frames,source,refined,out,CFG['keyframe_budget'])
    sheet(frames,sorted({i for g in ids['groups'] for i in g['prototype_samples']}),out/'prototype_contact.png')
    for g in ids['groups']:
        g['occurrence_ids']=[r['occurrence_id'] for r in coarse['occurrences'] if r['scene_id']==g['scene_id']]
    dump(out/'scene_ids.json',ids)
    dump(out/'stage_status.json',{'failures':failures,'completed':['sampling','identity_or_fallback','intervals_or_fallback','actual_keyframes','no_split_controls']})
    return source,ids,coarse,refined,selected


def resample(path,window,out,previous_frames=None,fps=10,budget=8):
    if not math.isfinite(fps) or fps<=0 or len(window)!=2 or not all(math.isfinite(t) for t in window) or window[1]<=window[0]:
        raise ValueError('require positive fps and finite increasing source-time window')
    if out.exists(): raise ValueError('resample output must be new; refuses overwrite')
    out.mkdir(parents=True)
    frames,source=load_source(path,out,fps,window,160)
    actual_window=[max(window[0],source['start']),min(window[1],source['end'])]
    s={'segments':[{'segment_id':'local','start':actual_window[0],'end':actual_window[1],'state':'unknown'}]}
    result=deliver(frames,source,s,out,budget)
    old={f['source_frame'] for f in (previous_frames or [])}
    new=[f['source_frame'] for f in frames if f['source_frame'] not in old]
    trace={'request':{'source':str(path),'window':window,'fps':fps,'budget':budget},'effective_window':actual_window,
           'reader':'p1.io.read_video fresh source decode + exact-index RGB/native export; full sequential read, not seek-optimized',
           'source_sha256':source['sha256'],'candidate_count':len(frames),'new_source_frames_vs_coarse':new,
           'selected_new_source_frames':[f['source_frame'] for f in result['frames'] if f['source_frame'] not in old],
           'comparison_available':previous_frames is not None,'test_type':'scripted tool-loop, no Agent understanding evaluation',
           'stop_reason':'one request, bounded candidates, no recursive calls'}
    dump(out/'resample_trace.json',trace)
    return trace
