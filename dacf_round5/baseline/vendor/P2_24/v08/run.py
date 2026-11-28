import copy
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from p1.io import dump
from v05.engine import deliver,resample,sheet
from v06.engine import recognize,occurrences,boundaries,extend,refine,conflicts
from v06.run import timeline
from v07.regional import full_pair,reciprocal_ids,SIFT_CFG
from .common import PROJECT,V05,V06,V07,OUT,read,sha
from .triples import CFG,select_triplet,build_triplets,local_loop
from .core_gate import gated_cores,boundary_changes


def fingerprint(det):
    kp,desc=det;data=np.asarray([[*k.pt,k.size,k.angle,k.response,k.octave,k.class_id] for k in kp],dtype=np.float64).tobytes()
    return hashlib.sha256(data+(desc.tobytes() if desc is not None else b'')).hexdigest()


def validate_ids(pair,descriptors):
    errors=[];a,b=pair['samples'];qkp=descriptors[a][0];rkp=descriptors[b][0]
    if len(pair['matches'])!=pair['full_coordinate_count']:errors.append('coordinate_count')
    for m in pair['matches']:
        i,j=m['query_keypoint_id'],m['reference_keypoint_id']
        if not(0<=i<len(qkp) and 0<=j<len(rkp)):errors.append('keypoint_id_range');break
        if not np.allclose(qkp[i].pt,m['query_xy'],atol=1e-6,rtol=0) or not np.allclose(rkp[j].pt,m['reference_xy'],atol=1e-6,rtol=0):errors.append('keypoint_id_coordinate');break
    if pair['matches'] and not errors:
        mapping=reciprocal_ids(descriptors[a],descriptors[b]);allowed={ij for q in mapping.values() for ij in q}
        if any((m['query_keypoint_id'],m['reference_keypoint_id']) not in allowed for m in pair['matches']):errors.append('descriptor_reciprocal_pair')
    return errors


def local_pairs(cid,needed,old,gray,root):
    detector=cv2.SIFT_create(nfeatures=SIFT_CFG['nfeatures']);descriptors=[detector.detectAndCompute(g,None) for g in gray]
    dump(root/'feature_detection.json',{'opencv_version':cv2.__version__,'SIFT_parameters':SIFT_CFG,'gray_cache_sha256':sha(V06/cid/'shared/gray_pixels.npz'),
        'frames':[{'sample':i,'keypoints':len(d[0]),'detector_descriptor_sha256':fingerprint(d)} for i,d in enumerate(descriptors)],
        'identity_rule':'IDs qualified by frame and this detector fingerprint; reused v07 IDs checked against coordinates AND reciprocal descriptor pairs'})
    cache={}
    with (V07/cid/'full_pair_evidence.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            p=json.loads(line)
            if tuple(p['samples']) in needed:cache[tuple(p['samples'])]=p
    pairs={};counts=Counter();failures=[]
    with (root/'local_pairs_full.jsonl').open('w',encoding='utf-8') as output:
        for a,b in sorted(needed):
            if (a,b) in cache:
                p=cache[(a,b)];p['v08_coordinate_origin']='reused_v07_full';counts['reused_v07_full']+=1
            else:
                p=full_pair(a,b,old[(a,b)],descriptors,gray);p['v08_coordinate_origin']=p['coordinate_source'];counts[p['coordinate_source']]+=1
            errors=validate_ids(p,descriptors) if p['metric_agreement'] else ['original_metric_or_prefix_not_reproducible']
            p['feature_ids_valid']=not errors;p['feature_id_validation_errors']=errors
            if errors:
                p['usable_regional_geometry']=False;failures.append({'pair':p['pair_id'],'errors':errors})
            pairs[(a,b)]=p;output.write(json.dumps(p,allow_nan=False)+'\n')
    dump(root/'local_pair_audit.json',{'needed_pairs':len(needed),'coordinate_origins':dict(counts),'failures':failures,
        'v07_cache_sha256':sha(V07/cid/'full_pair_evidence.jsonl'),'v05_global_pair_sha256':sha(V05/cid/'pair_evidence.jsonl'),
        'new_area_threshold_for_C':False,'missing_policy':'quarantine local coordinates only; global direct evidence and C core gate unchanged'})
    return pairs,descriptors


def draw_triplet(record,local,frames,rgb,path):
    h,w=rgb[0].shape[:2];im=Image.new('RGB',(3*w,h+68),'#dddddd');d=ImageDraw.Draw(im)
    for panel,i in enumerate(record['samples']):
        if i is not None:im.paste(Image.fromarray(rgb[i]),(panel*w,36));label=f"f={frames[i]['source_frame']} t={frames[i]['t']:.6f}"
        else:label='unavailable source neighbor'
        d.text((panel*w+3,h+41),label,fill='black')
    d.text((3,3),f"{record['triplet_id']} {record['state']} P12/P23/P13="+str([record['edges'][e]['support'] for e in ('12','23','13')]),fill='black')
    d.text((3,18),f"exact feature loops={local['exact_closed_center_features']} shared center features={local['shared_center_features']} (green closed/orange not closed)",fill='black')
    for f in local['feature_loops']:
        color='#31e471' if f['exact_three_edge_closed'] else '#ffaf38';x,y=f['center_xy'];center=(x+w,y+36)
        for side,offset in [('left_feature_matches',0),('right_feature_matches',2*w)]:
            for m in f.get(side,[]):
                px,py=m['xy'];p=(px+offset,py+36);d.line((*p,*center),fill=color,width=1);d.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),outline=color)
        if f['exact_three_edge_closed']:
            for left in f.get('left_feature_matches',[]):
                for right in f.get('right_feature_matches',[]):
                    if any(e['left_frame_keypoint_id']==left['keypoint_id'] and e['right_frame_keypoint_id']==right['keypoint_id'] for e in f['exact_loop_evidence']):
                        d.line((left['xy'][0],left['xy'][1]+36,right['xy'][0]+2*w,right['xy'][1]+36),fill=color,width=1)
        d.ellipse((center[0]-2,center[1]-2,center[0]+2,center[1]+2),outline=color)
    im.save(path)


def write_triplet_csv(root,records,local_summaries):
    with (root/'triplet_edges.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        fields=['triplet_id','center_sample','offset_s','evaluable','state','edge','from_sample','to_sample','from_source_frame','to_source_frame','from_pts','to_pts','from_t','to_t','support','reason','pair_id','hsv_distance','gray_mad','sift_verified','sift_reason','near_duplicate','low_sharpness_samples']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in records:
            for edge,e in r['edges'].items():
                a,b=e['source_refs'];row={k:r[k] for k in ('triplet_id','center_sample','offset_s','evaluable','state')}
                row.update(edge=edge,from_sample=e['samples'][0],to_sample=e['samples'][1],from_source_frame=a['source_frame'] if a else None,to_source_frame=b['source_frame'] if b else None,
                    from_pts=a['pts'] if a else None,to_pts=b['pts'] if b else None,from_t=a['t'] if a else None,to_t=b['t'] if b else None)
                row.update({k:e[k] for k in fields if k in e});writer.writerow(row)
    with (root/'local_loop_summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        fields=['triplet_id','center_sample','offset_s','global_state','available','missing_reasons','shared_center_features','exact_closed_center_features','geometry_composition_checks','geometry_composition_consistent','geometry_composition_median_px']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in local_summaries:writer.writerow({k:r.get(k) for k in fields})


def run_case(cid,root):
    root.mkdir();meta=read(V06/cid/'full/frames.json');frames=meta['frames'];source=meta['source'];gray=np.load(V06/cid/'shared/gray_pixels.npz')['gray']
    rgb=[np.asarray(Image.open(f['image']).convert('RGB')) for f in frames];runtime=[{**f,'gray':gray[i],'rgb':rgb[i]} for i,f in enumerate(frames)]
    z=np.load(V05/cid/'similarity_matrices.npz');matrices=tuple(z[k] for k in ('hsv','gray','sift_verified','support'))
    oldcores=read(V06/cid/'shared/cores_frozen.json');needed=set()
    for i in range(len(frames)):
        for scale in CFG['offsets_s']:
            a,b,c=select_triplet(frames,i,scale)['samples']
            for x,y in [(a,b),(b,c),(a,c)]:
                if x is not None and y is not None and x!=y:needed.add(tuple(sorted((x,y))))
    old={}
    with (V05/cid/'pair_evidence.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            p=json.loads(line)
            if tuple(p['samples']) in needed:old[tuple(p['samples'])]=p
    if set(old)!=needed:raise ValueError('Frozen global direct pair evidence missing')
    records,eligibility=build_triplets(frames,old,matrices,oldcores)
    dump(root/'triplets.json',records);dump(root/'core_eligibility.json',eligibility)
    with (root/'core_eligibility.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(eligibility[0]));writer.writeheader();writer.writerows(eligibility)
    for arm in ('A','B'):
        out=root/arm;out.mkdir()
        for name in ('frames.json','scene_ids.json','segments_coarse.json','segments_refined.json','keyframes.json','conflicts.json','extension_trace.json'):
            shutil.copy2(V06/cid/'full'/name,out/name)
    dump(root/'input_manifest.json',{'source_sha256':source['sha256'],'v05_matrix_sha256':sha(V05/cid/'similarity_matrices.npz'),
        'v06_core_sha256':sha(V06/cid/'shared/cores_frozen.json'),'v06_rules_sha256':sha(PROJECT/'v06/protocol.json'),
        'same_frozen_candidates':True,'v07_regional_evidence':str((V07/cid/'regional_evidence.json').resolve())})
    # C executes irrespective of B accuracy, before reading any evaluation labels.
    C=root/'C';C.mkdir();cores=gated_cores(runtime,oldcores,matrices,eligibility);dump(C/'cores_frozen.json',cores)
    initial=recognize(runtime,cores,matrices);motion=read(V05/cid/'motion_quality.json')
    final,extension=extend(initial,occurrences(initial),boundaries(occurrences(initial)),motion,matrices)
    coarse,refined=refine(runtime,source,initial,final,motion,True)
    shutil.copy2(V06/cid/'full/frames.json',C/'frames.json')
    dump(C/'initial_recognition.json',{'assignments':initial});dump(C/'scene_ids.json',{'groups':cores['groups'],'assignments':final,'core_sample_union':sorted({i for g in cores['groups'] for i in g['frozen_core_samples']}),'core_changes_after_recognition':0})
    dump(C/'extension_trace.json',extension);dump(C/'conflicts.json',conflicts(final,cores));dump(C/'segments_coarse.json',coarse);dump(C/'segments_refined.json',refined)
    selected=deliver(runtime,source,refined,C,8)
    sheet(runtime,sorted({i for g in cores['groups'] for i in g['frozen_core_samples']}),C/'prototype_contact.png');timeline(runtime,source,final,coarse,refined,C)
    oldrefined=read(V06/cid/'full/segments_refined.json');changes=boundary_changes(oldrefined['boundaries'],refined['boundaries'],cores['old_to_new_ids'])
    dump(C/'revision_log.json',{'old_to_new_ids':cores['old_to_new_ids'],'candidate_qualification':cores['candidate_qualification'],'boundary_changes':changes,
         'A_assignments':read(V06/cid/'full/scene_ids.json')['assignments'],'C_assignments':final,'core_promotions':0,'replacement_refs':0,'manual_merge_split':False})
    dump(C/'stage_status.json',{'core_gate':'executed_once','recognition':'executed','extension':'executed_once','midpoint_transition':'executed_original_v06_rules',
        'actual_keyframes':len(selected['frames']),'groups':len(cores['groups']),'initial_midpoints':len(coarse['boundaries']),'transition_checks':len(refined['diagnostics']),
        'blocking_reason':refined['blocking_reason'],'no_extra_regional_gate':True})
    # All local triples are auxiliary diagnostics; never feed C decisions.
    pairs,descriptors=local_pairs(cid,needed,old,gray,root);local_summaries=[];(root/'triplet_overlays').mkdir()
    example_ids=set();first={}
    old_refs={i for g in oldcores['groups'] for i in g['frozen_core_samples']}
    for r in records:
        key=(r['offset_s'],r['state'])
        if key not in first:first[key]=r['triplet_id'];example_ids.add(r['triplet_id'])
        if r['center_sample'] in old_refs:example_ids.add(r['triplet_id'])
    with (root/'local_loops.jsonl').open('w',encoding='utf-8') as output:
        for r in records:
            local=local_loop(r,pairs,descriptors,gray[r['center_sample']].shape);output.write(json.dumps(local,allow_nan=False)+'\n')
            local_summaries.append({k:v for k,v in local.items() if k not in ('feature_loops','cells')})
            if r['triplet_id'] in example_ids:draw_triplet(r,local,frames,rgb,root/'triplet_overlays'/f"s{r['center_sample']:04d}_dt{r['offset_s']}.png")
    dump(root/'local_loop_summary.json',local_summaries);write_triplet_csv(root,records,local_summaries)
    dump(root/'scale_summary.json',{str(scale):dict(Counter(r['state'] for r in records if r['offset_s']==scale)) for scale in CFG['offsets_s']})
    # Same predeclared request rule per arm; A/B share one actual re-read.
    queries={}
    for arm in ('AB','C'):
        bs=read(root/('A' if arm=='AB' else 'C')/'segments_coarse.json')['boundaries']
        if bs:a,b=bs[0]['bracket'];window=[max(source['start'],a-.5),min(source['end'],b+.5)]
        else:a,b=source['start'],source['end'];window=[a+(b-a)/3,a+2*(b-a)/3]
        trace=resample(source['path'],window,root/(arm+'_resample'),frames,10,8);queries[arm]=trace
    dump(root/'resample_comparison.json',{'A':queries['AB'],'B':queries['AB'],'C':queries['C'],'A_B_shared_actual_read':True})
    return {'case':cid,'frames':len(frames),'triplets':len(records),'old_core_refs':len(old_refs),'old_core_refs_rejected':sum(i in old_refs and not e['eligible'] for i,e in enumerate(eligibility)),
        'C_groups':len(cores['groups']),'C_core_refs':sum(len(g['frozen_core_samples']) for g in cores['groups']),'C_assigned':sum(bool(r['scene_id']) for r in final),'C_keyframes':len(selected['frames']),
        'local_pair_failures':len(read(root/'local_pair_audit.json')['failures']),'local_exact_loop_triples':sum(r['exact_closed_center_features']>0 for r in local_summaries)}


def main():
    freeze=read(OUT/'history_freeze.json')
    if (OUT/'run_manifest.json').exists():raise SystemExit('Refuse second v08 prediction pass in same output')
    for rel,digest in freeze['files'].items():
        if sha(PROJECT/rel)!=digest:raise ValueError(f'Historical file changed: {rel}')
    shutil.copy2(Path(__file__).parent/'protocol.json',OUT/'protocol_locked.json')
    dump(OUT/'run_manifest.json',{'status':'running_B_and_C','cases':freeze['cases'],'protocol_sha256':sha(OUT/'protocol_locked.json'),
        'implementation':{str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'v08').glob('*.py')},'python':sys.version,'platform':platform.platform(),
        'C_mandatory_development_run':True,'evaluation_references_read':False,'passes':{'B':1,'C':1}})
    results=[];failures=[];start=time.monotonic()
    for c in freeze['cases']:
        cid=c['id'];print('RUN B+C',cid,flush=True);tick=time.monotonic()
        try:
            result=run_case(cid,OUT/cid);results.append(result);print('DONE',result,round(time.monotonic()-tick,2),flush=True)
        except Exception:
            error=traceback.format_exc();failures.append({'case':cid,'error':error});print(error,flush=True);dump(OUT/f'{cid}_failure.json',{'error':error,'other_cases_continue':True})
    dump(OUT/'run_summary.json',results);m=read(OUT/'run_manifest.json');m.update(status='B_C_predictions_complete',completed=[r['case'] for r in results],failed=failures,elapsed_seconds=time.monotonic()-start)
    dump(OUT/'run_manifest.json',m);print('COMPLETE',len(results),'failed',len(failures),flush=True)


if __name__=='__main__':main()
