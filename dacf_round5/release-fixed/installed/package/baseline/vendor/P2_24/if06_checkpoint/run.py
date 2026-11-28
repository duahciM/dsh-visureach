"""Immutable first run: python -m if06_checkpoint.run --out DIR [--media-dir DIR]."""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from identity_catalog_matching.common import read,sha
from identity_small_support_spatial.memory import direct_pairs
from p1.io import dump
from v05.engine import load_source,compare,motion_diagnostics,deliver,choose,resample,export_native,sheet,public
from v06.engine import refine
from .spatial import direct_new,space_interface,abrupt_pairs,memory,associations
from .p2 import state_select

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
OLD=PROJECT/'identity_small_support_spatial_20261118'
COARSE_OLD=PROJECT/'v06_first_run_20261117'
CFG=read(HERE/'protocol.json')
FOCUS={'dissolve','local_object','ppt','room_corridor_dev'}


def media_files(p):
    return [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in {'.mp4','.mkv','.mov','.m4v','.avi','.webm'}]


def historic(out):
    return {str(p.relative_to(PROJECT)):sha(p) for p in PROJECT.rglob('*') if p.is_file() and
            not p.is_relative_to(HERE) and not p.is_relative_to(out) and not {'.venv','__pycache__'}.intersection(p.parts)}


def window(s,coarse):
    if coarse['boundaries']:
        a,b=coarse['boundaries'][0]['bracket']
        return [max(s['start'],a-.5),min(s['end'],b+.5)]
    span=s['end']-s['start']
    return [s['start']+span/3,s['start']+span*2/3]


def old_pool(case):
    meta=read(OLD/case/'Cmax/frames.json')
    with np.load(COARSE_OLD/case/'shared/gray_pixels.npz') as gray:
        fs=[f|{'gray':g,'rgb':np.asarray(Image.open(f['image']).convert('RGB'))} for f,g in zip(meta['frames'],gray['gray'])]
    return fs,meta


def picks(data): return [f['sample'] for f in data['frames']]


def a_case(out,case,frames,meta,source_path):
    base=OLD/case
    for ver,direct_arm,time_arm in [('A0','Cmax','Cmax'),('A1','Cadd','Cadd'),('A2','Cadd','Cmax'),('A3','Cmin','Cmax')]:
        root=out/'IF06_A'/case/ver;root.mkdir(parents=True)
        srcs={name:base/arm/f'{name}.json' for name,arm in [('frames',time_arm),('scene_ids',direct_arm),
                   ('segments_coarse',time_arm),('segments_refined',time_arm),('keyframes',time_arm)]}
        srcs.update(all_channel_match_evidence=base/direct_arm/'all_channel_match_evidence.jsonl',
                    direct_catalog=base/f'{direct_arm}_catalog.json',direct_templates=base/f'{direct_arm}_templates.json')
        if ver=='A0':
            srcs.update(lh_safe_evidence=base/'LH_safe/stage1_match_evidence.jsonl',lh_safe_scene_ids=base/'LH_safe/scene_ids.json',
                        lh_safe_segments_coarse=base/'LH_safe/segments_coarse.json',lh_safe_segments_refined=base/'LH_safe/segments_refined.json',
                        lh_safe_keyframes=base/'LH_safe/keyframes.json')
        for key,p in srcs.items(): shutil.copyfile(p,root/f'{key}{p.suffix}')
        coarse,refined=map(read,(srcs['segments_coarse'],srcs['segments_refined'])); ids=read(srcs['scene_ids'])
        expected,_,v,uniform=choose(frames,refined['segments'],8); k=read(srcs['keyframes'])
        assert expected==picks(k) and v['budgets']['8']['sample_indices']==[f['sample'] for f in k['controls']['v']] and uniform==[f['sample'] for f in k['controls']['uniform']]
        assert all(sha(p)==sha(root/f'{key}{p.suffix}') for key,p in srcs.items())
        interface=space_interface(coarse,refined,ids['assignments'],time_arm)
        existing={g['scene_id'] for g in read(base/'Cmax/scene_ids.json')['groups']}
        fresh=[g['scene_id'] for g in ids['groups'] if g['scene_id'] not in existing]
        for name in ('integrated_contact.png','baseline_v_contact.png','baseline_uniform_contact.png','timeline.png'):
            shutil.copyfile(base/time_arm/name,root/name)
        dump(root/'interfaces.json',interface|{'new_local_evidence_ids':fresh,'new_evidence_without_independent_scene_support':fresh if ver in ('A2','A3') else [],
               'direct_input':direct_arm,'time_input':time_arm,'source_input_hashes':{k:sha(v) for k,v in srcs.items()},
               'time_uses_new_evidence_ids':bool(fresh and ver=='A1'), 'no_new_physical_scene_inference':True})
        if case in FOCUS and case!='local_object' and case!='ppt':
            resample(source_path,window(meta['source'],coarse),root/'resample',meta['frames'],10,8)
    print('A complete:',case,flush=True)


def draw_pairs(frames,raw,chosen,path):
    if not chosen:return
    w=320; h=frames[0]['rgb'].shape[0]; canvas=Image.new('RGB',(2*w,(h+24)*len(chosen)),'#18252c');d=ImageDraw.Draw(canvas)
    for n,key in enumerate(chosen):
        row=next(x for x in raw.values() if x['id']==key)
        i,j=row['samples']; y=n*(h+24)
        for offset,k in ((0,i),(w,j)):
            canvas.paste(Image.fromarray(frames[k]['rgb']),(offset,y));d.text((offset+3,y+h+2),f"{key} sample {k} | {frames[k]['t']:.3f}s PTS {frames[k]['pts']}",fill='white')
        for m in row['sift'].get('inlier_pairs',[]):
            a,b=m['query_xy'],m['reference_xy']; ax,ay=a[0],y+a[1]; bx,by=w+b[0],y+b[1]
            d.line((ax,ay,bx,by),fill='#e9aa4d');d.ellipse((ax-2,ay-2,ax+2,ay+2),fill='#67e5c6');d.ellipse((bx-2,by-2,bx+2,by+2),fill='#67e5c6')
    canvas.save(path)


def b_case(out,code,path):
    root=out/'IF06_B'/code; shared=root/'shared';shared.mkdir(parents=True)
    frames,source=load_source(str(path),shared,5,limit=240)
    matrices=compare(frames,shared)
    cores,templates,rows,evidence=direct_new(frames,matrices)
    ids={'groups':cores['groups'],'assignments':rows}
    for name,obj in [('cores',cores),('templates',templates),('scene_ids',ids),('channel_evidence',evidence)]:dump(shared/f'{name}.json',obj)
    raw=direct_pairs(shared/'pair_evidence.jsonl',lambda key:True)
    motion=motion_diagnostics(frames);dump(shared/'motion_quality.json',motion)
    coarse,refined=refine(frames,source,rows,rows,motion,True)
    dump(shared/'segments_coarse.json',coarse);dump(shared/'segments_refined.json',refined)
    abrupt=abrupt_pairs(frames,motion,raw,CFG['B_association']['max_contrast_link_mad']);dump(shared/'abrupt_edit_or_motion.json',abrupt)
    selected=deliver(frames,source,refined,shared,8)
    for arm in ('B0','B1','B2'):
        dest=root/arm;dest.mkdir()
        for name in ('scene_ids.json','channel_evidence.json','segments_coarse.json','segments_refined.json','keyframes.json'):
            shutil.copyfile(shared/name,dest/name)
        mem=memory(frames,motion,raw,rows,refined,arm,abrupt);dump(dest/'landmark_memory.json',mem)
        rel=associations(frames,raw,rows,refined,abrupt,mem,CFG['B_association']) if arm=='B2' else {'accepted':None,'relations':[],'reason':'base arm does not propose cross-evidence association'}
        dump(dest/'association_evidence.json',rel)
        dump(dest/'interfaces.json',space_interface(coarse,refined,rows,'new',rel['accepted']))
        if arm=='B2':
            vis=[v['pair'] for v in rel['all_nonlocal_pair_gate'] if v['status']=='wide_direct']
            qualifying=rel['accepted']['witness_pairs'] if rel['accepted'] else []
            chosen=qualifying or (vis[:1]+vis[-1:] if len(vis)>1 else vis)
            err=[e['direct_pair'] for e in mem['events'] if e['kind'] in {'confirmed_reentry','retracted_edge_prediction'}]
            draw_pairs(frames,raw,list(dict.fromkeys(chosen+err))[:8],dest/'direct_pair_overlays.png')
    q=resample(str(path),window(source,coarse),shared/'resample',frames,10,8)
    print('B complete:',code,len(frames),len(cores['groups']),len(refined['occurrences']),len(selected['frames']),q['candidate_count'],flush=True)


def p2_case(out,case,frames,meta,source_path):
    old=OLD/case/'Cmax';write=out/'P2_01'/case;write.mkdir(parents=True)
    for arm in ('P0','P1'):(write/arm).mkdir()
    coarse,refined,baseline=map(read,(old/'segments_coarse.json',old/'segments_refined.json',old/'keyframes.json'))
    motion=read(COARSE_OLD/case/'shared/motion_quality.json')
    fs=frames;s=meta['source']; cfg=CFG['P2']
    chosen,reasons,roles,trace=state_select(fs,coarse,refined,motion,cfg)
    assert len(chosen)==len(baseline['frames'])
    for file in ('scene_ids.json','segments_coarse.json','segments_refined.json','frames.json'):
        shutil.copyfile(old/file,write/file)
    for file in ('keyframes.json','integrated_contact.png','baseline_v_contact.png','baseline_uniform_contact.png'):
        shutil.copyfile(old/file,write/'P0'/file)
    assets=export_native(s|{'path':source_path},fs,chosen,write/'P1'/'native_frames')
    result=baseline|{'frames':[f|{'selection_reason':reasons[f['sample']]} for f in assets],
       'unrepresented_intervals':[r for r in refined['segments'] if not any(r['start']<=fs[i]['t']<r['end'] for i in chosen)],
       'limitation':'visual state proxies, no semantic relation inference'}
    dump(write/'P1'/'keyframes.json',result);sheet(fs,chosen,write/'P1'/'integrated_contact.png')
    dump(write/'role_trace.json',{'state_pairs':roles,'candidates':trace,'locked_pool_sha256':sha(old/'frames.json'),
             'locked_time_sha256':sha(old/'segments_refined.json'),'locked_direct_sha256':sha(old/'scene_ids.json'),
             'old_selected':picks(baseline),'new_selected':chosen,'removed':sorted(set(picks(baseline))-set(chosen)),
             'added':sorted(set(chosen)-set(picks(baseline)))})
    if case in FOCUS:
        a,b=window(s,coarse); fallback=[a+(b-a)/4,b-(b-a)/4]
        for arm,centers in [('P0',fallback),('P1',[fs[i]['t'] for i in roles[0]['samples']] if roles else fallback)]:
            seen=0
            for k,t in enumerate(centers[:cfg['query_centers']],1):
                d=cfg['query_halfwidth_s'];w=[max(s['start'],t-d),min(s['end'],t+d)]
                resample(source_path,w,write/arm/f'resample_{k}',fs,10,8);seen+=1
            assert seen==cfg['query_centers']
    print('P2 complete:',case,'roles',len(roles),flush=True)


def main():
    args=argparse.ArgumentParser(description=__doc__);args.add_argument('--out',type=Path,required=True);args.add_argument('--media-dir',type=Path,default=Path(CFG['media_directory']));a=args.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=out/'prediction_manifest.json'
    if manifest.exists(): raise SystemExit(f'Will not overwrite existing prediction: {manifest}')
    inputs=read(OLD/'run_manifest.json')['cases'];new=media_files(a.media_dir)
    expected=read(OLD/'history_freeze.json')['sources']
    by_name={str(p.resolve()):sha(p) for p in new}
    current={}
    for r in inputs:
        p,digest=r['path'],expected[r['path']]
        if Path(p).is_file() and sha(p)==digest:
            current[p]=p
        else:
            copies=[q for q,h in by_name.items() if h==digest]
            current[p]=copies[0] if len(copies)==1 else None
    oldsrc={p:{'expected_sha256':h,'resolved_source':current[p],'relocated':current[p] is not None and current[p]!=p} for p,h in expected.items()}
    newsrc={f'N{i:02d}':{'path':p,'sha256':h,'equals_historical_source':h in expected.values()} for i,(p,h) in enumerate(by_name.items(),1)}
    history=historic(out)
    labels={name:sha(out/name) for name in ('labels_B_sealed.json','labels_P2_sealed.json') if (out/name).exists()}
    code={p.name:sha(p) for p in HERE.glob('*') if p.suffix in ('.py','.json')}
    dump(out/'lock.json',{'source_directory':str(a.media_dir.resolve()),'historical_files':history,'historical_sources':oldsrc,
                           'new_sources':newsrc,'prediction_code_sha256':code,'sealed_label_hashes_only':labels})
    shutil.copyfile(HERE/'protocol.json',out/'protocol_locked.json')
    dump(manifest,{'status':'running','A_cases':[r['id'] for r in inputs],'B_sources':newsrc,'labels_opened_by_prediction':False})
    failures=[];done={'A':[],'B':[],'P2':[]};start=time.monotonic()
    def execute(group,case,fn):
        try:fn();done[group].append(case)
        except Exception:
            failures.append({'branch':group,'case':case,'traceback':traceback.format_exc()});print(failures[-1]['traceback'],flush=True)
    for item in inputs:
        c=item['id']
        try: fs,meta=old_pool(c)
        except Exception:
            failures.append({'branch':'A/P2','case':c,'traceback':traceback.format_exc()});continue
        execute('A',c,lambda:a_case(out,c,fs,meta,current[item['path']]))
        execute('P2',c,lambda:p2_case(out,c,fs,meta,current[item['path']]))
    for c,r in newsrc.items():execute('B',c,lambda:b_case(out,c,Path(r['path'])))
    dump(manifest,{'status':'predictions_complete' if not failures else 'complete_with_failures','completed':done,
                   'B_sources':newsrc,'seconds':time.monotonic()-start,'failures':failures,'labels_opened_by_prediction':False})
    print('Completed',list(map(lambda x:(x,len(done[x])),done)), 'failures',len(failures),flush=True)
    if failures:sys.exit(1)


if __name__=='__main__':main()
