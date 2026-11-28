"""One locked core/full pass with frozen v05 evidence and unchanged downstream."""
import copy
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from p1.io import dump,read_video
from v05.engine import deliver,resample,sheet
from .engine import CFG,build_cores,recognize,occurrences,boundaries,extend,conflicts,refine
from .freeze import PROJECT,OLD,OUT,read,sha,paths


def load_pool(cid,shared,freeze):
    old=OLD/cid
    for name,digest in freeze['candidate_pools'][cid].items():
        if sha(old/name)!=digest:raise ValueError(f'Frozen evidence changed: {cid}/{name}')
    meta=read(old/'frames.json')
    fresh,source=read_video(meta['source']['path'],fps=5,width=320,max_frames=5000)
    if [(f['source_frame'],f['t'],f['pts']) for f in fresh]!=[(f['source_frame'],f['t'],f['pts']) for f in meta['frames']]:
        raise ValueError('Fresh source candidate PTS differs from locked v05 pool')
    if sha(source['path'])!=freeze['sources'][next(k for k in freeze['sources'] if Path(k).resolve()==Path(source['path']).resolve())]:
        raise ValueError('Source changed')
    frames=[]
    for original,new in zip(meta['frames'],fresh):
        f={**original,'gray':new['gray'],'rgb':np.asarray(Image.open(original['image']).convert('RGB'))}
        frames.append(f)
    shared.mkdir(parents=True)
    shutil.copy2(old/'frames.json',shared/'frames.json')
    shutil.copy2(old/'similarity_matrices.npz',shared/'similarity_matrices.npz')
    shutil.copy2(old/'motion_quality.json',shared/'motion_quality.json')
    np.savez_compressed(shared/'gray_pixels.npz',gray=np.asarray([f['gray'] for f in frames]))
    dump(shared/'input_evidence.json',{'candidate_pool':'exact locked v05 source/frame/PTS/RGB pool; grayscale re-read from original video',
         'pair_evidence_path':str((old/'pair_evidence.jsonl').resolve()),'pair_evidence_sha256':sha(old/'pair_evidence.jsonl'),
         'frozen_input_hashes':freeze['candidate_pools'][cid],'source_sha256':source['sha256'],'fresh_source_pts_match':True})
    z=np.load(shared/'similarity_matrices.npz')
    matrices=tuple(z[key].copy() for key in ('hsv','gray','sift_verified','support'))
    return frames,meta['source'],read(old/'scene_ids.json'),matrices,read(old/'motion_quality.json')


def timeline(frames,source,rows,coarse,refined,out):
    im=Image.new('RGB',(1320,310),'white');d=ImageDraw.Draw(im);a,b=source['start'],source['end']
    def x(t):return 150+int((t-a)/(b-a)*1140)
    colors=['#6c9bcf','#efa76a','#83bb86','#b998cf','#d6c66b','#76bbb6']
    for label,y,segs in [('coarse',35,coarse['segments']),('refined',100,refined['segments'])]:
        d.text((5,y+5),label,fill='black')
        for s in segs:
            sid=s['scene_id'];color=colors[(int(sid[1:])-1)%len(colors)] if sid else ('#e9bb5c' if s['state']=='transition' else '#cccccc')
            d.rectangle((x(s['start']),y,x(s['end']),y+35),fill=color,outline='white')
    d.text((5,170),'evidence high/mid/low',fill='black')
    for r in rows:
        y={'high':170,'medium':195,'low':220}[r['confidence_level']]
        color=colors[(int(r['scene_id'][1:])-1)%len(colors)] if r['scene_id'] else '#999999'
        d.ellipse((x(r['t'])-2,y,x(r['t'])+2,y+5),fill=color)
    selected=read(out/'keyframes.json')['frames'];d.text((5,263),'selected PTS',fill='black')
    for f in selected:d.line((x(f['t']),255,x(f['t']),280),fill='black');d.text((x(f['t']),285),f"{f['t']:.1f}",fill='black')
    d.text((150,5),f'{a:.6f} .. {b:.6f} source seconds; gray unknown/ambiguous, amber transition',fill='black');im.save(out/'timeline.png')


def run_case(cid,root,freeze):
    frames,source,old_ids,matrices,motion=load_pool(cid,root/'shared',freeze)
    cores=build_cores(frames,old_ids,matrices)
    initial=recognize(frames,cores,matrices)
    initial_runs=occurrences(initial);initial_bs=boundaries(initial_runs)
    extended,trace=extend(initial,initial_runs,initial_bs,motion,matrices)
    dump(root/'shared/cores_frozen.json',cores)
    dump(root/'shared/initial_recognition.json',{'assignments':initial})
    dump(root/'shared/extension_trace.json',trace)
    core_hash=sha(root/'shared/cores_frozen.json')
    for variant,rows,full in [('core',initial,False),('full',extended,True)]:
        out=root/variant;out.mkdir()
        coarse,refined=refine(frames,source,initial,rows,motion,full)
        dump(out/'frames.json',{'source':source,'frames':[{k:v for k,v in f.items() if k not in ('gray','rgb')} for f in frames]})
        dump(out/'scene_ids.json',{'version':f'v06-{variant}','groups':cores['groups'],'assignments':rows,'frozen_core_sha256':core_hash,
             'core_sample_union':sorted({i for g in cores['groups'] for i in g['frozen_core_samples']}),'core_changes_after_recognition':0})
        dump(out/'segments_coarse.json',coarse);dump(out/'segments_refined.json',refined)
        dump(out/'conflicts.json',conflicts(rows,cores))
        dump(out/'extension_trace.json',trace if full else {'passes':0,'accepted':0,'reason':'disabled core control'})
        dump(out/'input_evidence.json',{'shared':str((root/'shared').resolve()),'frozen_core_sha256':core_hash,'pair_evidence':str((OLD/cid/'pair_evidence.jsonl').resolve())})
        selected=deliver(frames,source,refined,out,8)
        sheet(frames,sorted({i for g in cores['groups'] for i in g['frozen_core_samples']}),out/'prototype_contact.png')
        timeline(frames,source,rows,coarse,refined,out)
        dump(out/'stage_status.json',{'core':'executed','recognition':'executed','extension':'executed_once' if full else 'disabled_control',
             'midpoints':len(coarse['boundaries']),'transition_checks':len(refined['diagnostics']),'keyframes':len(selected['frames']),
             'blocking_reason':refined['blocking_reason'],'selector':'unchanged v05.engine.deliver budget8'})
    return source


def main():
    freeze=read(OUT/'history_freeze.json')
    if (OUT/'run_manifest.json').exists():raise SystemExit('Refuse to overwrite existing v06 predictions')
    for rel,digest in freeze['historical_files'].items():
        if sha(PROJECT/rel)!=digest:raise ValueError(f'Historical file changed before predictions: {rel}')
    shutil.copy2(Path(__file__).parent/'protocol.json',OUT/'protocol_locked.json')
    # Review rules are fixed before first recognition. Reference contents stay on evaluation side.
    dump(OUT/'review_plan_locked.json',{'reference_sources':{'p14_frame_labels_sha256':sha(PROJECT/'p14_scene_prototypes/frame_labels.json'),
         'full_transition_sha256':sha(PROJECT/'validation_full_scene_20261116/annotations.json'),'synthetic_manifest_sha256':sha(PROJECT/'artifacts/videos/manifest.json')},
         'primary':'fair noncore A/B reviewed-point same/different pairs; all reviewed points and each-version noncore separately',
         'fair_exclusion':'union v05 prototype samples and v06 frozen-core samples; no held-out claim',
         'extension':'new assignment evaluated as correct/wrong/unknown using evaluation-only majority map on reviewed core labels; unmapped/conflicting group -> unknown',
         'transition':'first chronological accepted bracket vs locked reference; no best-IoU cherry picking; absent null',
         'downstream':'same pool/budget8, exact selected source indices vs old/V/uniform, event before/after proxy + contact review',
         'negative':'same-space fragments/boundaries warnings; no physical-scene accuracy claim'})
    dump(OUT/'run_manifest.json',{'status':'running','cases':freeze['cases'],'versions':['frozen-v05','v06-core','v06-full'],
         'python':sys.version,'platform':platform.platform(),'protocol_sha256':sha(OUT/'protocol_locked.json'),
         'predictor_implementation':{str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'v06').glob('*.py')},
         'selector_sha256':sha(PROJECT/'v05/engine.py'),'prediction_label_access':False})
    start=time.monotonic();completed=[];failed=[]
    for case in freeze['cases']:
        cid=case['id'];tick=time.monotonic();print('RUN',cid,flush=True)
        try:
            root=OUT/cid;root.mkdir();run_case(cid,root,freeze);completed.append(cid)
            print('DONE core+full',cid,round(time.monotonic()-tick,2),flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True);failed.append({'id':cid,'error':error});dump(OUT/f'{cid}_failure.json',{'error':error,'continuation':'other materials continue; no false completion claim'})
    # Actual new source re-read for each variant, same scripted request for comparability.
    if 'room_corridor_dev' in completed:
        root=OUT/'room_corridor_dev';source=read(root/'core/frames.json')['source']
        bs=read(root/'core/segments_coarse.json')['boundaries']
        if bs:
            a,b=bs[0]['bracket'];window=[max(source['start'],a-.5),min(source['end'],b+.5)]
        else:
            a,b=source['start'],source['end'];window=[a+(b-a)/3,a+2*(b-a)/3]
        for variant in ('core','full'):
            prior=read(root/variant/'frames.json')['frames']
            resample(source['path'],window,root/variant/'resample',prior,10,8)
    proc=subprocess.run([sys.executable,'-B','-m','unittest','v06.test_contracts','-v'],cwd=PROJECT,capture_output=True)
    (OUT/'tests.log').write_bytes(proc.stdout+proc.stderr)
    current={str(p.relative_to(PROJECT)):sha(p) for p in paths()}
    changed=[p for p,h in freeze['historical_files'].items() if current.get(p)!=h]
    added=sorted(set(current)-set(freeze['historical_files']))
    source_changed=[p for p,h in freeze['sources'].items() if sha(p)!=h]
    dump(OUT/'history_verification.json',{'count':len(freeze['historical_files']),'changed':changed,'added':added,'source_changed':source_changed,'pass':not(changed or added or source_changed)})
    m=read(OUT/'run_manifest.json');m.update(status='one_round_predictions_complete',completed=completed,failed=failed,elapsed_seconds=time.monotonic()-start,tests_exit_code=proc.returncode)
    dump(OUT/'run_manifest.json',m)
    print('COMPLETE',len(completed),'failed',len(failed),'tests exit',proc.returncode,flush=True)


if __name__=='__main__':main()
