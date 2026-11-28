"""Run IF-07 once into a fresh, separate directory; no evaluation imports."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import shutil
import sys
import time
import traceback

from p1.io import dump,read_video
from v05.engine import export_native,sheet,resample
from if06_checkpoint.run import media_files,old_pool,window,draw_pairs
from if06_checkpoint.spatial import associations,abrupt_pairs
from identity_small_support_spatial.memory import direct_pairs,motion_links
from identity_catalog_matching.common import read,sha
from .ledger import restore,emit_ledger
from .inference import graph,infer,proposals,supported_pose
from .p2 import small_states,query_plan

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREV=ROOT/'if06_checkpoint_20261118'
CFG=read(HERE/'protocol.json')
OLD=ROOT/'v05_first_run_20261117'
LOCK=read(PREV/'lock.json')
STAGES=('A','B','C','P2')


def historic(out):
    return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and not p.is_relative_to(HERE)
        and not p.is_relative_to(out) and not {'.venv','__pycache__'}.intersection(p.parts)}


def peak_mb():
    if sys.platform!='win32':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    import ctypes as c
    from ctypes import wintypes as w
    class Memory(c.Structure):
        _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD)]+[(k,c.c_size_t) for k in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    get=c.windll.kernel32.GetCurrentProcess;get.restype=w.HANDLE
    api=c.windll.psapi.GetProcessMemoryInfo;api.argtypes=[w.HANDLE,c.POINTER(Memory),w.DWORD];api.restype=w.BOOL
    m=Memory();m.cb=c.sizeof(m)
    return m.PeakWorkingSetSize/(1024**2) if api(get(),c.byref(m),m.cb) else None


def sources(media):
    listed=[{'path':str(p.resolve()),'sha256':sha(p)} for p in media_files(media)]
    iso={}
    for p,expected in LOCK['historical_sources'].items():
        candidates=[x['path'] for x in listed if x['sha256']==expected['expected_sha256']]
        keep=expected['resolved_source']
        fallback=next((x for x in (p,keep) if x and Path(x).is_file() and sha(x)==expected['expected_sha256']),None)
        iso[p]={'path':fallback or (candidates[0] if len(candidates)==1 else None),'sha256':expected['expected_sha256']}
    return listed,iso


def prior_frames(code,source_path):
    path=PREV/'IF06_B'/code/'shared'
    meta=read(path/'frames.json')
    frames,source=read_video(source_path,5,320)
    if [(f['source_frame'],f['pts']) for f in frames]!=[(f['source_frame'],f['pts']) for f in meta['frames']]:raise ValueError('PTS lock mismatch')
    assert source['sha256']==meta['source']['sha256']
    return [x|{'gray':f['gray']} for x,f in zip(meta['frames'],frames)],meta,path


def frozen_old(case):
    frames,meta=old_pool(case)
    path=PREV/'IF06_A'/case/'A2'
    motion=read(OLD/case/'motion_quality.json')
    ids=read(path/'scene_ids.json'); temporal_rows=read(PREV/'IF06_A'/case/'A0/scene_ids.json')['assignments']
    return frames,meta,path,motion,ids,temporal_rows


def a_case(out,name,frames,folder,original,full,motion,rows,refined):
    old=PREV/'IF06_B'/name/'B2'
    mem=read(old/'landmark_memory.json')
    policy=read(PREV/'protocol_locked.json')['B_association']
    edits=read(folder/'abrupt_edit_or_motion.json')
    a0=associations(frames,original,rows,refined,edits,mem,policy)
    if a0!=read(old/'association_evidence.json'):raise AssertionError('historical A0 replay differed')
    a1=associations(frames,full,rows,refined,edits,mem,policy)
    by={x['pair']:x['status'] for x in a0['all_nonlocal_pair_gate']}
    statuses=Counter(); changes=[]
    keys={p['id']:pair for pair,p in full.items()}
    for x in a1['all_nonlocal_pair_gate']:
        k=x['pair']; old_gate=by[k]; new_gate=x['status']; p=full[keys[k]]
        if p['distribution_state'] in ('budget_unassessed','recomputation_error','recomputation_inconsistent','old_computation_failed'):
            new_gate='budget_unassessed' if p['distribution_state']=='budget_unassessed' else 'not_evaluable'
        statuses[new_gate]+=1
        if old_gate!=new_gate:changes.append({'pair':k,'before':old_gate,'after':new_gate,'source_samples':p['samples']})
    root=out/'AB'/name;root.mkdir(parents=True,exist_ok=True)
    dump(root/'A0.json',a0);dump(root/'A1.json',a1)
    dump(root/'A_input_diff.json',{'changes':changes,'a0_by_gate':dict(Counter(by.values())),'a1_by_gate':dict(statuses),
        'a0_cross_id_with_wide':sum(x['evidence_ids'][0]!=x['evidence_ids'][1] and bool(x['wide_direct_pairs']) for x in a0['relations']),
        'a1_cross_id_with_wide':sum(x['evidence_ids'][0]!=x['evidence_ids'][1] and bool(x['wide_direct_pairs']) for x in a1['relations']),
        'historical_A0_replayed_exactly':True,'memory_input_sha256':sha(old/'landmark_memory.json')})
    return mem


def core_case(out,name,frames,meta,path,motion,ids,temporal_rows,raw,full,restore_trace,source_path,a_mem=None):
    c=out/'C'/name;c.mkdir(parents=True)
    c0=read(path/'segments_coarse.json');r=read(path/'segments_refined.json')
    edits=abrupt_pairs(frames,motion,raw,CFG['B']['abrupt_raw_mad'])
    edits_idx={x['sample'] for x in edits}
    if a_mem is None:
        adj={k:p for k,p in raw.items() if k[1]-k[0]==1}
        old_motion=motion_links(frames,motion,adj,temporal_rows,[d['bracket'] for d in r['diagnostics'] if d['transition_accepted']])
    else:old_motion=a_mem['links']
    led=c/'evidence_ledger.jsonl'; start=time.monotonic()
    b1=emit_ledger(led,frames,raw,full,motion,r,{x['sample']:x for x in old_motion},sha(out/'protocol_locked.json'),CFG)
    elapsed=time.monotonic()-start
    shutil.copyfile(path/'frames.json',c/'frames.json')
    for f in ('segments_coarse.json','segments_refined.json','keyframes.json','integrated_contact.png','baseline_v_contact.png','baseline_uniform_contact.png'):
        origin=path/f
        if not origin.exists() and name in ('N01','N02','N03'):origin=PREV/'IF06_B'/name/'shared'/f
        shutil.copyfile(origin,c/f)
    for f in ('scene_ids.json','all_channel_match_evidence.jsonl','channel_evidence.json','cores.json','templates.json','direct_catalog.json','direct_templates.json'):
        if (path/f).exists():shutil.copyfile(path/f,c/f)
    dump(c/'B0_legacy_observation.json',{'memory_decisions':old_motion,'direct_corpus':str(path.absolute()),
          'comparison_pair_count':len(raw),'original_sha256':meta['source']['sha256'],'carry_rule':'old timing and direct decisions are a single historical input, not independent geometry votes'})
    locked={'B1':b1,'restoration':restore_trace,'ledger_sha256':sha(led),'direct_lock':sha(path/'scene_ids.json'),
        'pool_sha256':sha(path/'frames.json'),'legacy_timing_sha256':sha(path/'segments_refined.json'),
        'suspected_abrupt_conflicts':edits,'ledger_wall_seconds':elapsed,'peak_working_set_mib_cumulative':peak_mb()}
    dump(c/'freeze_B1.json',locked)
    stage=time.monotonic()
    g=graph(frames,full,edits_idx,CFG['R'])
    dump(c/'anchor_pose_graph.json',g)
    duties,wide,gates=proposals(frames,temporal_rows,r,full,CFG['R'])
    dump(c/'proposal_gate.json',{'nonlocal_counts':gates,'eligible_wide_count':len(wide),'occurrence_pairs':len(duties),
         'deterministic_graph_and_gate_seconds':time.monotonic()-stage,
         'wide_pairs':[p['id'] for p in wide.values()],'lane_budget':CFG['R']['edge_visits_per_relation']})
    dump(c/'graph_performance.json',{'frames':len(frames),'pose_edges':len(g['pose_hypotheses']),
         'pose_spatial_edges':sum(p['state']=='image_relative_pose' for p in g['pose_hypotheses']),
         'landmark_count':len(g['landmarks']),'peak_working_set_mib_cumulative':peak_mb()})
    diagnostic=[]
    by_start=defaultdict(list);by_end=defaultdict(list)
    for edge in g['pose_hypotheses']:
        if edge['state']=='image_relative_pose':
            i,j=edge['samples'];by_start[i].append(edge);by_end[j].append(edge)
    long_pairs=sorted(wide,key=lambda ij:(-(frames[ij[1]]['t']-frames[ij[0]]['t']),ij))
    for i,j in long_pairs[:2]:
        p=wide[i,j]
        diagnostic.append({'pair':p['id'],'sources':[frames[i]['t'],frames[j]['t']],
            'matching_across_abrupt_candidate':any(i<k<=j for k in edits_idx),
            'relative_loop':supported_pose(i,j,p,frames,g,by_start,by_end,True,'forward',CFG['R']),
            'meaning':'diagnostic direct reobservation even without a chain or different ID; not scene assertion'})
    dump(c/'long_closure_diagnostics.json',diagnostic)
    checks={}
    for arm in ('R0','R1','R2','R3'):
        armroot=c/arm;armroot.mkdir()
        for order in ('forward','reverse','shuffle'):
            z=infer(frames,ids['assignments'],c0,r,full,g,duties,edits_idx,arm,order,CFG['R'])
            dump(armroot/f'{order}.json',z)
        base=read(armroot/'forward.json')
        ref=lambda x:(x['interface']['sample_to_segment'],x['association']['occurrences'] if x['association'] else None)
        checks[arm]={o:ref(base)==ref(read(armroot/f'{o}.json')) for o in ('reverse','shuffle')}
    dump(c/'order_sensitivity.json',checks)
    choose=[p['id'] for p in wide.values() if p['samples'][1]-p['samples'][0]>len(frames)/2]
    if choose:
        for f in frames:
            if 'rgb' not in f:
                from PIL import Image
                import numpy as np
                f['rgb']=np.asarray(Image.open(f['image']).convert('RGB'))
        draw_pairs(frames,full,[choose[0]] if len(choose)==1 else [choose[0],choose[-1]],c/'long_recurrence_overlay.png')
    resample(source_path,window(meta['source'],c0),c/'resample',meta['frames'],10,8)
    print('B/C',name,'pairs',len(raw),'wide',len(wide),'pose edges',sum(p['state']=='image_relative_pose' for p in g['pose_hypotheses']),flush=True)


def p2_case(out,case,frames,meta,source_path):
    h=PREV/'P2_01'/case; path=out/'P2'/case; path.mkdir(parents=True)
    cfg=CFG['P2']; s=meta['source']
    coarse=read(h/'segments_coarse.json');r=read(h/'segments_refined.json'); motion=read(OLD/case/'motion_quality.json')
    k=read(h/'P1/keyframes.json')
    oldroles=read(h/'role_trace.json')['state_pairs']
    picks,chosen,trace=small_states(frames,motion,[f['sample'] for f in k['frames']],oldroles,cfg)
    z=path/'Psmall';z.mkdir()
    rows=export_native(s|{'path':source_path},frames,picks,z/'native_frames')
    role_by={i:f'connected_persistent_change_{n}:{tag}' for n,p in enumerate(chosen,1) for i,tag in zip(p['samples'],('before','after'))}
    oldwhy={f['sample']:f['selection_reason'] for f in k['frames']}
    for f in rows:f['selection_reason']=role_by.get(f['sample'],oldwhy.get(f['sample'],'legacy_frozen_support'))
    dump(z/'keyframes.json',k|{'frames':rows,'unrepresented_intervals':[p for p in r['segments'] if not any(p['start']<=f['t']<p['end'] for f in rows)],'limitation':'two persistent signed image states; no event semantics'})
    sheet(frames,picks,z/'integrated_contact.png')
    dump(z/'role_trace.json',{'broader_P1_roles':oldroles,'reserved':chosen,'audit':trace,'frozen_pool_sha256':sha(h/'frames.json'),'frozen_segments_sha256':sha(h/'segments_refined.json')})
    for arm in ('P0','P1'):
        dst=path/arm;dst.mkdir()
        for f in ('keyframes.json','integrated_contact.png'):
            shutil.copyfile(h/arm/f,dst/f)
    z=path/'Pquery';z.mkdir();shutil.copyfile(h/'P0/keyframes.json',z/'keyframes.json')
    centers,trace=query_plan(frames,s,coarse,r,motion,cfg)
    dump(z/'query_plan.json',trace|{'centers':centers,'budget':cfg['query_budget'],'window_width_s':2*cfg['query_half_s']})
    for i,t in enumerate(centers):resample(source_path,[max(s['start'],t-cfg['query_half_s']),min(s['end'],t+cfg['query_half_s'])],z/f'resample_{i+1}',meta['frames'],10,8)
    print('P2',case,'local paired roles',len(chosen),flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('--out',type=Path,required=True);a.add_argument('--media-dir',type=Path,default=Path(CFG['media_dir']));p=a.parse_args()
    out=p.out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=out/'run_manifest.json'
    if manifest.exists():raise SystemExit('Refuses to overwrite existing run')
    t=time.monotonic();listed,old=sources(p.media_dir)
    historical=historic(out)
    prediction_code={name:sha(HERE/name) for name in ('run.py','ledger.py','inference.py','p2.py','protocol.json','__init__.py')}
    locks={'historical_files':historical,'historical_sources':old,'media_snapshot':listed,'pred_code':prediction_code,
       'external_task_card_sha256':sha(Path('C:/Users/Administrator/Desktop/视觉插件/video-keyframes-lightweight-master-IF07-next.md')),
       'review_file_hashes_only':{str(PREV/f):sha(PREV/f) for f in ('labels_B_sealed.json','labels_P2_sealed.json')}}
    dump(out/'lock.json',locks);shutil.copyfile(HERE/'protocol.json',out/'protocol_locked.json')
    d={'status':'running','completed':{k:[] for k in STAGES},'failures':[],'prediction_opened_labels':False}
    dump(manifest,d)
    inputs=read(ROOT/'identity_small_support_spatial_20261118/run_manifest.json')['cases']
    expected=read(PREV/'prediction_manifest.json')['B_sources']; by_sha={x['sha256']:x['path'] for x in listed}
    active_b={c:by_sha.get(s['sha256']) for c,s in expected.items()}
    supplemental=[x for x in listed if x['sha256'] not in {y['sha256'] for y in expected.values()}|{y['sha256'] for y in old.values()}]
    dump(out/'media_status.json',{'historical_AB_inputs':active_b,'additional_content_needs_separate_source_setup':supplemental,
       'independent_new_crossing_evaluation':False,'calibrated_real_angles':False,'notes':'No frozen test source or tags invented, N04 only AB historical control.'})
    def attempt(tag,name,fn):
        try:
            x=fn();d['completed'][tag].append(name);return x
        except Exception:
            err=traceback.format_exc();d['failures'].append({'stage':tag,'case':name,'trace':err});print(err,flush=True)
    for code,source in active_b.items():
        if source is None:
            d['failures'].append({'stage':'A','case':code,'trace':'Frozen source unavailable'});continue
        def b():
            frames,meta,root=prior_frames(code,source)
            raw=direct_pairs(root/'pair_evidence.jsonl',lambda x:True)
            t0=time.monotonic();full,trace=restore(frames,raw,CFG['A']['max_full_recomputations_per_clip']);trace['elapsed_s']=time.monotonic()-t0
            motion=read(root/'motion_quality.json');rows=read(root/'scene_ids.json')['assignments'];r=read(root/'segments_refined.json')
            mem=attempt('A',code,lambda:a_case(out,code,frames,root,raw,full,motion,rows,r))
            dump(out/'AB'/code/'coordinate_recovery.json',trace)
            if code!='N04':
                attempt('C',code,lambda:core_case(out,code,frames,meta,root,motion,{'assignments':rows},rows,raw,full,trace,source,mem))
        attempt('B',code,b)
    for item in inputs:
        c=item['id'];source=old[item['path']]['path']
        if not source:
            d['failures'].append({'stage':'B/C/P2','case':c,'trace':'Historical source unavailable'});continue
        try: frames,meta,path,motion,ids,rows=frozen_old(c)
        except Exception:
            d['failures'].append({'stage':'B/C/P2','case':c,'trace':traceback.format_exc()});continue
        def old_c():
            raw=direct_pairs(OLD/c/'pair_evidence.jsonl',lambda x:True)
            t0=time.monotonic();full,trace=restore(frames,raw,CFG['A']['max_full_recomputations_per_clip']);trace['elapsed_s']=time.monotonic()-t0
            return core_case(out,c,frames,meta,path,motion,ids,rows,raw,full,trace,source)
        attempt('C',c,old_c)
        attempt('P2',c,lambda:p2_case(out,c,frames,meta,source))
    d.update(status='predictions_complete' if not d['failures'] else 'finished_with_failures',elapsed_s=time.monotonic()-t,
             peak_working_set_mib=peak_mb(),distinct_content_count=len({x['sha256'] for x in listed}|{x['sha256'] for x in old.values()}))
    dump(manifest,d);print('Complete',[(a,len(b)) for a,b in d['completed'].items()],d['elapsed_s'],flush=True)
    if d['failures']:raise SystemExit(1)


if __name__=='__main__':main()
