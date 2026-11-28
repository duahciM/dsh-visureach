"""Run the locked four-way local anchor / competition decomposition."""

from collections import Counter
from pathlib import Path
from time import perf_counter
import traceback

import cv2

from dacf_research.field import Bank, clip_encoder, detect_groups
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_research.run import gz_dump, raw_frames
from dacf_round2.regions import discover
from evidenweft_phase1.source_review import timeline
from partial_modules_stage6.downstream import frames, nominate_arm
from partial_modules_stage6.envelopes import qualify as old_qualify
from partial_modules_stage6.predict import warm_bank
from partial_modules_stage7.dacf import qualify
from partial_modules_stage7.inputs import H
from partial_modules_stage7.local_observation import Layout, local_groups


def working_frames(name,spec,out):
    if 'stage6' in spec:
        folder=ROOT/spec['stage6']
        previous=read(folder/'auto.json')
        assert previous['sha256']==spec['sha256']
        catalogue={r['pts']:r for r in read(ROOT/previous['initial_index']/'catalog.json')['frames']}
        windows=previous['evaluated_windows']
        return frames(folder),catalogue,previous['r2_regions'],previous,windows
    assert name in ('new_recording','reverse_transfer')
    source=ROOT/spec['source']
    info,clock=timeline(source)
    chosen=[]; last=float('-inf')
    for r in clock:
        if r['pts']>spec['extent'][1]:
            break
        if r['seconds']>=last+.20-1e-7:
            chosen.append(r);last=r['seconds']
    raw_frames(source,info,clock,{r['frame'] for r in chosen},out,spec['sha256'],320)
    rows=frames(out)
    first=nominate_arm(rows,spec,'first-control-index',None)
    catalogue={}
    for group in first['nomination_groups']:
        for r in group['nominations']:
            catalogue.setdefault(r['pts'],{'pts':r['pts'],'nominations':[]})['nominations'].append(r)
    write(out/'first_control_index.json',list(catalogue.values()))
    return rows,catalogue,None,None,[spec['extent']]


def old_view(old):
    return {'accepted':old['accepted'],'closed_scope':old['appearance_scope'],
        'audit':[{'pts':r['bounds_pts'],'state':r['state'],'cause':r['reasons']} for r in old['all_candidates']],
        'R1_dropped':[[x['reason'] for x in group['rejected']] for group in old['all_adjacent_groups']]}


def audit_groups(bank,layout,pool):
    old=bank.config
    rows=[]
    for a,b in zip(pool,pool[1:]):
        lm=layout.compare(a,b)
        gray=bank.metric(a,b)['gray']
        rows.append({'pts':lm['pts'],'geometry_parents':lm['source_roots'],'measured_geometry':lm['id'],
                   'local_direct_supported':lm['supported'],'mutual_points':lm['matched_count'],
                   'gray_mad':gray,'above_old_within_gray':gray>old['automatic']['within_gray']})
    return rows


def build_feature_input(rows,spans,encoder):
    raw={r['frame']:r|{'rgb':cv2.cvtColor(r['rgb'],cv2.COLOR_BGR2RGB)}
         for r in rows if any(a<=r['pts']<=b for a,b in spans)}
    bank=Bank(raw,read(ROOT/'dacf_research/protocol.json'),encoder)
    windows=[[i for i,r in raw.items() if a<=r['pts']<=b] for a,b in spans]
    return bank,windows


def run(name,spec,out,encoder):
    out.mkdir()
    start=perf_counter()
    rows,catalogue,regions,previous,spans=working_frames(name,spec,out)
    prepare=perf_counter()-start
    start=perf_counter()
    bank,windows=build_feature_input(rows,spans,encoder)
    features=perf_counter()-start
    r2=read(ROOT/'dacf_round2/protocol.json')
    start=perf_counter()
    warm={}
    if previous and spec.get('r2_history'):
        warm=warm_bank(bank,ROOT/spec['r2_history'])
    elif name=='reverse_transfer':
        saved=ROOT/'dacf_round2/replay/reverse_doorway'
        warm=warm_bank(bank,saved)
    if regions is None:
        regions=[]
        for i,pool in enumerate(windows):
            regions.extend(x|{'id':f'{i}-{x["id"]}'} for x in discover(bank,pool,catalogue,r2)['regions'])
    init=len(bank.evidence);old_hold=len(bank.links)
    if previous is None:
        previous=old_qualify(bank,windows,catalogue,regions,r2,spec['sha256'],spec['extent'])
    r2_seconds=perf_counter()-start
    start=perf_counter();layout=Layout(bank)
    old_groups=previous['all_adjacent_groups']
    new_groups=[local_groups(layout,x,bank.config['maximum_gap_seconds'],bank.config['automatic']['dwell_seconds'],
                r2['prototype_neighborhood']['radius_s']) for x in windows]
    spatial=perf_counter()-start
    branches={'OLD':old_view(previous)};times={}
    factors={'L':(new_groups,True,False),'P':(old_groups,False,True),'LP':(new_groups,True,True)}
    for arm,(groups,local,new) in factors.items():
        start=perf_counter()
        branches[arm]=qualify(bank,layout,windows,groups,catalogue,regions,r2,spec['sha256'],spec['extent'],local,new)
        times[arm]=perf_counter()-start
    diagnostics=[audit_groups(bank,layout,w) for w in windows]
    result={'source_sha256':spec['sha256'],'native_extent':spec['extent'],'pool_pts':[r['pts'] for r in rows],
        'examined_windows_pts':spans,'first_pts':list(catalogue),'arms':branches,
        'old_groups_scope':old_groups,'development_local_groups':new_groups,'adjacent_group_diagnostic':diagnostics,
        'cost':{'shared_source_raster_or_reader_seconds':prepare,'prepared_feature_seconds':features,'r2_seconds':r2_seconds,
            'shared_group_generation_seconds':spatial,'per_branch_seconds_with_shared_comparison_cache':times,
            'total_new_layout_pairs':len(layout.pairs),'new_pair_records':len(bank.evidence)-init,
            'new_cell_holdout_records':len(bank.links)-old_hold,'features_used':len(bank.rows),'clip':bank.clip_state,
            'frozen_native_images_reused':len(rows) if spec.get('stage6') else 0,
            'new_retained_originals':len(rows) if not spec.get('stage6') else 0,'historical_bank_reuse':warm}}
    gz_dump(out/'layout_fit.json.gz',list(layout.pairs.values()))
    gz_dump(out/'raw_photometry.json.gz',list(bank.evidence.values()))
    write(out/'results.json',result)
    return {arm:{'accepted':[x['pts'] for x in branch['accepted']],'closed_count':len(branch['closed_scope']['closed']),
        'unknown':dict(Counter(r for x in branch['audit'] for r in x.get('cause',[])))} for arm,branch in branches.items()}


if __name__=='__main__':
    target=H/'automatic';target.mkdir()
    lock=read(H/'freeze.json')
    assert all(digest(ROOT/key)==value for key,value in lock['prediction'].items())
    cv2.setNumThreads(1)
    try:
        clip=clip_encoder()
    except (OSError,ImportError,RuntimeError,ValueError) as exc:
        clip=None;write(target/'clip_failure.json',{'exception':str(exc)})
    results={};failure={}
    for name,spec in read(H/'inputs/cases.json').items():
        cv2.setRNGSeed(730)
        try:
            results[name]=run(name,spec,target/name,clip)
            print(name,{arm:res['accepted'] for arm,res in results[name].items()},flush=True)
        except (KeyError,AssertionError,ValueError,RuntimeError,OSError,cv2.error):
            failure[name]=traceback.format_exc()
            print(failure[name],flush=True)
    write(target/'summary.json',{'cases':results,'failures':failure,'read_assessor':False})
