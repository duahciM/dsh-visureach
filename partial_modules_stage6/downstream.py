"""Actually rerun the frozen nomination and local measurement algorithms per scope arm."""

from argparse import ArgumentParser
from collections import Counter
from copy import copy
import importlib
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from dacf_research.field import quality
from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage1.study import root
from partial_modules_stage3.observation import Media, arms_from, role_discovery, scoped_identity
from partial_modules_stage3.run import m5
from partial_modules_stage5.isolation import raw_measurements
from partial_modules_stage5.scopes import derive, context
from partial_modules_stage6.material import H

nomination=importlib.import_module('if16_research.nominate')


def reference_scope(sha,extent,bounds,wrong=False):
    selection=bounds
    if wrong and bounds:
        midpoints=[(a+b)/2 for a,b in bounds]
        distances=[b-a for a,b in zip(midpoints,midpoints[1:])]
        shift=round(.30*median(distances)) if distances else round(.30*(extent[1]-extent[0]))
        selection=[[max(extent[0],lo+shift-round(.15*(hi-lo))),
                    min(extent[1],hi+shift+round(.15*(hi-lo)))] for lo,hi in bounds]
        selection=[r for r in selection if r[0]<=r[1]]
    intervals=[{'id':root('wrong' if wrong else 'manual',sha,r),'pts':r,'source_sha256':sha,
                'kind':'observed_spatial_context_change'} for r in selection]
    return derive(sha,extent,intervals),selection


def frames(folder):
    rows=read(folder/'frame_receipts.json')
    prepared=[]
    for r in rows:
        path=folder/r['png']
        if digest(path)!=r['png_sha256']:
            raise ValueError(f'Changed shared exact original PTS {r["pts"]}')
        with Image.open(path) as source:
            rgb=np.asarray(source.convert('RGB').resize(tuple(r['analysis_size']),Image.Resampling.BILINEAR))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        prepared.append(r|{'source_frame':r['frame'],'rgb':cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),
                           'gray':gray,'t':r['seconds'], **quality(gray)})
    return prepared


def blocks(rows,scope):
    if scope is None:
        chunks={}
        for r in rows:
            chunks.setdefault(int(r['seconds']//30),[]).append(r)
        return [(f'global-window-{key}',group) for key,group in chunks.items()]
    return [(episode['id'],[r for r in rows if episode['pts'][0]<r['pts']<episode['pts'][1]])
            for episode in scope['closed']]


def nominate_arm(rows,spec,arm,scope):
    receipts=[];by_reason=Counter()
    sums=Counter(); t0=perf_counter()
    for sid,samples in blocks(rows,scope):
        if not samples:
            receipts.append({'scope_id':sid,'state':'strict_interior_has_no_coarse_sample','positions':[],'nominations':[]})
            continue
        keys=[r['id'] for r in samples]
        provenance=root('scope-input-v1',spec['sha256'],keys)
        groups=[]
        with patch.object(nomination,'register',wraps=nomination.register) as register, \
             patch.object(nomination,'pair_metrics',wraps=nomination.pair_metrics) as pair:
            output,foreground,_=nomination.nominate(samples,{'sha256':spec['sha256']})
        sums['actual_phase_registration_calls']+=register.call_count
        sums['actual_paired_diagnostic_calls']+=pair.call_count
        for dimension,rows_of_type in output.items():
            by_reason[dimension]+=len(rows_of_type)
            for row in rows_of_type:
                nominal=row['sample']; source=samples[nominal]
                input_roots=row['comparison_frame_ids']
                # Original routines name a registered neighborhood by center only.
                # A changed set of neighbors requires a new calculation reference.
                source_refs=row['evidence_root_ids']
                contextual=root('scoped-nomination-v1',provenance,row)
                groups.append(row|{'pts':source['pts'],'source_root':source['id'],'scope_id':sid,
                    'raw_nominator_roots':source_refs,'calculation_id':contextual,
                    'analysis_context_id':provenance, 'comparison_frame_ids':input_roots})
        receipts.append({'scope_id':sid,'state':'computed','positions':[r['pts'] for r in samples],
            'analysis_context_id':provenance,'exact_context_input_frame_ids':keys,
            'nominations':groups,'foreground_states':[{'pts':samples[z['i']]['pts'],'state':z['state'],
                'context_frames':[r['frame_id'] for r in z['comparisons']]} for z in foreground]})
    return {'arm':arm,'nomination_groups':receipts,'nominations_by_dimension':dict(by_reason),
        'independent_from_frozen_first':True,'raw_source_access_remains_available':True,
        'additional_source_decodes':0,'actual_operations':dict(sums),'seconds':perf_counter()-t0,
        'scope_state':'computed' if receipts else 'no_closed_context_no_supplement'}


def single_m2(media,arm,scope):
    start=perf_counter()
    partitions=[]
    for sid,stamps in blocks([{'pts':p,'seconds':media.raw_reader.receipts[p]['seconds']} for p in media.ticks],scope):
        if not stamps:
            continue
        shard=copy(media)
        shard.ticks=[r['pts'] for r in stamps]
        # Give each scope the same original rasters; recompute dependent cache only.
        shard.chart_cache={}
        shard.feature_pairs={}
        work,diag,_,_=role_discovery(shard,[])
        choices=arms_from(work)
        identity=scoped_identity(shard,choices['A3'],'A3')
        with_id=m5(shard,choices['A3-no-role'],identity)
        raw=m5(shard,choices['A3-no-role'],{'links':[]})
        assert with_id['same_tick_projected']==raw['same_tick_projected']
        assert raw_measurements(with_id)==raw_measurements(raw)
        partitions.append({'scope_id':sid,'pts':shard.ticks, 'identity_work_and_competition':identity,
           'm5_with_identity':with_id,'raw_m5_without_identity':raw,'recomputed_background_charts':list(shard.chart_cache.values()),
           'actual_feature_pair_builds':len(shard.feature_pairs),'registered_diagnostics':diag})
    return {'arm':arm,'partitions':partitions,'seconds':perf_counter()-start,
        'source_decode_reused':True,'strict_contextual_candidates_do_not_replace_global_source_evidence':True}


def write_arm(target,row):
    if target.is_file():
        raise FileExistsError(target)
    write(target,row)


def run_case(name,spec,stage,results,reference=None):
    base=H/'automatic'/name
    source=ROOT/spec['source']
    prediction=read(base/'auto.json')
    target=results/name;target.mkdir()
    index=ROOT/prediction['initial_index']
    def index_signature():
        return {k:digest(index/k) for k in prediction['index_sha']}
    assert index_signature()==prediction['index_sha'] and digest(source)==spec['sha256']
    start=perf_counter()
    rows=frames(base)
    raster_s=perf_counter()-start
    scopes={'GLOBAL':None,'AUTO':prediction['appearance_scope']}
    bounds={}
    if stage=='reference':
        scopes={}
        for arm in ('MANUAL','WRONG'):
            scopes[arm],bounds[arm]=reference_scope(spec['sha256'],spec['extent'],reference['bounds'],wrong=arm=='WRONG')
    local_cost={}
    # Two sources have distinct independently labeled positive endpoints; only those run the local reanalysis.
    if name in ('reel','real_return'):
        start=perf_counter()
        media=Media({'path':spec['source'],'sha256':spec['sha256'],'pts':prediction['pool_pts']},target)
        inputs=list(media.raw_reader.receipts.values())
        local_cost={
            'measured_once_seconds':perf_counter()-start,
            'new_native_reads_for_this_batch':len(inputs),
            'filter_frames':sum(a['frames_at_input_filter'] or 0 for r in inputs for a in r['decode_attempts'])}
        write(target/'local_source_replay.json',inputs)
        for arm,index_scope in scopes.items():
            local=single_m2(media,arm,index_scope)
            write_arm(target/f'{arm}-local.json',local)
    for arm,index_scope in scopes.items():
        written=nominate_arm(rows,spec,arm,index_scope)
        write_arm(target/f'{arm}.json',written|{'scope':index_scope,'input_hashes':prediction['index_sha'],
            'source_sha256':spec['sha256'],'reference_envelopes':bounds.get(arm)})
    assert index_signature()==prediction['index_sha']
    write(target/'run.json',{'coarse_raster_reuse_seconds':raster_s,'original_group_preparation':local_cost,
           'first_digest_unchanged':prediction['index_sha'],'first_count':len(prediction['first_pts']),
           'arms':list(scopes),'stage':stage})
    print(stage,name,'completed',flush=True)


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--stage',choices=('automatic','reference'),required=True);args=parser.parse_args()
    lock=read(H/'freeze.json')
    assert all(digest(ROOT/key)==sha for key,sha in lock['predictors'].items())
    downstream=read(H/'downstream_lock.json')
    assert digest(Path(__file__))==downstream['code_sha256']
    assert all(digest(H/'automatic'/name/'auto.json')==sig for name,sig in downstream['automatic_outputs'].items())
    if args.stage=='automatic':
        reference=None
    else:
        assert (H/'observations/automatic/run_complete.json').is_file(), 'Run the blind conditions first.'
        path=ROOT/lock['assessor']['file']
        assert digest(path)==lock['assessor']['sha256']
        reference=read(path)
    results=H/'observations'/args.stage;results.mkdir(parents=True)
    cv2.setNumThreads(1)
    cases=read(H/'inputs/cases.json')
    usable={r['case'] for r in read(H/'automatic/summary.json')['results']}
    for name,spec in cases.items():
        if name not in usable:
            continue
        cv2.setRNGSeed(730)
        run_case(name,spec,args.stage,results,reference[name] if reference else None)
    write(results/'run_complete.json',{'cases':sorted(usable),'read_assessor':reference is not None})
