"""Recompute A/B/C/D/E and equal-volume window variants on the same exact rasters."""

from argparse import ArgumentParser
import gzip
import json
from time import perf_counter

import cv2

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage6.downstream import frames, reference_scope
from partial_modules_stage7.inputs import H
from partial_modules_stage7.windowed import Foreground, adjacent_reasons, arm, stable_lookup


def by_time(run):
    return {int(k):v for k,v in run['independent_schedule_per_pts'].items()}


def historical_wrong(source,run):
    baseline=read(ROOT/f'partial_modules_stage6/observations/reference/{source}/WRONG.json')
    old={r['pts']:r['context_frames'] for g in baseline['nomination_groups'] for r in g['foreground_states']}
    new={r['pts']:[e['root'] for e in r['fit_refs']] for r in run['measurements']}
    assert old==new,(source,'different historical registered-neighborhood support')
    prior={(r['pts'],tuple(r['applicable_region']),r['dimension_id'])
           for g in baseline['nomination_groups'] for r in g['nominations'] if r['dimension_id'] in ('D1','D3')}
    replay={(r['pts'],tuple(r['applicable_region']),r['dimension_id']) for r in run['nominations'] if r['dimension_id'] in ('D1','D3')}
    assert prior==replay,(source,'different observed candidate geometry')
    return {'same_historical_supported_refs':len(old),'same_historical_D1_D3_keys':len(prior)}


def run(source,spec,out,stage,reference):
    out.mkdir()
    index=ROOT/spec['stage6'] if 'stage6' in spec else H/'automatic'/source
    t0=perf_counter();rows=frames(index); worker=Foreground(rows)
    before=perf_counter()-t0
    scope_result=read(H/'automatic'/source/'results.json')
    primary=scope_result['arms']['LP']['closed_scope']
    with gzip.open(H/'automatic'/source/'layout_fit.json.gz','rt',encoding='utf8') as stream:
        geometry=stable_lookup(json.load(stream))
    first=None;wrong=None
    if reference is not None:
        first,_=reference_scope(spec['sha256'],spec['extent'],reference['bounds'])
        wrong,_=reference_scope(spec['sha256'],spec['extent'],reference['bounds'],wrong=True)
    config=(
        {'A':(None,'old',False),'B':(None,'forward',False),'B_back':(None,'backward',False),
         'B_wide':(None,'wide',False),'B_split':(None,'split',False),'D0':(None,'forward',True),
         'E':(primary,'forward',True)}
        if stage=='blind' else {'C':(first,'old',False),'D':(first,'forward',True),'WRONG':(wrong,'old',False)})
    checks={};common_pairs={}
    counts={}
    for name,(scope,choice,expansion) in config.items():
        t0=perf_counter()
        report=arm(worker,rows,scope,choice,expansion,geometry)
        # Pair facts are common in equal-window comparisons with the same bounds.
        # Keep them out of per-window marginal costs.
        places=scope['closed'] if scope else 'none'
        group_key=name if name=='WRONG' else 'FULL' if scope is None else repr(places)
        if group_key not in common_pairs:
            begin=perf_counter()
            p,n=adjacent_reasons(worker,rows,scope)
            common_pairs[group_key]={'rows':p,'actual_pair_calls':n,'seconds':perf_counter()-begin}
        if name=='WRONG' and 'stage6' in spec and source in ('reel','real_return'):
            checks['WRONG']=historical_wrong(source,report)
        if name=='A':
            base=by_time(report)
        if name in ('B','B_back','B_wide','B_split'):
            q=by_time(report)
            assert all(base[ts]==k for ts,k in q.items())
            assert (name=='B_split' or q==base)
            checks[name]={'same_scheduled_comparisons_for_all_shared_targets':len(q),
                'executed_calls':report['executed_phase_calls'],'A_equivalent_execution_for_those_targets':sum(
                   k for ts,k in q.items() if worker.frames[ts]['sharpness']>=8)}
            assert checks[name]['executed_calls']==checks[name]['A_equivalent_execution_for_those_targets']
        counts[name]={'count':len(report['nominations']),'executed_phase_calls':report['executed_phase_calls'],'seconds_with_new_common_pair_only_once':perf_counter()-t0}
        write(out/f'{name}.json',report|{'condition':name,'scope':scope,'pair_channel':group_key,'input_source_sha256':spec['sha256']})
    write(out/'common_pair_facts.json',common_pairs)
    write(out/'summary.json',{'arms':counts,'checks':checks,'shared_raster_seconds':before,
          'source_additional_decode_calls':0,'source_sha256':spec['sha256'],'target_PTS':[r['pts'] for r in rows]})


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--stage',choices=('blind','reference'),required=True)
    args=parser.parse_args()
    const=read(H/'window_lock.json')
    assert all(digest(H/key)==v for key,v in const['source_code'].items())
    assert all(digest(H/'automatic'/case/'results.json')==v for case,v in const['automatic'].items())
    if args.stage=='reference':
        assert (H/'windows/blind/completed.json').is_file()
        lock=read(H/'freeze.json')
        assert all(digest(ROOT/key)==v for key,v in lock['assessor_files'].items())
        truth=read(ROOT/'partial_modules_stage6/assessment/sealed.json')
        truth['new_recording']=read(H/'assessment_extra.json')['new_recording']
    else:
        truth=None
    out=H/'windows'/args.stage;out.mkdir(parents=True)
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    specs=read(H/'inputs/cases.json')
    for name in ('reel','real_return','new_recording'):
        run(name,specs[name],out/name,args.stage,truth[name] if truth else None)
        print(args.stage,name,flush=True)
    write(out/'completed.json',{'assessor_opened':truth is not None})
