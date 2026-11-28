"""Run the rectangular region / role factorial with Stage 3 scene and identity code."""

from argparse import ArgumentParser
from time import perf_counter

import cv2

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage3.observation import Media, eligible, role_discovery, scoped_identity
from partial_modules_stage3.run import m5
from partial_modules_stage4.prepare import H
from partial_modules_stage4.regions import discover_extra, policy


def selected_ids(rows):
    return sorted(r['id'] for r in rows)


def run_case(case, spec, out):
    out.mkdir()
    start=perf_counter()
    media=Media(spec,out)
    preparation=perf_counter()-start
    start=perf_counter()
    old,diagnostics,_,cost=role_discovery(media,spec['explicit_goal_queries'])
    legacy_time=perf_counter()-start
    start=perf_counter()
    extra,geometry=discover_extra(media,old)
    formation_time=perf_counter()-start
    pools,arms=policy(old,extra)
    start=perf_counter()
    frozen_identity=scoped_identity(media,arms['R0P0'],'A3')
    frozen_time=perf_counter()-start
    for arm,inventory in arms.items():
        start=perf_counter()
        actual=frozen_identity if arm=='R0P0' else scoped_identity(media,inventory,'A3')
        called_time=frozen_time if arm=='R0P0' else perf_counter()-start
        actual_rows=[r for rows in inventory.values() for r in rows]
        start=perf_counter()
        relations=m5(media,inventory,actual)
        fixed=m5(media,inventory,frozen_identity)
        retained=[r for r in actual_rows if r['id'] in frozen_identity['identifiers']]
        write(out/f'{arm}.json',{'arm':arm,'source_sha256':media.sha,'regions':actual_rows,
            'identity_follows_role':actual,'same_identity_for_all_conditions':{
                'arm':'R0P0','eligible_visible':selected_ids(retained),'eligible_original_count':len(frozen_identity['identifiers'])},
            'm5':relations,'m5_fixed_identity':fixed,'seconds_identity':called_time,
            'seconds_relations_both':perf_counter()-start})
    for arm in ('R0P0','R1P1'):
        begin=perf_counter()
        v=scoped_identity(media,arms[arm],'A3-no-flow')
        write(out/f'{arm}-no-flow.json',{'identity':v,'seconds_identity':perf_counter()-begin})
    for arm,inventory in pools.items():
        write(out/f'{arm}-ledger.json',{'candidate_rows':inventory,
            'frozen_tuple_by_pts':{t:selected_ids(rows) for t,rows in inventory.items()}})
    for t,item in extra.items():
        item['lines']=[line.tolist() for line in item['lines']]
    write(out/'region_features.json',{'formation_inputs':extra,'extra_geometry':geometry,
          'legacy_regions':old,'compensation':diagnostics,'charts':list(media.chart_cache.values())})
    write(out/'base_receipts.json',{'source_sha256':media.sha,'native_receipts':list(media.raw_reader.receipts.values()),
          'seconds_shared_preparation':preparation,'seconds_legacy_proposals':legacy_time,'seconds_new_formation':formation_time,
          'legacy_timing':cost,'fixed_identity_eligible_by_pts':{t:selected_ids(eligible('A3',rs)) for t,rs in arms['R0P0'].items()}})
    print(case,len(media.ticks),len(geometry['feature_patches']),len(geometry['stable_edges']),flush=True)


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--out',default='predictions')
    args=parser.parse_args();out=H/args.out
    if out.parent!=H:
        raise ValueError('Use a distinct research output directory.')
    lock=read(H/'freeze.json')
    assert all(digest(ROOT/file)==sha for file,sha in lock['predictor_and_inputs'].items())
    out.mkdir()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(730)
    data=read(H/'inputs/source_inventory.json')
    for key,value in data.items():
        run_case(key,value,out/key)
    write(out/'completed.json',{'cases':list(data),'input_hash':digest(H/'inputs/source_inventory.json')})
