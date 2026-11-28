"""Replay the existing local candidate and M5 mechanisms with read-only scene metadata."""

from collections import Counter
from time import perf_counter

import cv2

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage3.observation import Media, arms_from, role_discovery, scoped_identity
from partial_modules_stage3.run import m5
from partial_modules_stage5.isolation import raw_measurements
from partial_modules_stage5.review_sources import H
from partial_modules_stage5.scopes import context, priority


def compare_case(case, info, out):
    actual=read(H/'source_review'/case/'receipts.json')
    if case=='camera':
        spec=read(ROOT/'partial_modules_stage4/inputs/source_inventory.json')['camera']
    elif case=='real_return':
        spec={'path':actual['source'],'sha256':actual['sha256'],
              'pts':read(H/'real_round2/regions.json')['coarse_pts'],'explicit_goal_queries':[]}
    else:
        prior=read(ROOT/f'dacf_round3/run/{case}/initial.json')
        spec={'path':actual['source'],'sha256':actual['sha256'],
              'pts':prior['coarse_pts'],'explicit_goal_queries':[]}
    folder=out/case;folder.mkdir()
    start=perf_counter()
    media=Media(spec,folder)
    time_prep=perf_counter()-start
    start=perf_counter()
    discovered,diag,_,_=role_discovery(media,spec['explicit_goal_queries'])
    arms=arms_from(discovered)
    # Same independent evidence for all workload arms. Identity uses the current
    # old eligible set, so a role change never alters this comparison's input.
    evidence=[r for t in media.ticks for r in arms['A3-no-role'][t]]
    identity=scoped_identity(media,arms['A3'],'A3')
    base=m5(media,arms['A3-no-role'],identity)
    without=m5(media,arms['A3-no-role'],{'links':[]})
    assert raw_measurements(base)==raw_measurements(without)
    assert base['same_tick_projected']==without['same_tick_projected']
    proof=info['reviewed_scope']
    plans=[]
    for pair in identity['pair_audit']:
        a,b=pair['pts']
        plans.append({'views':pair['views'],'pts':pair['pts'],'source_roots':pair['source_roots'],
                      'source_reference':pair['calculation_root'],
                      'ordinary_priority':'existing_adjacent_work',
                      'scene_priority':priority(a,b,proof),'eligible_for_explicit_query':True,
                      'comparison_outcome':pair['local']})
    scene_contexts={str(t):context(t,proof) for t in media.ticks}
    adjustments=sum(a['interpretation_scope']!=b['interpretation_scope'] for a,b in zip(base['raw_cross_time'],without['raw_cross_time']))
    records=list(media.raw_reader.receipts.values())
    commands={'local_current':{'attempts_by_reason':identity['attempts_by_reason'],'local_jobs_executed':len(plans)},
        'reviewed_scene_annotation':{'local_jobs_shared':len(plans),'changed_priority':dict(Counter(p['scene_priority'] for p in plans)),
            'bytes_and_measurements_retained':True,'priority_not_consumed_by_frozen_M2':True},
        'independent_without_identity':{'local_jobs_executed':0,'proposals_retained':len(evidence),
            'identity_dependent_results_kept_as_conditional':adjustments}}
    findings={'real_source_sha256':media.sha,'fixed_nomination_pts':info['first_nomination_pts'],
        'local_source_pts':media.ticks,'context_by_pts':scene_contexts,'roles_and_candidate_rows':evidence,
        'actual_local_jobs':plans,'identity':identity,'independent_raw_m5':without,
        'm5_with_local_interpretation':base,'retained_across_all_conditions':{
            'single_time_measurements':len(base['same_tick_projected']),'two_time_raw_rows':len(base['raw_cross_time']),
            'boundaries_and_unknown_source_unrestricted':True},'arms':commands,'input_receipts':records,
        'measured_cost':{'exact_native_frame_count':len(records),'native_prepare_seconds':time_prep,
            'shared_replay_seconds':perf_counter()-start,
            'decode_filter_input_count':sum(d['frames_at_input_filter'] or 0 for r in records for d in r['decode_attempts'])}}
    write(folder/'replay.json',findings)
    return {k:findings[k] for k in ('retained_across_all_conditions','arms','measured_cost')}


if __name__=='__main__':
    home=H/'local_replay';home.mkdir()
    scopes=read(H/'scene_scope.json')
    lock=read(H/'freeze.json')
    manifest=H/'real_round2/index/manifest.json'
    catalog=H/'real_round2/index/catalog.json'
    before={p.name:digest(p) for p in (manifest,catalog)}
    assert before==read(H/'real_round2/regions.json')['frozen_catalogue_and_pool']
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    scores={k:compare_case(k,v,home) for k,v in scopes['cases'].items()}
    assert before=={p.name:digest(p) for p in (manifest,catalog)}
    write(home/'summary.json',{'cases':scores,'fixed_catalogue':before,'freeze_history_count':lock['baseline_count']})
