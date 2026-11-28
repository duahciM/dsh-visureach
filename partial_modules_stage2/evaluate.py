"""Verify source provenance and score frozen hypotheses, scopes, measurements and requests."""
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import ROOT,digest,read,write
from evidenweft_phase1.source_review import native_stream,timeline
from partial_modules_stage2.inputs import H


def source_audit(result_sets, source_paths):
    wanted=defaultdict(dict)
    for array in result_sets:
        for item in array:
            assert digest(Path(item['path']))==item['png_sha256']
            known=wanted[item['source_sha256']]
            known.setdefault(item['pts'],item['rgb_sha256'])
            assert known[item['pts']]==item['rgb_sha256']
    out={}
    for source in source_paths:
        hash_=digest(source); expect=wanted[hash_]; proved=set()
        info,clock=timeline(source)
        for row,rgb in native_stream(source,info,clock):
            if row['pts'] in expect:
                assert sha256(rgb).hexdigest()==expect[row['pts']]
                proved.add(row['pts'])
            if len(proved)==len(expect):break
        assert set(expect)==proved
        out[hash_]={'source':str(source),'distinct_pts':sorted(proved)}
    return out


def local_counts(run):
    inv={x['view_key']:x for views in run['view_inventory'].values() for x in views}
    pair_audit=[]
    for row in run['scoped']['relations']:
        a,b=(inv[k] for k in row['views'])
        pair_audit.append({'pts':[a['pts'],b['pts']],'episodes':[a['episode'],b['episode']],
                           'same_supplied_appearance':a['episode']==b['episode'],'visual_fit':row['analysis_fit']})
    return {'accepted_local_links':len(pair_audit),'ungated_local_links':len(run['ungated_edges']),
            'scopes':len(run['scoped']['scopes']),'unresolved_transitions':len(run['scoped']['unknown_transitions']),
            'coappearance_projections':len(run['scoped']['coappearance']),
            'valid_projected_endpoints':sum(e['axes'][ch] is not None for s in run['relations'].values() for e in s for ch in ('ref_x','peer_x')),
            'ungated_projected_endpoints':sum(e['axes'][ch] is not None for s in run['ungated_relations'].values() for e in s for ch in ('ref_x','peer_x')),
            'context_states':dict(Counter(r['state'] for r in run['context_calculations'])),
            'each_local_relation':pair_audit,'runtime':run['runtime']}


def controlled_error(run, observed):
    rows={(r['episode'],r['pts']):r for r in observed}
    values={'ref_x':[],'ref_y':[],'peer_x':[],'peer_y':[]}
    errors=[]
    def center(v):
        a,b,w,h=v['native_xywh'];return np.array([a+w/2,b+h/2])
    for episode,edges in run['relations'].items():
        for e in edges:
            a,b=e['pts']
            for role,tag in [('rail','ref'),('dock','peer')]:
                if not all((role,t) in rows for t in (a,b)):continue
                expected=(center(rows[episode,b])-center(rows[role,b]))-(center(rows[episode,a])-center(rows[role,a]))
                for i,key in enumerate(('_x','_y')):
                    axis=tag+key
                    result=e['axes'][axis]
                    if result is not None:
                        delta=abs(result-expected[i]);values[axis].append(float(delta))
                        errors.append({'axis':axis,'pts':[a,b],'abs_err_native_px':float(delta)})
    return {'axes':{k:{'n':len(v),'median_abs_native_px':median(v) if v else None,'max_abs_native_px':max(v) if v else None,
                       'within_3px':sum(a<=3 for a in v)} for k,v in values.items()},'records':errors}


def old_association_diagnostic():
    old=read(ROOT/'partial_modules_stage1/evaluation/results.json')
    out=[]
    for case in old['rows']:
        for row in case['bridge_details']:
            pair=row['episodes']
            demanded=pair in (['g0','g2'],['g2','g3'])
            out.append({'case':case['source'],'episodes':pair,'stage1_physical_mismatch':not row['requested_link_was_same_fixture_instance'],
                        'local_continuity_error':False,'cross_comparison_needed_under_fixed_test_goal':demanded,
                        'local_only_goal':'no cross-range need','phase2_default':'does not create cross-scope explanation'})
    return out


def sampling_score():
    stages=H/'sampling'; setup=read(stages/'run_lock.json')
    plan=read(H/'plan/tree.json'); timing={x['pts']:x['frame'] for x in plan['clock']}
    roles=read(H/'inputs/sealed_roles.json')['control']['relation_roles']
    initial={timing[t] for t in setup['anchors']}; nom={timing[t] for t in setup['original_nominees']}; sampled={timing[t] for t in setup['original_coarse_grid']}
    has=lambda fr,span:any(span[0]<=f<=span[1] for f in fr)
    a0={k for k,span in roles.items() if has(initial,span)}
    rows=[];receipts=[]
    for p in sorted(stages.glob('*/result.json')):
        line=read(p); req=line['requests']; assert len(req)==3 and len({r['original']['pts'] for r in req})==3
        frames=[r['original']['frame'] for r in req]; rec=[r['original'] for r in req];receipts.extend(rec)
        obtained={k for k,span in roles.items() if has(frames,span)}
        incidental=[e for series in line['revised_relation_edges_incident_to_supplement'].values() for e in series]
        per_role={}
        for role,key in [('near_rail_x_apex','ref_x'),('vertical_lane_apex','ref_y'),('local_state_marker','own_color')]:
            span=roles[role]
            per_role[role]=[edge['measurement_ref'] for edge in incidental if any(
                 t in {v['pts'] for v in rec} and span[0]<=timing[t]<=span[1] and edge['axes'][key] is not None for t in edge['pts'])]
        rows.append({'route':line['route'],'new_frame_indices':frames,'new_pts':[r['pts'] for r in rec],
            'new_roles':sorted(obtained-a0),'hit_roles_including_precovered':sorted(obtained),
            'supported_relation_receipts':per_role,'cold_exact_reads':sum(x['new_exact_decode'] for x in req),
            'cold_decode_filter_inputs':sum(a['frames_at_input_filter'] or 0 for x in req if x['new_exact_decode'] for a in x['original']['decode_attempts']),
            'bytes_of_selected_originals':sum(r['png_bytes'] for r in rec),'wall_seconds':line['runtime_seconds'],
            'used_axes':[r['used_axis'] for r in req],
            'unknown_axis_fallbacks':sum(r['used_axis']=='time_fallback' for r in req)})
    catalogue={k:{'truth_frame_window':span,'coarse_count':sum(span[0]<=f<=span[1] for f in sampled),
        'nomination_count':sum(span[0]<=f<=span[1] for f in nom),'visible_in_common_initial_bundle':k in a0}
      for k,span in roles.items()}
    return {'catalogue_roles':catalogue,'initial_seed_pts':setup['anchors'],'original_nominees':len(nom),
       'coarse_candidates':len(sampled),'arms':rows,'initial_covered_roles':sorted(a0),'local_receipts':receipts}


if __name__=='__main__':
    dest=H/'evaluation';dest.mkdir()
    runs={p.parent.name:read(p) for p in sorted((H/'scope_run').glob('*/result.json'))}
    selection=sampling_score()
    delayed=read(H/'inputs/control/observations-dense.json')
    local={key:local_counts(run) for key,run in runs.items()}
    scores={k:controlled_error(runs[k],delayed['observations']) for k in ('control','control-jitter')}
    pairs=old_association_diagnostic()
    auto=[read(H/'automatic'/k/'prediction.json')['predecode_receipts'] for k in ('control','street')]
    sources=[ROOT/'partial_modules_stage1/inputs'/name/'source.mkv' for name in ('paired','panning')]
    sources.extend([H/'inputs/control/source.mkv',H/'inputs/street/source.avi'])
    rgb=source_audit([r['receipts'] for r in runs.values()]+auto+[selection.pop('local_receipts')],sources)
    engineering=read(H/'engineering/result.json')
    assert engineering['passed'] and engineering['novel_native_tick']
    for hash_, relative in engineering['transport_files'].items():
        assert digest(H/'engineering'/relative)==hash_
    result={'scoped_comparison':local,'roi_error':scores,'first_stage_cross_scope_reaudit':pairs,'equal_originals':selection,
            'same_pixel_worlds':read(H/'inputs/sealed_roles.json')['control']['indistinguishable_worlds'],
            'scope_only_linked_by_visual_support':runs['control']['requested_cross_control'],
            'independent_source_rgb_audit':rgb,'amr_transport_hashes_verified':len(engineering['transport_files'])}
    write(dest/'result.json',result)
