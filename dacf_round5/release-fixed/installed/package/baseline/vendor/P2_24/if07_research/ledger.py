"""Restore audited complete geometry and emit immutable independent observations."""
from collections import Counter
import json

import cv2
from p14_scene_prototypes.features import sift_pair
from v05.engine import FEATURE_CFG

METRICS=('distance','verified','query_keypoints','reference_keypoints','ratio_forward_count','ratio_backward_count',
         'mutual_count','inlier_count','inlier_fraction','query_hull_coverage','reference_hull_coverage','reason','homography')


def restore(frames, raw, budget):
    restored={}; trace=[]; counter=Counter()
    sift=cv2.SIFT_create(nfeatures=FEATURE_CFG['sift']['nfeatures'])
    desc={}
    def get(k):
        if k not in desc: desc[k]=sift.detectAndCompute(frames[k]['gray'],None)
        return desc[k]
    used=0
    for k,p in raw.items():
        o=p['sift']; preview=o.get('inlier_pairs',[]); actual=o.get('inlier_count')
        status='old_computation_failed' if actual is None else 'complete_reused'
        found=o
        if actual is not None and actual!=len(preview):
            status='nonverified_distribution_not_applicable'
            if o['verified']:
                if used>=budget:
                    status='budget_unassessed'
                else:
                    used+=1
                    try:
                        i,j=k
                        full=sift_pair(get(i),get(j),frames[i]['gray'].shape,frames[j]['gray'].shape,FEATURE_CFG['sift'],f'{i}:{j}')
                        if not all(full.get(f)==o.get(f) for f in METRICS) or full['inlier_pairs'][:len(preview)]!=preview:
                            status='recomputation_inconsistent'
                        else:
                            found=full
                            status='recomputed_verified_complete'
                    except Exception as exc:
                        trace.append({'pair':p['id'],'detail':repr(exc)})
                        status='recomputation_error'
        if found is o and actual!=len(preview):
            found={**o,'inlier_pairs':[]}
        restored[k]={**p,'sift':found,'distribution_state':status,'preview_count':len(preview)}
        counter[status]+=1
        if status not in ('complete_reused','nonverified_distribution_not_applicable'):
            trace.append({'pair':p['id'],'status':status,'old_count':len(preview),'expected_count':actual,
                          'full_count':len(found.get('inlier_pairs',[]))})
    return restored,{'counts':dict(counter),'attempts':used,'budget':budget,'nonstandard_geometry':trace}


def stamp(f):
    return {k:f[k] for k in ('sample','source_frame','pts','time_base','t','image')}


def emit_ledger(dst, frames, old, complete, motion, refined, old_memory, code, cfg):
    ids=[stamp(f) for f in frames]; counts=Counter(); budgeted=set()
    cutoff=cfg['B']['minimum_sharpness']; cap=cfg['B']['old_hsv_prefilter']
    with dst.open('w',encoding='utf8') as out:
        def put(r): out.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        for (i,j),p in complete.items():
            raw=old[i,j]; s=p['sift']; k=p['id']; state=p['distribution_state']
            clear=min(frames[i]['sharpness'],frames[j]['sharpness'])>=cutoff
            status='supported' if s['verified'] else 'not_supported'
            if state=='old_computation_failed':status='not_evaluable'
            if s['verified'] and state not in ('complete_reused','recomputed_verified_complete'):budgeted.add((i,j))
            if s['verified'] and raw['hsv_distance']>cap and clear:counts['restored_geometry_outside_hsv_prefilter']+=1
            if not clear:counts['low_quality_relation_not_eligible']+=1
            for claim,channel,value,unit,st in [('local_correspondence','SIFT',s,'pixels and dimensionless geometry',status),
                ('color_distance','HSV',p['hsv_distance'],'Bhattacharyya distance','observed'),
                ('pixel_difference','gray',p['gray_mad'],'fractional mean absolute difference','observed')]:
                put({'record_id':f'{k}:{channel}','claim_type':claim,'source_channel':channel,'subject_ids':[i,j],
                      'sources':[ids[i],ids[j]],'support_state':st,'measurement':value,'unit':unit,
                      'quality_eligible':clear,'distribution_state':state if channel=='SIFT' else None,
                      'coordinates_are_complete':state in ('complete_reused','recomputed_verified_complete') if channel=='SIFT' else None,
                      'old_preview_count':p['preview_count'] if channel=='SIFT' else None,
                      'old_hsv_prefilter_pass':bool(raw['hsv_distance']<=cap) if channel=='SIFT' else None,
                      'old_composite_support':p['supported'],'dependency_ids':[],'locked_algorithm':code,
                      'reason':s.get('reason') if channel=='SIFT' else None})
        for q in motion['pair_metrics'][1:]:
            i=q['sample']; k=f'pair:{i-1}:{i}'; prior=old_memory.get(i)
            if prior and prior['reason'] in ('accepted_transition_bracket','two_different_direct_contexts') and q.get('motion_supported'):
                counts['restored_registered_motion_observations_behind_legacy_guard']+=1
            for channel,measurement in [('registered_motion',{k:v for k,v in q.items() if k!='brightness'}),('photometry',q.get('brightness',{}))]:
                put({'record_id':f'{k}:{channel}','claim_type':channel,'source_channel':channel,
                     'subject_ids':[i-1,i],'sources':[ids[i-1],ids[i]],'support_state':('supported' if
                       (q.get('motion_supported') if channel=='registered_motion' else measurement.get('supported')) else 'not_supported'),
                     'measurement':measurement,'unit':'pixels and fractional residual','dependency_ids':[] if channel=='registered_motion' else [f'{k}:registered_motion'],
                     'old_legacy_motion_guard':prior,'locked_algorithm':code})
        for b in refined.get('diagnostics',[]):
            i,j=b['left_sample'],b['right_sample']
            put({'record_id':f"legacy:{b['boundary_id']}",'claim_type':'historical_transition_candidate','source_channel':'frozen_timing',
                 'subject_ids':[i,j],'sources':[ids[i],ids[j]],'support_state':'inherited_candidate',
                 'measurement':b,'dependency_ids':[f'pair:{k-1}:{k}:registered_motion' for k in range(i+1,j+1)],
                 'dependency_note':'timing also depends on the frozen appearance classifications; neither an independent vote nor a geometric veto',
                 'locked_algorithm':code})
    return {'counts':dict(counts),'distribution_unavailable_for_verified_pairs':sorted(budgeted),
            'pair_count':len(complete),'root_identity':'Each pair and channel has one record; observation direction does not create a new record.'}
