"""Direct-edge tests and temporal sampling. No reference annotations imported."""
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from v07.regional import orient,homography_from,cell
from .common import read

CFG=read(Path(__file__).parent/'protocol.json')


def frame_ref(frames,i):
    return None if i is None else {k:frames[i][k] for k in ('sample','source_frame','pts','pts_time','t','time_base')}


def select_triplet(frames,center,offset):
    t=frames[center]['t'];chosen=[];errors=[];reasons=[]
    for direction in (-1,1):
        target=t+direction*offset
        pool=[i for i,f in enumerate(frames) if (f['t']<t if direction<0 else f['t']>t)]
        best=min(pool,key=lambda i:(round(abs(frames[i]['t']-target),9),frames[i]['t'])) if pool else None
        error=abs(frames[best]['t']-target) if best is not None else None
        if best is None:reason='no_source_frame_on_this_side'
        elif error>CFG['time_tolerance_s']+1e-9:reason='nearest_source_time_outside_tolerance';best=None
        else:reason=None
        chosen.append(best);errors.append(error);reasons.append(reason)
    samples=[chosen[0],center,chosen[1]]
    valid=all(i is not None for i in samples)
    if valid:
        if len({frames[i]['source_frame'] for i in samples})<3:valid=False;reasons.append('duplicate_source_frame')
        if not frames[samples[0]]['t']<t<frames[samples[2]]['t']:valid=False;reasons.append('unordered_source_pts')
    return {'center_sample':center,'offset_s':offset,'samples':samples,'source_refs':[frame_ref(frames,i) for i in samples],
            'target_times':[t-offset,t,t+offset],'side_time_errors_s':errors,
            'actual_intervals_s':[t-frames[chosen[0]]['t'] if chosen[0] is not None else None,frames[chosen[1]]['t']-t if chosen[1] is not None else None],
            'evaluable':valid,'unavailable_reasons':[r for r in reasons if r]}


def direct_edge(a,b,frames,pairs,matrices):
    out={'samples':[a,b],'source_refs':[frame_ref(frames,a),frame_ref(frames,b)],'support':None,'pair_id':None}
    if a is None or b is None:out['reason']='missing_temporal_neighbor';return out
    if a==b or frames[a]['source_frame']==frames[b]['source_frame']:out['reason']='self_or_duplicate_source_frame_not_evidence';return out
    key=tuple(sorted((a,b)));old=pairs.get(key)
    if old is None:out['reason']='direct_pair_not_evaluated';return out
    hsv,gray,sift,support=matrices
    if bool(old['supported'])!=bool(support[a,b]):raise ValueError('Frozen pair/matrix direct-support disagreement')
    supported=bool(support[a,b]);s=old['sift'];reasons=[]
    if not supported:
        if not s['verified']:reasons.append('SIFT:'+s['reason'])
        elif old['hsv_distance']>.5:reasons.append('verified_SIFT_but_HSV_above_0.5')
        if old['hsv_distance']>.12:reasons.append('tight_HSV_cap_not_met')
        if old['gray_mad']>.035:reasons.append('tight_gray_cap_not_met')
    out.update(pair_id=old['id'],support=supported,reason='frozen_direct_positive_support' if supported else 'no_positive_support; '+','.join(reasons),
        hsv_distance=old['hsv_distance'],gray_mad=old['gray_mad'],sift_verified=s['verified'],sift_reason=s['reason'],
        sift_inliers=s['inlier_count'],sift_coverage_each=[s['query_hull_coverage'],s['reference_hull_coverage']],
        near_duplicate=bool(hsv[a,b]<=.04 and gray[a,b]<=.02),
        low_sharpness_samples=[i for i in (a,b) if frames[i]['sharpness']<8.],
        identity_relation='unknown; missing positive edge is not a different-scene label')
    return out


def classify_triplet(edges,evaluable=True):
    if not evaluable:return 'unavailable'
    a,b,c=[edges[k]['support'] for k in ('12','23','13')]
    if a is True and b is True and c is True:return 'closed'
    if a is True and b is True and c is not True:return 'open_bridge'
    if sum(x is True for x in (a,b))==1:return 'single_side'
    return 'insufficient'


def build_triplets(frames,pairs,matrices,old_cores):
    records=[]
    for i in range(len(frames)):
        for offset in CFG['offsets_s']:
            r=select_triplet(frames,i,offset);a,b,c=r['samples']
            edges={name:direct_edge(x,y,frames,pairs,matrices) for name,x,y in [('12',a,b),('23',b,c),('13',a,c)]}
            r.update(triplet_id=f't{i}:s{offset}',edges=edges,state=classify_triplet(edges,r['evaluable']))
            records.append(r)
    old_refs={i for g in old_cores['groups'] for i in g['frozen_core_samples']}
    old_candidate_refs={i for c in old_cores['candidate_cores'] for i in c['refs']}
    eligibility=[]
    for i,f in enumerate(frames):
        rs=[r for r in records if r['center_sample']==i];closed=[r['offset_s'] for r in rs if r['state']=='closed'];bridges=[r['offset_s'] for r in rs if r['state']=='open_bridge']
        eligibility.append({**frame_ref(frames,i),'eligible':bool(closed),'closed_scales':closed,'open_bridge_scales':bridges,
            'evaluable_scales':[r['offset_s'] for r in rs if r['evaluable']],'unavailable_scales':[r['offset_s'] for r in rs if not r['evaluable']],
            'scale_states':{str(r['offset_s']):r['state'] for r in rs},'mixed_scale_conflict':bool(closed and bridges),
            'old_frozen_core':i in old_refs,'original_candidate_reference':i in old_candidate_refs,'still_member_and_keyframe_candidate':True,
            'reason':'at_least_one_scale_three_direct_edges' if closed else ('no_evaluable_scale' if not any(r['evaluable'] for r in rs) else 'no_scale_with_three_direct_edges'),
            'not_a_mixed_or_different_scene_label':True})
    return records,eligibility


def local_loop(record,pairs,descriptors,shape):
    a,b,c=record['samples'];base={'triplet_id':record['triplet_id'],'center_sample':b,'offset_s':record['offset_s'],'global_state':record['state']}
    if not record['evaluable']:return {**base,'available':False,'missing_reasons':record['unavailable_reasons'],'shared_center_features':0,'exact_closed_center_features':0,'feature_loops':[],'cells':[]}
    pe={name:pairs.get(tuple(sorted((x,y)))) for name,x,y in [('12',a,b),('23',b,c),('13',a,c)]}
    missing=[f'{name}:'+('missing_full_coordinate_record' if p is None else 'full_coordinates_or_feature_ID_not_reproducible' if not p.get('metric_agreement') or not p.get('feature_ids_valid',True) else 'no_verified_local_geometry') for name,p in pe.items() if p is None or not p.get('usable_regional_geometry') or not p.get('feature_ids_valid',True)]
    left=orient(pe['12'],b) if pe['12'] and pe['12'].get('feature_ids_valid',True) else []
    right=orient(pe['23'],b) if pe['23'] and pe['23'].get('feature_ids_valid',True) else []
    endpoint=orient(pe['13'],a) if pe['13'] and pe['13'].get('feature_ids_valid',True) else []
    lm=defaultdict(list);rm=defaultdict(list);em=defaultdict(list)
    for m in left:lm[m['query_keypoint_id']].append(m)
    for m in right:rm[m['query_keypoint_id']].append(m)
    for m in endpoint:em[(m['query_keypoint_id'],m['reference_keypoint_id'])].append(m)
    h=homography_from(pe['13'],a) if pe['13'] else None;loops=[];residuals=[]
    for center_id in sorted(set(lm)&set(rm)):
        exact=[];checks=[]
        for l in lm[center_id]:
            for r in rm[center_id]:
                matched=em.get((l['reference_keypoint_id'],r['reference_keypoint_id']),[])
                exact.extend({'left_frame_keypoint_id':l['reference_keypoint_id'],'right_frame_keypoint_id':r['reference_keypoint_id'],'direct_13_match_ids':[m['match_id'] for m in matched],
                    'edge12_match_id':l['match_id'],'edge23_match_id':r['match_id']} for _ in [0] if matched)
                if h is not None:
                    p=h@np.array(l['reference_xy']+[1.])
                    if abs(p[2])>1e-9:
                        error=float(np.linalg.norm(p[:2]/p[2]-np.asarray(r['reference_xy'])));residuals.append(error);checks.append(error)
        xy=list(descriptors[b][0][center_id].pt)
        loops.append({'center_feature_ref':f'frame:{b}:kp:{center_id}','center_keypoint_id':center_id,'center_xy':xy,'cell':cell(xy,shape),
            'exact_three_edge_closed':bool(exact),'exact_loop_evidence':exact,'composition_residuals_px':checks,
            'left_feature_matches':[{'frame':a,'keypoint_id':m['reference_keypoint_id'],'xy':m['reference_xy'],'edge_match_id':m['match_id']} for m in lm[center_id]],
            'right_feature_matches':[{'frame':c,'keypoint_id':m['reference_keypoint_id'],'xy':m['reference_xy'],'edge_match_id':m['match_id']} for m in rm[center_id]],
            'reason':'actual_endpoint_feature_pair_present_on13' if exact else 'no_matching_endpoint_feature_pair_on_direct13; same region/homography is not same feature loop'})
    cells=[]
    for n in range(16):
        shared=[l for l in loops if l['cell']==n];effective={tuple(int(round(x)) for x in k.pt) for k in descriptors[b][0] if cell(k.pt,shape)==n}
        cells.append({'cell':n,'effective_center_sites':len(effective),'shared_center_keypoints':len(shared),'exact_closed_keypoints':sum(l['exact_three_edge_closed'] for l in shared),
            'unknown_reason':None if shared else ('low_texture_no_detected_features' if not effective else 'no_two_side_verified_same_center_feature')})
    return {**base,'available':not missing,'missing_reasons':missing,'shared_center_features':len(loops),'exact_closed_center_features':sum(l['exact_three_edge_closed'] for l in loops),
        'geometry_composition_checks':len(residuals),'geometry_composition_consistent':sum(v<=CFG['geometry_diagnostic_tolerance_px'] for v in residuals),
        'geometry_composition_median_px':float(np.median(residuals)) if residuals else None,'feature_loops':loops,'cells':cells,
        'interpretation':'frame-qualified exact feature loops and geometric composition are distinct diagnostics; neither is semantic identity probability; not used by C'}
