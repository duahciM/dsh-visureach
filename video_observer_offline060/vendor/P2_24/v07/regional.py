"""Spatial correspondence evidence only. No scene truth or automatic merge."""
from collections import defaultdict,deque
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from p14_scene_prototypes.features import sift_pair,hull_coverage
from .common import PROJECT,read

CFG=read(Path(__file__).parent/'protocol_B.json')
SIFT_CFG=read(PROJECT/'p14_scene_prototypes/protocol.json')['sift']


def cell(point,shape):
    h,w=shape;cols,rows=CFG['grid']
    return min(rows-1,max(0,int(point[1]*rows/h)))*cols+min(cols-1,max(0,int(point[0]*cols/w)))


def site(point):return (int(round(point[0])),int(round(point[1])))


def reciprocal_ids(query,reference):
    qkp,qd=query;rkp,rd=reference
    if qd is None or rd is None or min(len(qd),len(rd))<2:return {}
    matcher=cv2.BFMatcher(cv2.NORM_L2);forward=matcher.knnMatch(qd,rd,k=2);backward=matcher.knnMatch(rd,qd,k=2)
    f={(p[0].queryIdx,p[0].trainIdx) for p in forward if len(p)==2 and p[0].distance<SIFT_CFG['ratio_test']*p[1].distance}
    b={(p[0].trainIdx,p[0].queryIdx) for p in backward if len(p)==2 and p[0].distance<SIFT_CFG['ratio_test']*p[1].distance}
    result=defaultdict(deque)
    for i,j in sorted(f&b):
        key=tuple(np.float32(qkp[i].pt).tolist()+np.float32(rkp[j].pt).tolist())
        result[key].append((i,j))
    return result


def metrics_agree(old,new):
    errors=[]
    for key in ('verified','reason','query_keypoints','reference_keypoints','ratio_forward_count','ratio_backward_count','mutual_count','inlier_count'):
        if old.get(key)!=new.get(key):errors.append(key)
    for key in ('distance','inlier_fraction','query_hull_coverage','reference_hull_coverage'):
        if abs(old.get(key,0)-new.get(key,0))>1e-6:errors.append(key)
    a,b=old.get('homography'),new.get('homography')
    if (a is None)!=(b is None) or (a is not None and not np.allclose(a,b,atol=1e-6,rtol=0)):errors.append('homography')
    prefix=old.get('inlier_pairs',[])
    if prefix!=new.get('inlier_pairs',[])[:len(prefix)]:errors.append('retained_coordinate_prefix')
    return errors


def full_pair(a,b,old,descriptors,gray):
    before=old['sift'];complete=len(before.get('inlier_pairs',[]))==before['inlier_count']
    result=before if complete else sift_pair(descriptors[a],descriptors[b],gray[a].shape,gray[b].shape,SIFT_CFG,f'{a}:{b}')
    errors=[] if complete else metrics_agree(before,result)
    coords=result.get('inlier_pairs',[])
    if len(coords)!=result['inlier_count']:errors.append('incomplete_new_coordinates')
    matches=[]
    if coords and not errors:
        mapping=reciprocal_ids(descriptors[a],descriptors[b])
        for n,p in enumerate(coords):
            key=tuple(p['query_xy']+p['reference_xy'])
            if key not in mapping or not mapping[key]:errors.append('missing_reciprocal_feature_id');break
            qi,ri=mapping[key].popleft()
            matches.append({'match_id':f'{a}:{b}:{n}','query_keypoint_id':qi,'reference_keypoint_id':ri,**p})
    return {'pair_id':f'pair:{a}:{b}','samples':[a,b],'coordinate_source':'reused_complete_old' if complete else 'recomputed_same_frozen_sift',
            'old_coordinate_count':len(before.get('inlier_pairs',[])),'full_coordinate_count':len(coords),'metric_agreement':not errors,'mismatch_fields':errors,
            'hsv_distance':old['hsv_distance'],'gray_mad':old['gray_mad'],'global_v05_supported':old['supported'],
            'geometry':{k:v for k,v in result.items() if k not in ('inlier_pairs','inlier_coordinates_recorded','inlier_coordinates_note')},
            'matches':matches if not errors else [],'raw_full_coordinates':coords if errors else None,
            'usable_regional_geometry':bool(result['verified'] and not errors),'note':'geometry is local correspondence support, not physical-region ownership'}


def orient(pair,query):
    if not pair['usable_regional_geometry']:return []
    if query==pair['samples'][0]:return pair['matches']
    return [{'match_id':m['match_id'],'query_keypoint_id':m['reference_keypoint_id'],'reference_keypoint_id':m['query_keypoint_id'],
             'query_xy':m['reference_xy'],'reference_xy':m['query_xy']} for m in pair['matches']]


def homography_from(pair,source):
    if pair is None or not pair['usable_regional_geometry']:return None
    h=np.asarray(pair['geometry']['homography']).reshape(3,3)
    if source==pair['samples'][0]:return h
    try:return np.linalg.inv(h)
    except np.linalg.LinAlgError:return None


def correspondence_consistency(first,second,pairs):
    by_id=defaultdict(list)
    for m in second:by_id[m['query_keypoint_id']].append(m)
    residuals=[];missing=0
    for a in first:
        for b in by_id.get(a['query_keypoint_id'],[]):
            if a['core_sample']==b['core_sample']:continue
            key=tuple(sorted((a['core_sample'],b['core_sample'])))
            h=homography_from(pairs.get(key),a['core_sample'])
            if h is None:missing+=1;continue
            p=h@np.array(a['reference_xy']+[1.]);q=b['reference_xy']
            if abs(p[2])<1e-9:missing+=1;continue
            residuals.append(float(np.linalg.norm(p[:2]/p[2]-q)))
    return {'cross_core_checks':len(residuals),'missing_cross_core_geometry':missing,
            'median_reprojection_px':float(np.median(residuals)) if residuals else None,
            'consistent_checks':sum(v<=CFG['reprojection_consistency_px'] for v in residuals),
            'inconsistent_checks':sum(v>CFG['reprojection_consistency_px'] for v in residuals),
            'interpretation':'homography-composition consistency diagnostic; parallax/nonplanarity can disagree; not proof of false match or distinct scene'}


def frame_evidence(i,frame,groups,pairs,descriptor,shape,old_assignment):
    group_maps={};group_records=[];cells=[]
    locations={j:site(k.pt) for j,k in enumerate(descriptor[0])}
    effective=defaultdict(set)
    for j,k in enumerate(descriptor[0]):effective[cell(k.pt,shape)].add(locations[j])
    for group in groups:
        sid=group['scene_id'];matches=[]
        for ref in group['frozen_core_samples']:
            if ref==i:continue
            p=pairs.get(tuple(sorted((i,ref))))
            if p is None:continue
            unit=next(u for u,v in enumerate(group['reference_view_units']) if ref in v)
            for m in orient(p,i):
                matches.append({**m,'core_sample':ref,'view_unit':unit,'pair_id':p['pair_id'],'global_pair_supported':p['global_v05_supported']})
        ids={m['query_keypoint_id'] for m in matches};sites={site(m['query_xy']) for m in matches}
        points=[list(p) for p in sites];coverage=hull_coverage(points,shape)
        record={'scene_id':sid,'unique_query_keypoints':len(ids),'unique_query_sites':len(sites),'different_core_sources':sorted({m['core_sample'] for m in matches}),
                'different_view_units':sorted({m['view_unit'] for m in matches}),'query_hull_coverage':coverage,
                'effective_detected_sites':len(set(locations.values())),'support_site_fraction':len(sites)/max(len(set(locations.values())),1),
                'pair_evidence':sorted({m['pair_id'] for m in matches}),'global_eligible_site_count':len({site(m['query_xy']) for m in matches if m['global_pair_supported']}),
                'unknown_reason':None if sites else 'no_reproducible_verified_local_correspondence; no proof of different scene'}
        group_records.append(record);group_maps[sid]={'ids':ids,'sites':sites,'matches':matches}
    for index in range(16):
        regions=[]
        for g in group_records:
            sid=g['scene_id'];ms=[m for m in group_maps[sid]['matches'] if cell(m['query_xy'],shape)==index];sites={site(m['query_xy']) for m in ms}
            regions.append({'scene_id':sid,'unique_query_keypoints':len({m['query_keypoint_id'] for m in ms}),'unique_query_sites':len(sites),
                            'core_sources':sorted({m['core_sample'] for m in ms}),'view_units':sorted({m['view_unit'] for m in ms}),
                            'effective_feature_sites':len(effective[index]),'support_fraction':len(sites)/max(len(effective[index]),1),
                            'query_hull_coverage_full_frame':hull_coverage([list(p) for p in sites],shape),
                            'unknown_reason':None if sites else ('low_texture_no_detected_sites' if not effective[index] else 'no_verified_local_support'),
                            'raw_match_refs':[m['match_id'] for m in ms]})
        cells.append({'cell':index,'row':index//4,'col':index%4,'effective_feature_sites':len(effective[index]),'groups':regions})
    overlap=[]
    for a,b in combinations(group_maps,2):
        ga,gb=group_maps[a],group_maps[b];shared=ga['sites']&gb['sites'];sameids=ga['ids']&gb['ids'];union=ga['sites']|gb['sites']
        other_sites=set().union(*(g['sites'] for sid,g in group_maps.items() if sid not in (a,b))) if len(group_maps)>2 else set()
        exclusive=shared-other_sites
        occupied_a={cell(p,shape) for p in ga['sites']};occupied_b={cell(p,shape) for p in gb['sites']}
        if not (ga['sites'] or gb['sites']):kind='unknown'
        elif min(len(ga['sites']),len(gb['sites']))<CFG['diagnostic_min_sites']:kind='single_or_weak_support'
        elif len(sameids)>=CFG['diagnostic_shared_sites']:kind='same_query_structure_multigroup'
        elif len(shared)/max(len(union),1)<CFG['diagnostic_overlap_ratio']:kind='different_query_locations_multigroup'
        else:kind='mixed_spatial_support'
        overlap.append({'groups':[a,b],'kind':kind,'shared_query_keypoints':len(sameids),'shared_query_sites':len(shared),
                        'query_site_jaccard':len(shared)/max(len(union),1),'same_grid_cells':sorted(occupied_a&occupied_b),
                        'shared_by_additional_groups':len(shared&other_sites),'pair_exclusive_sites':len(exclusive),'pair_exclusive_cells':len({cell(p,shape) for p in exclusive}),
                        'only_a_sites':len(ga['sites']-gb['sites']),'only_b_sites':len(gb['sites']-ga['sites']),
                        'shared_query_keypoint_ids':sorted(sameids),'composition_consistency':correspondence_consistency(ga['matches'],gb['matches'],pairs) if sameids else None})
    multi=[g for g in group_records if g['unique_query_sites']>=CFG['diagnostic_min_sites']]
    if not multi:category='weak_support' if any(g['unique_query_sites'] for g in group_records) else 'unknown'
    elif len(multi)==1:category='single_group_local_support'
    elif any(o['kind']=='same_query_structure_multigroup' for o in overlap):category='same_location_multigroup'
    else:category='different_location_multigroup'
    return {'sample':i,'source_frame':frame['source_frame'],'pts':frame['pts'],'t':frame['t'],'time_base':frame['time_base'],
            'image':frame['image'],'shape':list(shape),'global_assignment':old_assignment['scene_id'],'global_candidates':old_assignment['candidate_scene_ids'],
            'global_conflict':len(old_assignment['candidate_scene_ids'])>1,'global_unresolved':len(old_assignment['candidate_scene_ids'])>1 and old_assignment['scene_id'] is None,
            'quality':old_assignment['quality'],'category':category,'effective_query_keypoints':len(descriptor[0]),'effective_query_sites':len(set(locations.values())),
            'groups':group_records,'cells':cells,'group_overlaps':overlap,
            'note':'positions support local visual correspondence only; grid fractions are feature support, not area probabilities'},group_maps
