"""Bounded forward/backward inference over measured projective observations."""
from collections import Counter, defaultdict, deque
from itertools import combinations
import hashlib
import json
import math
from time import perf_counter

import cv2
import numpy as np

from if06_checkpoint.spatial import space_interface
from .ledger import stamp


def qcount(matches, shape, side):
    h,w=shape
    return len({(x>=w/2,y>=h/2) for x,y in (m[side] for m in matches)})


def decision(p, frames, cfg, legacy=False):
    i,j=p['samples']; s=p['sift']; m=s.get('inlier_pairs',[]); shape=frames[i]['gray'].shape
    if min(frames[i]['sharpness'],frames[j]['sharpness'])<8:return 'low_quality'
    if p.get('distribution_state')=='old_computation_failed':return 'not_evaluable_computation'
    if not s['verified']:return 'no_verified_geometry'
    if s['inlier_count']<int(cfg['wide_min_inliers']) or s['inlier_fraction']<cfg['wide_inlier_fraction'] or (legacy and p['hsv_distance']>.35):return 'insufficient_strength'
    if min(s['query_hull_coverage'],s['reference_hull_coverage'])<cfg['wide_min_hull']:return 'localized_hull'
    if len(m)!=s['inlier_count']:return 'budget_unassessed' if p.get('distribution_state')=='budget_unassessed' else 'distribution_unavailable'
    if min(qcount(m,shape,side) for side in ('query_xy','reference_xy'))<cfg['wide_quadrants']:return 'insufficient_full_quadrants'
    return 'wide_direct'


def project(h, xy):
    p=np.c_[xy,np.ones(len(xy))] @ h.T
    if np.min(np.abs(p[:,2]))<1e-8 or not np.isfinite(p).all():return None
    return p[:,:2]/p[:,2:]


def fit_pose(p,frames,abrupt,cfg):
    i,j=p['samples']; s=p['sift']; m=s.get('inlier_pairs',[])
    row={'pair':p['id'],'samples':[i,j],'state':'not_evaluable','dependency_ids':[p['id']+':SIFT']}
    if any(i<cut<=j for cut in abrupt): return row|{'reason':'suspected_abrupt_or_unexplained_motion'}
    if min(frames[i]['sharpness'],frames[j]['sharpness'])<8 or not s['verified'] or len(m)!=s['inlier_count']:
        return row|{'reason':'quality_geometry_or_complete_coordinates_missing'}
    if len(m)<cfg['local_min_inliers'] or s['inlier_fraction']<cfg['min_pose_inlier_fraction'] or min(s['query_hull_coverage'],s['reference_hull_coverage'])<cfg['local_min_hull']:
        return row|{'reason':'too_few_spread_landmarks'}
    a=np.float64([x['query_xy'] for x in m]);b=np.float64([x['reference_xy'] for x in m]); f=np.arange(len(m))%5!=0
    ht,_=cv2.findHomography(a[f],b[f],0)
    if ht is None:return row|{'reason':'degenerate_fit'}
    h,w=frames[i]['gray'].shape; coord=np.float64([[.2*w,.2*h],[.8*w,.2*h],[.8*w,.8*h],[.2*w,.8*h],[.5*w,.5*h]])
    pred=project(ht,a[~f]); moved=project(ht,coord)
    if pred is None or moved is None:return row|{'reason':'unstable_projection'}
    error=float(np.median(np.linalg.norm(pred-b[~f],axis=1)))
    delta=np.linalg.norm(moved-coord,axis=1); cap=float(np.max(delta)/max(w,h))
    if error>cfg['max_holdout_error_px'] or cap>cfg['max_local_corner_shift_fraction'] or np.linalg.det(ht)<=0:
        return row|{'reason':'reprojection_or_projective_motion_outside_domain','held_position_error_px':error,'projected_corner_shift_fraction':cap}
    dx,dy=(moved[-1]-coord[-1]).tolist()
    return row|{'state':'image_relative_pose','reason':'multi_landmark_projective_alignment',
        'homography':ht.ravel().tolist(),'fit_points':int(f.sum()),'held_position_points':int((~f).sum()),
        'held_position_error_px':error,'projected_corner_shift_fraction':cap,'image_relative_center_delta_px':[dx,dy],
        'geometry_note':'Held-out fit points share original RANSAC inlier selection, not independent calibration.',
        'rotation_DEG':None,'translation_metric':None,'decomposition_status':'unknown_rotation_vs_translation_without_intrinsics_or_parallax_validation'}


def graph(frames,raw,edits,cfg):
    poses=[]; points={}; station={}; deps={}
    def connect(i,x,j,y,key):
        a=(i,*x); b=(j,*y); ka=points.get(a); kb=points.get(b)
        if ka is None and kb is None:
            ka=len(station);station[ka]={i:a,j:b};deps[ka]={key}
        else:
            if ka is None:
                ka,kb=kb,None
            left=station[ka]
            if any(p[0] in left and left[p[0]]!=p for p in (a,b)):return
            if kb is not None and kb!=ka:
                # Merge compatible point tracks only, never scene labels.
                right=station[kb]
                if any(left[z]!=right[z] for z in left.keys()&right.keys()):return
                left.update(right);deps[ka]|=deps[kb];station[kb]={}
                for p in right.values():points[p]=ka
            left[i]=a;left[j]=b;deps[ka].add(key)
        points[a]=points[b]=ka
    for (i,j),p in raw.items():
        if frames[j]['t']-frames[i]['t']>cfg['local_gap_s'] and j!=i+1:continue
        if frames[j]['t']-frames[i]['t']>cfg['local_gap_s']:
            e={'pair':p['id'],'samples':[i,j],'state':'not_evaluable','reason':'cadence_gap','dependency_ids':[p['id']+':SIFT']}
        else:
            e=fit_pose(p,frames,edits,cfg)
        poses.append(e)
        if e['state']=='image_relative_pose':
            for m in p['sift']['inlier_pairs']:
                connect(i,m['query_xy'],j,m['reference_xy'],p['id']+':SIFT')
    good=sorted((k for k,p in station.items() if len(p)>=cfg['anchor_min_observations']),key=lambda k:(-len(station[k]),k))
    landmarks=[]
    for n,k in enumerate(good[:cfg['anchor_max_per_clip']]):
        tracks=station[k]; seen=sorted(tracks)
        landmarks.append({'landmark_instance_hypothesis':f'L{n:03d}','class':None,'stationarity':'unknown',
                          'first_observed':seen[0],'last_observed':seen[-1],
                          'unobserved_gap_samples':sorted(set(range(seen[0],seen[-1]))-set(seen)),
                          'gap_visibility':'unknown_not_absence','root_evidence_ids':sorted(deps[k]),
                          'observations':[{'sample':i,'xy':list(tracks[i][1:])} for i in seen]})
    return {'frames':[stamp(f) for f in frames],'pose_hypotheses':poses,'landmarks':landmarks,
            'eligible_tracks_before_budget':len(good),'unassessed_tracks':max(0,len(good)-len(landmarks)),
            'counter':dict(Counter(e.get('reason') for e in poses)),
            'metric_world_frames':0,'physical_instance_confirmed':0}


def pathfind(start,stop,by_start,by_end,both,order,cfg):
    fronts=[deque([start]),deque([stop])]
    paths=[{start:[]},{stop:[]}]; log=[[],[]]; used=set()
    step=0; winner=None
    while len(used)<cfg['edge_visits_per_relation'] and (fronts[0] or (both and fronts[1])):
        r=step%2 if both else 0; step+=1
        q=fronts[r]
        if not q:continue
        k=q.popleft()
        if len(paths[r][k])>=cfg['max_path_hops']:continue
        candidates=by_end[k] if r else by_start[k]
        intended=list(x for x in candidates if start<=x['samples'][0]<x['samples'][1]<=stop)
        def rank(e):
            i,j=e['samples']
            if order=='shuffle':return hashlib.sha256(e['pair'].encode()).hexdigest()
            return (i,j) if order=='forward' else (-i,-j)
        for edge in sorted(intended,key=rank):
            edge_id=edge['pair']
            if len(used)>=cfg['edge_visits_per_relation'] and edge_id not in used:break
            used.add(edge_id)
            nxt=edge['samples'][0 if r else 1]
            log[r].append(edge_id+':SIFT')
            if nxt not in paths[r]:
                paths[r][nxt]=paths[r][k]+[edge]
                q.append(nxt)
            if nxt in paths[1-r] and len(paths[0][nxt])+len(paths[1][nxt])<=cfg['max_path_hops']:
                winner=paths[0][nxt]+list(reversed(paths[1][nxt]));break
        if winner:break
    return winner,log,step


def supported_pose(start,stop,direct,frames,graph_data,by_start,by_end,both,order,cfg):
    path,logs,updates=pathfind(start,stop,by_start,by_end,both,order,cfg)
    rec={'source_samples':[start,stop],'forward_roots':logs[0],'reverse_roots':logs[1],'front_updates':updates,
         'unique_roots_visited':len(set(logs[0]+logs[1])),'repeat_accesses':len(logs[0])+len(logs[1])-len(set(logs[0]+logs[1]))}
    if not path or len(path)<2:return rec|{'status':'no_bounded_multi_edge_route'}
    H=np.eye(3)
    for e in path: H=np.array(e['homography']).reshape(3,3) @ H
    roots={k for e in path for k in e['dependency_ids']}
    s=direct['sift']; points=s['inlier_pairs']; hs=frames[start]['gray'].shape
    a=np.float64([p['query_xy'] for p in points]);b=np.float64([p['reference_xy'] for p in points])
    after=project(H,a)
    if after is None:return rec|{'status':'unstable_composed_pose'}
    err=float(np.median(np.linalg.norm(after-b,axis=1)))
    middle=path[len(path)//2]['samples'][0]
    anchors=[]
    for l in graph_data['landmarks']:
        ob=[x for x in l['observations'] if start<=x['sample']<=stop]
        if len(ob)<3 or not (ob[0]['sample']<middle<ob[-1]['sample']):continue
        mid=min(ob,key=lambda x:abs(x['sample']-middle))
        anchors.append((l['landmark_instance_hypothesis'],tuple(mid['xy'])))
    separated=next(([a[0],b[0]] for a,b in combinations(anchors,2) if math.dist(a[1],b[1])>=hs[1]*.2),[])
    return rec|{'status':'supported_projective_loop' if err<=cfg['max_holdout_error_px'] and len(separated)>=2 else 'loop_residual_or_anchor_diversity',
                'closure_median_error_px':err,'path_root_evidence_ids':sorted(roots),'direct_root':direct['id']+':SIFT',
                'spatially_separated_landmark_hypotheses':separated,'path_edges':[e['pair'] for e in path]}


def proposals(frames,rows,refined,raw,cfg):
    wide={}; gates=Counter()
    for (i,j),p in raw.items():
        if frames[j]['t']-frames[i]['t']>=cfg['wide_min_gap_s']:
            grade=decision(p,frames,cfg);gates[grade]+=1
            if grade=='wide_direct':wide[i,j]=p
    occ=refined['occurrences']; budgets=cfg['max_occurrence_pairs']; duties=[]
    for k,(a,b) in enumerate(combinations(occ,2)):
        aa=[i for i in a['samples'] if rows[i]['confidence_level']=='high']
        bb=[j for j in b['samples'] if rows[j]['confidence_level']=='high']
        pairs=[(i,j) for i in aa for j in bb if (i,j) in wide]
        fixed='budget_unassessed' if k>=budgets else 'same_fallback_scene' if a['scene_id']==b['scene_id'] else \
            'short_occurrence' if min(a['end']-a['start'],b['end']-b['start'])<cfg['minimum_occurrence_span_s']-1e-8 else None
        duties.append((a,b,aa,bb,pairs,fixed))
    return duties,wide,dict(gates)


def infer(frames,rows,coarse,refined,raw,graph_data,duties,edits,arm,order,cfg):
    t0=perf_counter(); both=arm in ('R1','R3'); pose=arm in ('R2','R3')
    by_start=defaultdict(list);by_end=defaultdict(list)
    if pose:
        for e in graph_data['pose_hypotheses']:
            if e['state']=='image_relative_pose':
                i,j=e['samples'];by_start[i].append(e);by_end[j].append(e)
    ret=[]; updates=0; visits=0; repeats=0; fresh_roots=set();direct_visits=0
    for a,b,aa,bb,pairs,fixed in duties:
        rec={'occurrences':[a['occurrence_id'],b['occurrence_id']],'evidence_ids':[a['scene_id'],b['scene_id']],
             'forward':[],'reverse':[],'relationship':'unknown','reason':fixed,'root_evidence_ids':[],
             'same_pool_wide_pairs':[raw[z]['id'] for z in pairs]}
        ret.append(rec)
        if fixed:continue
        if not pairs:rec['reason']='no_wide_direct_pair';continue
        # Read immutable direct roots once; reverse exchanges ordered summaries rather than voting.
        perm=sorted(pairs,key=lambda ij: (ij if order=='forward' else tuple(-k for k in ij) if order=='reverse' else hashlib.sha256(str(ij).encode()).hexdigest()))
        if both:
            tail=perm[len(perm)//2:]; head=perm[:len(perm)//2]
        else:head,tail=perm,[]
        direct_visits+=len(perm)
        rec['forward']=[raw[z]['id']+':SIFT' for z in head];rec['reverse']=[raw[z]['id']+':SIFT' for z in reversed(tail)]
        direct=sorted(head+tail)
        fallback=None
        blocked=any(aa[0]<cut<=bb[-1] for cut in edits)
        rec['observed_abrupt_model_conflict']=blocked
        for (i,j),(k,l) in combinations(direct,2):
            if min(frames[k]['t']-frames[i]['t'],frames[l]['t']-frames[j]['t'])+1e-8<cfg['min_witness_spacing_s']:continue
            inter=[x for x in range(aa[-1]+1,bb[0]) if min(frames[x]['sharpness'],frames[i]['sharpness'])>=8 and not raw[i,x]['sift']['verified'] and not raw[x,j]['sift']['verified']]
            if not blocked and len(inter)>=2 and frames[inter[-1]]['t']-frames[inter[0]]['t']+1e-8>=cfg['min_witness_spacing_s']:
                fallback=[raw[i,j]['id'],raw[k,l]['id']];break
        if fallback:
            rec.update(relationship='tentative_same_space',reason='spaced_wide_views_with_distinct_intervening_views',
                       root_evidence_ids=[k+':SIFT' for k in fallback],witness_pairs=fallback)
        else:
            rec['reason']='suspected_discontinuity' if blocked else 'unresolved_independent_context'
        if pose and not fallback and not blocked:
            i,j=direct[0]
            route=supported_pose(i,j,raw[i,j],frames,graph_data,by_start,by_end,both,order,cfg)
            rec['pose_route']=route;updates+=route['front_updates'];visits+=route['unique_roots_visited'];repeats+=route['repeat_accesses']
            fresh_roots.update(route['forward_roots']+route['reverse_roots'])
            independent_intermediate=any(o['occurrence_id'] not in rec['occurrences'] and o['start']>a['end'] and o['end']<b['start'] and
                                         o['scene_id'] not in rec['evidence_ids'] for o in refined['occurrences'])
            rec['independently_assigned_intermediate_appearance']=independent_intermediate
            if route['status']=='supported_projective_loop' and not independent_intermediate:
                rec.update(relationship='tentative_same_space',reason='wide_direct_plus_bounded_anchor_loop',
                    root_evidence_ids=sorted(set(route['path_root_evidence_ids']+[route['direct_root']])),witness_pairs=[raw[i,j]['id']])
    qualified=[r for r in ret if r['relationship']=='tentative_same_space'];accepted=qualified[0] if qualified else None
    for r in qualified[1:]:
        r.update(relationship='unknown',reason='one_relation_commit_budget')
    output=space_interface(coarse,refined,rows,'frozen',accepted)
    return {'arm':arm,'physical_input_order':order,'relations':ret,'stages':['freeze shared observations','directional proposals','single canonical judgment'],
        'association':accepted,'interface':output,'judgment_rounds':1,'direct_root_accesses':direct_visits,
        'new_source_observations':0,'pose_unique_visits_per_candidate_sum':visits,'pose_globally_distinct_roots':len(fresh_roots),
        'repeat_pose_accesses_within_candidate':repeats,'pose_updates':updates,'elapsed_seconds':perf_counter()-t0,
        'revision_log':[{'reason':accepted['reason'],'old_to_new':accepted['occurrences'],'witnesses':accepted['root_evidence_ids'],
                         'withdrawal_conditions':'inconsistent wide match or new independently verified spatial conflict; retain both original time occurrences'}]
                        if accepted else []}
