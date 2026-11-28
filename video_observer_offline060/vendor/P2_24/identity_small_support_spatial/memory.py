from collections import Counter
import json
import math

import numpy as np

from identity_first.controlled import direction_step
from p1.io import dump
from .common import CFG, FIRST, read, source_ref


POLICY=CFG['spatial']


def finish_prediction(expected, actual, independent):
    if not independent or actual not in ('left','right'):
        return 'unknown_unverified_reentry'
    if expected is None:
        return 'direct_reobserved_no_prediction'
    return 'confirmed_reentry' if expected==actual else 'retracted_edge_prediction'


def read_context(state, observation):
    """Controlled scene readout consumes independent instance/context observables."""
    if any(observation.get(flag) for flag in ('cut','dissolve','doorway','moves','ambiguous','occluded')):
        return {'candidate':None,'basis':'explicit_blocker_keep_scene_distinct','write_direct_id':False}
    if observation.get('location')=='out_of_view_candidate' and observation.get('verified_exit'):
        return {'candidate':state.get('scene'),'basis':'indirect_last_observed_context_only','write_direct_id':False}
    if (observation.get('reobserved_instance') and observation.get('continuity') and
            len(set(observation.get('independent_context_cues',[])))>=2 and state.get('scene')):
        return {'candidate':state['scene'],'basis':'independently_corroborated_revisit','write_direct_id':False}
    return {'candidate':None,'basis':'insufficient_independent_scene_evidence','write_direct_id':False}


def run_controlled(out):
    original=read(FIRST/'controlled/B_locked_observation_fixtures.json')
    old=[]
    for row in original:
        previous={'appearance':None,'location':'unknown'}
        actual=[]
        for observation in row['observations']:
            previous=direction_step(previous,observation)
            actual.append(previous)
        old.append({'case':row['name'],'trace':actual,'expected':row['expected'],
                    'pass':[r['location'] for r in actual]==row['expected'] and
                    (row['name']!='contradictory_direction' or actual[-1]['reason']=='contradictory_reentry_revises_direction')})
    state={'scene':'room_sofa_and_table'}
    controls=[
        ('sofa_exits_right',{'location':'out_of_view_candidate','verified_exit':True},state['scene']),
        ('table_exits_left',{'location':'out_of_view_candidate','verified_exit':True},state['scene']),
        ('short_occlusion',{'occluded':True,'location':'out_of_view_candidate','verified_exit':True},None),
        ('similar_other_instance',{'ambiguous':True,'reobserved_instance':True,'continuity':True,'independent_context_cues':['wall','table']},None),
        ('moving_object_at_door',{'moves':True,'doorway':True,'reobserved_instance':True,'continuity':True,'independent_context_cues':['wall','table']},None),
        ('through_door_shared_object',{'doorway':True,'reobserved_instance':True,'continuity':True,'independent_context_cues':['wall','table']},None),
        ('return_without_mid_view_overlap',{'reobserved_instance':True,'continuity':True,'independent_context_cues':['original_wall_marker','original_fixed_table']},state['scene']),
        ('same_single_landmark',{'reobserved_instance':True,'continuity':True,'independent_context_cues':['original_wall_marker']},None),
        ('cut',{'cut':True,'reobserved_instance':True,'continuity':True,'independent_context_cues':['wall','table']},None),
        ('dissolve',{'dissolve':True,'reobserved_instance':True,'continuity':True,'independent_context_cues':['wall','table']},None),
    ]
    edge=[('reverse_returns_same_edge','right','right',True,'confirmed_reentry'),
          ('wrong_edge_retracts_once','left','right',True,'retracted_edge_prediction'),
          ('full_turn_without_pose','left',None,False,'unknown_unverified_reentry'),
          ('similar_appearance_no_identity','left','left',False,'unknown_unverified_reentry')]
    dump(out/'controlled_fixtures_locked.json',{'source':'scripted observable directions, visibility and identity; not automatic vision',
                                                 'old_inputs':original,'context':controls,'edge':edge})
    extra=[]
    for case,observation,expected in controls:
        result=read_context(state,observation)
        extra.append({'case':case,'result':result,'expected':expected,'pass':result['candidate']==expected and not result['write_direct_id']})
    for case,expected,actual,independent,outcome in edge:
        result=finish_prediction(expected,actual,independent)
        extra.append({'case':case,'result':result,'expected':outcome,'pass':result==outcome})
    dump(out/'controlled_results.json',{'IF01_B':old,'additional':extra})


def edge_at(x,width):
    margin=width*POLICY['edge_fraction']
    if x<=margin:
        return 'left'
    return 'right' if x>=width-margin else None


def closest(point, matches, side):
    candidates=sorted((math.dist(point,m[side]),i) for i,m in enumerate(matches))
    if not candidates or candidates[0][0]>POLICY['tracking_tolerance_px']:
        return None
    return candidates[0][1]


def direct_pairs(path, query):
    records={}
    with path.open(encoding='utf-8') as source:
        for line in source:
            record=json.loads(line)
            key=tuple(record['samples'])
            if query(key):
                records[key]=record
    return records


def motion_links(frames,motion,adjacent,ids,transitions):
    links=[]
    for i,frame in enumerate(frames):
        if not i:
            links.append({'sample':0,'direction':None,'continuous':False,'reason':'start_of_source'})
            continue
        pair=adjacent[i-1,i]['sift']
        m=motion['pair_metrics'][i]
        matches=pair.get('inlier_pairs',[])
        a,b=ids[i-1],ids[i]
        switch=a['scene_id'] and b['scene_id'] and a['scene_id']!=b['scene_id'] and a['confidence_level']==b['confidence_level']=='high'
        accepted=any(start<=a['t'] and b['t']<=stop for start,stop in transitions)
        reason=('two_different_direct_contexts' if switch else 'accepted_transition_bracket' if accepted else
                'gap_or_quality' if frame['t']-frames[i-1]['t']>POLICY['max_link_gap_s'] or min(frame['sharpness'],frames[i-1]['sharpness'])<8 else
                'no_verified_geometry' if not pair.get('verified') else 'continuous_direct_geometry')
        continuity=reason=='continuous_direct_geometry'
        shift=np.median([np.subtract(o['reference_xy'],o['query_xy']) for o in matches],axis=0).tolist() if matches else [None,None]
        dx=m.get('dx',0.)
        horizontal=(continuity and m['motion_supported'] and m.get('registration_valid') and len(matches)>=POLICY['minimum_adjacent_matches'] and
                    abs(dx)>=POLICY['min_horizontal_px'] and abs(m.get('dy',0.))<=abs(dx)*POLICY['max_vertical_ratio'] and
                    shift[0]*dx>0 and abs(shift[0])>=POLICY['min_horizontal_px'] and abs(shift[1])<=abs(shift[0])*POLICY['max_vertical_ratio'])
        direction=('left' if dx>0 else 'right') if horizontal else None
        links.append({'sample':i,'pair':adjacent[i-1,i]['id'],'continuous':continuity,'reason':reason,'direction':direction,
                      'stored_sift_points':len(matches),'median_image_dx_dy':shift,'phase_dx':dx,'phase_dy':m.get('dy'),
                      'old_registration_valid':m.get('registration_valid'),'old_motion_supported':m['motion_supported'],
                      'from':source_ref(frames[i-1]),'to':source_ref(frame)})
    return links


def track(frames,links,adjacent,lookup,ids):
    height,width=frames[0]['gray'].shape
    tracked={}
    all_tracks={}
    pending={}
    events=[]
    for i in range(1,len(frames)):
        frame,link=frames[i],links[i]
        if not link['continuous']:
            for key,rec in pending.items():
                events.append({'landmark':key,'kind':'prediction_withdrawn_on_continuity_break','query':source_ref(frame),
                               'reason':link['reason'],'exit_edge':rec['edge'],'new_state':'occluded_or_unknown','version':2})
            pending.clear()
            tracked.clear()
            continue
        matches=adjacent[i-1,i]['sift'].get('inlier_pairs',[])
        used=set()
        for key,rec in list(pending.items()):
            before=rec['last']['sample']
            if frame['t']-rec['last']['t']>POLICY['prediction_horizon_s']:
                events.append({'landmark':key,'kind':'expired_outside_prediction_horizon','query':source_ref(frame),
                               'new_state':'occluded_or_unknown','version':2})
                del pending[key]
                continue
            if not rec.get('predicted') and link['direction'] and link['direction']!=rec['direction']:
                rec['predicted']=True
                events.append({'landmark':key,'kind':'predicted_reentry_after_reversal','expected_edge':rec['edge'],
                               'query':source_ref(frame),'last_observation':rec['last'],'pair':link['pair'],'version':1})
            if not rec.get('predicted'):
                continue
            direct=lookup.get((before,i),{}).get('sift',{})
            if not direct.get('verified'):
                continue
            options=direct.get('inlier_pairs',[])
            idx=closest(rec['xy'],options,'query_xy')
            if idx is None:
                continue
            pt=options[idx]['reference_xy']
            actual=edge_at(pt[0],width)
            if actual is None:
                continue
            if i-before<=1:
                continue
            outcome=finish_prediction(rec['edge'],actual,True)
            proof={'landmark':key,'kind':outcome,'expected_edge':rec['edge'],'observed_edge':actual,'new_state':'observed',
                   'last_observation':rec['last'],'query':source_ref(frame),'xy':pt,'version':2,
                   'direct_pair':lookup[before,i]['id'],'direct_match':options[idx],
                   'scene_hypothesis_committed':False,'scene_use':'unverified_physical_instance_and_context'}
            events.append(proof)
            all_tracks[key]['observations'].append({'source':source_ref(frame),'xy':pt,'pair':proof['direct_pair']})
            del pending[key]
        next_tracks={}
        for key,rec in tracked.items():
            j=closest(rec['xy'],matches,'query_xy')
            if j is not None and j not in used:
                used.add(j)
                xy=matches[j]['reference_xy']
                next_tracks[key]={'xy':xy,'steps':rec['steps']+1}
                all_tracks[key]['observations'].append({'source':source_ref(frame),'xy':xy,'pair':link['pair']})
                continue
            direction=link['direction']
            x=rec['xy'][0]
            projected=x+(link['phase_dx'] or 0.)
            which=edge_at(x,width)
            crossing=(which=='left' and projected<0) or (which=='right' and projected>width-1)
            if direction and crossing and rec['steps']>=POLICY['minimum_track_links']:
                last=source_ref(frames[i-1])
                sid=ids[i-1]['scene_id'] if ids[i-1]['confidence_level']=='high' else None
                pending[key]={'edge':which,'direction':direction,'last':last,'xy':rec['xy'],'context':sid}
                events.append({'landmark':key,'kind':'out_of_view_candidate','query':source_ref(frame),'last_observation':last,
                               'exit_edge':which,'estimated_turn':direction,'projected_xy':[projected,rec['xy'][1]],
                               'inferred_view_bearing_relative_to_landmark':direction,'scene_context_candidate':sid,
                               'visibility':'out_of_view_candidate','pair':link['pair'],'version':1})
        # Replenish from actual edge-near correspondences under the fixed budget.
        for j,m in sorted(enumerate(matches),key=lambda x:min(x[1]['reference_xy'][0],width-1-x[1]['reference_xy'][0])):
            if len(all_tracks)>=POLICY['landmark_limit_per_clip'] or len(next_tracks)>=16:
                break
            xy=m['reference_xy']
            if j in used or min(xy[0],width-1-xy[0])>width*.30:
                continue
            key=f'K{len(all_tracks)+1:03}'
            next_tracks[key]={'xy':xy,'steps':1}
            all_tracks[key]={'landmark':key,'type':'nonsemantic_sift_coordinate_track','classification':None,'object_instance':None,
                             'observations':[{'source':source_ref(frames[i-1]),'xy':m['query_xy'],'pair':link['pair']},
                                             {'source':source_ref(frame),'xy':xy,'pair':link['pair']}]}
        tracked=next_tracks
    for key,rec in pending.items():
        events.append({'landmark':key,'kind':'clip_ended_without_reentry','query':source_ref(frames[-1]),
                       'exit_edge':rec['edge'],'new_state':'occluded_or_unknown','version':2})
    return all_tracks,events


def spatial(state, ids, refined, raw_path):
    frames=state['frames']
    old=direct_pairs(raw_path,lambda ij: frames[ij[1]]['t']-frames[ij[0]]['t']<=POLICY['prediction_horizon_s'] or ij[1]-ij[0]==1)
    adjacent={k:p for k,p in old.items() if k[1]-k[0]==1}
    selected=[r['bracket'] for r in refined['diagnostics'] if r['transition_accepted']]
    assignments=ids['assignments']
    links=motion_links(frames,state['motion'],adjacent,assignments,selected)
    tracks,events=track(frames,links,adjacent,old,assignments)
    graph={'scene_hypotheses':[{'scene_hypothesis_id':f'H:{g["scene_id"]}', 'evidence_ids':[g['scene_id']],
                                'state':'tentative_direct_seed'} for g in ids['groups']],
           'segments':[{'segment_id':s['segment_id'],'start':s['start'],'end':s['end'],
                        'scene_hypothesis_id':f'H:{s["scene_id"]}' if s['scene_id'] else None} for s in refined['segments']],
           'context_edges':[e for e in events if e.get('scene_context_candidate')],
           'scene_relation_revisions':[], 'reason':'old videos lack verified instance/pose and independent scene bridge'}
    stats={'landmarks':len(tracks),'pose_unknown_links':sum(not link['direction'] for link in links[1:]),
           'motion_direction_proxies':dict(Counter(link['direction'] for link in links if link['direction'])),
           'events':dict(Counter(e['kind'] for e in events)),'indirect_context_hints':len(graph['context_edges']),
           'scene_assignments_changed':0,'absence_assertions':0,'actual_360_pose_or_semantic_instance_verified':False}
    return {'links':links,'landmarks':list(tracks.values()),'events':events,'summary':stats},graph
