"""IF-06: keep direct evidence, tentative space, and ordered occurrence separate."""
import json
from itertools import combinations
from pathlib import Path

from identity_catalog_matching.match import CFG as MATCH_CFG, hsv_candidate, hsv_decide, fuse
from identity_catalog_matching.common import duplicate, source_ref
from identity_small_support_spatial.memory import direct_pairs, motion_links, track, POLICY
from v05.engine import group_and_assign
from v06.engine import build_cores, recognize, refine


def direct_new(frames, matrices):
    hsv, gray, geom, supported = matrices
    cores = build_cores(frames, group_and_assign(frames, matrices), matrices)
    old = recognize(frames, cores, matrices)
    bins = [int((f['t']-frames[0]['t'])/.8 + 1e-9) for f in frames]
    catalog = []
    for g in cores['groups']:
        sid, refs = g['scene_id'], g['frozen_core_samples']
        catalog.append({'scene_id': sid, 'reason':'frozen_automatic_core', 'reference_views':[{'sample':j} for j in refs],
                        'views':[{'anchor':j,'features':[{'present':[i for i in range(len(frames)) if i!=j and geom[i,j] and hsv[i,j]<=.5 and frames[i]['sharpness']>=8]}]} for j in refs]})
    # Same IF04 independent witness, dedup and medoid rule; B has whole-view direct observations only.
    anchors={r['sample'] for c in catalog for r in c['reference_views']}
    profiles={c['scene_id']:set().union(*(set(v['features'][0]['present']) for v in c['views'])) for c in catalog}
    templates={}
    for sid,pos in profiles.items():
        rivals=set().union(*(p for k,p in profiles.items() if k!=sid))
        witnesses=[i for i in sorted(pos-rivals-anchors) if not any(duplicate(hsv,gray,i,j) for j in rivals)]
        windows={b:[i for i in witnesses if bins[i]==b] for b in sorted({bins[i] for i in witnesses})}
        reps=[min(xs,key=lambda i:(sum(hsv[i,j] for j in xs),i)) for xs in windows.values()]
        picked=[min(reps,key=lambda i:(sum(hsv[i,j] for j in reps),i))] if reps else []
        while picked and len(picked)<MATCH_CFG['hsv']['templates_per_id']:
            available=[i for i in reps if i not in picked and not any(duplicate(hsv,gray,i,j) for j in picked)]
            if not available: break
            picked.append(min(available,key=lambda i:(-min(hsv[i,j] for j in picked),i)))
        eligible=len(windows)>=MATCH_CFG['hsv']['minimum_exclusive_windows']
        templates[sid]={'templates':[source_ref(frames[i]) for i in picked] if eligible else [],'windows':windows,'witnesses':witnesses,
                        'unusable_reason':None if eligible else 'insufficient_distinct_witness_templates'}
    evidence = []
    direct = []
    for r,f in zip(old,frames):
        i=r['sample']
        hs={sid:hsv_candidate(i,[t['sample'] for t in s['templates']],hsv,gray,f['sharpness']>=8) for sid,s in templates.items()}
        hw,hp,margin,hreason=hsv_decide(hs)
        lw=r['scene_id']; lp=r['candidate_scene_ids']
        winner,conf,reason=fuse(lw,lp,hw)
        if reason=='channel_winner_conflict' and hs[lw]['distance'] is None:
            winner,conf,reason=lw,'high','local_retained_hsv_not_comparable'
        per={s['scene_id']:{'geometry_and_color_core_support':s,'hsv':hs[s['scene_id']],
              'geometry_state': 'quality_ineligible' if f['sharpness']<8 else 'self_only' if s['nearest_core_sample'] is None else
              'direct_positive' if s['absolute_supported'] else 'no_positive_support',
              'pairs':[f'pair:{min(i,j)}:{max(i,j)}' for j in next(g['frozen_core_samples'] for g in cores['groups'] if g['scene_id']==s['scene_id']) if j!=i]}
             for s in r['scores']}
        evidence.append({'sample':i,'source_frame':f['source_frame'],'t':f['t'],'pts':f['pts'],'time_base':f['time_base'],
                         'evidence':per,'winners':{'L':lw,'H':hw,'LH_safe':winner},'hsv_reason':hreason,'hsv_margin':margin,'decision_reason':reason})
        direct.append(r|{'scene_id':winner,'confidence_level':conf,'reason':reason,
                        'assignment_state':'supported' if winner else 'unknown',
                        'assignment_origin':'frozen_new_media_direct' if winner else 'abstain',
                        'candidate_scene_ids':sorted(set(lp+hp))})
    return cores,templates,direct,evidence


def space_interface(coarse, refined, direct, label_prefix, association=None):
    occurrences=refined.get('occurrences',coarse.get('occurrences',[]))
    mapping={o['occurrence_id']:f'H:{label_prefix}:{o["scene_id"]}' for o in occurrences}
    if association:
        a,b=association['occurrences']
        mapping[b]=mapping[a]
    sis=[]
    for s in refined['segments']:
        partners=[o for o in occurrences if s['scene_id']==o['scene_id'] and min(s['end'],o['end'])>=max(s['start'],o['start'])]
        owners=sorted(partners,key=lambda o:(-min(s['end'],o['end'])+max(s['start'],o['start']),o['start']))
        hyp=mapping[owners[0]['occurrence_id']] if owners else (f'H:{label_prefix}:{s["scene_id"]}' if s['scene_id'] else None)
        observed={r['scene_id'] for r in direct if s['start']<=r['t']<s['end'] and r['scene_id']}
        positive={q for r in direct if s['start']<=r['t']<s['end'] for q in r.get('candidate_scene_ids',[])}
        sis.append({'segment_id':s['segment_id'],'range':[s['start'],s['end']],'state':s['state'],
                    'scene_hypothesis_id':hyp,'legacy_time_label':s['scene_id'],
                    'occurrences':[o['occurrence_id'] for o in owners], 'direct_evidence_winners':sorted(observed),
                    'all_positive_evidence_ids':sorted(positive)})
    occurrence_rows=[{'occurrence_id':o['occurrence_id'],'scene_hypothesis_id':mapping[o['occurrence_id']],
              'evidence_id_used_by_fallback':o['scene_id'],'range':[o['start'],o['end']],'samples':o['samples']} for o in occurrences]
    sample_to_segment=[]
    for r in direct:
        side=next((s for s in sis if s['range'][0]<=r['t']<s['range'][1]),None)
        sample_to_segment.append({'sample':r['sample'],'t':r['t'],'legacy_direct_winner':r['scene_id'],'evidence_ids':r.get('candidate_scene_ids',[]),
                                  'segment_id':side['segment_id'] if side else None,'scene_hypothesis_id':side['scene_hypothesis_id'] if side else None})
    return {'segments':sis,'scene_hypotheses':sorted(set(mapping.values())|{s['scene_hypothesis_id'] for s in sis if s['scene_hypothesis_id']}),
            'occurrences':occurrence_rows,'sample_to_segment':sample_to_segment,'accepted_relation':association,
            'legacy_evidence_repeat_policy':'same legacy time label shares tentative default ID; no physical proof',
            'time_policy':'frozen fallback; association never removes a temporal occurrence or changes a cut'}


def abrupt_pairs(frames,motion,raw, threshold):
    output=[]
    for i in range(1,len(frames)):
        m=motion['pair_metrics'][i]
        if (m['raw_mad']>=threshold and not raw[i-1,i]['sift']['verified'] and
            not m.get('brightness',{}).get('supported') and not (m.get('motion_supported') and m.get('registration_valid'))):
            output.append({'sample':i,'pair':raw[i-1,i]['id'],'t':[frames[i-1]['t'],frames[i]['t']],'raw_mad':m['raw_mad'],
                           'reason':'abrupt_edit_or_unexplained_motion_not_certified_cut'})
    return output


def memory(frames,motion,raw,rows,refined,arm,abrupt):
    if arm=='B0': return {'links':[],'tracks':[],'events':[],'reason':'memory_disabled'}
    adj={k:p for k,p in raw.items() if k[1]-k[0]==1}
    transitions=[r['bracket'] for r in refined['diagnostics'] if r['transition_accepted']]
    base=motion_links(frames,motion,adj,rows,transitions)
    links=base
    if arm=='B2':
        neutral=[r|{'scene_id':None} for r in rows]
        links=motion_links(frames,motion,adj,neutral,transitions)
        for edit in abrupt:
            i=edit['sample']
            links[i].update(continuous=False,direction=None,reason=edit['reason'])
    lands,events=track(frames,links,adj,raw,rows)
    return {'links':links,'tracks':list(lands.values()),'events':events,
            'changed_motion_links':[{'sample':o['sample'],'old':o['reason'],'new':n['reason'],'pair':n.get('pair')}
                                    for o,n in zip(base,links) if o['reason']!=n['reason']]}


def broad(p, frames, policy):
    if not p: return 'pair_not_available'
    i,j=p['samples']; s=p['sift']; m=s.get('inlier_pairs',[]); h,w=frames[i]['gray'].shape
    if min(frames[i]['sharpness'],frames[j]['sharpness'])<8: return 'low_quality'
    if not s.get('verified'): return 'geometry_unverified'
    if p['hsv_distance']>policy['max_hsv'] or s['inlier_count']<policy['min_inliers'] or s['inlier_fraction']<policy['min_inlier_fraction']:
        return 'insufficient_match_strength'
    if min(s['query_hull_coverage'],s['reference_hull_coverage'])<policy['min_hull_each']:
        return 'localized_hull'
    for side in ('query_xy','reference_xy'):
        if len({(x>=w/2,y>=h/2) for x,y in (r[side] for r in m)})<policy['min_lateral_quarters']:
            return 'insufficient_recorded_quadrants'
    return 'wide_direct'


def associations(frames,raw,rows,refined,abrupt,mem,policy):
    # We log closure independently of the gate; it need not have a second scene hypothesis.
    gap=policy['min_gap_s']; spacing=policy['min_witness_spacing_s']
    qualified={}
    eligible={}
    for (i,j),p in raw.items():
        if frames[j]['t']-frames[i]['t']>=gap:
            status=broad(p,frames,policy)
            eligible[i,j]=status
            if status=='wide_direct': qualified[i,j]=p
    occ=refined['occurrences']; controls=[]; accepted=None
    spans={o['occurrence_id']:{i for i in o['samples'] if rows[i]['confidence_level']=='high'} for o in occ}
    for index,(a,b) in enumerate(combinations(occ,2)):
        aa,bb=a['occurrence_id'],b['occurrence_id']; ai,bi=spans[aa],spans[bb]
        rec={'occurrences':[aa,bb],'evidence_ids':[a['scene_id'],b['scene_id']],
             'ranges':[[a['start'],a['end']],[b['start'],b['end']]],'relationship':'unknown',
             'wide_direct_pairs':[],'witness_pairs':[],'intervening_different_view_samples':[],'reason':None}
        controls.append(rec)
        if index>=policy['max_candidate_occurrence_pairs']:
            rec['reason']='candidate_occurrence_budget'; continue
        if min(a['end']-a['start'],b['end']-b['start'])+1e-9<policy['min_time_occurrence_span_s']:
            rec['reason']='insufficient_occurrence_span'; continue
        pairs=sorted((i,j) for i,j in qualified if i in ai and j in bi)
        rec['wide_direct_pairs']=[raw[i,j]['id'] for i,j in pairs]
        if not pairs:
            rec['reason']='no_wide_direct_closure'; continue
        if a['scene_id']==b['scene_id']:
            rec['reason']='same_fallback_hypothesis_keep_time_episodes_separate'; continue
        for (i,j),(k,l) in combinations(pairs,2):
            if frames[k]['t']-frames[i]['t']+1e-9<spacing or frames[l]['t']-frames[j]['t']+1e-9<spacing: continue
            if any(i<e['sample']<=k or j<e['sample']<=l for e in abrupt): continue
            if any(i<e['sample']<=l for e in abrupt): continue
            def no_support(x,z):
                return not raw[min(x,z),max(x,z)]['sift']['verified']
            divers=[z for z in range(max(ai,default=k)+1,min(bi,default=j)) if frames[z]['sharpness']>=8 and no_support(i,z) and no_support(j,z)]
            reentry=[e for e in mem['events'] if e['kind']=='confirmed_reentry' and e['last_observation']['sample'] in ai and e['query']['sample'] in bi]
            if not reentry and not any(frames[z]['t']-frames[divers[0]]['t']+1e-9>=spacing for z in divers[1:]): continue
            rec['witness_pairs']=[raw[i,j]['id'],raw[k,l]['id']];rec['intervening_different_view_samples']=divers
            rec['reentry_evidence']=[e['direct_pair'] for e in reentry]
            break
        if not rec['witness_pairs']:
            rec['reason']='no_independent_spaced_witness_pair_or_intervening_context'; continue
        if accepted:
            rec['reason']='one_association_budget'; continue
        rec['relationship']='tentative_same_space';rec['reason']='two_separated_wide_view_direct_witnesses_with_intervening_context';accepted=dict(rec)
    return {'relations':controls,'accepted':accepted,
            'all_nonlocal_pair_gate':[{'pair':raw[k]['id'],'samples':k,'status':v} for k,v in eligible.items()],
            'direct_closure_count':len(qualified), 'different_space_claims':0,
            'note':'Failures and visual diversity are not evidence of a different location; matching structure is not semantic instance identification.'}
