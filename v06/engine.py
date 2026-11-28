"""No evaluation labels, manual anchors, reference event times or model calls."""
import copy
import json
import math
from pathlib import Path

import numpy as np

from p1.io import dump
from v05.engine import public, sheet

CFG=json.loads((Path(__file__).parent/'protocol.json').read_text())


def pair_ref(a,b):return f'pair:{min(a,b)}:{max(a,b)}'


def choose_refs(members,frames,hsv):
    refs=[min(members,key=lambda i:(sum(hsv[i,j] for j in members),-frames[i]['sharpness'],i))]
    if len(members)>1:
        refs.append(max((i for i in members if i not in refs),key=lambda i:(min(hsv[i,j] for j in refs),frames[i]['sharpness'],-i)))
    if len(members)>2:
        refs.append(max((i for i in members if i not in refs),key=lambda i:(min(abs(frames[i]['t']-frames[j]['t']) for j in refs),frames[i]['sharpness'],-i)))
    return sorted(refs)


def associate_candidates(candidates,support,cfg=CFG):
    n=len(candidates); relation=np.eye(n,dtype=bool); logs=[]
    for i in range(n):
        for j in range(i+1,n):
            a,b=candidates[i]['refs'],candidates[j]['refs']
            pairs=[(x,y) for x in a for y in b if x!=y and support[x,y]]
            covered_a={x for x,y in pairs};covered_b={y for x,y in pairs}
            fraction=len(pairs)/max(len(a)*len(b),1)
            accept=len(covered_a)>=cfg['association_min_covered_refs'] and len(covered_b)>=cfg['association_min_covered_refs'] and fraction>=cfg['association_fraction']
            relation[i,j]=relation[j,i]=accept
            logs.append({'candidate_a':i,'candidate_b':j,'supported_fraction':fraction,'covered_refs_a':sorted(covered_a),'covered_refs_b':sorted(covered_b),
                         'accepted':bool(accept),'support_pairs':[pair_ref(x,y) for x,y in pairs],
                         'decision':'multiple_direct_core_correspondences' if accept else 'insufficient_direct_evidence_not_proof_of_different_scene'})
    remaining=set(range(n));groups=[]
    while remaining:
        seed=min(remaining,key=lambda i:(-sum(bool(relation[i,j]) for j in remaining),i))
        members=[seed]
        for i in sorted(remaining-{seed},key=lambda i:(-sum(bool(relation[i,j]) for j in remaining),i)):
            if all(relation[i,j] for j in members):
                refs={r for j in members+[i] for r in candidates[j]['refs']}
                if len(refs)<=cfg['max_core_refs']:members.append(i)
        remaining.difference_update(members);groups.append(members)
    return groups,logs,relation


def build_cores(frames,old_ids,matrices):
    hsv,gray,sift,support=matrices
    candidates=[]
    for old in old_ids['raw_groups']:
        m=[i for i in old['members'] if frames[i]['sharpness']>=CFG['minimum_sharpness']]
        if len(m)<4 or frames[max(m)]['t']-frames[min(m)]['t']<CFG['minimum_core_span_s']:continue
        refs=choose_refs(m,frames,hsv)
        if not all(any(i!=j and support[i,j] for j in refs) for i in refs):continue
        candidates.append({'candidate_id':len(candidates),'v05_raw_group':old['raw_group'],'members':m,'refs':refs,
                           'source':'frozen automatic v05 complete-link fragment, reselected refs without labels',
                           'self_excluded_support':[{'sample':i,'evidence':[pair_ref(i,j) for j in refs if j!=i and support[i,j]]} for i in refs]})
    associations,logs,relation=associate_candidates(candidates,support)
    associations.sort(key=lambda m:min(i for c in m for i in candidates[c]['refs']))
    groups=[]
    for n,m in enumerate(associations):
        refs=sorted({i for c in m for i in candidates[c]['refs']})
        units=[]
        for i in refs:
            same=next((u for u in units if hsv[i,u[0]]<=CFG['near_duplicate_hsv'] and gray[i,u[0]]<=CFG['near_duplicate_gray']),None)
            if same is None:units.append([i])
            else:same.append(i)
        groups.append({'scene_id':f'S{n+1:02d}','candidate_ids':m,'frozen_core_samples':refs,'reference_view_units':units,
                       'prototype_samples':refs,'prototype_frames':[{**public(frames[i]),'selection_basis':'frozen automatic core; not promoted from extension'} for i in refs],
                       'physical_identity':'unknown','core_revision':0})
    return {'groups':groups,'candidate_cores':candidates,'association_evidence':logs,'association_relation':relation.tolist(),
            'freeze_policy':'unchanged across core/full; no core promotion; candidate relation complete-link, not transitive closure'}


def recognize(frames,cores,matrices):
    hsv,gray,sift,support=matrices;rows=[]
    allcore={i for g in cores['groups'] for i in g['frozen_core_samples']}
    for i,f in enumerate(frames):
        scores=[]
        for g in cores['groups']:
            refs=[j for j in g['frozen_core_samples'] if j!=i]
            matches=[j for j in refs if support[i,j]]
            best=min(matches or refs,key=lambda j:(hsv[i,j],j)) if refs else None
            span=max((frames[j]['t'] for j in matches),default=0)-min((frames[j]['t'] for j in matches),default=0)
            units=[u for u in g['reference_view_units'] if any(j in matches for j in u)]
            distance=float(hsv[i,best]) if best is not None else 1.
            absolute=bool(matches) and distance<=CFG['absolute_hsv_cap']
            corroborated=len(matches)>=2 and span+1e-9>=CFG['minimum_core_span_s']
            scores.append({'scene_id':g['scene_id'],'distance':distance,'absolute_supported':bool(absolute),
                           'direct_core_samples':matches,'direct_core_evidence':[pair_ref(i,j) for j in matches],
                           'supported_view_units':len(units),'total_view_units':len(g['reference_view_units']),
                           'view_unit_fraction':len(units)/max(len(g['reference_view_units']),1),
                           'temporally_corroborated':bool(corroborated),'support_span_s':span,
                           'nearest_core_sample':best,'sift_core_samples':[j for j in matches if sift[i,j]],
                           'extension_only_support':False,'self_match_excluded':i in g['frozen_core_samples']})
        scores.sort(key=lambda r:(not r['absolute_supported'],r['distance'],r['scene_id']))
        good=[s for s in scores if s['absolute_supported']]
        margin=None;winner=None;level='low';reason='no_direct_core_support'
        if good:
            a=good[0];b=good[1] if len(good)>1 else None
            margin=(b['distance']-a['distance'])/(b['distance']+a['distance']+1e-9) if b else 1.
            level='medium';reason='direct_core_support_needs_corroboration'
            if margin<CFG['minimum_relative_margin']:reason='unresolved_competing_core_groups'
            elif a['temporally_corroborated']:
                winner=a['scene_id'];level='high';reason='corroborated_direct_core_and_decisive_HSV_margin'
        if f['sharpness']<CFG['minimum_sharpness']:
            winner=None;level='low';reason='low_sharpness_or_low_texture'
        rows.append({'sample':i,'t':f['t'],'pts':f['pts'],'source_frame':f['source_frame'],'scene_id':winner,
                     'confidence_level':level,'relative_margin':margin,'reason':reason,'scores':scores,
                     'candidate_scene_ids':[s['scene_id'] for s in good],'core_frame':i in allcore,
                     'assignment_state':'supported' if winner else 'unknown','assignment_origin':'initial_frozen_core' if winner else 'abstain',
                     'quality':{k:f[k] for k in ('sharpness','contrast')},'promotion_depth':0})
    return rows


def occurrences(rows,include_extended=False):
    runs=[];current=None
    for r in rows:
        reliable=r['scene_id'] and (r['confidence_level']=='high' or (include_extended and r.get('assignment_origin')=='one_pass_extension'))
        if not reliable:continue
        if current and current['scene_id']==r['scene_id'] and r['t']-current['end']<=CFG['run_max_gap_s']+1e-9:
            current['end']=r['t'];current['samples'].append(r['sample'])
        else:
            current={'scene_id':r['scene_id'],'start':r['t'],'end':r['t'],'samples':[r['sample']]};runs.append(current)
    runs=[r for r in runs if len(r['samples'])>=CFG['run_min_samples']]
    for n,r in enumerate(runs):r['occurrence_id']=f'I{n+1:03d}'
    return runs


def boundaries(runs):
    result=[]
    for a,b in zip(runs,runs[1:]):
        if a['scene_id']==b['scene_id']:continue
        result.append({'boundary_id':f'B{len(result)+1:03d}','left_scene_id':a['scene_id'],'right_scene_id':b['scene_id'],
                       'left_occurrence':a['occurrence_id'],'right_occurrence':b['occurrence_id'],
                       'left_sample':a['samples'][-1],'right_sample':b['samples'][0],'bracket':[a['end'],b['start']],
                       'midpoint':(a['end']+b['start'])/2})
    return result


def extend(rows,initial_runs,initial_boundaries,motion,matrices,cfg=CFG):
    # Read only immutable original rows. A result from this pass is never an input.
    original=copy.deepcopy(rows);updated=copy.deepcopy(rows);trace=[];accepted=0
    cap=min(cfg['extension_max_frames'],math.ceil(cfg['extension_max_fraction']*len(rows)))
    for r in original:
        if r['scene_id']:continue
        i=r['sample']; good=[s for s in r['scores'] if s['absolute_supported']]
        record={'sample':i,'t':r['t'],'old_scene_id':None,'accepted':False,'core_evidence':[], 'initial_high_neighbor':None}
        if not good or r['confidence_level']=='low':record['reason']='no_usable_direct_core_evidence'
        elif r['relative_margin'] is None or r['relative_margin']<cfg['minimum_relative_margin']:record['reason']='unresolved_competition'
        else:
            best=good[0];sid=best['scene_id'];record.update(candidate_scene_id=sid,core_evidence=best['direct_core_evidence'],core_samples=best['direct_core_samples'],core_HSV_distance=best['distance'])
            territories=[]
            for run in initial_runs:
                if run['scene_id']!=sid:continue
                lo=run['start']-cfg['extension_max_distance_from_run_s'];hi=run['end']+cfg['extension_max_distance_from_run_s']
                for b in initial_boundaries:
                    if b['left_occurrence']==run['occurrence_id']:hi=min(hi,b['midpoint'])
                    if b['right_occurrence']==run['occurrence_id']:lo=max(lo,b['midpoint'])
                territories.append([lo,hi])
            record['allowed_territories']=territories
            if not any(lo<=r['t']<=hi for lo,hi in territories):record['reason']='outside_initial_midpoint_territory_direct_evidence_retained'
            else:
                neighbors=[x for x in original if x['scene_id']==sid and x['confidence_level']=='high' and abs(x['t']-r['t'])<=cfg['extension_neighbor_s']+1e-9]
                eligible=[]
                for neighbor in neighbors:
                    a,b=sorted((i,neighbor['sample']));links=motion['pair_metrics'][a+1:b+1]
                    continuous=bool(links) and len(links)==b-a and all(m.get('registration_valid') and (m.get('motion_supported') or m['raw_mad']<=0.035 or m.get('brightness',{}).get('supported')) for m in links)
                    if continuous:eligible.append(neighbor)
                if not eligible:record['reason']='no_initial_high_neighbor_with_continuity'
                elif accepted>=cap:record['reason']='extension_budget_exhausted'
                else:
                    n=min(eligible,key=lambda x:(abs(x['t']-r['t']),x['sample']))
                    updated[i].update(scene_id=sid,confidence_level='medium',assignment_state='supported',assignment_origin='one_pass_extension',reason='direct_core_plus_initial_high_continuity',promotion_depth=1)
                    record.update(accepted=True,new_scene_id=sid,initial_high_neighbor=n['sample'],continuity_metric_samples=list(range(min(i,n['sample'])+1,max(i,n['sample'])+1)),reason='direct_core_plus_initial_high_continuity')
                    accepted+=1
        trace.append(record)
    return updated,{'max_accepted':cap,'accepted':accepted,'records':trace,'core_promotions':0,'passes':1,'snowball_policy':'ordinary additions never authorize another addition; direct frozen-core edge required'}


def conflicts(rows,cores):
    records=[]
    for r in rows:
        if len(r['candidate_scene_ids'])<2:continue
        records.append({'sample':r['sample'],'t':r['t'],'groups':r['candidate_scene_ids'],'resolved_scene_id':r['scene_id'],
                        'reason':r['reason'],'relative_margin':r['relative_margin'],
                        'core_support':[{'scene_id':s['scene_id'],'core_samples':s['direct_core_samples'],'evidence':s['direct_core_evidence']} for s in r['scores'] if s['absolute_supported']]})
    return {'conflicts':records,'association_evidence':cores['association_evidence'],'unresolved_count':sum(r['resolved_scene_id'] is None for r in records),
            'note':'multi support may be duplicated ID, viewpoint overlap or ambiguity; never automatically transition; no post-recognition merge'}


def partition(frames,source,rows,runs,bs,diagnostics=None,coarse=False):
    cell_edges=[source['start']]+[(a['t']+b['t'])/2 for a,b in zip(frames,frames[1:])]+[source['end']]
    extra=[v for b in bs for v in b['bracket']+[b['midpoint']]]
    edges=sorted(set(cell_edges+extra));segments=[]
    for lo,hi in zip(edges,edges[1:]):
        if hi<=lo:continue
        t=(lo+hi)/2;i=min(range(len(frames)),key=lambda j:abs(frames[j]['t']-t));r=rows[i]
        sid=r['scene_id'];state='supported' if sid else 'unknown';refs=[]
        # No one-run extrapolation: cells only, plus explicitly two-sided brackets.
        if coarse:
            reliable=any(i in run['samples'] for run in runs)
            sid=r['scene_id'] if reliable else None;state='supported' if sid else 'unknown'
            for b in bs:
                if b['bracket'][0]<=t<b['bracket'][1]:
                    sid=b['left_scene_id'] if t<b['midpoint'] else b['right_scene_id'];state='provisional_midpoint';refs=[b['boundary_id']]
        else:
            for d in diagnostics or []:
                if d['bracket'][0]<=t<d['bracket'][1]:
                    sid=None;state='transition' if d['transition_accepted'] else 'ambiguous';refs.append(d['boundary_id'])
        key=(sid,state,tuple(refs))
        if segments and segments[-1]['_key']==key:segments[-1]['end']=hi;segments[-1]['evidence_samples']=sorted(set(segments[-1]['evidence_samples']+[i]))
        else:segments.append({'start':lo,'end':hi,'scene_id':sid,'state':state,'boundary_refs':refs,'evidence_samples':[i],'_key':key})
    for n,s in enumerate(segments):s.pop('_key');s['segment_id']=f'{"C" if coarse else "R"}{n+1:03d}'
    return segments


def refine(frames,source,initial_rows,rows,motion,full):
    initial_runs=occurrences(initial_rows);initial_bs=boundaries(initial_runs)
    runs=occurrences(rows,include_extended=full);bs=boundaries(runs);diagnostics=[]
    if full:
        for b in bs:
            curves=[]
            for r in rows[b['left_sample']:b['right_sample']+1]:
                scores={s['scene_id']:s for s in r['scores']};a=scores[b['left_scene_id']];c=scores[b['right_scene_id']]
                curves.append({'sample':r['sample'],'t':r['t'],'d_A':a['distance'],'d_B':c['distance'],'S':(a['distance']-c['distance'])/(a['distance']+c['distance']+1e-9),
                               'absolute_support_A':a['absolute_supported'],'absolute_support_B':c['absolute_supported']})
            links=motion['pair_metrics'][b['left_sample']+1:b['right_sample']+1]
            progression=curves[-1]['S']-curves[0]['S'] if curves else None
            change=any(m['raw_mad']>=CFG['activity_mad'] and not m.get('brightness',{}).get('supported') for m in links)
            accepted=bool(progression is not None and progression>=CFG['minimum_contrast_progression'] and change)
            matching=[x for x in initial_bs if x['left_scene_id']==b['left_scene_id'] and x['right_scene_id']==b['right_scene_id']]
            old=min(matching,key=lambda x:abs(x['midpoint']-b['midpoint'])) if matching else None
            diagnostics.append({**b,'initial_bracket':old['bracket'] if old else None,'initial_midpoint':old['midpoint'] if old else None,
                                'curves':curves,'contrast_progression':progression,'nonphotometric_visual_change':change,'transition_accepted':accepted,
                                'transition_range':b['bracket'] if accepted else None,'reason':'two_sided_contrast_and_visual_change' if accepted else 'insufficient_transition_structure_keep_ambiguous',
                                'metric_samples':[m['sample'] for m in links],'stop_reason':'one_revision_no_recursive_refinement'})
    coarse={'segments':partition(frames,source,initial_rows,initial_runs,initial_bs,coarse=True),'occurrences':initial_runs,'boundaries':initial_bs,
            'safety_fix':'single occurrence has no midpoint; unsupported leading/trailing cells remain unknown'}
    refined={'segments':partition(frames,source,rows,runs,bs,diagnostics),'occurrences':runs,'boundaries':bs,'diagnostics':diagnostics,
             'module_status':'one extension/revision executed' if full else 'core-only; extension/transition revision disabled by design',
             'blocking_reason':None if bs else 'no_two_different_reliable_occurrences; no midpoint or transition candidate',
             'new_group_reclustering':'not implemented; unresolved candidate cores/conflicts retained'}
    return coarse,refined
