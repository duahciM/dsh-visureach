from identity_feature_grouping.grouping import positive
from identity_first.recognition import measure
from .common import CFG, duplicate, source_ref


def make_templates(catalog, frames, bins, hsv, gray):
    active=catalog['competitive']
    anchors={r['sample'] for candidate in catalog['candidates'] for r in candidate['reference_views']
             if candidate['reason']!='reference_budget'}
    profiles={c['scene_id']:positive(c['views'],len(frames)) for c in active}
    summary={}
    for c in active:
        sid=c['scene_id']
        rivals=set().union(*(s for name,s in profiles.items() if name!=sid))
        witnesses=sorted(i for i in profiles[sid]-rivals-anchors if not any(duplicate(hsv,gray,i,j) for j in rivals))
        windows=sorted({bins[i] for i in witnesses})
        by_window={b:[i for i in witnesses if bins[i]==b] for b in windows}
        reps=[min(members,key=lambda i:(sum(hsv[i,j] for j in members),i)) for members in by_window.values()]
        chosen=[]
        if reps:
            chosen.append(min(reps,key=lambda i:(sum(hsv[i,j] for j in reps),i)))
        while chosen and len(chosen)<CFG['hsv']['templates_per_id']:
            remaining=[i for i in reps if i not in chosen and not any(duplicate(hsv,gray,i,j) for j in chosen)]
            if not remaining:
                break
            chosen.append(min(remaining,key=lambda i:(-min(hsv[i,j] for j in chosen),i)))
        usable=bool(chosen) and len(windows)>=CFG['hsv']['minimum_exclusive_windows']
        summary[sid]={'templates':[source_ref(frames[i]) | {'window':bins[i]} for i in chosen] if usable else [],
                      'candidates_before_independence_check':chosen,'exclusive_witnesses':witnesses,
                      'window_medoids':reps,'status':'available' if usable else 'insufficient_distinct_witness_templates',
                      'common_anchor_color':'shared_context_only'}
    return summary


def hsv_candidate(sample, templates, hsv, gray, quality):
    checks=[{'sample':j,'distance':float(hsv[sample,j]) if sample!=j else None,
             'self_reference_excluded':sample==j,'query_near_duplicate':duplicate(hsv,gray,sample,j)} for j in templates]
    distances=[r['distance'] for r in checks if r['distance'] is not None]
    distance=min(distances,default=None) if quality else None
    reason=('low_quality' if not quality else 'no_independent_frozen_templates' if not templates else
            'self_excluded_no_remaining_template' if not distances else 'calculated_from_frozen_hsv_matrix')
    return {'distance':distance,'absolute_supported':distance is not None and distance<=CFG['hsv']['absolute_cap'],
            'template_comparisons':checks,'state':reason}


def hsv_decide(scores):
    ranked=sorted((s['distance'],cid) for cid,s in scores.items() if s['distance'] is not None)
    positives=[cid for cid,s in scores.items() if s['absolute_supported']]
    if not positives:
        return None,positives,None,'no_absolute_hsv_support'
    d,cid=ranked[0]
    runner=ranked[1][0] if len(ranked)>1 else 1.
    margin=(runner-d)/(runner+d+1e-9)
    if len(ranked)>1 and runner<=CFG['hsv']['both_strong_cap']:
        return None,positives,margin,'multiple_strong_hsv_candidates'
    if margin<CFG['hsv']['relative_margin']:
        return None,positives,margin,'insufficient_hsv_margin'
    return cid,positives,margin,'unique_hsv_margin'


def fuse(local, local_positive, hue):
    if local is not None and hue not in (None,local):
        return None,'low','channel_winner_conflict'
    if local is not None:
        return local,'high','channels_agree' if hue==local else 'local_winner_hsv_abstains'
    if local_positive:
        return None,'low','unresolved_local_support' if hue is None else 'hsv_cannot_resolve_local_ambiguity'
    if hue is not None:
        return hue,'medium','hsv_only_tentative'
    return None,'low','no_unique_support'


def local_evidence(comp, row, frames, hsv, gray, geom, pairs, historical_index):
    i=row['sample']
    views=[]
    for v in comp['feature_groups']:
        a=v['anchor']
        key=tuple(sorted((a,i)))
        pair=pairs.get(key)
        historic=historical_index.get((a,i)) if comp['origin_type'].startswith('historic') else None
        local_support=[f for f in v['features'] if i in f['present'] and i!=a]
        if i==a:
            reason='self_reference_excluded'
        elif frames[i]['sharpness']<8:
            reason='low_quality'
        elif comp['reason']=='reference_budget':
            reason='unassessed_reference_budget'
        elif not geom[a,i]:
            reason='frozen_geometry_not_verified'
        elif hsv[a,i]>.5:
            reason='verified_geometry_old_hsv_prefilter_failed'
        elif pair is not None and comp['origin_type'] in ('suppressed_reference','alternative_feature_group'):
            reason='alternate_pair_validated' if pair['usable_regional_geometry'] else 'quarantined_pair'
        elif historic and historic.get('local_verified'):
            reason='reused_frozen_direct_pair'
        else:
            reason='unassessed_no_verified_full_coordinates'
        views.append({'anchor':a,'state':reason,'local_feature_ids':[f['id'] for f in local_support],
                      'match_ids':[m for f in local_support for m in f.get('match_ids_by_sample',{}).get(str(i),[])],
                      'frozen_pair_id':f'pair:{key[0]}:{key[1]}' if i!=a else None,
                      'frozen_geometry_verified':bool(geom[a,i]) if i!=a else None,
                      'reference_hsv_context':float(hsv[a,i]) if i!=a else None,
                      'shares_reference_with':[r['candidate_id'] for r in comp['duplicate_or_related_candidates']
                                               if a in r['shared_reference_samples']],
                      'gray_mad':float(gray[a,i]) if i!=a else None})
    raw=row['positive'].get(comp['candidate_id'])
    if raw is None:
        v=[v for v in comp['feature_groups'] if v['features']]
        raw=measure(v,i) if v else (0.,0.,[],[])
    evaluated=any(v['state'] in ('reused_frozen_direct_pair','alternate_pair_validated') for v in views)
    return {'views':views,'absolute_supported':raw[0]>0,'local_score':raw[0] if evaluated else None,
            'unweighted_fraction':raw[1] if evaluated else None,
            'positive_view_samples':raw[2], 'positive_feature_ids':raw[3],
            'state':'positive' if raw[0]>0 else 'compared_no_absolute_support' if evaluated else 'unavailable',
            'final_pairwise':[p for p in row['pairwise'] if p['own']==comp['candidate_id']]}


def scoring_tables(catalog, frames, matrices, local_rows, templates, pairs, historical_index, motion):
    hsv,gray,geom,_=matrices
    for f,l in zip(frames,local_rows):
        i=f['sample']
        evidence={}
        options={}
        for c in catalog['candidates']:
            cid=c['candidate_id']
            local=local_evidence(c,l,frames,hsv,gray,geom,pairs,historical_index)
            record={'candidate_id':cid,'active_status':c['active_status'],'local':local,
                    'used_in_candidate_generation':i in {w['sample'] for w in c['generation_witnesses']}}
            if templates is not None:
                these=[p['sample'] for p in templates.get(cid,{}).get('templates',[])]
                record['hsv']=hsv_candidate(i,these,hsv,gray,f['sharpness']>=8.)
                record['used_in_hsv_templates']=i in these
                if c['active_status']=='competing':
                    options[cid]=record['hsv']
            evidence[cid]=record
        ldecision,lpositive=l['assignment'],l['candidate_ids']
        result={'query':source_ref(f),'sharpness':f['sharpness'],'contrast':f['contrast'],
                'local_winner':ldecision,'local_positive':lpositive,'pair_evidence':evidence,
                'neighbor_motion_and_photometry':motion['pair_metrics'][i]}
        if templates is not None:
            winner,positive_ids,margin,reason=hsv_decide(options)
            fused,confidence,f_reason=fuse(ldecision,lpositive,winner)
            result.update(hsv_winner=winner,hsv_positive=positive_ids,hsv_margin=margin,hsv_reason=reason,
                          fused_winner=fused,fused_confidence=confidence,fused_reason=f_reason,
                          positive_union=sorted(set(lpositive)|set(positive_ids)))
        yield result
