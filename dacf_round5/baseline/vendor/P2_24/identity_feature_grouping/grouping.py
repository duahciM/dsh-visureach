from collections import defaultdict, deque
from itertools import combinations

from identity_feature_decomposition.split import diagnose_parent, jaccard, positive, views_for
from .common import CFG


def point_budget(sites):
    profiles = defaultdict(deque)
    for item in sorted(sites,key=lambda s:s['keypoint_id']):
        if len(item['positive_bins'])>=2:
            profiles[tuple(item['positive_bins'])].append(item)
    ordered = sorted(profiles,key=lambda p:(-len(p),profiles[p][0]['keypoint_id']))
    kept = []
    limit = CFG['point_budget_per_reference']
    while ordered and len(kept)<limit:
        for profile in ordered:
            if len(kept)==limit:
                break
            kept.append(profiles[profile].popleft())
        ordered = [p for p in ordered if profiles[p]]
    return kept


def point_groups(sites):
    clusters = []
    for item in sorted(sites,key=lambda s:(-len(s['positive_bins']),s['keypoint_id'])):
        profile = set(item['positive_bins'])
        fits = []
        for i,cluster in enumerate(clusters):
            if cluster[0]['anchor']!=item['anchor']:
                continue
            other = [set(s['positive_bins']) for s in cluster]
            if all(len(p & profile)>=2 and jaccard(p,profile)>=.60 for p in other):
                fits.append((min(jaccard(p,profile) for p in other),-i))
        if fits:
            clusters[-max(fits)[1]].append(item)
        else:
            clusters.append([item])
    return clusters


def point_view(cluster):
    """Project only view-qualified frames into frozen region-oriented readout."""
    hits = defaultdict(int)
    for site in cluster:
        for sample in site['positive_samples']:
            hits[sample] += 1
    valid = {sample for sample,n in hits.items() if n>=CFG['point_query_hits'] and n/len(cluster)>=.25}
    features = [{'id':site['id'],'present':sorted(valid & set(site['positive_samples'])),
                 'source':'frame_scoped_direct_site'} for site in cluster]
    return [{'anchor':cluster[0]['anchor'],'features':features}]


def build_groups(parent, representation, bins, hsv, gray, sites=None):
    if representation=='G':
        frozen = diagnose_parent(parent,bins,hsv,gray)
        groups = [(item['regions'],views_for(parent,item['regions']),True) for item in frozen['clusters']]
        relationships = frozen['region_pair_overlap']
    else:
        groups = []
        for view in parent['views']:
            for cluster in point_groups(point_budget(sites[view['anchor']])):
                groups.append(([s['id'] for s in cluster],point_view(cluster),len(cluster)>=CFG['point_min_cluster_sites']))
        relationships = None
    rows = []
    for i,(ids,views,large_enough) in enumerate(groups):
        samples = positive(views,len(bins)) if large_enough else set()
        rows.append({'group':i,'atoms':ids,'views':views,'positive_samples':sorted(samples),
                     'positive_bins':sorted({bins[j] for j in samples}),
                     'qualified':large_enough and len({bins[j] for j in samples})>=2,
                     'sufficient_atoms':large_enough})

    if representation=='G':
        eligible = [g for g in rows if g['qualified']]
        if len(eligible)>=2:
            feature_by_id = {f['id']:f for v in parent['views'] for f in v['features']}
            support = {k:{bins[i] for i in f['present']} for k,f in feature_by_id.items()}
            occupied = {fid for g in eligible for fid in g['atoms']}
            shared = [fid for fid in sorted(set(support)-occupied)
                      if all(len(support[fid]&support[r])>=2 and jaccard(support[fid],support[r])>=.60 for r in occupied)]
            for g in eligible:
                g['shared_atoms'] = shared
                g['views'] = views_for(parent,g['atoms']+shared)
                samples = positive(g['views'],len(bins))
                g['positive_samples'] = sorted(samples)
                g['positive_bins'] = sorted({bins[i] for i in samples})
    contenders = [row for row in rows if row['qualified']]
    pair_evidence = []
    for a,b in combinations(rows,2):
        a_bins,b_bins = set(a['positive_bins']),set(b['positive_bins'])
        pair_evidence.append({'groups':[a['group'],b['group']],'window_jaccard':jaccard(a_bins,b_bins),
                              'same_samples':sorted(set(a['positive_samples'])&set(b['positive_samples'])),
                              'exclusive_bins_each':[sorted(a_bins-b_bins),sorted(b_bins-a_bins)]})
    for row in rows:
        others = set().union(*(set(item['positive_samples']) for item in contenders if item is not row))
        own = set(row['positive_samples'])
        witness = sorted(i for i in own-others if not any(hsv[i,j]<=.04 and gray[i,j]<=.02 for j in others))
        row['exclusive_positive_samples'] = witness
        row['exclusive_positive_bins'] = sorted({bins[i] for i in witness})
        row['status'] = 'unqualified_support' if not row['qualified'] else 'diagnostic_no_exclusive_bins'
    ranked = sorted((row for row in contenders if len(contenders)>=2 and len(row['exclusive_positive_bins'])>=2),
                    key=lambda row:(-len(row['exclusive_positive_bins']),row['group']))
    accepted = []
    for row in ranked:
        if any(jaccard(set(row['positive_bins']),set(old['positive_bins']))>.50 for old in accepted):
            row['status'] = 'diagnostic_positive_window_overlap'
        else:
            row['status'] = 'eligible_child'
            accepted.append(row)
    return {'parent':parent['scene_id'],'representation':representation,'groups':rows,'cross_group_support':pair_evidence,
            'region_pair_overlap':relationships,'eligible_children':[r['group'] for r in accepted],
            'status':'propose_replacement' if len(accepted)>=2 else 'fallback_original_parent'}


def execute(parents, diagnostics, capacity=6):
    available = max(0,capacity-len(parents))
    next_id = max((int(p['scene_id'][1:]) for p in parents),default=0)+1
    by_id = {p['scene_id']:p for p in parents}
    selections = {}
    qualified = {d['parent']:[d['groups'][i] for i in d['eligible_children']] for d in diagnostics}
    order = sorted(by_id,key=lambda sid:(-len(qualified[sid][1]['exclusive_positive_bins']) if len(qualified[sid])>=2 else 0,sid))
    for sid in order:
        parent = by_id[sid]
        rows = qualified[sid]
        if len(rows)<2 or available<1:
            selections[sid] = ([parent],{'parent':sid,'status':'capacity_blocked' if len(rows)>=2 else 'fallback_original_parent',
                'children':[],'blocked_groups':[r['group'] for r in rows] if len(rows)>=2 else [],'fallback':True})
            continue
        active = rows[:1+available]
        children = [sid,*[f'S{i:02}' for i in range(next_id,next_id+len(active)-1)]]
        next_id += len(active)-1
        available -= len(active)-1
        next_parents = [{'scene_id':name,'views':r['views']} for name,r in zip(children,active)]
        selections[sid] = (next_parents,{'parent':sid,'status':'executed','fallback':False,'children':children,
            'groups_in_competition':[r['group'] for r in active],'blocked_groups':[r['group'] for r in rows[len(active):]],
            'restore':parent})
    return [view for sid in by_id for view in selections[sid][0]], [selections[sid][1] for sid in by_id]
