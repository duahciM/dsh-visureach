from itertools import combinations

from identity_first.recognition import measure
from identity_first.common import CFG as OLD
from .common import CFG


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else None


def views_for(parent, regions):
    return [{'anchor': view['anchor'], 'features': [f for f in view['features'] if f['id'] in regions]}
            for view in parent['views'] if any(f['id'] in regions for f in view['features'])]


def positive(views, count):
    return {i for i in range(count) if measure(views, i)[0] > 0}


def diagnose_parent(parent, bins, hsv, gray):
    features = [f for view in parent['views'] for f in view['features']]
    support = {f['id']: set(f['present']) for f in features}
    windows = {fid: {bins[i] for i in samples} for fid, samples in support.items()}
    relationships = []
    for a, b in combinations(sorted(support), 2):
        wa, wb = windows[a], windows[b]
        fa, fb = support[a], support[b]
        relationships.append({'regions':[a,b], 'same_reference':a.split(':')[0]==b.split(':')[0],
                              'same_query_samples':sorted(fa & fb), 'same_windows':sorted(wa & wb),
                              'left_only_windows':sorted(wa-wb), 'right_only_windows':sorted(wb-wa),
                              'frame_jaccard':jaccard(fa,fb), 'window_jaccard':jaccard(wa,wb)})

    def compatible(a, b):
        wa, wb = windows[a], windows[b]
        return len(wa & wb) >= CFG['common_bins_min'] and jaccard(wa,wb) >= CFG['within_cluster_jaccard']

    groups = []
    for fid in sorted(support, key=lambda f: (-len(windows[f]), f)):
        fits = [i for i,g in enumerate(groups) if all(compatible(fid, gfid) for gfid in g)]
        if fits:
            idx = max(fits, key=lambda i: (min(jaccard(windows[fid], windows[gfid]) for gfid in groups[i]), -i))
            groups[idx].append(fid)
        else:
            groups.append([fid])
    cluster_rows = []
    children = []
    for group in groups:
        views = views_for(parent, group)
        matched = positive(views, len(bins))
        active_bins = sorted({bins[i] for i in matched})
        eligible = bool(matched) and len(active_bins) >= CFG['exclusive_bins_min']
        cluster_rows.append({'regions':group,'positive_samples':sorted(matched),'positive_bins':active_bins,
                             'eligible':eligible,'reason':None if eligible else 'insufficient_same_view_regions_or_positive_windows'})
        if eligible:
            children.append(set(group))
    result = {'parent':parent['scene_id'],'region_pair_overlap':relationships,'clusters':cluster_rows,
              'proposal':None,'status':'not_exactly_two_independently_supported_clusters'}
    if len(children) != 2:
        return result
    unused = set(support)-set.union(*children)
    shared = sorted(fid for fid in unused if all(compatible(fid,gfid) for child in children for gfid in child))
    a,b = (views_for(parent, child | set(shared)) for child in children)
    regions_a, regions_b = positive(a,len(bins)),positive(b,len(bins))

    def witnesses(own, other):
        return sorted(i for i in own-other if not any(hsv[i,j]<=CFG['near_duplicate_hsv'] and gray[i,j]<=CFG['near_duplicate_gray'] for j in other))

    distinct = [witnesses(regions_a,regions_b), witnesses(regions_b,regions_a)]
    window_counts = [len({bins[i] for i in refs}) for refs in distinct]
    bin_jaccard = jaccard({bins[i] for i in regions_a},{bins[i] for i in regions_b})
    result['candidate_evidence'] = {'child_views':[a,b], 'exclusive_positive_samples':distinct,'exclusive_window_counts':window_counts,
                                    'double_positive_samples':sorted(regions_a & regions_b),'positive_window_jaccard':bin_jaccard,
                                    'shared_regions':shared,'regions_unassigned':sorted(unused-set(shared))}
    if min(window_counts) < CFG['exclusive_bins_min']:
        result['status'] = 'insufficient_exclusive_nonnear_duplicate_positive_windows'
    elif bin_jaccard > CFG['between_cluster_jaccard_max']:
        result['status'] = 'too_much_observed_positive_overlap'
    else:
        result.update(status='proposed', proposal={'views':[a,b],'priority':min(window_counts)})
    return result


def execute(candidates, diagnostics, capacity=OLD['max_candidates']):
    available = capacity - len(candidates)
    ready = sorted((d for d in diagnostics if d['proposal']), key=lambda d:(-d['proposal']['priority'],d['parent']))
    chosen = {d['parent']:d for d in ready[:max(0, available)]}
    next_id = max((int(c['scene_id'][1:]) for c in candidates), default=0) + 1
    revised, decisions = [], []
    for parent, diagnostic in zip(candidates, diagnostics):
        sid = parent['scene_id']
        if sid not in chosen:
            revised.append(parent)
            decisions.append({'parent':sid,'children':[],'status':'capacity_blocked' if diagnostic['proposal'] else diagnostic['status']})
            continue
        left, right = chosen[sid]['proposal']['views']
        ids = [sid, f'S{next_id:02}']
        next_id += 1
        revised.extend({'scene_id':i,'views':v} for i,v in zip(ids,(left,right)))
        decisions.append({'parent':sid,'children':ids,'status':'executed','restore':parent})
    return revised, decisions
