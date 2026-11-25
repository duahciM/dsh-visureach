from collections import Counter, defaultdict
from itertools import permutations

import cv2

from v07.regional import SIFT_CFG, cell, full_pair, orient, site
from v08.run import validate_ids
from .common import CFG, V05, V07, V08
import json


def temporal_bins(frames):
    start = frames[0]['t']
    return [int((f['t'] - start + 1e-8) / CFG['bins_seconds']) for f in frames]


def overlap(a, b):
    return len(a & b) / max(1, len(a | b))


def seed_plan(frames, matrices, old_groups):
    hsv, _, verified, _ = matrices
    bins = temporal_bins(frames)
    clear = {i for i, f in enumerate(frames) if f['sharpness'] >= CFG['minimum_sharpness']}
    direct = {i: {bins[j] for j in clear if i != j and verified[i, j] and hsv[i, j] <= .5} for i in clear}
    a1 = [sorted((i for i in group['frozen_core_samples'] if i in clear),
                 key=lambda i: (-len(direct[i]), i))[:CFG['max_views']]
          for group in old_groups[:CFG['max_candidates']]]
    pool = {i for i in clear if len(direct[i]) >= CFG['min_support_bins']}
    seeds = []
    seen = set()
    while pool and len(seeds) < CFG['max_A2_seeds']:
        anchor = min(pool, key=lambda i: (-len(direct[i] - seen), -len(direct[i]), i))
        seeds.append(anchor)
        seen.update(direct[anchor])
        pool = {i for i in pool if i != anchor and overlap(direct[i], direct[anchor]) < .8}
    return a1, seeds, bins


def read_needed(path, needed):
    result = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            key = tuple(record['samples'])
            if key in needed:
                result[key] = record
    return result


def regional_bank(case, frames, matrices, anchors, bins):
    hsv, _, verified, _ = matrices
    gray = [f['gray'] for f in frames]
    needed = {tuple(sorted((a, i))) for a in anchors for i, f in enumerate(frames)
              if a != i and verified[a, i] and hsv[a, i] <= .5 and f['sharpness'] >= CFG['minimum_sharpness']}
    records = read_needed(V07 / case / 'full_pair_evidence.jsonl', needed)
    records.update(read_needed(V08 / case / 'local_pairs_full.jsonl', needed - records.keys()))
    missing = needed - records.keys()
    originals = read_needed(V05 / case / 'pair_evidence.jsonl', missing) if missing else {}
    detector = cv2.SIFT_create(nfeatures=SIFT_CFG['nfeatures'])
    descriptors = [detector.detectAndCompute(im, None) for im in gray]
    stats = Counter()
    failures = []
    for a, b in sorted(needed):
        if (a, b) in records:
            pair = records[a, b]
            stats['historical_full_coordinates'] += 1
        else:
            pair = full_pair(a, b, originals[a, b], descriptors, gray)
            records[a, b] = pair
            stats[pair['coordinate_source']] += 1
        problems = validate_ids(pair, descriptors) if pair['metric_agreement'] else pair['mismatch_fields']
        if problems:
            pair['usable_regional_geometry'] = False
            failures.append({'pair': pair['pair_id'], 'issues': problems})

    bank = {}
    for anchor in anchors:
        observations = defaultdict(dict)
        for i in range(len(frames)):
            if i == anchor:
                continue
            pair = records.get(tuple(sorted((i, anchor))))
            if not pair or not pair['usable_regional_geometry']:
                continue
            for match in orient(pair, anchor):
                coord = site(match['query_xy'])
                region = cell(match['query_xy'], gray[anchor].shape)
                observations[region].setdefault(i, {}).setdefault(coord, match['match_id'])
        features = []
        for region, by_query in observations.items():
            by_site = defaultdict(set)
            for query, sites in by_query.items():
                for coord in sites:
                    by_site[coord].add(bins[query])
            stable = {coord for coord, windows in by_site.items() if len(windows) >= CFG['min_support_bins']}
            if len(stable) < CFG['min_stable_sites']:
                continue
            observed = [i for i, coords in sorted(by_query.items()) if len(stable & coords.keys()) >= CFG['min_query_sites']]
            if len({bins[i] for i in observed}) < CFG['min_support_bins']:
                continue
            features.append({'id': f's{anchor}:r{region}', 'reference': anchor, 'region': region,
                             'stable_coordinate_sites': sorted(stable), 'present': observed,
                             'match_ids_by_sample': {str(i): [m for coord, m in by_query[i].items() if coord in stable] for i in observed},
                             'absent_supported': [], 'unspecified_observation': 'unknown'})
        features.sort(key=lambda f: (-len({bins[i] for i in f['present']}), -len(f['stable_coordinate_sites']), f['region']))
        bank[anchor] = features[:CFG['max_features_per_view']]
    return bank, records, {'needed': len(needed), 'origins': dict(stats), 'quarantined': failures}


def measure(views, sample, weights=None):
    weights = weights or {}
    best = 0.
    raw_best = 0.
    support_views = []
    feature_refs = []
    for view in views:
        present = [f['id'] for f in view['features'] if sample in f['present']]
        raw = len(present) / len(view['features'])
        if len(present) < 2 or raw < CFG['min_raw_support']:
            continue
        raw_best = max(raw_best, raw)
        best = max(best, sum(weights.get(fid, 1.) for fid in present) / len(view['features']))
        support_views.append(view['anchor'])
        feature_refs.extend(present)
    return best, raw_best, support_views, feature_refs


def build_candidates(a1_plan, seeds, bank, matrices, bins):
    hsv, _, verified, _ = matrices
    views = {a: {'anchor': a, 'features': fs} for a, fs in bank.items() if len(fs) >= 2}
    profiles = {a: {bins[i] for i in range(len(bins)) if measure([v], i)[0] > 0} for a, v in views.items()}
    a1 = [[views[i] for i in refs if i in views] for refs in a1_plan]
    pool = {a for a in seeds if a in views and len(profiles[a]) >= CFG['min_support_bins']}
    a2 = []
    seen = set()
    while pool and len(a2) < CFG['max_candidates']:
        seed = min(pool, key=lambda a: (-len(profiles[a] - seen), -len(profiles[a]), a))
        new = profiles[seed] - seen
        if not new:
            break
        group = [views[seed]]
        complements = [a for a in pool - {seed} if verified[seed, a] and hsv[seed, a] <= .5
                       and profiles[a] & profiles[seed] and profiles[a] - profiles[seed]]
        if complements and CFG['max_views'] > 1:
            extra = min(complements, key=lambda a: (-len(profiles[a] - profiles[seed]), a))
            group.append(views[extra])
        a2.append(group)
        consumed = set().union(*(profiles[v['anchor']] for v in group))
        seen.update(consumed)
        pool = {a for a in pool if a not in {v['anchor'] for v in group} and overlap(profiles[a], consumed) < .8}
    return [[{'scene_id': f'S{i+1:02}', 'views': g} for i, g in enumerate(groups) if g] for groups in (a1, a2)]


def comparisons(candidates, weights, count):
    positive = [{c['scene_id']: measure(c['views'], i) for c in candidates} for i in range(count)]
    rows = []
    for i, support in enumerate(positive):
        contenders = [c for c in candidates if support[c['scene_id']][0] > 0]
        pairwise = []
        margins = defaultdict(list)
        for own, rival in permutations(contenders, 2):
            a, b = own['scene_id'], rival['scene_id']
            left = measure(own['views'], i, weights.get((a, b)))[0]
            right = measure(rival['views'], i, weights.get((b, a)))[0]
            margin = left - right
            margins[a].append(margin)
            pairwise.append({'own': a, 'rival': b, 'own_support': left, 'rival_support': right, 'margin': margin})
        winning = [a for a, evidence in support.items() if evidence[0] > 0 and
                   all(m >= CFG['margin'] for m in margins[a])]
        rows.append({'sample': i, 'assignment': winning[0] if len(winning) == 1 else None,
                     'positive': support, 'pairwise': pairwise,
                     'candidate_ids': [c['scene_id'] for c in contenders], 'absent_supported': []})
    return rows


def refine_features(candidates, bins):
    initial = comparisons(candidates, {}, len(bins))
    weights = {}
    logs = []
    for own, rival in permutations(candidates, 2):
        a, b = own['scene_id'], rival['scene_id']
        rival_bins = {bins[i] for i, row in enumerate(initial) if b in row['candidate_ids']}
        weights[a, b] = {}
        for view in own['views']:
            for feature in view['features']:
                common = rival_bins & {bins[i] for i in feature['present']}
                shared = len(common) >= 2 and len(common) / max(1, len(rival_bins)) >= CFG['shared_ratio']
                weight = CFG['shared_weight'] if shared else 1.
                if shared:
                    weights[a, b][feature['id']] = weight
                logs.append({'round': 1, 'own': a, 'rival': b, 'feature': feature['id'], 'weight': weight,
                             'rival_observation_bins': sorted(common), 'rival_profile_bins': sorted(rival_bins),
                             'reason': 'observed_repeatedly_in_rival_scope' if shared else 'retain_no_reliable_sharing_signal'})
    round1 = comparisons(candidates, weights, len(bins))
    by_id = {c['scene_id']: c for c in candidates}
    for (a, b), adjusted in weights.items():
        if not adjusted:
            continue
        fid = sorted(adjusted)[0]
        old_weight = adjusted[fid]
        old_wins, new_wins = [], []
        lost = []
        for i, row in enumerate(initial):
            if a not in row['candidate_ids']:
                continue
            before = measure(by_id[a]['views'], i, adjusted)[0]
            after = measure(by_id[a]['views'], i, {**adjusted, fid: 0.})[0]
            if before > 0 and after == 0:
                lost.append(i)
            if b not in row['candidate_ids']:
                continue
            competitor = measure(by_id[b]['views'], i, weights[b, a])[0]
            old_wins.append((i, before - competitor >= CFG['margin'], competitor - before >= CFG['margin']))
            new_wins.append((i, after - competitor >= CFG['margin'], competitor - after >= CFG['margin']))
        keep_removal = old_wins == new_wins and not lost and any(row[1] or row[2] for row in old_wins)
        if keep_removal:
            adjusted[fid] = 0.
        logs.append({'round': 2, 'own': a, 'rival': b, 'feature': fid, 'weight_before': old_weight,
                     'tested_weight': 0., 'applied': keep_removal, 'lost_positive_samples': lost,
                     'tested_before': old_wins, 'tested_after': new_wins,
                     'reason': 'empirical_comparisons_preserved' if keep_removal else 'insufficient_or_changed_empirical_comparisons_reverted'})
    final = comparisons(candidates, weights, len(bins))
    return [initial, round1, final], logs


def apply_absence(row, observations, required_by_candidate):
    """Consume qualified observation records, never missing recognition evidence."""
    original = row['assignment']
    vetoes = []
    for obs in observations:
        if obs.get('state') != 'absent_supported' or not obs.get('scope_visible') or not obs.get('verified'):
            continue
        if not all(obs.get(key) for key in ('source_ref', 'scope', 'feature', 'basis')):
            continue
        for cid in row['candidate_ids']:
            if (obs['feature'], obs['scope']) in required_by_candidate.get(cid, ()):
                vetoes.append({'candidate': cid, 'observation': obs})
    survivors = [cid for cid in row['candidate_ids'] if cid not in {v['candidate'] for v in vetoes}]
    winner = original if original in survivors else None
    if vetoes and len(survivors) == 1:
        winner = survivors[0]
    return {**row, 'assignment': winner, 'absent_supported': vetoes}
