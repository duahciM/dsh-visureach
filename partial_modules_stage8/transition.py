"""Development R1 and R2 changes, sharing the Stage 7 evidence and qualification primitives."""

from itertools import combinations, pairwise

from dacf_round2.regions import anchor, distinguish
from partial_modules_stage1.study import root
from partial_modules_stage5.scopes import derive
from partial_modules_stage6.envelopes import investigation_chains
from partial_modules_stage7.dacf import competition, support_at


def reference_options(bank, layout, group, side, policy):
    original = group['exit_prototypes' if side == 'left' else 'entry_prototypes']
    yield original
    reference = original[-1] if side == 'left' else original[0]
    idx = group['ranks'].index(reference)
    direction = -1 if side == 'left' else 1
    yielded = 1
    for hop in range(2, 5):
        endpoint = idx + hop * direction
        ranks = group['ranks'][endpoint - 2:endpoint + 1] if side == 'left' else group['ranks'][endpoint:endpoint + 3]
        if endpoint < 0 or len(ranks) != 3:
            continue
        stamps = [bank.rows[p]['seconds'] for p in ranks]
        if abs(bank.rows[reference]['seconds'] - bank.rows[group['ranks'][endpoint]]['seconds']) > .85:
            break
        if not bank.config['automatic']['dwell_seconds'] <= stamps[-1] - stamps[0] <= policy['prototype_neighborhood']['radius_s']:
            continue
        if any(b - a > bank.config['maximum_gap_seconds'] for a, b in pairwise(stamps)):
            continue
        if all(layout.compare(a, b)['supported'] for a, b in combinations(ranks, 2)):
            yield ranks
            yielded += 1
            if yielded == 3:
                break


def candidate(bank, layout, pool, a, b, catalogue, regions, policy, sha, local, release_r2):
    lo, hi = bank.rows[a[-1]]['pts'], bank.rows[b[0]]['pts']
    linked = [r['id'] for r in regions if max(lo, r['bounds_pts'][0]) < min(hi, r['bounds_pts'][1])]
    ref = root('context-envelope-v7', sha, [lo, hi], local, True)
    record = {'id': ref, 'pts': [lo, hi], 'matched_R2': linked,
              'source_prototype_pts': [[bank.rows[p]['pts'] for p in group] for group in (a, b)],
              'source_prototype_roots': [[bank.rows[p]['id'] for p in group] for group in (a, b)]}
    if not linked and not release_r2:
        return record | {'state': 'unknown', 'cause': ['no intersecting existing DACF investigation']}
    first, last = anchor(bank, a[-1], a), anchor(bank, b[0], b)
    comparison = distinguish(bank, first, last, catalogue, policy)
    record.update(first=first, last=last, comparison=comparison)
    if not {'hsv', 'clip'} & set(comparison['supported_by']):
        return record | {'state': 'unknown', 'cause': ['no corroborating contrast in existing appearance channels']}
    query = [p for p in pool if lo <= bank.rows[p]['pts'] <= hi]
    probes = [support_at(bank, layout, p, a, b, local) for p in sorted(set(a[-2:] + b[:2] + query))]
    by_rank = {r['rank']: r for r in probes}
    states = {k: [] for k in ('left', 'right', 'both', 'neither', 'shared_direct_structure')}
    for row in probes:
        left, right = row['left'], row['right']
        side = 'both' if left and right else 'left' if left else 'right' if right else 'neither'
        states[side].append(row['pts'])
        if row['evidence'].get('shared_supported_cells'):
            states['shared_direct_structure'].append(row['pts'])
    causes = []
    if not (all(by_rank[p]['left'] and not by_rank[p]['right'] for p in a[-2:]) and
            all(by_rank[p]['right'] and not by_rank[p]['left'] for p in b[:2])):
        causes.append('two separate locally supported prototypes per side not established')
    if any(bank.rows[j]['seconds'] - bank.rows[i]['seconds'] > bank.config['maximum_gap_seconds']
           for i, j in pairwise(query)):
        causes.append('missing coarse observations within the span')
    if states['left'] and states['right'] and max(states['left']) > min(states['right']):
        causes.append('local support reverses along the sampled timeline')
    adjacent = [bank.metrics_on_pair(i, j) for i, j in pairwise(query)]
    competing = competition(bank, layout, a, b, adjacent, True)
    if competing['withheld_context_envelope']:
        causes.append('cross-side same-structure with measured view/light alternative')
    if local:
        # A stable, separate middle can be investigated independently. It is not silently merged into
        # either side when growing the outer bracket. Two interior samples are only a hypothesis.
        middle = [p for p in query[1:-1] if not by_rank[p]['left'] and not by_rank[p]['right']]
        record['middle_candidates'] = [
            {'pts': [bank.rows[i]['pts'], bank.rows[j]['pts']], 'root': layout.compare(i, j)['id'],
             'scope': 'direct middle geometry only, identity and stability unknown'}
            for i, j in pairwise(middle) if query.index(j) - query.index(i) == 1 and layout.compare(i, j)['supported']]
    status = 'unknown' if causes else 'visual_context_change_candidate' if linked else 'new_investigation_unverified'
    investigation = None
    if not linked:
        investigation = root('r2-independent-investigation-v8', sha, [lo, hi], record['source_prototype_roots'])
    return record | {'cause': causes, 'state': status, 'support_profiles': probes, 'support_states': states,
                     'competition': competing, 'independent_investigation_root': investigation,
                     'publication_basis': 'historical and current evidence' if linked else 'requires separate source verification'}


def qualify(bank, layout, windows, groups, catalogue, regions, policy, sha, extent, *,
            local=False, outward=False, release_r2=False):
    accepted, audit = [], []
    for pool, bundle in zip(windows, groups):
        for left, right in pairwise(bundle['groups']):
            if outward:
                lefts = list(reference_options(bank, layout, left, 'left', policy))
                rights = list(reference_options(bank, layout, right, 'right', policy))
            elif local:
                lefts, rights = [left['exit_prototypes']], [right['entry_prototypes']]
            else:
                lefts, rights = [left['prototypes']], [right['prototypes']]
            sequence = sorted(((i, j) for i in range(len(lefts)) for j in range(len(rights))),
                              key=lambda item: (sum(item), max(item), item))
            for i, j in sequence:
                row = candidate(bank, layout, pool, lefts[i], rights[j], catalogue, regions, policy, sha, local, release_r2)
                row.update(groups=[left['id'], right['id']], outward_steps=[i, j], attempted_for=sequence[0])
                audit.append(row)
                if row['state'] == 'visual_context_change_candidate':
                    accepted.append({'id': row['id'], 'pts': row['pts'], 'kind': 'observed_spatial_context_change',
                        'source_sha256': sha, 'scope': 'observed projection context only; mechanism and physical scene identity remain unknown'})
                    break
                if row['state'] == 'new_investigation_unverified':
                    break
    return {'accepted': accepted, 'audit': audit, 'closed_scope': derive(sha, extent, accepted),
            'R2_chains': investigation_chains(regions)}
