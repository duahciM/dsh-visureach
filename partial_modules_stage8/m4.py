"""Scoped reference selection inside the existing M4 residual / D1 and D3 consumers."""

from collections import Counter
from time import perf_counter

import cv2
import numpy as np

from partial_modules_stage1.study import root
from partial_modules_stage6.downstream import blocks
from partial_modules_stage7.windowed import Foreground, legal_refs, measure_reasons


def references(stamps, i, option):
    old, = legal_refs(stamps, i, 'old')
    k = len(old)
    if option in ('old', 'wide'):
        return legal_refs(stamps, i, option)
    past, future = stamps[:i], stamps[i + 1:]
    if option == 'past' and len(past) >= k:
        return [past[-k:]] if k else []
    if option == 'future' and len(future) >= k:
        return [future[:k]] if k else []
    if option == 'complement' and k >= 6 and len(past) >= k // 2 and len(future) >= k - k // 2:
        return [past[-(k // 2):], future[:k - k // 2]]
    return []


class ScopedForeground(Foreground):
    def measure_domain(self, pts, refs, kind):
        measured = self.measure(pts, refs)
        proof = root('m4-reference-domain-v8', measured['rank_independent_root'], kind)
        scope = {'root': proof, 'prior_m4_root': measured['rank_independent_root'], 'selection': kind,
                 'comparability': measured['state'], 'fits': measured['fit_refs'],
                 'capture_pts': pts, 'requested_pts': refs,
                 'unknown_reference_pts': sorted(set(refs) - {r['pts'] for r in measured['fit_refs']}),
                 'interpretation': 'registered reference states and local contrasts in the current target image only'}
        if measured['state'] in ('registration_abstention', 'quality_abstention'):
            return measured | {'domain': scope}
        w, h = self.size
        stack, masks, edges = [], [], []
        for ref in measured['fit_refs']:
            matrix = np.array(ref['matrix'], dtype=np.float32)
            aligned = cv2.warpAffine(self.colors[ref['pts']], matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
            stack.append(aligned)
            masks.append(cv2.warpAffine(np.ones((h, w), np.float32), matrix, (w, h)) > .999)
            edges.append(cv2.dilate(cv2.Canny(np.uint8(np.clip(aligned[:, :, 0] * 255, 0, 255)), 48, 124),
                                    np.ones((5, 5), np.uint8)) > 0)
        valid = np.logical_and.reduce(masks)
        median = np.median(stack, axis=0)
        delta = np.max(np.abs(np.asarray(stack) - self.colors[pts]), axis=-1)
        from_median = np.max(np.abs(median - self.colors[pts]), axis=-1)
        tile_rows = []
        for y in range(2):
            for x in range(3):
                region = np.s_[y*h//2:(y+1)*h//2, x*w//3:(x+1)*w//3]
                keep = valid[region]
                if not keep.any():
                    continue
                disagreement = [float(np.mean(d[region][keep])) for d in delta]
                tile_rows.append({'cell': [x, y], 'analysis_roi': [x*w//3, y*h//2,
                    (x+1)*w//3-x*w//3, (y+1)*h//2-y*h//2],
                    'overlap_pixels': int(keep.sum()), 'registered_reference_mad': disagreement,
                    'median_reference_mad': float(np.mean(from_median[region][keep])),
                    'reference_target_mad_range': max(disagreement) - min(disagreement)})
        for item in measured['candidates']:
            x, y, rw, rh = item['bbox']
            patch = np.s_[y:y+rh, x:x+rw]
            keep = valid[patch]
            item['registered_reference_mad'] = [float(np.mean(d[patch][keep])) for d in delta] if keep.any() else []
            if 'context_edge_xy' in item:
                x, y = item['context_edge_xy']
                item['direct_edge_roots'] = [r['root'] for r, edge, mask in zip(measured['fit_refs'], edges, masks)
                                             if edge[y, x] and mask[y, x]]
        return measured | {'domain': scope | {'tiles': tile_rows,
            'statistic': 'mean maximum-channel absolute YCrCb difference on actual common overlap, normalized 0–1',
            'compensation': 'legacy translation registration, without a fitted gain/offset in this measurement',
            'world_stability_inferred': False}}


def nominations(worker, measured, option):
    old = measure_reasons(worker, measured)
    if option == 'old':
        return [r for r in old if r['dimension_id'] in ('D1', 'D3')]
    result = []
    domain = measured['domain']
    by_box = {tuple(row['bbox']): row for row in measured['candidates']}
    for row in old:
        if row['dimension_id'] not in ('D1', 'D3'):
            continue
        direct = by_box[tuple(row['applicable_region'])].get('direct_edge_roots', [])
        if row['dimension_id'] == 'D3' and len(set(direct)) < 2:
            continue
        roots = [ref['root'] for ref in domain['fits']]
        reasons = {'nomination_reference_domain': domain['root'], 'actual_comparison_roots': roots,
                   'identity_scheduled': False, 'independent_dimension': True}
        if row['dimension_id'] == 'D3':
            reasons['direct_reference_edge_roots'] = direct
        result.append(row | reasons | {'calculation_id': root('scoped-nomination-v8', domain['root'],
                       row['dimension_id'], row['applicable_region'], direct if row['dimension_id'] == 'D3' else [])})
    return result


def audit(worker, frames, option, scope=None):
    start = perf_counter()
    before = worker.calls
    records, proposed, requested, unsupported = [], [], {}, []
    for context, rows in blocks(frames, scope):
        stamps = [frame['pts'] for frame in rows]
        for i, pts in enumerate(stamps):
            choice = 'old' if option == 'natural' else option
            neighborhoods = references(stamps, i, choice)
            if not neighborhoods:
                unsupported.append(pts)
                continue
            used = set()
            for refs in neighborhoods:
                measured = worker.measure_domain(pts, refs, choice)
                records.append(measured | {'context': context})
                proposed.extend(nominations(worker, measured, choice))
                used.update(refs)
            if option == 'natural':
                tiles = measured['domain'].get('tiles', [])
                is_unknown = measured['state'] == 'registration_abstention'
                has_distinct_ref_states = any(t['reference_target_mad_range'] >= .10 for t in tiles)
                if is_unknown or has_distinct_ref_states:
                    wide, = references(stamps, i, 'wide')
                    if set(wide) != used:
                        measured = worker.measure_domain(pts, wide, 'wide')
                        records.append(measured | {'context': context,
                            'continuation_reason': 'unknown_registration' if is_unknown else 'reference_state_heterogeneity'})
                        proposed.extend(nominations(worker, measured, 'wide'))
                        used.update(wide)
            requested[pts] = sum(len(r['comparison_pts_requested']) for r in records[-2:]
                                 if r['pts'] == pts)
    return {'measurements': records, 'nominations': proposed, 'unsupported_schedule_pts': unsupported,
            'requested_by_pts': requested, 'actual_registration_attempts': worker.calls - before,
            'additional_decode_calls': 0, 'identity_attempts': 0, 'seconds': perf_counter() - start,
            'measurement_statuses': dict(Counter(row['state'] for row in records))}
