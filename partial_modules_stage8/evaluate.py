"""Score sealed responses, gate independent factors, rerun the eligible combination."""

from collections import Counter
from time import perf_counter

import cv2

from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_research.run import gz_dump
from partial_modules_stage6.evaluate import match, role_rows, summarized_roles
from partial_modules_stage6.predict import gzip_json
from . import m4, run

H = run.H
METRIC = 'assessed_native_ROI_support_pts'
CHOICES = ('old', 'wide', 'past', 'future', 'complement', 'natural')


def first_rows(name, spec, rows):
    catalog = run.get_catalogue(name, spec, rows)
    roots = {row['pts']: row['id'] for row in rows}
    return [n | {'pts': p, 'source_root': frame.get('frame_id', roots.get(p))}
            for p, frame in catalog.items() for n in frame['nominations']]


def scored(items, truth, dimensions):
    roles = role_rows(items, truth, dimensions)
    return {'supports': roles, 'metrics': summarized_roles(roles, truth)}


def positives(score):
    return {key for key, value in score['supports'].items() if value[METRIC]}


def changes(old, new):
    return {'gained_roles': sorted(positives(new) - positives(old)),
            'lost_roles': sorted(positives(old) - positives(new)),
            'combination_delta': new['metrics']['complete_assessed_combinations'][METRIC]['covered'] -
                                 old['metrics']['complete_assessed_combinations'][METRIC]['covered']}


def m4_case(name, spec, truth):
    folder = H / 'm4' / name
    results = {key: gzip_json(folder / f'{key}.json.gz') for key in CHOICES}
    rows = run.get_frames(name, spec)
    dims = rows[0]['native_size'], rows[0]['analysis_size']
    first = first_rows(name, spec, rows)
    common = set.intersection(*(set(result['requested_by_pts']) for key, result in results.items() if key != 'natural'))
    common = {int(p) for p in common
              if len({results[key]['requested_by_pts'][p] for key in CHOICES[:-1]}) == 1}
    budgets = {}
    for key, data in results.items():
        budgets[key] = sum(len(row['comparison_pts_requested']) for row in data['measurements']
                           if row['pts'] in common and row['state'] != 'quality_abstention')
    assert len({budgets[key] for key in CHOICES[:-1]}) <= 1
    assert results['old']['actual_registration_attempts'] == results['wide']['actual_registration_attempts']
    if truth is None:
        truth = {'roles': [], 'relations': []}
    labelled = bool(truth['roles'])
    scores = {}
    for subset, allowed in (('full', None), ('matched_neighborhoods', common)):
        initial = [row for row in first if allowed is None or row['pts'] in allowed]
        scores[subset] = {'FIRST': scored(initial, truth, dims)}
        for key, res in results.items():
            proposed = [row for row in res['nominations'] if allowed is None or row['pts'] in allowed]
            scores[subset][key] = {'supplement': scored(proposed, truth, dims), 'combined': scored(initial + proposed, truth, dims),
                'candidate_counts': dict(Counter(row['dimension_id'] for row in proposed)),
                'unique_nominated_pts': len({row['pts'] for row in proposed})}
        for key in CHOICES:
            scores[subset][key]['changed_from_old_plus_FIRST'] = changes(scores[subset]['old']['combined'],
                                                                       scores[subset][key]['combined'])
    summary = read(folder / 'summary.json')
    summary['worker_initialization_seconds'] = {key: res['worker_initialization_seconds'] for key, res in results.items()}
    return {'source_coarse_count': len(rows), 'frozen_first_records': len(first), 'roi_assessed': labelled,
            'equal_schedule_pts': sorted(common), 'equal_schedule_actual_registrations': budgets,
            'scores': scores, 'execution': summary}


def transition_case(name, truth):
    data = read(H / 'automatic' / name / 'result.json')
    gold = truth['bounds'] if truth else None
    arms = {}
    for key, res in data['branches'].items():
        bouts = [v['pts'] for v in res['accepted']]
        score = match(bouts, gold) if gold is not None else None
        unmatched = [] if gold is None else score['unmatched_outputs']
        causes = dict(Counter(s for row in res['audit'] for s in row['cause']))
        crossings = [{'closed_scope': s['pts'], 'enclosed_missing_reference': bound}
                     for s in res['closed_scope']['closed'] for bound in gold or []
                     if s['pts'][0] < bound[0] and bound[-1] < s['pts'][-1]]
        arms[key] = {'score': score, 'published_to_research': bouts, 'unmatched': unmatched,
            'closed_scopes': res['closed_scope']['closed'], 'swallowed_reference_envelopes': crossings,
            'rejected_diagnostics': causes, 'outward_attempts': sum(any(row.get('outward_steps', [])) for row in res['audit']),
            'accepted_after_reselection': [row['pts'] for row in res['audit']
                                            if any(row.get('outward_steps', [])) and row['state'] == 'visual_context_change_candidate'],
            'measured_shared_structure_rows': sum(bool(row.get('support_states', {}).get('shared_direct_structure')) for row in res['audit']),
            'middle_hypothesis_rows': sum(bool(row.get('middle_candidates')) for row in res['audit']),
            'new_gap_qualifies_investigation': [row['pts'] for row in res['audit'] if row['state'] == 'new_investigation_unverified'],
            'new_gap_measured_then_unknown': [row['pts'] for row in res['audit']
                if row['state'] == 'unknown' and row.get('independent_investigation_root')]}
    return {'reference': gold, 'arms': arms, 'cost': data['cost']}


def gate(boundaries, windows, refined):
    cs = windows['shared_structure']['scores']
    full, same = cs['full'], cs['matched_neighborhoods']
    full_delta = full['wide']['changed_from_old_plus_FIRST']
    same_delta = same['wide']['changed_from_old_plus_FIRST']
    positive = lambda delta: bool(delta['gained_roles']) or delta['combination_delta'] > 0
    clean_full = not full_delta['lost_roles'] and full_delta['combination_delta'] >= 0
    first_kept = (positives(full['FIRST']) <= positives(full['wide']['combined']) and
        full['FIRST']['metrics']['complete_assessed_combinations'][METRIC]['covered']
        <= full['wide']['combined']['metrics']['complete_assessed_combinations'][METRIC]['covered'])
    b = first_kept and (positive(same_delta) or (clean_full and positive(full_delta)))
    counts = {}
    for key in ('A', 'C', 'D', 'LP_plus_R2_diagnostic'):
        current = [row['arms'][key]['score'] for row in boundaries.values() if row['reference'] is not None]
        counts[key] = {field: sum(len(item[field]) for item in current)
                       for field in ('matches', 'unmatched_outputs', 'missed_reference')}
    improvements = [key for key, row in boundaries.items() if row['reference'] is not None and
        len(row['arms']['C']['score']['matches']) > len(row['arms']['A']['score']['matches'])]
    c = counts['C']['matches'] >= counts['A']['matches'] and counts['C']['unmatched_outputs'] <= counts['A']['unmatched_outputs'] and bool(improvements)
    d = refined.get('status') == 'verified_new_boundary'
    return {'B': b, 'C': c, 'D': d, 'C_new_supported_matches': improvements, 'boundary_totals': counts,
            'B_reference_roles_preserved': first_kept, 'B_full_delta': full_delta, 'B_equal_target_delta': same_delta,
            'D_validation_status': refined['status']}


def combination(specs, assessor, decision, matrix):
    output = H / 'combination'
    output.mkdir()
    envelope_arm = 'C' if decision['C'] else 'A'
    selected_m4 = 'wide' if decision['B'] else 'old'
    assert not decision['D'], 'Verified D publication requires a new independent qualification'
    result = {}
    for name, spec in specs.items():
        if name not in matrix:
            continue
        source = H / 'automatic' / name / 'result.json'
        base = digest(source)
        auto = read(source)
        current = auto['branches'][envelope_arm]
        record = {'native_scope': current['closed_scope'], 'envelopes': current['accepted'],
                  'automatic_inputs_sha256': base, 'FIRST_pts': auto['frozen_first_pts']}
        if name in run.M4_CASES:
            rows = run.get_frames(name, spec)
            old = first_rows(name, spec, rows)
            worker = m4.ScopedForeground(rows)
            full = m4.audit(worker, rows, selected_m4)
            frozen = gzip_json(H / 'm4' / name / f'{selected_m4}.json.gz')
            assert full['nominations'] == frozen['nominations']
            scoped = m4.audit(worker, rows, selected_m4, current['closed_scope'])
            scopes = {q['id']: q['pts'] for q in current['closed_scope']['closed']}
            for proof in scoped['measurements']:
                lo, hi = scopes[proof['context']]
                assert all(lo < p < hi for p in [proof['pts'], *proof['comparison_pts_requested']])
            dims = rows[0]['native_size'], rows[0]['analysis_size']
            truth = assessor[name]
            record['M4'] = {
                'global_measures': len(full['measurements']), 'scoped_measures': len(scoped['measurements']),
                'global_operations': full['actual_registration_attempts'], 'scoped_operations': scoped['actual_registration_attempts'],
                'global_seconds': full['seconds'], 'scoped_seconds': scoped['seconds'],
                'global_with_first': scored(old + full['nominations'], truth, dims),
                'strict_scope_with_first': scored(old + scoped['nominations'], truth, dims),
                'strict_scope_alone': scored(scoped['nominations'], truth, dims)}
            gz_dump(output / f'{name}-m4.json.gz', {'global': full, 'scoped': scoped, 'FIRST': old})
        assert digest(source) == base
        result[name] = record
    return {'selected': {'R1_candidate': envelope_arm, 'M4_schedule': selected_m4, 'R2_publication': 'P'},
            'cases': result, 'formal_release_changed': False, 'scope_is_optional': True}


def main():
    start = perf_counter()
    locked = read(H / 'freeze.json')
    assert all(digest(ROOT / path) == sha for path, sha in locked['assessor'].items())
    assert all(not read(H / route / 'summary.json')['read_assessor'] for route in ('automatic', 'm4'))
    cases = read(H / 'inputs/cases.json')
    truth = read(ROOT / 'partial_modules_stage6/assessment/sealed.json')
    truth.update(read(ROOT / 'partial_modules_stage7/assessment_extra.json'))
    truth.update(read(H / 'assessment/sealed.json'))
    truth = {key: truth[key] for key in cases}
    boundaries = {name: transition_case(name, truth[name]) for name in cases}
    cv2.setNumThreads(1)
    windows = {name: m4_case(name, cases[name], truth[name]) for name in run.M4_CASES}
    decision = gate(boundaries, windows, read(H / 'refinement/result.json'))
    combined = combination(cases, truth, decision, boundaries)
    write(H / 'evaluation.json', {'transition': boundaries, 'windows': windows, 'decision': decision, 'E': combined,
        'recovered_input_contract': read(H / 'automatic/recovery.json'), 'evaluation_and_E_seconds': perf_counter() - start})
    print(decision)


if __name__ == '__main__':
    main()
