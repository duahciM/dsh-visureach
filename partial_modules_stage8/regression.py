"""Repeat prior isolation checks and verify Stage 8 source and dependency contracts."""

import subprocess
import traceback

import cv2

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage5.isolation import boundary_contract, influence_graph
from partial_modules_stage5.verify import verify_six
from partial_modules_stage6.pre_release import first_undeclared_change, scoped_outputs
from partial_modules_stage6.predict import gzip_json
from partial_modules_stage7.regression import check_source_history, window_comparability
from .run import H, M4_CASES, NEW, PREVIOUS


def signed_release():
    release = read(ROOT / 'evidenweft_phase1/protocol_lock.json')['baselines']
    archive = ROOT / 'plugin_offline/release/local-dsh-video-observer-0.6.0.tgz'
    snapshot = digest(archive)
    assert snapshot == release['visureach_archive_sha256']
    evidence = subprocess.check_output(['git', '-C', 'plugin_offline/github/EvidenWeft', 'rev-parse', 'HEAD'],
                                       cwd=ROOT, text=True).strip()
    assert evidence == release['evidenweft_main']
    return {'release_sha256': snapshot, 'EvidenWeft_head': evidence, 'inherited': check_source_history()}


def files_and_roots():
    lock = read(H / 'freeze.json')
    for section in ('prediction', 'assessor'):
        assert all(digest(ROOT / path) == sha for path, sha in lock[section].items())
    recovery = read(H / 'automatic/recovery.json')
    assert recovery['recovery_source_sha256'] == digest(H / 'replay_binding.py')
    assert recovery['initial_summary_sha256'] == digest(H / 'automatic/summary.json')
    source_count = verified = measures = nominations = direct_edges = gaps = 0
    for name, spec in read(H / 'inputs/cases.json').items():
        assert digest(ROOT / spec['source']) == spec['sha256']
        folder = (ROOT / spec['stage6'] if 'stage6' in spec else
                  H / 'bound' / name if name in NEW else PREVIOUS / 'automatic' / name)
        rows = read(folder / 'frame_receipts.json')
        assert all(r['id'].startswith(spec['sha256'] + ':') and r['time_base'] == spec['time_base'] for r in rows)
        assert all(digest(folder / r['png']) == r['png_sha256'] for r in rows)
        by_pts = {r['pts']: r['id'] for r in rows}
        source_ids = set(by_pts.values())
        source_count += len(rows)
        output = H / 'automatic' / name / 'result.json'
        data = read(output)
        assert data['pool_pts'] == list(by_pts)
        for arm in ('C', 'D', 'LP_plus_R2_diagnostic'):
            for result in data['branches'][arm]['audit']:
                for pts_list, ref_list in zip(result['source_prototype_pts'], result['source_prototype_roots']):
                    assert [by_pts[p] for p in pts_list] == ref_list
                if result.get('independent_investigation_root'):
                    assert not result['matched_R2']
                    assert not any(r['id'] == result['id'] for r in data['branches'][arm]['accepted'])
                    gaps += 1
        if name not in NEW:
            assert data['branches']['A']['accepted'] == read(PREVIOUS / 'automatic' / name / 'results.json')['arms']['P']['accepted']
            verified += 1
        if name not in M4_CASES:
            continue
        for opt in ('old', 'wide', 'past', 'future', 'complement', 'natural'):
            arm = gzip_json(H / 'm4' / name / f'{opt}.json.gz')
            domains = {}
            for row in arm['measurements']:
                assert row['original'] == by_pts[row['pts']]
                assert all(by_pts[q['pts']] == q['root'] for q in row['fit_refs'])
                assert row['pts'] not in row['comparison_pts_requested']
                domains[row['domain']['root']] = row
                measures += 1
            for item in arm['nominations']:
                assert item['dimension_id'] in ('D1', 'D3') and item['source_root'] in source_ids
                if opt != 'old' and 'nomination_reference_domain' in item:
                    evidence = domains[item['nomination_reference_domain']]
                    assert item['actual_comparison_roots'] == [r['root'] for r in evidence['fit_refs']]
                    assert not item['identity_scheduled']
                    if item['dimension_id'] == 'D3':
                        assert len(set(item['direct_reference_edge_roots'])) >= 2
                        assert set(item['direct_reference_edge_roots']) <= set(item['actual_comparison_roots'])
                        direct_edges += 1
                nominations += 1
            assert arm['identity_attempts'] == arm['additional_decode_calls'] == 0
    return {'verified_original_png_records': source_count, 'P_exact_replays': verified, 'M4_measures': measures,
            'D1_D3_checked': nominations, 'D3_direct_edge_checks': direct_edges,
            'unpublished_no_R2_measurements': gaps, 'prediction_and_labels_sha_checked': len(lock['prediction']) + len(lock['assessor'])}


def host(out):
    code = (ROOT / 'partial_modules_stage5/host_regression.mjs').read_text(encoding='utf8')
    key = "const study=resolve('partial_modules_stage5')"
    assert code.count(key) == 1
    script = H / 'host-regression.mjs'
    with script.open('x', encoding='utf8') as f:
        f.write(code.replace(key, f"const study=resolve('{H.name}')"))
    with (out / 'host.log').open('x', encoding='utf8') as log:
        subprocess.run(['node', str(script), 'host-regression'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=120)
    result = read(H / 'host-regression/result.json')
    assert all(x['state'] == 'stopped' for x in result['checked_stopped_node']['policy_hint']['statuses'].values())
    original = read(ROOT / 'partial_modules_stage5/host-verified/result.json')['original_attachment_without_scene']
    assert result['original_attachment_without_scene'] == original
    return {'same_attachment_hash_as_stage5': original, 'still_open_for_refinement_after_S0': True, 'PTS': result['probe']}


def main():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(730)
    out = H / 'regression'
    out.mkdir()
    results, failures = {}, {}
    def check(key, callback):
        try:
            results[key] = callback()
        except (AssertionError, IndexError, KeyError, OSError, ValueError, RuntimeError, subprocess.SubprocessError, cv2.error):
            failures[key] = traceback.format_exc()
    check('historical_and_release_signatures', signed_release)
    check('source_hash_PTS_registered_comparisons_nomination_ancestry', files_and_roots)
    check('D1_to_D6_independent_frozen_first', verify_six)
    check('strict_scene_scope_open_endpoints', boundary_contract)
    check('historical_scope_outputs', scoped_outputs)
    check('historical_budget_and_dependencies', window_comparability)
    specs = read(ROOT / 'partial_modules_stage4/inputs/source_inventory.json')
    for name in ('roles', 'camera', 'street-repeat'):
        check('independent_M5_' + name, lambda: influence_graph(name, specs[name], out))
    check('direct_native_read_and_refinement_after_S0', lambda: host(out))
    previous = {('source', 'PTS'): 8, ('observation', 'interpretation'): True}
    changed = previous | {('source', 'PTS'): 7, ('observation', 'interpretation'): False}
    failure = first_undeclared_change(previous, changed, {'observation'})
    assert failure['stage'] == 'source'
    results['injected_first_change'] = failure
    write(out / 'result.json', {'passed': not failures, 'checks': results, 'failures': failures})
    print('passed', len(results), 'failures', list(failures))
    return not failures


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
