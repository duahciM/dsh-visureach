"""Run blind Stage 8 module ablations without opening any assessment file."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import traceback

import cv2

from dacf_research.field import clip_encoder
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_research.run import gz_dump, raw_frames
from dacf_round2.regions import discover
from evidenweft_phase1.source_review import timeline
from partial_modules_stage6.downstream import frames, nominate_arm
from partial_modules_stage6.envelopes import qualify as old_qualify
from partial_modules_stage6.predict import gzip_json
from partial_modules_stage7 import dacf
from partial_modules_stage7.local_observation import Layout, local_groups
from partial_modules_stage7.predict import build_feature_input
from . import m4, transition

H = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'partial_modules_stage7'
NEW = ('shared_structure', 'real_holdout_window')
M4_CASES = ('reel', 'real_return', 'real_pan', 'new_recording', *NEW)


def bind_inputs():
    result = {}
    for name, spec in read(H / 'inputs/cases.json').items():
        source = ROOT / spec['source']
        assert digest(source) == spec['sha256']
        if name not in NEW:
            continue
        out = H / 'bound' / name
        out.mkdir(parents=True)
        start = perf_counter()
        info, clock = timeline(source)
        chosen = set()
        last = float('-inf')
        for row in clock:
            if spec['extent'][0] <= row['pts'] <= spec['extent'][1] and row['seconds'] >= last + .2 - 1e-7:
                chosen.add(row['frame'])
                last = row['seconds']
        raw_frames(source, info, clock, chosen, out, spec['sha256'], 320)
        raster = perf_counter() - start
        rows = frames(out)
        first = nominate_arm(rows, spec, 'frozen-first', None)
        write(out / 'first.json', first)
        result[name] = {'source_frames_sequentially_decoded': len(clock), 'coarse_originals': len(chosen),
                        'raster_seconds': raster, 'first_seconds': first['seconds'],
                        'first_sha256': digest(out / 'first.json'), 'image_receipts_sha256': digest(out / 'frame_receipts.json')}
    write(H / 'bound' / 'manifest.json', result)


def get_frames(name, spec):
    old = PREVIOUS / 'automatic' / name
    if 'stage6' in spec:
        return frames(ROOT / spec['stage6'])
    return frames(H / 'bound' / name if name in NEW else old)


def get_catalogue(name, spec, rows):
    if 'stage6' in spec:
        initial = read(ROOT / spec['stage6'] / 'auto.json')['initial_index']
        return {row['pts']: row for row in read(ROOT / initial / 'catalog.json')['frames']}
    if name in NEW:
        values = [row for bundle in read(H / 'bound' / name / 'first.json')['nomination_groups']
                  for row in bundle['nominations']]
        catalog = {}
        for row in values:
            catalog.setdefault(row['pts'], {'pts': row['pts'], 'nominations': []})['nominations'].append(row)
        return catalog
    return {row['pts']: row for row in read(PREVIOUS / 'automatic' / name / 'first_control_index.json')}


def warm_history(bank, layout, old):
    available = {r['id']: rank for rank, r in bank.rows.items()}
    for row in gzip_json(old / 'layout_fit.json.gz'):
        if all(root in available for root in row['source_roots']):
            x, y = (available[root] for root in row['source_roots'])
            assert [x, y] == row['ranks']
            layout.pairs[x, y] = row
    for row in gzip_json(old / 'raw_photometry.json.gz'):
        if all(root in available for root in row['frames']):
            x, y = (available[root] for root in row['frames'])
            bank.evidence[x, y] = row
            bank.matrix[tuple(sorted((x, y)))] = row['metrics']


def counts(bank, layout):
    return (len(bank.evidence), len(bank.links), len(layout.pairs), len(bank.matrix))


def automatic(name, spec, out, encoder):
    start = perf_counter()
    rows = get_frames(name, spec)
    catalogue = get_catalogue(name, spec, rows)
    historic = PREVIOUS / 'automatic' / name
    prior = read(historic / 'results.json') if name not in NEW else None
    previous_r2 = read(ROOT / spec['stage6'] / 'auto.json') if 'stage6' in spec else None
    spans = prior['examined_windows_pts'] if prior else [spec['extent']]
    source_io = perf_counter() - start
    start = perf_counter()
    bank, windows = build_feature_input(rows, spans, encoder)
    preparation = perf_counter() - start
    layout = Layout(bank)
    start = perf_counter()
    if prior:
        warm_history(bank, layout, historic)
    historical_cache_time = perf_counter() - start
    warmed = counts(bank, layout)
    policy = read(ROOT / 'dacf_round2/protocol.json')
    sha, extent = spec['sha256'], spec['extent']
    start = perf_counter()
    if previous_r2:
        regions = previous_r2['r2_regions']
    else:
        regions = [row | {'id': f'{i}-{row["id"]}'} for i, pool in enumerate(windows)
                   for row in discover(bank, pool, catalogue, policy)['regions']]
    if prior:
        old_groups, local = prior['old_groups_scope'], prior['development_local_groups']
    else:
        old_groups = old_qualify(bank, windows, catalogue, regions, policy, sha, extent)['all_adjacent_groups']
        local = [local_groups(layout, pool, bank.config['maximum_gap_seconds'],
                 bank.config['automatic']['dwell_seconds'], policy['prototype_neighborhood']['radius_s']) for pool in windows]
    prep_groups = perf_counter() - start
    common = (catalogue, regions, policy, sha, extent)
    branches, timings = {}, {}
    calls = {
        'A': lambda: dacf.qualify(bank, layout, windows, old_groups, *common, False, True),
        'C': lambda: transition.qualify(bank, layout, windows, local, *common, local=True, outward=True),
        'D': lambda: transition.qualify(bank, layout, windows, old_groups, *common, release_r2=True),
        'LP_plus_R2_diagnostic': lambda: transition.qualify(bank, layout, windows, local, *common,
                                                            local=True, release_r2=True),
    }
    for arm, run in calls.items():
        before, start = counts(bank, layout), perf_counter()
        branches[arm] = run()
        timings[arm] = {'seconds': perf_counter() - start,
            'additional_new_pair_local_fit_layout_distance_counts': [b-a for a, b in zip(before, counts(bank, layout))]}
    if prior:
        assert branches['A']['accepted'] == prior['arms']['P']['accepted'], name
    assert branches['A']['accepted'] == branches['D']['accepted']
    out.mkdir()
    gz_dump(out / 'layout.json.gz', list(layout.pairs.values()))
    gz_dump(out / 'pair_photometry.json.gz', list(bank.evidence.values()))
    gz_dump(out / 'local_fits.json.gz', list(bank.links.values()))
    write(out / 'result.json', {'source_sha256': sha, 'time_base': spec['time_base'],
        'pool_pts': [r['pts'] for r in rows], 'scanned_windows': spans, 'frozen_first_pts': sorted(catalogue),
        'branches': branches, 'historical_r2': regions, 'default_baseline_exact_replayed': bool(prior),
        'cost': {'source_png_audit_and_raster_seconds': source_io, 'feature_seconds': preparation,
            'historical_cache_load_seconds': historical_cache_time, 'prior_pair_local_fit_layout_distance_counts': warmed,
            'group_and_R2_preparation_seconds': prep_groups, 'by_arm': timings, 'clip': bank.clip_state,
            'original_video_decodes_in_this_predictor': 0}})
    return {key: [row['pts'] for row in val['accepted']] for key, val in branches.items()}


def m4_case(name, spec, out):
    start = perf_counter()
    rows = get_frames(name, spec)
    raster = perf_counter() - start
    out.mkdir()
    counts_by = {}
    for choice in ('old', 'wide', 'past', 'future', 'complement', 'natural'):
        start = perf_counter()
        worker = m4.ScopedForeground(rows)
        initialization = perf_counter() - start
        result = m4.audit(worker, rows, choice) | {'worker_initialization_seconds': initialization}
        gz_dump(out / f'{choice}.json.gz', result)
        counts_by[choice] = {k: result[k] for k in ('actual_registration_attempts', 'seconds', 'measurement_statuses')}
    write(out / 'summary.json', {'options': counts_by, 'common_source_raster_hash_and_load_s': raster})
    return counts_by


def main(kind):
    lock = read(H / 'freeze.json')
    assert all(digest(ROOT / path) == sha for path, sha in lock['prediction'].items())
    cv2.setNumThreads(1)
    if kind == 'bind':
        bind_inputs()
        return
    assert all(digest(H / 'bound' / name / 'first.json') == entry['first_sha256']
               and digest(H / 'bound' / name / 'frame_receipts.json') == entry['image_receipts_sha256']
               for name, entry in read(H / 'bound/manifest.json').items())
    target = H / kind
    target.mkdir()
    if kind == 'automatic':
        try:
            encoder = clip_encoder()
        except (ImportError, OSError, ValueError, RuntimeError):
            encoder = None
            write(target / 'clip_failure.json', {'details': traceback.format_exc()})
    completed, failures = {}, {}
    for name, spec in read(H / 'inputs/cases.json').items():
        if kind == 'm4' and name not in M4_CASES:
            continue
        cv2.setRNGSeed(730)
        try:
            completed[name] = automatic(name, spec, target / name, encoder) if kind == 'automatic' else m4_case(name, spec, target / name)
            print(name, completed[name] if kind == 'automatic' else 'complete', flush=True)
        except (ArithmeticError, AssertionError, KeyError, IndexError, ValueError, RuntimeError, OSError, cv2.error):
            failures[name] = traceback.format_exc()
            print(name, failures[name], flush=True)
    write(target / 'summary.json', {'completed': completed, 'failures': failures, 'read_assessor': False})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('kind', choices=['bind', 'automatic', 'm4'])
    main(parser.parse_args().kind)
