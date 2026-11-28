"""Verify one independent gap through a real AMR tool call and an independent decoder."""

from fractions import Fraction
import subprocess
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round3.acquire import Reader
from evidenweft_phase1.source_review import timeline
from partial_modules_stage7.local_observation import Layout
from partial_modules_stage7.predict import build_feature_input
from .run import H, get_frames


def plan_node(key, pts, time_base, parent, children, first, known):
    return {'id': key, 'kind': 'independent_gap' if parent else 'source', 'bounds_pts': pts,
            'start': float(pts[0] * time_base), 'end': float(pts[1] * time_base), 'parent_id': parent,
            'children': children, 'nominees': first, 'known_pts': known, 'stops': {}}


def choose(cases):
    eligible = []
    for name, spec in cases.items():
        result_path = H / 'automatic' / name / 'result.json'
        if not result_path.exists():
            continue
        result = read(result_path)
        for arm in ('D', 'LP_plus_R2_diagnostic'):
            for row in result['branches'][arm]['audit']:
                if row['matched_R2'] or not row.get('independent_investigation_root'):
                    continue
                support = row['support_states']
                lo, hi = row['pts']
                key = (row['state'] != 'new_investigation_unverified', not (support['left'] and support['right']),
                       arm != 'D', name, lo, hi)
                eligible.append((key, name, spec, result_path, result, arm, row))
    for _, name, spec, path, result, arm, row in sorted(eligible, key=lambda x: x[0]):
        movie = ROOT / spec['source']
        info, clock = timeline(movie)
        lo, hi = row['pts']
        pool = set(result['pool_pts'])
        novel = [r for r in clock if lo < r['pts'] < hi and r['pts'] not in pool]
        if novel:
            midpoint = min(novel, key=lambda p: (abs(p['pts'] - (lo+hi)/2), p['pts']))
            return name, spec, path, result, arm, row, movie, info, clock, midpoint
    return None


def run():
    out = H / 'refinement'
    out.mkdir()
    selected = choose(read(H / 'inputs/cases.json'))
    if selected is None:
        write(out / 'result.json', {'status': 'unknown_no_measured_new_gap_with_interior_source_frame'})
        return
    name, spec, path, coarse, arm, row, movie, info, clock, point = selected
    before = digest(path)
    assert digest(movie) == spec['sha256']
    time_base = Fraction(info['time_base'])
    key = row['independent_investigation_root']
    first = coarse['frozen_first_pts']
    lo, hi = row['pts']
    write(out / 'request.json', {'case': name, 'source': str(movie), 'arm': arm, 'node': key,
         'interior': point, 'bounds_pts': row['pts'], 'prediction_sha256': before, 'initial_status': row['state'],
         'initial_abstentions': row['cause'], 'prior_R2_ids': row['matched_R2']})
    write(out / 'plan.json', {'sha256': spec['sha256'], 'time_base': info['time_base'], 'clock': clock,
        'width': info['width'], 'height': info['height'], 'edges': [],
        'nodes': [plan_node('source', spec['extent'], time_base, None, [key], first, coarse['pool_pts']),
                  plan_node(key, [lo, hi], time_base, 'source', [], [x for x in first if lo <= x <= hi], [lo, hi])]})
    with (out / 'host.log').open('x', encoding='utf-8') as log:
        subprocess.run(['node', str(H / 'amr_probe.mjs'), str(out)], check=True, stdout=log,
                       stderr=subprocess.STDOUT, cwd=ROOT, timeout=120)
    host = read(out / 'host_receipt.json')
    start = perf_counter()
    registration = []
    reader = Reader(movie, spec['sha256'], info, clock, out / 'verified')
    fresh, _ = reader.read(point['pts'])
    with Image.open(fresh['path']) as original:
        native = original.convert('RGB')
        rgb = native.tobytes()
        width, height = original.size
        resized = np.asarray(native.resize((320, max(8, round(height * 320 / width))), Image.Resampling.BILINEAR))
    for ref in set(host['attachments']):
        with Image.open(out / (ref.removeprefix('sha256:') + '.png')) as image:
            if image.convert('RGB').tobytes() == rgb:
                registration.append(ref)
    assert registration, 'requested native original not verified in host payloads'
    proof = perf_counter() - start
    rows = get_frames(name, spec)
    prototypes = {p for side in row['source_prototype_pts'] for p in side}
    relevant = [r for r in rows if r['pts'] in prototypes]
    relevant.append(fresh | {'rgb': cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)})
    relevant.sort(key=lambda r: r['pts'])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(730)
    start = perf_counter()
    bank, _ = build_feature_input(relevant, [[relevant[0]['pts'], relevant[-1]['pts']]], None)
    layout = Layout(bank)
    rank = fresh['frame']
    comparisons = [layout.compare(rank, r) for r in bank.rows if r != rank]
    profiles = [layout.group_profile(rank, [r for r, f in bank.rows.items() if f['pts'] in side])
                for side in row['source_prototype_pts']]
    assert before == digest(path)
    write(out / 'result.json', {'status': 'added_native_observation_boundary_unpublished',
        'source_sha256': spec['sha256'], 'parent_unchanged_sha256': before, 'native_interior': fresh,
        'exact_matching_png_payloads': registration, 'supports_interior_to_each_side': profiles,
        'direct_geometry': comparisons, 'prior_status': row['state'], 'prior_cause': row['cause'],
        'cost': {'host_attachments': len(host['attachments']), 'new_unique_native_PTS': 1,
            'independent_decode_attempts': len(fresh['decode_attempts']), 'independent_source_proof_s': proof,
            'geometry_recompute_s': perf_counter() - start, 'host_elapsed_ms': host['host_elapsed_ms']}})


if __name__ == '__main__':
    run()
