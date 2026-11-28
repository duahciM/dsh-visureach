"""After prediction freeze, inspect specific registered control pixels and save source boards."""

from itertools import pairwise
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageDraw

from dacf_research.prepare_inputs import ROOT, read, write
from partial_modules_stage6.predict import gzip_json
from .run import H, get_frames

OPTIONS = ('old', 'wide', 'past', 'future', 'complement', 'natural')


def plate_audit(rows, labels):
    colors = {r['pts']: cv2.cvtColor(r['rgb'], cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255 for r in rows}
    x, y, w, h = labels['nominal_plate_roi']
    windows = {'stable': labels['fixed_environment'], 'plate_arrival': [1300, 1600], 'plate_persists': [2000, 2700],
               'plate_departure': [3500, 3800], 'exposure': labels['exposure_same_layout']}
    diagnostic = {}
    for key in OPTIONS:
        records = gzip_json(H / 'm4/shared_structure' / f'{key}.json.gz')['measurements']
        answer = {}
        for tag, (lo, hi) in windows.items():
            observed = []
            for row in records:
                p = row['pts']
                if not lo <= p <= hi or 'tiles' not in row['domain']:
                    continue
                shape = colors[p].shape[1], colors[p].shape[0]
                patch = np.s_[y:y+h, x:x+w]
                references, valid = [], []
                for fit in row['fit_refs']:
                    matrix = np.asarray(fit['matrix'], dtype=np.float32)
                    pixels = cv2.warpAffine(colors[fit['pts']], matrix, shape, borderMode=cv2.BORDER_REPLICATE)
                    mask = cv2.warpAffine(np.ones(shape[::-1], np.float32), matrix, shape) > .999
                    valid.append(mask[patch])
                    references.append(pixels[patch])
                applicable = np.logical_and.reduce(valid)
                if not applicable.any():
                    continue
                target = colors[p][patch]
                relative = np.max(np.abs(target - np.asarray(references)), axis=-1)
                med_delta = np.max(np.abs(target - np.median(references, axis=0)), axis=-1)
                observed.append({'target_PTS': p, 'source_root': row['original'], 'domain': row['domain']['root'],
                    'valid_pixels': int(applicable.sum()), 'actual_comparison_PTS': [z['pts'] for z in row['fit_refs']],
                    'target_vs_each_observed_reference_mad': [float(np.mean(q[applicable])) for q in relative],
                    'target_vs_neighbor_pixel_median_mad': float(np.mean(med_delta[applicable]))})
            answer[tag] = {'measurements': observed, 'count': len(observed),
                'median_over_receipts_of_measured_roi_mad': median([v['target_vs_neighbor_pixel_median_mad']
                                                                   for v in observed]) if observed else None}
        diagnostic[key] = answer
    return diagnostic


def photometry():
    found = {}
    for case in read(H / 'inputs/cases.json'):
        out = H / 'automatic' / case
        source = read(out / 'result.json')
        pairs = gzip_json(out / 'pair_photometry.json.gz')
        adjacent = {tuple(z) for seq in source['scanned_windows'] for z in pairwise(
            [p for p in source['pool_pts'] if seq[0] <= p <= seq[1]])}
        measures = [r for r in pairs if tuple(r['pts']) in adjacent]
        verified = [r for r in measures if r['photometry']['supported']]
        found[case] = {'measured_adjacent_unique_pairs': len(measures),
            'eligible_adjacent_positions': len(adjacent), 'adjacent_photo_support': len(verified),
            'adjacent_motion_support': sum(r['phase']['motion_supported'] for r in measures),
            'adjacent_registered_overlap': sum(r['aligned']['status'] == 'measured_overlap' for r in measures),
            'gain_offset_and_comparison_scope': [{'PTS': r['pts'], 'root': r['root'], 'photometry': r['photometry'],
                'aligned_MAD': r['aligned'].get('mad'), 'aligned_overlap_pixels': r['aligned'].get('overlap_pixels')}
                for r in verified],
            'candidate_scoped_competitions': {key: [{'PTS': row['pts'],
                'actual_adjacent_count': len(row['competition']['adjacent_measurements']),
                'cross_valid': sum(c['scope_supported'] for c in row['competition']['cross_observed_structure']),
                'cross_total': len(row['competition']['cross_observed_structure']),
                'competing_only_in_measured_scope': row['competition']['withheld_context_envelope'],
                'roots': row['competition']['adjacent_measured_photometry']} for row in dat['audit'] if 'competition' in row]
                for key, dat in source['branches'].items()}}
    return found


def draw_sheet(folder, frames, selection, name, groups=None, labeled_roi=None):
    lookup = {row['pts']: row for row in frames}
    cols, w, h = 5, 240, 166
    board = Image.new('RGB', (cols*w, ((len(selection)+cols-1)//cols)*h), '#101821')
    draw = ImageDraw.Draw(board)
    for i, pts in enumerate(selection):
        pos = i % cols * w, i // cols * h
        hist = lookup[pts]
        with Image.open(folder / hist['png']) as native:
            image = native.convert('RGB')
            image.thumbnail((w-8, h-29))
            ix, iy = pos[0] + 4, pos[1] + 4
            board.paste(image, (ix, iy))
            sx, sy = image.width/hist['analysis_size'][0], image.height/hist['analysis_size'][1]
            for option, color in (('old', '#eebf6f'), ('wide', '#54ddda'), ('past', '#f67ed2')):
                for r in (groups or {}).get(option, {}).get(pts, []):
                    x, y, ww, hh = r['bbox']
                    draw.rectangle((ix+x*sx, iy+y*sy, ix+(x+ww)*sx, iy+(y+hh)*sy), outline=color, width=2)
            if labeled_roi:
                x, y, ww, hh = labeled_roi
                draw.rectangle((ix+x*sx, iy+y*sy, ix+(x+ww)*sx, iy+(y+hh)*sy), outline='white', width=1)
        draw.text((pos[0]+4, pos[1]+h-22), f'PTS {pts}   {hist["seconds"]:.3f}s', fill='#dce7ed')
    board.save(H / 'diagnostics' / name)


def boards():
    control = H / 'bound/shared_structure'
    old = read(control / 'frame_receipts.json')
    groups = {}
    for key in ('old', 'wide', 'past'):
        groups[key] = {r['pts']: r['candidates'] for r in gzip_json(H / 'm4/shared_structure' / f'{key}.json.gz')['measurements']}
    draw_sheet(control, old, [600, 1200, 1400, 1800, 2000, 2200, 2600, 3200, 3600, 3800],
               'm4-plate.png', groups, [60, 36, 56, 52])
    holdout = H / 'bound/real_holdout_window'
    orig = read(holdout / 'frame_receipts.json')
    draw_sheet(holdout, orig, [r['pts'] for r in orig], 'real-holdout.png')
    for name in ('reel', 'real_return', 'shared_structure'):
        route = {'reel': ROOT / 'partial_modules_stage6/automatic/reel',
                 'real_return': ROOT / 'partial_modules_stage6/automatic/real_return',
                 'shared_structure': control}[name]
        pts_rows = read(route / 'frame_receipts.json')
        passage = read(H / 'automatic' / name / 'result.json')['branches']['C']['audit']
        chosen = {p for r in passage if r['state'] == 'visual_context_change_candidate'
                  for p in [*r['pts'], *r['source_prototype_pts'][0], *r['source_prototype_pts'][1]]}
        if chosen:
            draw_sheet(route, pts_rows, sorted(chosen), name + '-anchor-frames.png')


if __name__ == '__main__':
    cv2.setNumThreads(1)
    (H / 'diagnostics').mkdir()
    specs = read(H / 'inputs/cases.json')
    labels = read(H / 'assessment/sealed.json')['shared_structure']['structured_controls']
    write(H / 'diagnostics/m4_roi.json', plate_audit(get_frames('shared_structure', specs['shared_structure']), labels))
    write(H / 'diagnostics/photometry.json', photometry())
    boards()
