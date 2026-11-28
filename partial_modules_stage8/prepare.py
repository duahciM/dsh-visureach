"""Seal fixture construction, inputs, assessor and experimental rules."""

from pathlib import Path
import subprocess

import cv2
import numpy as np

from dacf_research.prepare_inputs import ROOT, digest, pattern, read, write
from evidenweft_phase1.source_review import WITHDRAWN, timeline
from partial_modules_stage7.regression import check_source_history

H = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'partial_modules_stage7'


def create_fixture(path):
    # Keep a common textured strip between separately constructed layouts.
    a = pattern(1638, (62, 96, 132))
    b = pattern(2841, (166, 116, 57))
    c = pattern(5219, (115, 70, 162))
    for board in (b, c):
        board[:, 136:181] = a[:, 136:181]
    plate = pattern(729, (185, 161, 92))[:52, :56]
    moving = pattern(911, (184, 64, 121))[:38, :40]
    images = []
    for i in range(166):
        if i < 40:
            image = a.copy()
            if 13 <= i <= 34:
                image[36:88, 60:116] = plate
        elif i <= 53:
            image = cv2.addWeighted(a, 1 - (i - 39) / 15, b, (i - 39) / 15, 0)
        elif i < 101:
            image = b.copy()
            if 62 <= i <= 74:
                image = np.uint8(np.clip(image.astype(float) * .72 + 35, 0, 255))
        elif i <= 114:
            image = cv2.addWeighted(b, 1 - (i - 100) / 15, c, (i - 100) / 15, 0)
        else:
            image = c.copy()
            if 125 <= i <= 143:
                x = 200 + (i - 125) * 3
                image[104:142, x:x + 40] = moving
        images.append(image)
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', '320x180', '-r', '10', '-i', 'pipe:0', '-threads', '1',
                    '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(path)],
                   input=b''.join(image.tobytes() for image in images), check=True)


def prepare():
    history = check_source_history()
    cases = read(PREVIOUS / 'inputs/cases.json')
    source = H / 'inputs/shared_structure.mkv'
    source.parent.mkdir(exist_ok=True)
    create_fixture(source)
    info, clock = timeline(source)
    cases['shared_structure'] = {'source': source.relative_to(ROOT).as_posix(),
        'sha256': digest(source), 'time_base': info['time_base'], 'extent': [clock[0]['pts'], clock[-1]['pts']],
        'type': 'independent_control'}
    spec = cases['new_recording']
    _, clock = timeline(ROOT / spec['source'])
    holdout = [r['pts'] for r in clock if 8.0 < r['seconds'] <= 14.0]
    cases['real_holdout_window'] = spec | {'extent': [holdout[0], holdout[-1]], 'type': 'independent_window_same_source'}
    for spec in cases.values():
        assert digest(ROOT / spec['source']) == spec['sha256'] != WITHDRAWN
    write(H / 'inputs/cases.json', cases)
    write(H / 'assessment/sealed.json', {
        'shared_structure': {'bounds': [[3900, 5400], [10000, 11500]],
            'roles': [{'id': name, 'pts': span, 'native_roi': roi} for name, span, roi in [
                ('plate_arrival', [1300, 1600], [60, 36, 56, 52]),
                ('plate_persists', [2000, 2700], [60, 36, 56, 52]),
                ('plate_departure', [3500, 3800], [60, 36, 56, 52]),
                ('moving_left', [12500, 12800], [200, 104, 49, 38]),
                ('moving_right', [14000, 14300], [245, 104, 55, 38]),
                ('post_motion', [14400, 14800], [248, 104, 48, 38])]],
            'relations': [['plate_arrival', 'plate_persists', 'plate_departure'],
                          ['moving_left', 'moving_right', 'post_motion']],
            'structured_controls': {'fixed_environment': [0, 1000], 'exposure_same_layout': [6200, 7400],
                'common_layout_strip_xywh': [136, 0, 45, 180],
                'nominal_plate_roi': [60, 36, 56, 52]}},
        'real_holdout_window': {'bounds': None, 'roles': [], 'relations': [],
            'annotation': 'New chronological raster window from the same prior raw recording. No negative labels inferred.'}
    })
    frozen = [*H.glob('*.py'), *H.glob('*.mjs'), H / 'PROTOCOL.md', H / 'inputs/cases.json']
    for name, spec in cases.items():
        if name in ('shared_structure', 'real_holdout_window'):
            continue
        old = PREVIOUS / 'automatic' / name
        frozen.extend(old / file for file in ('results.json', 'layout_fit.json.gz', 'raw_photometry.json.gz'))
        folder = ROOT / spec['stage6'] if 'stage6' in spec else old
        frozen.extend(folder / file for file in ['frame_receipts.json'])
        if 'stage6' in spec:
            frozen.append(folder / 'auto.json')
    frozen.extend(PREVIOUS / 'automatic' / name for name in ['summary.json'])
    labels = [H / 'assessment/sealed.json', ROOT / 'partial_modules_stage6/assessment/sealed.json',
              PREVIOUS / 'assessment_extra.json']
    signatures = lambda paths: {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    write(H / 'freeze.json', {'prediction': signatures(frozen), 'assessor': signatures(labels),
        'history_checked': history, 'existing_frozen_master': digest(PREVIOUS / 'freeze.json'),
        'protocol_order': 'sealed before stage8 predictions and developer branch inspection'})


if __name__ == '__main__':
    prepare()
