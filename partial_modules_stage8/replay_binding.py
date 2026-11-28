"""Recover the fixture's old R1 group binding after the v8 original attempt."""

from unittest.mock import patch

import cv2

from dacf_research.field import clip_encoder, detect_groups
from dacf_research.prepare_inputs import ROOT, digest, read, write
from . import run


def groups_only(bank, windows, *_):
    # Stage 6 qualification invokes the catalogue viewer as a side effect.
    # The only field Stage 8 consumes here is the preceding raw R1 result.
    return {'all_adjacent_groups': [detect_groups(bank, pool) for pool in windows]}


if __name__ == '__main__':
    home = run.H
    spec = read(home / 'inputs/cases.json')['shared_structure']
    lock = read(home / 'freeze.json')
    assert all(digest(ROOT / path) == value for path, value in lock['prediction'].items())
    original = read(home / 'automatic/summary.json')
    assert 'KeyError' in original['failures']['shared_structure']
    cv2.setNumThreads(1)
    cv2.setRNGSeed(730)
    with patch.object(run, 'old_qualify', groups_only):
        result = run.automatic('shared_structure', spec, home / 'automatic/shared_structure', clip_encoder())
    write(home / 'automatic/recovery.json', {'completed': result, 'initial_summary_sha256': digest(home / 'automatic/summary.json'),
        'change': 'Read the original R1 detect_groups output directly, avoiding an unrelated figure builder with a catalogue schema mismatch.',
        'thresholds_source_pts_model_and_factor_code_unchanged': True,
        'recovery_source_sha256': digest(home / 'replay_binding.py')})
