"""Small independent controlled checks with recorded image/observation inputs."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from p1.io import dump
from .recognition import apply_absence

PANEL = (36, 24, 116, 80)
PATCH = (57, 41, 95, 63)


def panel_image(path, kind):
    image = Image.new('RGB', (160, 108), '#7c8791')
    draw = ImageDraw.Draw(image)
    if kind != 'out_of_view':
        draw.rectangle(PANEL, fill='#e2e0d8', outline='#151515', width=3)
        for xy in ((43,30),(106,30),(43,72),(106,72)):
            draw.rectangle((xy[0], xy[1], xy[0]+4, xy[1]+4), fill='#1c7592')
        if kind in ('present', 'held_same_category'):
            draw.rectangle(PATCH, fill='#d5342e' if kind == 'present' else '#53a34a')
        elif kind == 'occluded':
            draw.rectangle((51,35,100,67), fill='#555555')
        elif kind == 'low_quality':
            draw.rectangle(PANEL, fill='#b2ada8')
    image.save(path)


def observe_panel(path):
    rgb = np.asarray(Image.open(path).convert('RGB'))
    marks = all(np.array_equal(rgb[y+2, x+2], [28,117,146]) for x,y in ((43,30),(106,30),(43,72),(106,72)))
    x0,y0,x1,y1 = PATCH
    pixels = rgb[y0:y1+1,x0:x1+1]
    red = float(np.mean(np.all(pixels == [213,52,46], axis=-1)))
    empty = float(np.mean(np.all(pixels == [226,224,216], axis=-1)))
    state = 'unknown'
    if marks and red >= .98:
        state = 'present'
    if marks and empty >= .98:
        state = 'absent_supported'
    return {'feature': 'fixed_red_tile', 'scope': 'calibrated_panel_slot', 'state': state,
            'scope_visible': bool(marks and (red >= .98 or empty >= .98)), 'verified': state != 'unknown',
            'source_ref': str(path.resolve()), 'scope_box_xyxy': list(PATCH),
            'basis': 'four visible fixed panel fiducials plus >=98% direct red or uncovered background pixels in predeclared slot',
            'verified_fiducials': marks, 'red_fraction': red, 'visible_empty_fraction': empty}


def direction_step(previous, observation):
    # Direction/edge and confidence are observation inputs. Object identity is an
    # appearance token in this small harness, with no claim of real-video tracking.
    result = dict(previous)
    reason = 'insufficient_evidence'
    token = observation.get('appearance')
    if observation.get('cut'):
        return {'appearance': None, 'location': 'unknown', 'reason': 'cut_invalidates_view_frame'}
    if observation.get('duplicate_appearance'):
        return {**result, 'location': 'unknown', 'reason': 'ambiguous_instance'}
    if observation.get('visible') and token:
        revision = previous.get('appearance') == token and previous.get('location') in ('left', 'right') and observation.get('entry') != previous['location']
        return {'appearance': token, 'location': 'visible', 'reason': 'contradictory_reentry_revises_direction' if revision else 'direct_reobservation',
                'expected_entry': previous.get('location') if previous.get('location') in ('left','right') else None,
                'actual_entry': observation.get('entry')}
    if observation.get('static_supported') and observation.get('camera_direction') and not observation.get('occluded'):
        expected = {'turn_left': 'right', 'turn_right': 'left'}[observation['camera_direction']]
        if observation.get('exit') == expected and previous.get('appearance'):
            return {**result, 'location': expected, 'reason': 'offscreen_hypothesis_from_edge_and_direction'}
    return {**result, 'location': 'unknown', 'reason': reason}


def run(root):
    root.mkdir()
    panel = root / 'A3_calibrated_panel'
    panel.mkdir()
    fixtures = {
        'present': ('present', ['I1', 'I2'], None),
        'visible_empty': ('absent_supported', ['I1', 'I2'], 'I2'),
        'out_of_view': ('unknown', ['I1', 'I2'], None),
        'occluded': ('unknown', ['I1', 'I2'], None),
        'low_quality': ('unknown', ['I1', 'I2'], None),
        'held_same_category': ('unknown', ['I1', 'I2'], None),
        'empty_without_positive_rival': ('absent_supported', ['I1'], None),
    }
    # Write the observation fixtures and their independent expectation before scoring.
    truth = []
    for name, (state, contenders, expected) in fixtures.items():
        kind = 'empty' if state == 'absent_supported' else name
        panel_image(panel / f'{name}.png', kind)
        truth.append({'fixture': name, 'expected_state': state, 'positive_candidates': contenders, 'expected_A3': expected})
    dump(panel / 'locked_expectations.json', truth)
    required = {'I1': [('fixed_red_tile', 'calibrated_panel_slot')]}
    dump(panel / 'candidate_requirements.json', {'required': required, 'basis': 'controlled physical panel fixture definition; scoped to current view'})
    rows = []
    for spec in truth:
        obs = observe_panel(panel / f"{spec['fixture']}.png")
        row = {'assignment': None, 'candidate_ids': spec['positive_candidates'], 'absent_supported': []}
        actual = apply_absence(row, [obs], required)
        rows.append({**spec, 'observed': obs, 'A2': row['assignment'], 'A3': actual['assignment'], 'vetoes': actual['absent_supported'],
                     'pass': obs['state'] == spec['expected_state'] and actual['assignment'] == spec['expected_A3']})
    dump(panel / 'results.json', rows)

    seen = {'visible': True, 'appearance': 'lamp_appearance'}
    left_exit = {'static_supported': True, 'camera_direction': 'turn_right', 'exit': 'left'}
    right_exit = {'static_supported': True, 'camera_direction': 'turn_left', 'exit': 'right'}
    cases = [
        ('left_pan_and_reverse', [seen, left_exit, {**seen,'entry':'left'}], ['visible','left','visible']),
        ('right_pan_and_reverse', [seen, right_exit, {**seen,'entry':'right'}], ['visible','right','visible']),
        ('revisit_loop', [seen, left_exit, {}, {}, {**seen,'entry':'right'}], ['visible','left','unknown','unknown','visible']),
        ('moved_object', [seen, {**left_exit,'static_supported':False}], ['visible','unknown']),
        ('occlusion', [seen, {**right_exit,'occluded':True}], ['visible','unknown']),
        ('similar_instances', [seen, {'visible':True,'appearance':'lamp_appearance','duplicate_appearance':True}], ['visible','unknown']),
        ('cut_or_door_with_no_usable_pose', [seen, {'cut':True}, {'exit':'right'}], ['visible','unknown','unknown']),
        ('contradictory_direction', [seen, right_exit, {**seen,'entry':'left'}], ['visible','right','visible']),
    ]
    dump(root / 'B_locked_observation_fixtures.json', [dict(name=n, observations=o, expected=e) for n,o,e in cases])
    results = []
    for name, observations, expected in cases:
        state = {'appearance': None, 'location': 'unknown'}
        trace = []
        for observation in observations:
            state = direction_step(state, observation)
            trace.append(state)
        revision_ok = name != 'contradictory_direction' or trace[-1]['reason'] == 'contradictory_reentry_revises_direction'
        results.append({'case': name, 'trace': trace, 'expected': expected,
                        'pass': revision_ok and [row['location'] for row in trace] == expected,
                        'input_type': 'scripted object/edge/camera observations, no automatic pose or instance estimation'})
    dump(root / 'B_direction_results.json', results)
    return {'A3_panel_checks': sum(r['pass'] for r in rows), 'A3_panel_total': len(rows),
            'B_checks': sum(r['pass'] for r in results), 'B_total': len(results)}
