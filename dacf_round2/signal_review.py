"""Describe saved signals without changing predicted regions or reading score roles."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import median

from dacf_research.prepare_inputs import read, write
from dacf_round2.frozen_inputs import HERE
from dacf_round2.verify_round2 import unpack


def summarize(regions):
    channel_states = {}
    absolute_support = {}
    path_ratios = {}
    max_interior_deviations = {}
    scale_extrema = {}
    comparable_pairs = Counter()
    overlapping_pairs = Counter()

    for region in regions:
        brackets = {}
        for channel, result in region['crossings'].items():
            channel_states.setdefault(channel, Counter()).update([result['state']])
            events = [event for event in result['events'] if event['kind'] != 'sampled_equality_only']
            absolute_support.setdefault(channel, Counter()).update(event['support'] for event in events)
            brackets[channel] = [(event['from_pts'], event['to_pts']) for event in events]
            ratio = region['paths'][channel]['path_to_endpoint_direct_ratio']
            if ratio is not None:
                path_ratios.setdefault(channel, []).append(ratio)
            deltas = [row['channels'][channel]['anchor_minus_path'] for row in region['rows'][1:-1]]
            deltas = [abs(value) for value in deltas if value is not None]
            if deltas:
                max_interior_deviations.setdefault(channel, []).append(max(deltas))
            for scale, extrema in region['multi_scale_extrema'][channel].items():
                if extrema:
                    scale_extrema.setdefault(channel, Counter()).update([scale])

        for first, second in combinations(sorted(brackets), 2):
            if not brackets[first] or not brackets[second]:
                continue
            pair = f'{first}/{second}'
            comparable_pairs[pair] += 1
            if any(max(a, c) <= min(b, d) for a, b in brackets[first] for c, d in brackets[second]):
                overlapping_pairs[pair] += 1

    def medians(values):
        return {channel: {'median': median(items), 'valid_windows': len(items)} for channel, items in values.items()}

    return {'windows': len(regions), 'crossing_states': channel_states, 'crossing_support_events': absolute_support,
            'channel_pairs_with_a_bracket_on_each_side': comparable_pairs,
            'channel_pairs_with_any_overlapping_brackets': overlapping_pairs,
            'median_path_to_direct_distance_ratio': medians(path_ratios),
            'median_maximum_absolute_interior_progress_deviation': medians(max_interior_deviations),
            'windows_with_interior_extrema_by_scale_seconds': scale_extrema}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    all_windows = {'with_interior': [], 'no_interior': []}
    cases = {}
    for name, case in read(args.run / 'lock.json')['cases'].items():
        cohort = Path('replay') if case['cohort'] == 'replay' else Path('independent_execution/independent')
        folder = args.run / cohort / name
        raw = {window['region_id']: window for window in unpack(folder / 'region_signals.json.gz')}
        windows = {'with_interior': [], 'no_interior': []}
        for region in read(folder / 'regions.json')['regions']:
            if region['kind'] == 'paired_investigation':
                group = 'with_interior' if region['coarse_interior_pts'] else 'no_interior'
                windows[group].append(raw[region['id']])
                all_windows[group].append(raw[region['id']])
        cases[name] = {group: summarize(rows) for group, rows in windows.items()}

    if args.out.exists():
        raise FileExistsError(args.out)
    write(args.out, {'purpose': 'descriptive diagnostics only; equality without a side change excluded from bracket overlap',
                     'aggregate': {group: summarize(rows) for group, rows in all_windows.items()}, 'by_source': cases})


if __name__ == '__main__':
    main()
