"""Build production catalogues, then apply the locked study's delivery lists."""
from collections import Counter
from pathlib import Path
import subprocess
import sys
from time import monotonic

from evidenweft_phase1.prepare import FROZEN, HERE, PUBLIC, digest, project, read, select_rank, select_time, verify, write


def main():
    verify()
    protocol = read(HERE / 'public_protocol.json')
    data = HERE / 'inventory'
    data.mkdir()
    for case, source_info in protocol['sources'].items():
        clip = PUBLIC / case / source_info['public_name']
        folder = data / case
        folder.mkdir()
        src = read(HERE / 'evaluation_only/source_review' / case / 'native_timeline.json')
        assert digest(clip) == src['sha256']
        index = folder / 'index'
        t = monotonic()
        with (folder / 'indexer.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-B', '-X', 'utf8', str(FROZEN / 'visureach/python/index_video.py'),
                '--video', str(clip), '--out', str(index), '--fps', str(protocol['coarse_fps']), '--keep-analysis'],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed = round(monotonic() - t, 3)
        manifest = read(index / 'manifest.json')
        catalogue = read(index / 'catalog.json')['frames']
        original_rows = src['frames']
        original = {r['pts']:r['frame'] for r in original_rows}
        pool = sorted(manifest['pool'], key=lambda r: r['pts'])
        for row in pool:
            assert row['source_frame'] == original[row['pts']]
        nominated = [r for r in pool if r['in_catalog']]
        assert {r['pts'] for r in nominated} == {r['pts'] for r in catalogue}
        n = len(nominated)
        coarse_uniform = select_time(pool, n)
        a0 = project(coarse_uniform)
        a1 = project(nominated)
        for name, frames in [('uniform', a0), ('multi', a1)]:
            write(PUBLIC / case / f'{name}.json', {'source_sha256': source_info['sha256'], 'source_frames': len(original_rows),
                'start': original_rows[0]['seconds'], 'end': original_rows[-1]['seconds'], 'frames': frames})
        rows = []
        for name in protocol['offline_counts']:
            k = n if name == 'inventory' else name
            assert k <= n
            lists = [('coarse_uniform', select_time(pool, k)), ('source_uniform', select_time(original_rows, k)),
                     ('evidenweft', select_rank(nominated, k))]
            for mode, selected in lists:
                assert len({r['pts'] for r in selected}) == k
                rows.append({'requested': name, 'count': k, 'mode': mode, 'pts': [r['pts'] for r in selected]})
        write(folder / 'inventory.json', {'source': source_info['sha256'], 'full_source_count': len(original_rows), 'coarse_pool_count': len(pool),
             'evidenweft_inventory_count': n, 'nomination_counts': manifest['nomination_counts'], 'coarse_pts': [r['pts'] for r in pool],
             'evidenweft_pts': [r['pts'] for r in nominated], 'source_pts': list(original), 'method_proposed_deliveries': rows,
             'compute_seconds': elapsed, 'algorithm': 'frozen VisuReach 0.6.0 P2_24 vendor; native IF-30 research checkout frozen separately'})
        print(case, len(original_rows), len(pool), n, manifest['nomination_counts'], elapsed, flush=True)
    lock_files = {str(p.relative_to(HERE).as_posix()): digest(p) for p in list(PUBLIC.rglob('*')) + list(data.rglob('*')) if p.is_file()}
    write(HERE / 'prediction_inputs_lock.json', lock_files)


if __name__ == '__main__':
    main()
