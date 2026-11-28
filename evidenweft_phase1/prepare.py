from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import shutil
import subprocess
import sys
import tarfile
from time import monotonic

from evidenweft_phase1.source_review import HERE, WITHDRAWN

ROOT = HERE.parent
PUBLIC = HERE / 'prediction_inputs'
LOCK = HERE / 'protocol_lock.json'
FROZEN = HERE / 'frozen'


def read(file):
    return json.loads(file.read_text(encoding='utf-8-sig'))


def write(file, obj):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(obj, out, ensure_ascii=False, indent=2)
        out.write('\n')


def digest(file):
    h = sha256()
    with file.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def select_time(rows, count):
    if count == 1:
        return [min(rows, key=lambda r: (abs(r['seconds'] - (rows[0]['seconds'] + rows[-1]['seconds']) / 2), r['pts']))]
    selected = []
    previous = -1
    for i in range(count):
        target = rows[0]['seconds'] + i * (rows[-1]['seconds'] - rows[0]['seconds']) / (count - 1)
        last = len(rows) - (count - i)
        idx = min(range(previous + 1, last + 1), key=lambda j: (abs(rows[j]['seconds'] - target), rows[j]['pts']))
        selected.append(rows[idx])
        previous = idx
    return selected


def select_rank(rows, count):
    return [rows[(i * (len(rows) - 1) * 2 + count - 1) // (2 * (count - 1))] for i in range(count)]


def project(rows):
    return [{'frame_id': r['frame_id'], 'pts': r['pts'], 'seconds': r['seconds'], 'source_frame': r['source_frame']} for r in rows]


def setup_schedule(protocol):
    state = Random(protocol['seed'])
    schedule = []
    for repetition in range(1, protocol['repetitions'] + 1):
        tickets = [(case, arm) for case in protocol['sources'] for arm in protocol['arms']]
        while True:
            state.shuffle(tickets)
            run = [v['arm'] for v in schedule[-1:]] + [arm for _, arm in tickets]
            if all(a != b for a, b in zip(run, run[1:])):
                break
        for case, arm in tickets:
            schedule.append({'case': case, 'arm': arm, 'repetition': repetition, 'blind_id': f'S{state.getrandbits(48):012x}'})
    return schedule


def verify(code_only=False):
    lock = read(LOCK)
    for rel, sig in lock['sha256'].items():
        if code_only and rel.startswith('evaluation_only/'):
            continue
        assert digest(HERE / rel) == sig, rel
    return lock


def main():
    if LOCK.exists():
        raise FileExistsError('Keep the frozen protocol. Use a new experiment for a different design.')
    protocol = read(HERE / 'public_protocol.json')
    rubric = read(HERE / 'evaluation_only/rubric.json')
    installed = Path('D:/LLM/dsh-video-observer-0.6.0')
    tarfile_path = ROOT / 'plugin_offline/release/local-dsh-video-observer-0.6.0.tgz'
    frozen_plugin = FROZEN / 'visureach'
    frozen_plugin.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        for entry in tar.getmembers():
            if entry.isfile():
                rel = Path(entry.name).relative_to('package')
                dst = frozen_plugin / rel
                content = tar.extractfile(entry).read()
                assert content == (installed / rel).read_bytes()
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    assert dst.read_bytes() == content
                else:
                    dst.write_bytes(content)
    shutil.copyfile(tarfile_path, FROZEN / tarfile_path.name)
    repo = ROOT / 'plugin_offline/github/EvidenWeft'
    run = lambda *args: subprocess.check_output(['git', '-C', str(repo), *args], encoding='utf-8').strip()
    assert not run('status', '--porcelain')
    head = run('rev-parse', 'HEAD')
    if30 = run('rev-list', '-1', 'if30')
    bundle = FROZEN / 'evidenweft.bundle'
    if not bundle.exists():
        subprocess.run(['git', '-C', str(repo), 'bundle', 'create', str(bundle), 'HEAD', 'refs/tags/if30'], check=True)
    bundle_heads = run('bundle', 'list-heads', str(bundle)).splitlines()
    assert f'{head} HEAD' in bundle_heads
    frozen_map = {'evidenweft_main': head, 'evidenweft_if30': if30, 'visureach_archive_sha256': digest(tarfile_path),
                  'visureach_source_commit': read(ROOT / 'plugin_offline/release/verification.json')['source_commit']}
    if (FROZEN / 'baselines.json').exists():
        assert read(FROZEN / 'baselines.json') == frozen_map
    else:
        write(FROZEN / 'baselines.json', frozen_map)
    date = datetime.now().astimezone().isoformat()
    evidence_files = {}
    for case, item in protocol['sources'].items():
        source = ROOT / item['file']
        assert digest(source) == item['sha256'] != WITHDRAWN
        public = PUBLIC / case
        public.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, public / item['public_name'])
        independent = HERE / 'evaluation_only/source_review' / case
        independent_frames = read(independent / 'native_timeline.json')
        assert independent_frames['sha256'] == item['sha256']
        all_pts = {r['pts'] for r in independent_frames['frames']}
        reviewed = {r['pts'] for name in ('dense', 'overview') for r in read(independent / f'{name}_reviewed_pts.json')}
        refs = rubric['cases'][case]
        for description in refs['slots'].values():
            lo, hi = description['pts']
            assert {lo, hi} <= all_pts and description['witness'] in reviewed and lo <= description['witness'] <= hi, description
        for pts in refs.get('exact_endpoints', {}).values():
            lo, hi = pts
            assert all_pts and {lo, hi} <= reviewed
            order = [r['pts'] for r in independent_frames['frames']]
            assert order.index(hi) == order.index(lo) + 1
        for path in independent.rglob('*'):
            if path.is_file():
                evidence_files[str(path.relative_to(HERE)).replace('\\', '/')] = digest(path)
    preprediction = ['public_protocol.json','evaluation_only/rubric.json','source_review.py','prepare.py','control_plugin.mjs',
                     'run_native.py','inventory.py'] 
    files = {rel: digest(HERE / rel) for rel in preprediction} | evidence_files
    for path in FROZEN.rglob('*'):
        if path.is_file():
            files[path.relative_to(HERE).as_posix()] = digest(path)
    write(LOCK, {'sealed_at': date, 'pre_prediction': True, 'baselines': frozen_map, 'sha256': files,
                 'source_copies_sha256': {case: item['sha256'] for case, item in protocol['sources'].items()},
                 'schedule': setup_schedule(protocol)})


if __name__ == '__main__':
    main()
