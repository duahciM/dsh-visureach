"""Create two independent sources and seal prediction inputs."""
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from dacf_research.freeze import PLUGIN, validate
from dacf_research.prepare_inputs import DESKTOP, ROOT, digest, read, write
from dacf_research.run import raw_frames
from evidenweft_phase1.source_review import WITHDRAWN, timeline

HERE = Path(__file__).resolve().parent
R2 = ROOT / 'dacf_round2'


def controls(path):
    rng = np.random.default_rng(31027)
    background = rng.integers(44, 135, (144, 256, 3), dtype=np.uint8)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '256x144',
                           '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', str(path)],
                          stdin=subprocess.PIPE)
    for i in range(360):
        x = (40+i//2 if i<72 else 76 if i<135 else 76+min(i-135,20)*5 if i<190 else
             176-(i-190) if i<290 else 76)
        shift = i % 5 - 2 if 287 <= i < 320 else 0
        gain = 26 if 305 <= i < 327 else 0
        frame = np.roll(background, shift, 1).copy() + gain
        frame[48:85, 119:125] = (200, 210, 240)
        frame[54:78, x:x+24] = (253, 25, 35)
        frame[62:68, x+8:x+16] = (254, 225, 70)
        if i == 331:
            frame[26:55, 197:228] = (38, 250, 76)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError('Could not encode independent control')


def verify(lock):
    old = validate('manual')
    assert digest(ROOT/'plugin_offline/release/local-dsh-video-observer-0.6.0.tgz') == old['baseline']['visureach_archive_sha256']
    assert all(digest(ROOT / path) == sig for path, sig in lock['inputs'].items())


def main():
    hold = HERE/'sealed_inputs'
    hold.mkdir()
    r2 = read(R2/'lock.json')
    cases = dict(r2['cases'])
    known = {digest(ROOT/row['source']) for row in cases.values()}
    known.update(digest(path) for path in (ROOT/'evidenweft_phase1/prediction_inputs').rglob('*.mp4'))
    sources = {'new_laptop': DESKTOP/'6.mp4', 'new_control': None}
    for name, original in sources.items():
        folder = hold/name
        folder.mkdir()
        path = folder/'video.mp4'
        if original is None:
            controls(path)
        else:
            shutil.copyfile(original, path)
        source_hash = digest(path)
        assert source_hash not in known and source_hash != WITHDRAWN
        index = folder/'index'
        subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),
                        '--video',str(path),'--out',str(index),'--fps','5'],check=True)
        manifest = read(index/'manifest.json')
        info, clock = timeline(path)
        raw_frames(path, info, clock, {r['source_frame'] for r in manifest['pool']}, folder, source_hash, 320)
        cases[name] = {k: v.relative_to(ROOT).as_posix() for k,v in {'source':path,'index':index,'pixels':folder}.items()} | {'cohort':'new'}
    pins = dict(r2['inputs']) | r2['preserved_round1_outputs']
    extra = [R2/'lock.json', R2/'verification.json', R2/'run_manifest.json', ROOT/'dacf_research/lock.json',
             HERE/'protocol.json', *HERE.glob('*.py')]
    for name, row in cases.items():
        if row['cohort']=='new':
            extra.extend((ROOT/row['source'], ROOT/row['index']/'manifest.json', ROOT/row['index']/'catalog.json',
                          ROOT/row['pixels']/'frame_receipts.json'))
        else:
            result = R2/('replay' if row['cohort']=='replay' else 'independent_execution/independent')/name
            extra.extend(result/f for f in ('regions.json','region_signals.json.gz','pair_evidence.json.gz','local_holdout.json.gz'))
    pins.update({f.relative_to(ROOT).as_posix():digest(f) for f in extra})
    lock = {'locked_at':datetime.now().astimezone().isoformat(),'inputs':pins,'cases':cases}
    verify(lock)
    write(HERE/'lock.json',lock)


if __name__ == '__main__':
    main()
