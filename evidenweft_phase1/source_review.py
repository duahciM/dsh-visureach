"""Independent frame review. No plugin, catalogue, experimental arm or model."""
import json
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
WITHDRAWN = '7f74580c0426420011267f3ad0c68bfa8a53c91b2ae74fa168912a72d4a201cf'


def timeline(source):
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                                   '-show_streams', '-show_entries', 'frame=best_effort_timestamp:stream=width,height,time_base', '-of', 'json', str(source)])
    v = json.loads(raw)
    stream = v['streams'][0]
    selected = [{'frame': i, 'pts': f['best_effort_timestamp'],
                 'seconds': float(Fraction(stream['time_base']) * f['best_effort_timestamp'])}
                for i, f in enumerate(v['frames']) if 'best_effort_timestamp' in f]
    assert len(selected) == len(v['frames']) and all(a['pts'] < b['pts'] for a, b in zip(selected, selected[1:]))
    return stream, selected


def native_stream(source, stream, rows):
    extract = ['ffmpeg', '-v', 'error', '-noautorotate', '-i', str(source), '-map', '0:v:0',
               '-vf', 'format=rgb24', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
    worker = subprocess.Popen(extract, stdout=subprocess.PIPE)
    try:
        size = stream['width'] * stream['height'] * 3
        for row in rows:
            bits = worker.stdout.read(size)
            assert len(bits) == size
            yield row, bits
        assert worker.stdout.read(1) == b'' and worker.wait() == 0
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def review(case, source, target_seconds=None):
    destination = HERE / 'evaluation_only/source_review' / case
    destination.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(source.read_bytes()).hexdigest()
    assert source_sha != WITHDRAWN
    stream, rows = timeline(source)
    if target_seconds is None:
        target_seconds = [i / 2 for i in range(37)] + [rows[-1]['seconds']]
    selected = {min(rows, key=lambda r: abs(r['seconds'] - t))['pts'] for t in target_seconds}
    columns = 5
    canvas = Image.new('RGB', (columns * 244, ((len(selected) + columns - 1) // columns) * 160), '#191f28')
    painter = ImageDraw.Draw(canvas)
    observed = []
    for row, bits in native_stream(source, stream, rows):
        row['decoded_rgb_sha256'] = sha256(bits).hexdigest()
        if row['pts'] not in selected:
            continue
        size = (stream['width'], stream['height'])
        original = Image.frombytes('RGB', size, bits)
        original.save(destination / f'pts-{row["pts"]}.png')
        i = len(observed)
        x, y = i % columns * 244, i // columns * 160
        original.thumbnail((238, 134))
        canvas.paste(original, (x + 3, y + 4))
        painter.text((x + 3, y + 140), f'{row["seconds"]:.5f}s  PTS {row["pts"]}', fill='white')
        observed.append(row)
    name = 'dense' if len(target_seconds) != 38 else 'overview'
    canvas.save(destination / f'{name}.png')
    (destination / 'native_timeline.json').write_text(json.dumps({'sha256': source_sha, 'format': stream, 'frames': rows}) + '\n', encoding='utf-8')
    (destination / f'{name}_reviewed_pts.json').write_text(json.dumps(observed, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    review(sys.argv[1], Path(sys.argv[2]), [float(x) for x in sys.argv[3:]] or None)
