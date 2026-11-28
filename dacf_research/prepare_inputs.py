"""Create independent research copies and generate controlled source clips."""
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from evidenweft_phase1.source_review import native_stream, timeline, WITHDRAWN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = HERE / 'inputs'
DESKTOP = Path('C:/Users/Administrator/Desktop') / '视觉插件' / '素材'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def copy_real():
    sources = {'bench':ROOT/'evidenweft_phase1/prediction_inputs/bench/video-a.mp4',
               'doorway':ROOT/'evidenweft_phase1/prediction_inputs/doorway/video-b.mp4',
               'reverse_doorway':DESKTOP / '13.mp4', 'pan_return':DESKTOP / '7.mp4'}
    for name, path in sources.items():
        signature = digest(path)
        assert signature != WITHDRAWN
        target = INPUTS / f'{name}.mp4'
        shutil.copyfile(path, target)
        assert digest(target) == signature


def pattern(seed, base):
    rng = np.random.default_rng(seed)
    arr = np.zeros((180, 320, 3),np.uint8)
    arr[:] = base
    for j in range(126):
        x, y = (int(k) for k in rng.integers([4,4], [318,178]))
        choice = tuple(int(z) for z in rng.integers(18, 247, size=3))
        cv2.circle(arr,(x,y),int(rng.integers(2,9)), choice, -1)
        if j%9==0:
            cv2.rectangle(arr,(x,y),(min(x+10,319),min(y+20,179)),choice,1)
    return arr


def create_controls():
    a = pattern(304, (34,49,114))
    b = pattern(831, (214,176,40))
    bridge = np.concatenate((a[:,:160], b[:,160:]), axis=1)
    mix = lambda x: np.uint8(np.round(a.astype(float)*(1-x)+b.astype(float)*x))
    linear = [0.]*6 + [i/20 for i in range(1,20)] + [1.]*6
    detour = [0.]*6 + [i/10 for i in range(1,11)] + [1.]*4 + [1-i/10 for i in range(1,11)] + [0.]*4 + [i/10 for i in range(1,11)] + [1.]*5
    controls = {'linear': ([mix(t) for t in linear], {'known_mix_proportions':linear}),
                'detour':([mix(t) for t in detour], {'known_mix_proportions':detour}),
                'cut':([a]*15 + [b]*15, {'cut_between_source_ranks':[14,15]}),
                'static':([a]*30, {'has_change':False}),
                'local_bridge':([a]*12+[bridge]*5+[b]*12, {'dual_local_pattern_source_ranks':[12,16]})}
    for name,(frames, truth) in controls.items():
        f = INPUTS / f'{name}.mkv'
        subprocess.run(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','320x180','-r','5',
                        '-i','pipe:0','-threads','1','-c:v','ffv1','-pix_fmt','bgr0',str(f)],
                        input=b''.join(im.tobytes() for im in frames), check=True)
        write(HERE/'evaluation_only'/f'{name}_construction.json', truth)


def sheet(source, out):
    stream, rows = timeline(source)
    ts = [j*.5 for j in range(int(rows[-1]['seconds']*2)+1)] + [rows[-1]['seconds']]
    pts = {min(rows,key=lambda v:abs(v['seconds']-t))['pts'] for t in ts}
    board = Image.new('RGB', (5*230, ((len(pts)+4)//5)*150), '#151c24')
    ink = ImageDraw.Draw(board)
    i=0
    for row, bits in native_stream(source, stream, rows):
        if row['pts'] not in pts:
            continue
        img=Image.frombytes('RGB',(stream['width'],stream['height']),bits)
        img.thumbnail((224,126))
        x,y=i%5*230,i//5*150
        board.paste(img,(x+3,y+4)); ink.text((x+3,y+132),f'{row["seconds"]:.3f}s  PTS {row["pts"]}',fill='white')
        i+=1
    board.save(out)
    return {'dimensions':stream,'frames':rows,'sha256':digest(source)}


def main():
    INPUTS.mkdir(exist_ok=False)
    copy_real(); create_controls()
    review = HERE / 'evaluation_only' / 'original_reviews'
    review.mkdir(parents=True,exist_ok=False)
    for source in sorted(INPUTS.iterdir()):
        info=sheet(source, review / f'{source.stem}.png')
        write(review / f'{source.stem}.json', info)
        print(source.stem,len(info['frames']),Fraction(info['dimensions']['time_base']),flush=True)


if __name__=='__main__':
    main()
