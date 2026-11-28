"""Build independent controlled sources and point to immutable replay footage."""

from pathlib import Path
import subprocess

import cv2
import numpy as np

from dacf_research.prepare_inputs import ROOT, WITHDRAWN, digest, pattern, write
from evidenweft_phase1.source_review import timeline

H=Path(__file__).resolve().parent


def render(path):
    bases=[pattern(seed,color) for seed,color in ((116,(39,61,122)),(402,(116,132,62)),(1014,(58,94,136)))]
    # Equal large silhouettes, but independently measurable surface patterns and accents.
    for i,img in enumerate(bases):
        cv2.rectangle(img,(22,32),(78,148),(29+i*9,39,56+i*19),4)
        cv2.rectangle(img,(240,32),(285,150),(181,184-i*16,80),4)
        cv2.line(img,(95,133),(222,133),(239,238,225),3)
        cv2.rectangle(img,(214,22),(243,45),(25,29,38),-1)
    out=[]
    for i in range(160):
        if i<26 or i>=116:
            frame=bases[0].copy()
        elif i<=65:
            frame=bases[1].copy()
            if 34<=i<=47:
                cv2.rectangle(frame,(218,25),(237,40),(250,235,81),-1)
            x=104 if i<=46 else min(185,104+(i-46)*8)
            cv2.rectangle(frame,(x,106),(x+21,126),(203,34,42),-1)
        elif i<76:
            ratio=(i-65)/11
            frame=np.uint8(np.round((1-ratio)*bases[1]+ratio*bases[2]))
        else:
            frame=bases[2].copy()
            if 86<=i<105:
                cv2.rectangle(frame,(27,57),(60,88),(100,195,215),-1)
            y=94 if i<96 else min(134,94+(i-96)*7)
            cv2.rectangle(frame,(175,y),(198,y+15),(210,48,124),-1)
        if i==133:
            frame=pattern(145,(187,23,136))
        out.append(frame.tobytes())
    subprocess.run(['ffmpeg','-v','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','320x180',
      '-framerate','10','-i','pipe:0','-an','-c:v','ffv1','-pix_fmt','bgr0',str(path)],input=b''.join(out),
      check=True,timeout=45)


def case(path,index=None,history=None,kind='control'):
    src=ROOT/path
    metadata,clock=timeline(src)
    sha=digest(src)
    assert sha!=WITHDRAWN
    return {'source':path,'sha256':sha,'extent':[clock[0]['pts'],clock[-1]['pts']], 'time_base':metadata['time_base'],
            'initial_index':index,'r2_history':history,'type':kind}


if __name__=='__main__':
    inputs=H/'inputs';inputs.mkdir()
    controlled='partial_modules_stage6/inputs/reel.mkv'
    render(ROOT/controlled)
    cases={
        'detour':case('dacf_research/inputs/detour.mkv','dacf_research/shared_catalogues/detour','dacf_round2/replay/detour'),
        'cut':case('dacf_research/inputs/cut.mkv','dacf_research/shared_catalogues/cut','dacf_round2/replay/cut'),
        'camera':case('partial_modules_stage3/inputs/camera/source.mkv'),
        'real_return':case('partial_modules_stage5/inputs/11.mp4','partial_modules_stage5/real_round2/index','partial_modules_stage5/real_round2','real'),
        'reel':case(controlled),
        'real_pan':case('dacf_research/inputs/pan_return.mp4','dacf_research/shared_catalogues/pan_return','dacf_round2/replay/pan_return','real'),
        'long_street':case('partial_modules_stage2/inputs/street/source.avi',kind='real')}
    write(inputs/'cases.json',cases)
