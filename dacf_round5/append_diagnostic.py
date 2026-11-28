"""Measured lossless prefix intervention with separate local and global scopes."""
import json
from hashlib import sha256
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

from dacf_research.field import Bank,METRICS,clip_encoder
from dacf_research.prepare_inputs import ROOT,read,write
from dacf_round5.setup import HERE,verify
from evidenweft_phase1.source_review import timeline,native_stream


def tail_encode(path, stream, rasters):
    size=f"{stream['width']}x{stream['height']}"
    process=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',size,'-r','5','-i','-',
                              '-c:v','ffv1','-pix_fmt','bgr0',str(path)],stdin=subprocess.PIPE)
    for raw in rasters: process.stdin.write(raw.tobytes())
    process.stdin.close()
    if process.wait(): raise RuntimeError('Lossless tail encode failed')


def concat(prefix, tail, out):
    filters='[0:v]setpts=PTS-STARTPTS,format=bgr0[p];[1:v]setpts=PTS-STARTPTS,format=bgr0[t];[p][t]concat=n=2:v=1:a=0[v]'
    subprocess.run(['ffmpeg','-v','error','-noautorotate','-i',str(prefix),'-i',str(tail),'-filter_complex',filters,'-map','[v]',
                    '-fps_mode','passthrough','-c:v','ffv1','-pix_fmt','bgr0',str(out)],check=True)


def main():
    verify(read(HERE/'lock.json'))
    out=HERE/'append';out.mkdir()
    reg=read(ROOT/'dacf_round4/lock.json')['cases']['doorway']
    p=ROOT/reg['source']; stream,clock=timeline(p)
    lasts=[]; hashes=[]
    for r,bits in native_stream(p,stream,clock):
        hashes.append(sha256(bits).hexdigest())
        if r['frame']==len(clock)-1:
            lasts.append(np.frombuffer(bits,np.uint8).reshape(stream['height'],stream['width'],3))
    static=lasts[0]
    h,w=static.shape[:2]
    texture=np.stack(np.mgrid[:h,:w][::-1],axis=-1)
    pieceA=np.stack([20+(texture[:,:,0]//25)%3*8,50+(texture[:,:,1]//23)%3*8,218-(texture[:,:,0]//11)%4*10],axis=-1).astype(np.uint8)
    pieceB=np.stack([225+(texture[:,:,1]//25)%3*8,190+(texture[:,:,0]//17)%3*8,18+(texture[:,:,1]//13)%4*10],axis=-1).astype(np.uint8)
    cv2.setNumThreads(1)
    encoder=clip_encoder()
    audit=read(ROOT/'dacf_round4/offline/doorway/initial_stops.json')
    result={}
    for name,frames in [('matched_static',[static]*40),('unrelated_high',[pieceA,pieceB]*20)]:
        src=out/f'{name}-tail.nut'; dst=out/f'{name}-joined.nut'
        tail_encode(src,stream,frames)
        concat(p,src,dst)
        st, rows=timeline(dst)
        prefix=[]; raw=[]
        for row,bits in native_stream(dst,st,rows):
            if row['frame'] < len(clock):
                prefix.append(sha256(bits).hexdigest())
                if row['frame']+1<len(clock): continue
            pixels=Image.frombytes('RGB',(w,h),bits)
            raw.append({'rgb':np.asarray(pixels.resize((320,round(h*320/w)),Image.Resampling.BILINEAR))})
        assert hashes==prefix and len(raw)==len(frames)+1
        bank=Bank(dict(enumerate(raw)),read(ROOT/'dacf_research/protocol.json'),encoder)
        result[name]={'exact_prefix_rgb_frames':len(prefix),'tail_frames':len(raw)-1,
                      'joined_time_max_abs_shift_s':max(abs(a['seconds']-b['seconds']) for a,b in zip(clock,rows[:len(clock)],strict=True)),
                      'scope':'Frozen prefix path and proposal geometry. Tail variation measured excluding seam.', 'channels':{}}
        for ch in METRICS:
            distances=[bank.metric(a,a+1)[ch] for a in range(len(raw)-1)]
            measured=sum(distances[1:]); initial=audit[ch]
            result[name]['channels'][ch]={'full_prefix_path':initial['global_change_total'],'prefix_candidate_gap':initial['largest_candidate_gap'],
              'prefix_global_threshold':initial['global_limit'],'tail_variation_without_join':measured,'join_distance':distances[0],
              'initial_global_policy':initial['s0_initial_stop'],'global_after_tail_without_join':initial['largest_candidate_gap']<=.12*(initial['global_change_total']+measured),
              'global_after_tail_with_join':initial['largest_candidate_gap']<=.12*(initial['global_change_total']+measured+distances[0]),
              'new_threshold_without_join':.12*(initial['global_change_total']+measured),
              'local_policy_before_and_after':[{'region':r['root'],'pts':r['pts'],'eligible_both':r['local']['eligible'],'ratio_both':r['local']['fraction']} for r in initial['candidates']]}
    write(out/'diagnostic.json',result)
    print('Prefix RGB verified; freeze original interval and local edge applicability. Separate global tail and join.')


if __name__=='__main__':main()
