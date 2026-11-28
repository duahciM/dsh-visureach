from pathlib import Path
import subprocess,sys

import cv2,numpy as np

from dacf_research.prepare_inputs import ROOT,digest,read,write
from dacf_research.freeze import PLUGIN
from dacf_research.run import raw_frames
from dacf_round4.offline import Experiment,clip_encoder
from dacf_round5.setup import nodes,verify
from evidenweft_phase1.source_review import timeline

HERE=Path(__file__).resolve().parent


def make_source(p):
    w,h=480,272
    rng=np.random.default_rng(608721)
    noise=rng.normal(0,4,(h,w,1))
    y,x=np.mgrid[:h,:w]
    tex=np.clip(np.array([153,147,129])+noise+((x-y)*.035)[:,:,None],0,255).astype('uint8')
    worker=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','25',
                               '-i','-','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',str(p)],stdin=subprocess.PIPE)
    for i in range(300):
        a=tex.copy()
        for x0,y0,on in ((48,60,79<=i<87 or 197<=i<230),(390,88,146<=i<151)):
            cv2.rectangle(a,(x0-23,y0-15),(x0+23,y0+15),(48,54,50),-1)
            cv2.circle(a,(x0,y0),5,(251,213,45) if on else (106,93,62),-1)
        x0=int(405-min(max(i-109,0),138)*2.25)
        cv2.rectangle(a,(61,172),(150,238),(73,121,160),-1)
        cv2.circle(a,(x0,205),12,(208,58,54),-1);cv2.circle(a,(x0-4,202),4,(253,177,135),-1)
        # Two separate occluders; no target labels are visible.
        cv2.rectangle(a,(275,132),(329,241),(68,78,83),-1)
        cv2.rectangle(a,(182,145),(212,239),(101,95,105),-1)
        if i>=259:cv2.rectangle(a,(352,166),(374,187),(87,181,126),-1)
        worker.stdin.write(a.tobytes())
    worker.stdin.close()
    if worker.wait():raise RuntimeError('ffmpeg failed')


def main():
    baseline=ROOT/'dacf_round5';verify(read(baseline/'lock.json'))
    home=HERE/'independent';home.mkdir()
    source=home/'source.mp4';make_source(source)
    index=home/'index'
    subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),'--video',str(source),'--out',str(index),'--fps','5','--keep-analysis'],check=True)
    meta=read(index/'manifest.json'); stream,clock=timeline(source)
    raw_frames(source,stream,clock,{i['source_frame'] for i in meta['pool']},home,digest(source),320)
    case={k:v.relative_to(ROOT).as_posix() for k,v in {'source':source,'index':index,'pixels':home}.items()}|{'cohort':'new'}
    stage=home/'diagnostics';stage.mkdir()
    cv2.setNumThreads(1)
    task=Experiment('independent',case,clip_encoder(),read(ROOT/'dacf_research/protocol.json'),stage)
    tree,edges=nodes(task,read(ROOT/'dacf_round4/protocol.json'))
    write(home/'plan.json',{'sha256':digest(source),'time_base':stream['time_base'],'width':stream['width'],'height':stream['height'],'clock':clock,'nodes':tree,'edges':edges})
    files=[baseline/'release-fixed/verification.json',baseline/'release-fixed/local-visureach-amr-research-0.6.0-r5.1.tgz',
       baseline/'run_manifest.json',baseline/'scored-audit.json',baseline/'inputs/fixture/source.mp4',baseline/'inputs/fixture/plan.json',
       source,home/'plan.json',HERE/'guide.json',HERE/'measure.mjs',HERE/'native.py']
    write(HERE/'lock.json',{'files':{p.relative_to(ROOT).as_posix():digest(p) for p in files},'scope':'Prompt-only comparisons on identical r5.1'} )

if __name__=='__main__':main()
