"""Create new sources and freeze study inputs without exposing score roles."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

from dacf_research.freeze import PLUGIN
from dacf_research.prepare_inputs import ROOT, DESKTOP, digest, read, write
from dacf_research.run import raw_frames
from dacf_round4.offline import verify as verify4, Experiment, clip_encoder, proposals, METRICS
from evidenweft_phase1.source_review import timeline, WITHDRAWN

HERE=Path(__file__).resolve().parent


def fixture(path):
    width,height,fps=640,360,25
    y,x=np.mgrid[:height,:width]
    rng=np.random.default_rng(511)
    texture=np.clip(168+.05*x-.12*y+rng.normal(0,3.,(height,width)),0,255)
    table=np.stack([texture+9,texture+2,texture-7],axis=-1).astype(np.uint8)
    for px,py in ((31,275),(560,101),(97,306),(585,307)):
        cv2.circle(table,(px,py),8,(124,128,125),1,cv2.LINE_AA)
        cv2.line(table,(px-14,py),(px+14,py),(140,137,131),1,cv2.LINE_AA)
    worker=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
                              '-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
    for frame in range(350):
        im=table.copy()
        cv2.rectangle(im,(75,74),(175,145),(49,58,58),-1)
        cv2.rectangle(im,(84,84),(166,134),(21,32,31),-1)
        cv2.circle(im,(154,104),6,(48,224,111) if 159<=frame<=166 else (63,89,77),-1,cv2.LINE_AA)
        cv2.rectangle(im,(66,247),(163,320),(57,119,165),-1)
        cv2.rectangle(im,(74,255),(155,312),(87,164,194),2)
        # The token travels continuously after a long hold. An opaque screen is drawn over its path.
        px=int(round(490-min(max(frame-182,0),115)*3.25))
        cv2.ellipse(im,(px+5,284),(19,9),0,0,360,(125,107,97),-1,cv2.LINE_AA)
        cv2.circle(im,(px,280),14,(214,51,47),-1,cv2.LINE_AA)
        cv2.circle(im,(px-3,277),4,(248,219,164),-1,cv2.LINE_AA)
        cv2.rectangle(im,(270,163),(362,303),(77,87,92),-1)
        cv2.rectangle(im,(278,170),(353,293),(111,122,125),-1)
        cv2.line(im,(286,183),(345,183),(145,153,153),2,cv2.LINE_AA)
        # An independent late action is a violet tag laid beside the indicator box.
        if frame>=315:
            cv2.rectangle(im,(184,89),(220,120),(116,73,164),-1)
            cv2.line(im,(188,94),(215,94),(191,154,225),2)
        worker.stdin.write(im.tobytes())
    worker.stdin.close()
    if worker.wait(): raise RuntimeError('fixture encoder failed')


def verify(lock):
    verify4(read(ROOT/'dacf_round4/lock.json'))
    assert all(digest(ROOT/p)==h for p,h in lock['sha256'].items())


def nodes(task, configuration):
    flags={}
    for channel in METRICS:
        axis, _, options, largest=proposals(task,task.coarse,task.initial,channel,configuration)
        flags[channel]={'whole_path':axis['total'],'largest_candidate_gap':largest,'threshold':.12*axis['total'],
                        'S0': 'stopped' if largest<=axis['total']*.12 else 'continue', 'options':options,
                        'unknown_edges':sum(not r['applicable'] for r in axis['edges'])}
    def make(key,reg,lo,hi,root=False):
        measured=[p for p in task.coarse if lo<=p<=hi]
        stops={}
        for ch,s in flags.items():
            options=[o for o in s['options'] if o['root']==key]
            usable=any(o['local']['eligible'] for o in options)
            unknown=any(o['local']['fraction'] is None for o in options)
            leaf=reg.get('kind')=='paired_investigation'
            stops[ch]={'S0':s['S0'],'S0_scope':'whole_source_policy','S1_scope':'paired_leaf_only',
                       'S1':('not_evaluated' if not leaf else 'continue' if usable else 'unknown' if unknown else 'stopped'),
                       'largest_leaf_gap':max((o['gap'] for o in options),default=None),'local_evidence':options[:3],
                       'global_gap':s['largest_candidate_gap'],'global_threshold':s['threshold']}
        data={'id':key,'kind':'source' if root else reg['kind'],'start':task.times[lo],'end':task.times[hi],
              'bounds_pts':[lo,hi],'nominees':[p for p in measured if p in task.initial], 'known_pts':measured,
              'unknown_interiors':[[a,b] for a,b in zip(measured,measured[1:]) if task.ranks[b]-task.ranks[a]>1],
              'anchor_pts':[r['pts'] for side in ('left','right') if (r:=reg.get(side))],
              'physical_continuity':'unknown','parent_id':None if root else 'source','children':list(reg.get('children',[])),
              'resolution_max_s':max((task.times[b]-task.times[a] for a,b in zip(measured,measured[1:])),default=None),'stops':stops,
              'spatial_native':[0,0,task.reader.stream['width'],task.reader.stream['height']]}
        f=task.signals.get(key)
        if f:
            data['anchor_and_signal']={n:f[n] for n in ('crossings','group_distance_by_channel','multi_scale_extrema')}
        return data
    all_nodes=[make('source',{},task.native[0],task.native[-1],True)]
    all_nodes.extend(make(k,r,*r['bounds_pts']) for k,r in task.plan.items())
    index={n['id']:n for n in all_nodes}
    for node in all_nodes[1:]:
        for child in node['children']: index[child]['parent_id']=node['id']
    all_nodes[0]['children']=[n['id'] for n in all_nodes[1:] if n['parent_id']=='source']
    joins=[]
    for i,n in enumerate(all_nodes[1:]):
        for m in all_nodes[i+2:]:
            shared=set(n['known_pts'])&set(m['known_pts'])
            if shared: joins.append({'nodes':[n['id'],m['id']],'pts':sorted(shared),'kind':'shared_exact_source_pts','identity':'unknown'})
    return all_nodes,joins


def main():
    prev=read(ROOT/'dacf_round4/lock.json'); verify4(prev)
    study=HERE/'inputs'; study.mkdir()
    policy=read(ROOT/'dacf_round4/protocol.json')
    cfg=read(ROOT/'dacf_research/protocol.json')
    cv2.setNumThreads(1)
    encoder=clip_encoder()
    cases={}; files=[HERE/'protocol.json',HERE/'setup.py',ROOT/'dacf_round4/run_manifest.json',ROOT/'dacf_round4/evaluation/summary.json',
                     ROOT/'dacf_round4/agent/interim-audit.json',ROOT/'dacf_round4/lock.json']
    known={digest(ROOT/c['source']) for c in prev['cases'].values()}
    for case, original in {'fixture':None,'arm_clip':DESKTOP/'4.mp4'}.items():
        folder=study/case; folder.mkdir()
        video=folder/'source.mp4'
        fixture(video) if original is None else shutil.copyfile(original,video)
        sig=digest(video)
        assert sig not in known and sig!=WITHDRAWN
        index=folder/'index'
        subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),'--video',str(video),'--out',str(index),'--fps','5','--keep-analysis'],check=True)
        stream, clock=timeline(video)
        mf=read(index/'manifest.json')
        raw_frames(video,stream,clock,{p['source_frame'] for p in mf['pool']},folder,sig,320)
        loc={k:v.relative_to(ROOT).as_posix() for k,v in {'source':video,'index':index,'pixels':folder}.items()}|{'cohort':'new'}
        build=folder/'plan_evidence'; build.mkdir()
        exp=Experiment(case,loc,encoder,cfg,build)
        leaf,edges=nodes(exp,policy)
        write(folder/'plan.json',{'sha256':sig,'time_base':stream['time_base'],'width':stream['width'],'height':stream['height'],
                                  'clock':clock,'nodes':leaf,'edges':edges})
        files.extend([video,index/'manifest.json',index/'catalog.json',folder/'plan.json',folder/'frame_receipts.json'])
        cases[case]=loc
        print(case,'source frames',len(clock),'nominees',len(exp.initial),'tree nodes',len(leaf),flush=True)
    write(HERE/'lock.json',{'date':datetime.now().astimezone().isoformat(),'cases':cases,
                            'sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in files}})


if __name__=='__main__':main()
