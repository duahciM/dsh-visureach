"""Pin the common source addresses and separate assessor targets from predictions."""
from collections import defaultdict
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw

from dacf_research.prepare_inputs import ROOT,digest,read,write
from evidenweft_phase1.source_review import native_stream,timeline
from partial_modules_stage1.fixtures import textile

H=Path(__file__).resolve().parent


def frame_box(x,y,patch):
    return [x,y,patch.shape[1],patch.shape[0]]


def render(case):
    out=H/'inputs'/case;out.mkdir(parents=True)
    src=out/'source.mkv'; w,h=800,448
    rng=np.random.default_rng(6183 if case=='roles' else 1974)
    scenes=[textile(rng,[(64,83,87),(91,93,100),(115,125,132),(139,137,144),(173,161,150)],h,w) for _ in range(2)]
    for canvas in scenes:
        for idx in range(95):
            x,y=map(int,rng.integers([14,14],[w-14,h-14]))
            cv2.line(canvas,(x-5,y-3),(x+7,y+3),(121,145,160),2)
    target=textile(rng,[(172,54,65),(242,86,111),(231,169,102),(180,106,57),(50,45,68)],84,104)
    rail=textile(rng,[(72,134,53),(117,172,50),(106,139,68),(135,160,80)],67,72)
    housing=textile(rng,[(113,59,177),(165,90,210),(138,88,194),(217,76,153)],76,91)
    post=textile(rng,[(45,115,185),(83,141,237),(103,176,210),(72,85,161)],65,99)
    shots=[]; tracks=[]
    for k in range(80):
        if case=='roles':
            im=scenes[0].copy()
            lamp=housing.copy()
            if 12<=k<28:cv2.rectangle(lamp,(14,13),(70,57),(250,196,71),-1)
            items=[('lamp',512,260,lamp),('rail',330,173,rail),('post',641,46,post)]
            if 44<=k<56:
                cv2.rectangle(im,(179,165),(333,265),(62,81,94),-1)
            elif k<44:
                x=67+min(k*5,160)
                items.append(('target_before',x,164,target))
            else:
                x=171+min((k-56)*4,63)
                items.append(('target_after',x,165,target))
            if 20<=k<43:items.append(('copy',599-min((k-20)*7,105),142,target))
            dx=dy=0
        else:
            im=scenes[1 if 32<=k<52 else 0].copy()
            items=[('post',605,50,post),('fixed',175,191,target),('rail',369,229,rail)]
            dx=min(40,k*2) if k<32 else 16 if k<52 else 38-min((k-52)*2,25)
            dy=min(17,k) if k<32 else 4
        annotations=[]
        for label,x,y,u in items:
            x0,y0,bw,bh=frame_box(x,y,u)
            im[y0:y0+bh,x0:x0+bw]=u
            annotations.append({'tag':label,'box':[x0+dx,y0+dy,bw,bh]})
        if dx or dy:im=cv2.warpAffine(im,np.float32([[1,0,dx],[0,1,dy]]),(w,h),borderMode=cv2.BORDER_REFLECT)
        if case=='camera' and 14<=k<26: im=np.uint8(np.clip(im.astype(np.int16)+24,0,255))
        shots.append(im.tobytes());tracks.append(annotations)
    subprocess.run(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','20','-i','-',
      '-an','-c:v','ffv1','-pix_fmt','bgr0',str(src)],input=b''.join(shots),check=True,timeout=50)
    _,clock=timeline(src); selected=clock[::4]
    truth=[]
    for row in selected:
        frame=row['frame']; t=row['pts']
        if case=='roles':
            role_map={'target_before':'independent_change','target_after':'independent_change','copy':'independent_change',
              'rail':'relation_participant' if 20<=frame<44 or frame>=64 else 'environment',
              'lamp':'independent_change' if 12<=frame<=28 else 'environment','post':'specified_goal' if frame==16 else 'environment'}
            possible=['target_before','target_after','copy','rail','lamp','post']
        else:
            role_map={'post':'specified_goal' if frame==16 else 'environment','fixed':'environment','rail':'environment'}
            possible=['post','fixed','rail']
        for roi in tracks[frame]:
            assert roi['tag'] in possible
            truth.append({'pts':t,'frame':frame,'role':role_map[roi['tag']],
                          'episode':roi['tag']+('_before' if frame<32 else '_away' if frame<52 else '_return') if case=='camera' else roi['tag'],**roi})
    goals=[{'pts':r['pts'],'native_xywh':r['box'],'intent':'Inspect the static local region at the supplied address without assuming its identity.'}
            for r in truth if r['role']=='specified_goal']
    return src,selected,truth,goals


def external(case):
    home=H/'external'/case; source=home/'source.webm';duration,clock=timeline(source)
    labels=[[int(float(p)) for p in re.split(r'[,\s]+',s.strip())[:4]] for s in (home/'gt.txt').read_text('utf8').splitlines() if s.strip()]
    start=int(re.search(r'start:\s*(\d+)',(home/'dataset.yml').read_text('utf8'))[1])
    # OpenCV tracker test ground-truth rows are addressed relative to its configured first frame.
    begin=start-1
    selected=clock[begin:begin+60:4]
    assert len(selected)==15 and len(labels)>56
    truth=[]
    for row in selected:
        x,y,w,h=labels[row['frame']-begin]
        x=max(0,x);y=max(0,y);w=min(w,duration['width']-x);h=min(h,duration['height']-y)
        assert min(w,h)>0
        truth.append({'pts':row['pts'],'frame':row['frame'],'role':'independent_change','episode':'visible_reference_region','box':[x,y,w,h],'label_index':row['frame']-begin})
    return source,selected,truth,[]


def replay():
    source=ROOT/'partial_modules_stage2/inputs/street/source.avi'
    data=read(ROOT/'partial_modules_stage2/inputs/street/observations.json')
    _,clock=timeline(source);ticks=set(data['sampled_pts'])
    truth=[{'pts':r['pts'],'frame':r['frame'],'role':{'candidate':'independent_change','reference':'environment','peer':'relation_participant'}[r['role']],
            'episode':r['episode'],'box':r['native_xywh']} for r in data['observations']]
    return source,[r for r in clock if r['pts'] in ticks],truth,[]


def review(source,selected,truth,name):
    # Contact views are prepared for assessment. Predictions receive only the source and chosen addresses.
    info,clock=timeline(source)
    chosen=[selected[i]['pts'] for i in range(0,len(selected),4)]
    labels=defaultdict(list)
    for y in truth:labels[y['pts']].append(y)
    canvas=Image.new('RGB',(880,540),'#18202b');g=ImageDraw.Draw(canvas)
    for row,rgb in native_stream(source,info,clock):
        t=row['pts']
        if t not in chosen:continue
        i=chosen.index(t); x,y=(i%3)*293,(i//3)*270
        frame=Image.frombytes('RGB',(info['width'],info['height']),rgb)
        frame.thumbnail((287,242)); d=ImageDraw.Draw(frame)
        for label in labels[t]:
            bx,by,bw,bh=label['box'];sx=frame.width/info['width'];sy=frame.height/info['height']
            d.rectangle((bx*sx,by*sy,(bx+bw)*sx,(by+bh)*sy),outline='#ffde86',width=3)
        canvas.paste(frame,(x,y)); g.text((x+2,y+245),f'PTS {t} | frame {row["frame"]}',fill='white')
        if t==chosen[-1]:break
    canvas.save(H/'references'/f'{name}.png')


if __name__=='__main__':
    (H/'inputs').mkdir();(H/'references').mkdir()
    selected_sources={'roles':render('roles'),'camera':render('camera'),'street-repeat':replay(),
                      'external-1':external('david'),'external-2':external('faceocc2')}
    inputs={}; reference={}
    for name,(source,times,truth,goals) in selected_sources.items():
        inputs[name]={'path':source.relative_to(ROOT).as_posix(),'sha256':digest(source),
                      'pts':[r['pts'] for r in times],'explicit_goal_queries':goals}
        reference[name]=truth
        review(source,times,truth,name)
    write(H/'inputs/source_inventory.json',inputs)
    write(H/'references/sealed.json',{'targets':reference,'scope':'Only listed independent visual goals. Unlabelled regions have unknown relevance; projected ROI is not physical identity.'})
