"""Source-only motion proposals and their independently audited local scopes."""
from collections import Counter
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import digest,read,write
from dacf_round3.acquire import Reader
from evidenweft_phase1.source_review import timeline
from partial_modules_stage2.scoped import H,Video,freeze_known,scopes


def proposals(images):
    background=np.median(images,axis=0).astype(np.uint8)
    for image in images:
        raw=(cv2.absdiff(background,image)>24).astype(np.uint8)*255
        opened=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask=cv2.morphologyEx(opened,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[cv2.boundingRect(c) for c in contours if 160<=cv2.contourArea(c)<=.08*image.size]
        rows=sorted(boxes,key=lambda q:(-q[2]*q[3],q[0],q[1]))
        yield rows[:8],len(rows)-min(8,len(rows))


def run_case(case,folder):
    folder.mkdir()
    requested=H/'inputs'/case
    source=requested/('source.avi' if case=='street' else 'source.mkv')
    stream,clock=timeline(source);sha=digest(source)
    selected=[r for r in clock if r['pts'] in range(14,59,4)] if case=='street' else clock[::4]
    readers=Reader(source,sha,stream,clock,folder/'candidate_rasters')
    raw=[]
    for row in selected:
        rec,_=readers.read(row['pts'])
        with Image.open(rec['path']) as img:raw.append(cv2.cvtColor(np.array(img),cv2.COLOR_RGB2GRAY))
    obs=[];discard={}
    for r,(boxes,extra) in zip(selected,proposals(np.stack(raw))):
        pts=r['pts'];discard[pts]=extra
        obs.extend(r|{'role':'candidate','episode':f'auto-source-{pts}-{i}','native_xywh':list(box)}
                   for i,box in enumerate(boxes))
    freeze_known({'source':source.name,'video_sha256':sha,'native_size':[stream['width'],stream['height']], 'time_base':stream['time_base'],
                  'sampled_pts':[r['pts'] for r in selected]},folder/'automatic_input',source,obs)
    t=perf_counter();video=Video(folder/'automatic_input',folder);result=scopes(video,supplied_labels=False);secs=perf_counter()-t
    serialized=result|{'relations':list(result['relations'].values())}
    write(folder/'prediction.json',{'proposals':obs,'discarded_contours':discard,'scoped':serialized, 'source_sha256':sha,
               'view_inventory':{p:[{'view_key':x['view_key'],'native_xywh':x['native_xywh']} for x in rows] for p,rows in video.views.items()},
               'scope_fit_seconds_with_source_preparation':secs,'predecode_receipts':list(readers.receipts.values()),'source_read_receipts':list(video.reader.receipts.values())})


def iou(a,b):
    x=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]));y=max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return x*y/(a[2]*a[3]+b[2]*b[3]-x*y)


def evaluate(case,folder):
    data=read(folder/'prediction.json')
    trusted=H/'inputs'/case/('observations.json' if case=='street' else 'observations-dense.json')
    roi=read(trusted)['observations']; result=[]; matched={}
    for time, items in data['view_inventory'].items():
        pts=int(time)
        base=[x for x in roi if x['role']=='candidate' and x['pts']==pts]
        for item in items:
            candidate=max(((iou(item['native_xywh'],truth['native_xywh']),truth['episode']) for truth in base),default=(0,None))
            if candidate[0]>=.5:matched[item['view_key']]=candidate[1]
            result.append({'pts':pts,'view':item['view_key'],'best_given_roi_iou':candidate[0],'recorded_episode':candidate[1]})
    associations=Counter()
    for edge in data['scoped']['relations']:
        a,b=(matched.get(k) for k in edge['views'])
        associations['unassessed_outside_sealed_RoI' if a is None or b is None else 'matched_same_local_label' if a==b else 'conflicts_with_local_labels']+=1
    all_ticks={r['pts'] for r in data['predecode_receipts']}
    possible={(r['pts'],r['episode']) for r in roi if r['role']=='candidate' and r['pts'] in all_ticks}
    recovered={(r['pts'],r['recorded_episode']) for r in result if r['best_given_roi_iou']>=.5}
    write(folder/'evaluation.json',{'supplied_roi_count':len(possible),'matched_supplied_roi_count':len(possible&recovered),
          'automatic_boxes':len(data['proposals']),'proposals_with_iou_at_least_half':sum(x['best_given_roi_iou']>=.5 for x in result),
          'automatic_scope_count':len(data['scoped']['scopes']),'link_evaluation':dict(associations),'rows':result,
          'limitation':'Other detected people or moving regions can be real. Withheld boxes cover only specific questions, not an exhaustive detection ground truth.'})


if __name__=='__main__':
    base=H/'automatic';base.mkdir()
    cv2.setNumThreads(1);cv2.setRNGSeed(628)
    for k in ('control','street'):run_case(k,base/k)
    for k in ('control','street'):evaluate(k,base/k)
