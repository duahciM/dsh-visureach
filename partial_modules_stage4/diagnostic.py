"""After prediction, isolate proposal failures and run manual-region diagnostic arms."""

from collections import Counter, defaultdict
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import ROOT, read, write
from partial_modules_stage1.study import box_at, fit, in_box
from partial_modules_stage3.evaluate import summarize
from partial_modules_stage3.observation import K5, Media, NEIGHBOR, contours, iou, scoped_identity
from partial_modules_stage3.run import m5
from partial_modules_stage4.prepare import H


def previous_rail_failure():
    trial=H/'predictions/roles'
    data=read(trial/'region_features.json')
    originals={r['pts']:r for r in read(trial/'base_receipts.json')['native_receipts']}
    gold=read(H/'inputs/sealed_assessment.json')['roles']['legacy_targets']
    result=[]
    for label in gold:
        if label['role']!=NEIGHBOR:
            continue
        t=label['pts'];box=label['box']
        with Image.open(originals[t]['path']) as original:
            rgb=np.asarray(original.convert('RGB'))
        sat=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1]
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        chroma=cv2.morphologyEx((sat>=55).astype(np.uint8)*255,cv2.MORPH_CLOSE,K5)
        edge=cv2.morphologyEx(cv2.Canny(gray,45,110),cv2.MORPH_CLOSE,K5)
        raw=[contours(chroma),contours(edge)]
        prior=[max((iou(q,box) for q in boxes),default=0) for boxes in raw]
        nxt=[max((iou(q['native_xywh'],box) for q in data['formation_inputs'][str(t)]['boundaries']),default=0)]
        result.append({'pts':t,'source_root':originals[t]['id'],'reference_xywh':box,
            'chroma_candidate_count_before_top8':len(raw[0]),'edge_components_before_top8':len(raw[1]),
            'best_old_component_iou_before_top8':prior,'best_new_closed_boundary_iou_before_role':nxt[0],
            'first_old_failure':'component_boundary_formation' if max(prior)<.5 else 'later_selection'})
    return result


def manual_arms(cases,labels,out):
    output={}
    for name,spec in cases.items():
        t0=perf_counter();dest=out/name;dest.mkdir()
        media=Media(spec,dest)
        entry=labels[name]
        by_time={t:[] for t in media.ticks}
        for r in entry['legacy_targets']:
            by_time[r['pts']].append(media.region(r['pts'],r['box'],[r['role']],
                [{'kind':'independently_supplied_manual_endpoint'}])|{'episode':r['episode']})
        identity=scoped_identity(media,by_time,'A4')
        data={'source_sha256':media.sha,'arm':'A4','regions':sum(list(by_time.values()),[]),'identity':identity,
              'relation':m5(media,by_time,identity,manual=True),'seconds_identity':None,'seconds_relation':None}
        authored=defaultdict(dict)
        for o in entry['local_anchors']:
            authored[o['key']][o['pts']]=o['roi']
        anchored=[]
        for a,b in zip(media.ticks,media.ticks[1:]):
            u,v=media.feature_pair(a,b)
            for key, refs in authored.items():
                if a not in refs or b not in refs:
                    continue
                support=in_box(u,box_at(refs[a],media.packets[a]))&in_box(v,box_at(refs[b],media.packets[b]))
                status,_=fit(u,v,support)
                global_unexcluded=media.chart(a,b)
                anchored.append({'key':key,'pts':[a,b],'source_roots':[media.source_root(a),media.source_root(b)],
                    'rois':[refs[a],refs[b]],'point_measurement':status,'all_frame_without_candidate_exclusion':global_unexcluded,
                    'scope':'Manually specified comparable area, no ROI appearance identification'} )
        write(dest/'result.json',data|{'human_local_reference_checks':anchored,'actual_originals':list(media.raw_reader.receipts.values())})
        output[name]={'manual_box_and_relation':summarize(data,entry['legacy_targets']),
             'manual_local_ref_status':dict(Counter(x['point_measurement']['state'] for x in anchored)),
             'original_count':len(media.ticks),'seconds':perf_counter()-t0}
    return output


if __name__=='__main__':
    assert (H/'predictions/completed.json').is_file() and (H/'evaluation.json').is_file()
    output=H/'manual';output.mkdir()
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    labels=read(H/'inputs/sealed_assessment.json')
    cases=read(H/'inputs/source_inventory.json')
    write(output/'first_failure.json',previous_rail_failure())
    write(output/'summary.json',manual_arms(cases,labels,output))
