"""Independent M2 and M5 experiments on compiled view receipts. Does not read evaluation truth."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
from math import hypot
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import ROOT,digest,read,write
from evidenweft_phase1.source_review import native_stream,timeline
from dacf_round3.acquire import Reader
from partial_modules_stage1.fixtures import HERE

sys.path.insert(0,str(ROOT/'video_observer_offline060/python'))
from local_observation import box_at, correspond, fit, in_box, pair, prepare
from p223_runtime import figure2


def root(prefix,*fields):
    return prefix+'-'+sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()[:32]


def box_distance(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    return hypot(max(ax-(bx+bw),bx-(ax+aw),0),max(ay-(by+bh),by-(ay+ah),0))


def centroid(box):
    x,y,w,h=box
    return np.array([x+w/2,y+h/2])


def patch_color(image,box):
    x,y,w,h=box
    tile=image[y:y+h,x:x+w]
    luma=float(cv2.cvtColor(tile,cv2.COLOR_RGB2GRAY).mean())
    hsv=cv2.cvtColor(tile,cv2.COLOR_RGB2HSV)
    bins=cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]).flatten()
    return {'mean_luma_8bit':luma,'hsv_h12s4':(bins/bins.sum()).tolist()}


class Video:
    def __init__(self, folder, output):
        manifest=read(folder/'observations.json')
        self.source=folder/manifest['source']; self.sha=digest(self.source)
        assert self.sha==manifest['video_sha256']
        stream,self.clock=timeline(self.source)
        self.reader=Reader(self.source,self.sha,stream,self.clock,output/'native')
        self.images={}; self.prepared={}; self.views=defaultdict(list); self.groups=defaultdict(list)
        self.comparisons={};self.background={};self.failures={}
        self.permits=set(manifest['sampled_pts'])
        for item in manifest['observations']:
            tick=item['pts']
            if tick not in self.images and tick not in self.failures:
                try:
                    rec,_=self.reader.read(tick)
                    with Image.open(rec['path']) as p:self.images[tick]=np.array(p.convert('RGB'))
                    self.prepared[tick]=prepare(rec['path'])
                except ValueError as e:
                    self.failures[tick]=str(e)
            if tick in self.failures:
                continue
            rec=self.reader.receipts[tick]
            roi=item['native_xywh']; name=item['episode']
            view=item|{'source_root':rec['id'],'source_sha256':self.sha,
                      'view_key':root('view',rec['id'],roi,name),'color':patch_color(self.images[tick],roi)}
            # These are conservative per-view aliases, matching the six roles in the installed ledger.
            for k in ('observation_id','appearance_id','slot_id','context_id','instance_candidate_id','track_id'):
                view[k]=root(k,view['view_key'])
            self.views[tick].append(view)
            if item['role']=='candidate':self.groups[name].append(view)
        for items in self.groups.values():items.sort(key=lambda v:v['frame'])

    def comparison(self,a,b):
        key=(a['view_key'],b['view_key'])
        if key not in self.comparisons:
            av,bv=a['native_xywh'],b['native_xywh']
            v=pair(self.prepared[a['pts']],self.prepared[b['pts']],av,bv)
            color=.5*float(np.abs(np.array(a['color']['hsv_h12s4'])-np.array(b['color']['hsv_h12s4'])).sum())
            self.comparisons[key]={'comparison_ref':root('local-fit',self.sha,*key),'source_roots':sorted({a['source_root'],b['source_root']}),
                              'pts':[a['pts'],b['pts']],'views':list(key),'hsv_total_variation':color,**v}
        return self.comparisons[key]

    def context(self,t0,t1):
        key=t0,t1
        if key in self.background:return self.background[key]
        a,b=self.prepared[t0],self.prepared[t1]
        pairs=correspond(a[3],b[3])
        ia,ib=(np.array([m[j] for m in pairs], dtype=int) for j in (0,1))
        xa,xb=a[2][ia],b[2][ib]
        allowed=np.ones(len(ia),dtype=bool)
        for ticks,coords,pre in ((t0,xa,a),(t1,xb,b)):
            for view in self.views[ticks]:
                allowed &= ~in_box(coords,box_at(view['native_xywh'],pre))
        measure,_=fit(xa,xb,allowed)
        result=measure|{'calculation_root':root('explicit-context',self.sha,t0,t1),
                         'source_roots':[self.reader.receipts[t]['id'] for t in key], 'pts':list(key),
                         'scope':'Background support excluding all supplied candidate and referent ROIs.'}
        self.background[key]=result
        return result

    def semantic_free_graph(self, pts):
        regions=[{'id':v['view_key'],'appearance':v['role'],'box':v['native_xywh'],
                  'roots':[v['view_key']],'roles':['independently_located_native_roi']} for v in self.views[pts]]
        graph=figure2({'regions':regions,'spatial_reference_observations':[],'candidate':str(pts)})
        return {'pts':pts,'source_root':self.reader.receipts[pts]['id'], 'calculation_root':root('figure2',self.sha,pts),
                'all_rois_share_original_root':True, 'geometry_units':'native pixels for supplied boxes',**graph}


def support(row):
    return row['local']['state']=='heldout_supported'


def choose_support(video,left,right):
    ra,rb=left[:2],right[:2]
    comparisons=[video.comparison(a,b) for a,b in product(ra,rb)]
    matches=[c for c in comparisons if support(c)]
    independent=next((list(p) for p in combinations(matches,2) if len(set(p[0]['source_roots']+p[1]['source_roots']))==4),[])
    return comparisons,matches,independent


def associate(video):
    results=[]
    for (e_a,left),(e_b,right) in combinations(sorted(video.groups.items()),2):
        comps,matches,independent=choose_support(video,left,right)
        exclude=[{'pts':a['pts'],'source_root':a['source_root'],'views':[a['view_key'],b['view_key']], 'gap_px':box_distance(a['native_xywh'],b['native_xywh'])}
                for a,b in product(left,right) if a['pts']==b['pts'] and box_distance(a['native_xywh'],b['native_xywh'])>0]
        # The future endpoint comparison is validation only; it does not change classification or promote prototypes.
        holdout=video.comparison(left[-1],right[-1]) if len(left)>2 and len(right)>2 else None
        if exclude:
            outcome='coappearance_conflict'
        elif independent:
            outcome='supported_visual_candidate'
        elif matches:
            outcome='partial_local_support'
        else:
            outcome='not_supported'
        results.append({'episodes':[e_a,e_b],'outcome':outcome,'supports':[c['comparison_ref'] for c in matches],
                        'counterevidence':exclude,'all_training_comparisons':[c['comparison_ref'] for c in comps],
                        'independent_comparison_refs':[c['comparison_ref'] for c in independent],
                        'holdout_comparison_ref':holdout['comparison_ref'] if holdout else None,
                        'holdout_support':support(holdout) if holdout else None,
                        'single_prototype_support':support(comps[0]),'physical_identity':'unknown',
                        'hypotheses':['reappearance_candidate','lookalike_or_replacement_candidate','view_or_quality_explanation'],
                        'missing_internal_interval':[left[-1]['pts'],right[0]['pts']] if left[-1]['pts']<right[0]['pts'] else None})
    return results


def snapshot(video, target):
    pts=target['pts']; center=centroid(target['native_xywh'])
    v={'object_luma_8bit':target['color']['mean_luma_8bit'],'object_hsv_h12s4':target['color']['hsv_h12s4']}
    refs={view['role']:view for view in video.views[pts] if view['role'] in ('reference','peer')}
    for role in ('reference','peer'):
        other=refs.get(role)
        v[role] = None if not other else {'offset_native_xy':(center-centroid(other['native_xywh'])).tolist(),
                  'bbox_gap_native_px':box_distance(target['native_xywh'],other['native_xywh']),'referent_episode':other['episode'],
                  'roots':[target['source_root']], 'roi_ids':[target['view_key'],other['view_key']], 'scope':'2D projection at exact sample'}
    return {'pts':pts,'frame':target['frame'],'target':target['view_key'],'target_region':target['native_xywh'],
            'source_root':target['source_root'],'episode':target['episode'],'values':v}


def difference(video,a,b,bridge=False):
    left,right=a['values'],b['values']; t0,t1=a['pts'],b['pts']
    context=video.context(t0,t1)
    applicable=context['state']=='heldout_supported'
    if applicable:
        transform=np.array(context['affine'])
        inverse=np.linalg.inv(transform[:,:2]); x_scale=np.array(video.prepared[t0][0].shape[::-1])/np.array(video.prepared[t0][1])
    channels={}
    for role in ('reference','peer'):
        l,r=left[role],right[role]
        if not l or not r or (l['referent_episode']!=r['referent_episode']):
            channels[role]={'state':'unknown','reason':'referent not observed or not comparable'}
            continue
        native_delta=np.array(r['offset_native_xy'])-np.array(l['offset_native_xy'])
        same=None if not applicable else ((inverse @ (np.array(r['offset_native_xy'])*x_scale))/x_scale-np.array(l['offset_native_xy'])).tolist()
        channels[role]={'state':'validated_projected_difference' if same is not None else 'unregistered_projected_difference',
                        'delta_offset_native_xy':native_delta.tolist(),'delta_offset_common_native_xy':same,
                        'delta_bbox_gap_native_px':r['bbox_gap_native_px']-l['bbox_gap_native_px'],
                        'local_rois':[l['roi_ids'],r['roi_ids']], 'units':'2D native pixels at endpoint views'}
    channels['object_luma']={'state':'raw_roi_appearance_only','delta_8bit':right['object_luma_8bit']-left['object_luma_8bit'],
                             'physical_state':'unknown','photometry_or_target_change':'unresolved'}
    return {'pts':[t0,t1],'source_roots':[a['source_root'],b['source_root']], 'context_root':context['calculation_root'],
            'context_applicability':context['state'],'gap_open_interval_pts':[t0,t1], 'channels':channels,
            'scope':'candidate cross-episode endpoints' if bridge else 'same supplied local episode',
            'physical_motion_or_contact':'unknown'}


def measure_relations(video,states):
    local=[]
    for episode,rows in states.items():
        events=[]
        for role in ('reference','peer'):
            last=None
            seen=False
            for row in rows:
                value=row['values'][role]
                if value:
                    event='first_confirmed_sample' if not seen else 'observed_again_after_sampled_gap' if last is None else 'sampled_measurement'
                    events.append({'role':role,'event':event,'pts':row['pts'],'source_root':row['source_root']})
                    seen=True
                else:
                    events.append({'role':role,'event':'unobserved_or_unavailable','pts':row['pts']})
                last=value
        local.append({'episode':episode,'sample_events':events,
                      'edges':[difference(video,a,b) for a,b in zip(rows,rows[1:])]})
    return local


def probe_bad_prototype(video, original):
    keys=list(sorted(video.groups))
    clean=video.groups[keys[0]]
    mislabeled=video.groups[keys[1]][0]
    rows=[]
    for episode in keys[2:]:
        query=video.groups[episode]
        trace,matches,stable=choose_support(video,[mislabeled,*clean[1:]],query)
        before=next(x for x in original if x['episodes']==[keys[0],episode])
        rows.append({'episode':episode,'clean_single':before['single_prototype_support'],'perturbed_single':support(trace[0]),
                     'clean_multi':bool(before['independent_comparison_refs']),'perturbed_multi':bool(stable),
                     'perturbed_refs':[c['comparison_ref'] for c in trace], 'promoted_views_count':0,
                     'scope':'deliberate provenance fault affects one source anchor, never feeds predictions'})
    return rows


def run_case(case, output):
    out=output/case;out.mkdir()
    start=perf_counter(); video=Video(HERE/'inputs'/case,out); source_sec=perf_counter()-start
    start=perf_counter()
    states={episode:[snapshot(video,v) for v in rows] for episode,rows in sorted(video.groups.items())}
    baseline={'m2':None,'m5':None,'bridges':None,'roi_state_samples':states,'graph_snapshots':[video.semantic_free_graph(t) for t in sorted(video.views)],
              'local_comparison_refs':[video.comparison(a,b)['comparison_ref'] for rows in video.groups.values() for a,b in zip(rows,rows[1:])]}
    base_seconds=perf_counter()-start
    start=perf_counter(); m2=associate(video); m2_seconds=perf_counter()-start
    start=perf_counter(); m5=measure_relations(video,states); m5_seconds=perf_counter()-start
    start=perf_counter(); bridges=[]
    for row in m2:
        a,b=row['episodes']; start_a,end_b=states[a][-1],states[b][0]
        if row['outcome']!='supported_visual_candidate' or start_a['pts']>=end_b['pts']:
            bridges.append({'episodes':[a,b],'state':'not_advanced','association':row['outcome']})
        else:
            bridges.append({'episodes':[a,b],'state':'candidate_endpoint_comparison','association':row['outcome'],
                            'measure':difference(video,start_a,end_b,True),'support_refs':row['independent_comparison_refs']})
    joint_seconds=perf_counter()-start
    stress=probe_bad_prototype(video,m2)
    for label,value in [('A0',baseline),('A1',{'m2':m2,'m5':None,'bridges':None}),('A2',{'m2':None,'m5':m5,'bridges':None}),
                        ('A3',{'m2':m2,'m5':m5,'bridges':bridges})]:
        write(out/f'{label}.json', {'arm':label,'source_sha256':video.sha,**value})
    observations_by_root={s['id']:s['rgb_sha256'] for s in video.reader.receipts.values()}
    verified=set()
    stream,clock=timeline(video.source)
    for row,pixels in native_stream(video.source,stream,clock):
        receipt=video.reader.receipts.get(row['pts'])
        if receipt:
            assert sha256(pixels).hexdigest()==observations_by_root[receipt['id']]
            verified.add(row['pts'])
    assert verified==set(video.reader.receipts)
    write(out/'evidence.json',{'source_sha256':video.sha,'source_receipts':list(video.reader.receipts.values()),
           'read_failures':video.failures,'views':dict(video.views), 'local_calculations':list(video.comparisons.values()),
           'context_calculations':list(video.background.values()),'independent_native_pts':sorted(verified),'fixed_anchor_stress':stress})
    write(out/'cost.json',{'seconds_shared_source_and_prepare':source_sec,'seconds_common_A0_measurement':base_seconds,
           'seconds_increment_m2':m2_seconds,'seconds_increment_m5':m5_seconds,'seconds_increment_bridge_incremental_after_m5':joint_seconds,
           'source_frames':len(verified),'decode_filter_inputs':sum(item['frames_at_input_filter'] or 0 for row in video.reader.receipts.values() for item in row['decode_attempts']),
           'delivered_native_png_bytes':sum(row['png_bytes'] for row in video.reader.receipts.values())})


def main():
    p=ArgumentParser();p.add_argument('--output', default='first_run');a=p.parse_args()
    output=HERE/a.output
    assert output.parent==HERE and not output.exists()
    freeze=read(HERE/'freeze.json')
    assert all(digest(ROOT/k)==v for k,v in freeze['paths'].items())
    cv2.setNumThreads(1);cv2.setRNGSeed(1128)
    output.mkdir()
    for case in ('paired','panning'):run_case(case,output)


if __name__=='__main__':main()
