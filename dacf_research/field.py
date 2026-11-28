"""Channel-preserving dual-anchor and actual sampled-path diagnostics."""
from collections import Counter
from itertools import combinations, product
from math import exp, sqrt
from statistics import median
import sys

import cv2
import numpy as np
from PIL import Image

from dacf_research.freeze import PLUGIN
from dacf_research.prepare_inputs import ROOT, read

sys.path[:0] = [str(PLUGIN/'vendor/P2_24'), str(PLUGIN/'python')]
from p1.core import Parameters, pair_metrics, quality
from p1.photometric import brightness_metrics
from p14_scene_prototypes.features import histogram, sift_pair
from if16_research.nominate import regions
from if16_research.common import RULES
from local_observation import correspond, fit
from p223_runtime import node_bundle, figure2, figure1


P14 = read(PLUGIN/'vendor/P2_24/p14_scene_prototypes/protocol.json')
METRICS = ('gray','hsv','edge','clip')


def clip_encoder():
    sys.path.append(str(ROOT/'if13_research_20261120/runtime_onnx'))
    import onnxruntime as ort
    from p215_research.semantic import tensor, unit
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/'p215_discovery_20261121/assets/vision.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    names={x.name for x in session.get_outputs()}
    def encode(images):
        data=np.stack([tensor(Image.fromarray(img),(0,0,img.shape[1],img.shape[0])) for img in images])
        predicted=session.run(['image_embeds'] if 'image_embeds' in names else None, {'pixel_values':data})[0]
        assert predicted.ndim==2
        return unit(predicted)
    return encode


class Bank:
    def __init__(self, raw, config, clip):
        self.rows=raw
        self.matrix={}
        self.evidence={}
        self.links={}
        self.config=config
        self.params=Parameters()
        sift=cv2.SIFT_create(nfeatures=P14['sift']['nfeatures'])
        for row in self.rows.values():
            gray=cv2.cvtColor(row['rgb'],cv2.COLOR_RGB2GRAY)
            row.update(gray=gray,gray_f=gray.astype(np.float32)/255,edge=cv2.Canny(gray,48,124)>0,
                       hist=histogram(row['rgb'],P14['histogram']),descriptor=sift.detectAndCompute(gray,None),**quality(gray))
            row['xy']=np.array([k.pt for k in row['descriptor'][0]],dtype=np.float32).reshape(-1,2)
        self.clip_state={'status':'unavailable'}
        if clip is not None:
            try:
                keys=list(self.rows)
                for start in range(0,len(keys),8):
                    part=keys[start:start+8]
                    for key,vec in zip(part,clip([self.rows[k]['rgb'] for k in part]),strict=True):
                        self.rows[key]['clip']=vec
                self.clip_state={'status':'computed','vectors':len(keys)}
            except (RuntimeError,ValueError,AssertionError) as exc:
                self.clip_state={'status':'failed','reason':str(exc)}

    def metric(self, x, y):
        key=tuple(sorted((x,y)))
        if key not in self.matrix:
            a,b=(self.rows[q] for q in key)
            p,q=a.get('clip'),b.get('clip')
            exact=x==y
            self.matrix[key]={
                'gray':float(np.mean(np.abs(a['gray_f']-b['gray_f']))),
                'hsv':0. if exact else sqrt(max(0.,1.-min(1.,float(np.sum(np.sqrt(a['hist']*b['hist'])))))),
                'edge':float(np.mean(a['edge']!=b['edge'])),
                'clip':float(np.linalg.norm(p-q)/2) if p is not None and q is not None else None}
        return self.matrix[key]

    def metrics_on_pair(self,x,y):
        if x==y:
            raise ValueError('Exclude exact self-reference from pair evidence.')
        key=(x,y)
        if key in self.evidence:
            return self.evidence[key]
        a,b=self.rows[x],self.rows[y]
        root=f'{a["id"]}>{b["id"]}'
        motion=pair_metrics(a['gray'],b['gray'],self.params)
        photometry=brightness_metrics(a['gray'],b['gray'],motion,self.params)
        geom=sift_pair(a['descriptor'],b['descriptor'],a['gray'].shape,b['gray'].shape,P14['sift'],root)
        regional={'status':'unassessable_registration'}
        if motion['registration_valid']:
            h,w=b['gray'].shape
            warp=np.float32([[1,0,motion['dx']],[0,1,motion['dy']]])
            mask=cv2.warpAffine(np.ones((h,w),np.float32),warp,(w,h))>.999
            warped=cv2.warpAffine(a['gray_f'],warp,(w,h))
            if photometry['supported']:
                warped=np.clip(warped * photometry['gain']+photometry['offset'],0,1)
            diff=np.abs(warped-b['gray_f'])
            diff[~mask]=0
            found,reason=regions(diff,RULES['local'])
            regional={'status':'measured_overlap','region_frame':b['id'],'reason':reason,
                      'overlap_pixels':int(mask.sum()),'mad':float(diff[mask].mean()),
                      'gain_offset_applied':photometry['supported'],'region_candidates':found}
        self.evidence[key]={'root':root,'frames':[a['id'],b['id']],'pts':[a['pts'],b['pts']],
            'metrics':self.metric(x,y),'sift':geom,'phase':motion,'photometry':photometry,'aligned':regional}
        return self.evidence[key]

    def measure_local(self, target, prototype):
        key=(target,prototype)
        if key in self.links:
            return self.links[key]
        source,ref=self.rows[target],self.rows[prototype]
        forward=correspond(source['descriptor'][1], ref['descriptor'][1])
        qi=np.array([i for i,j in forward],int)
        ri=np.array([j for i,j in forward],int)
        a=source['xy'][qi]; b=ref['xy'][ri]
        h,w=source['gray'].shape
        cells=[]
        for columns,lines in self.config['point_registration']['grids']:
            for iy,ix in product(range(lines),range(columns)):
                x0,x1=ix*w//columns,(ix+1)*w//columns
                y0,y1=iy*h//lines,(iy+1)*h//lines
                take=(x0<=a[:,0])&(a[:,0]<x1)&(y0<=a[:,1])&(a[:,1]<y1)
                measured, _ = fit(a,b,take)
                cells.append({'id':f'{columns}x{lines}:{ix},{iy}','grid':f'{columns}x{lines}',
                              'analysis_xywh':[x0,y0,x1-x0,y1-y0],**measured})
        result={'root':f'{source["id"]}>{ref["id"]}:local_heldout','pts':[source['pts'],ref['pts']],
                'scope':'source-image cells toward a single prototype, correspondence-point holdout','mutual_count':len(forward),'cells':cells}
        self.links[key]=result
        return result

    def figure(self, rank, catalogue):
        item=self.rows[rank]
        arch=catalogue.get(item['pts'])
        if arch is None:
            return {'status':'unassessed_no_catalogue_nominations','source_root':item['id']}
        class Book:
            def image(self,key):
                assert key==arch['frame_id']
                return item['rgb'],item['gray']
        bundle=node_bundle(Book(),{'candidate_id':arch['frame_id']},arch)
        graph=figure2(bundle)
        return {'status':'computed_frozen_nominations','source_root':item['id'],'figure1':figure1(bundle,graph),'figure2':graph}


def detect_groups(bank, coarse):
    cfg=bank.config['automatic']
    groups=[]; rejected=[]
    def save(window):
        if not window:
            return
        dwell=bank.rows[window[-1]]['seconds']-bank.rows[window[0]]['seconds']
        if len(window)<3 or dwell<cfg['dwell_seconds']:
            rejected.append({'ranks':window,'reason':'insufficient_stable_dwell','duration_s':dwell})
            return
        protos=[window[0],window[len(window)//2],window[-1]]
        values=[bank.metric(x,y)['gray'] for x,y in combinations(protos,2)]
        if max(values)>cfg['within_gray']:
            rejected.append({'ranks':window,'reason':'prototype_pair_gray_disagreement','distances':values})
            return
        groups.append({'id':f'G{len(groups)+1}','ranks':window,'prototypes':protos,'duration_s':dwell,
            'within_gray_distances':values,'pts':[bank.rows[q]['pts'] for q in protos]})
    window=[]
    for rank in coarse:
        row=bank.rows[rank]
        if row['sharpness']<cfg['minimum_sharpness']:
            save(window); window=[]
            rejected.append({'ranks':[rank],'reason':'quality'})
            continue
        if window and (row['seconds']-bank.rows[window[-1]]['seconds']>bank.config['maximum_gap_seconds']
                       or any(bank.metric(rank,q)['gray']>cfg['within_gray'] for q in {window[0],window[len(window)//2],window[-1]})):
            save(window); window=[]
        window.append(rank)
    save(window)
    relations=[]; pairs=[]
    for i,j in combinations(range(len(groups)),2):
        a,b=groups[i],groups[j]
        measures=[bank.metric(x,y) for x,y in product(a['prototypes'],b['prototypes'])]
        d={ch:median(v[ch] for v in measures if v[ch] is not None) if any(v[ch] is not None for v in measures) else None for ch in METRICS}
        adjacent=j==i+1
        adjacent_gap=[q for q in coarse if a['prototypes'][-1]<q<b['prototypes'][0]]
        adequate=d['gray']>=cfg['between_gray']
        link={'a':a['id'],'b':b['id'],'adjacent_stable_group_pair':adjacent,'distance_medians':d,
              'view_compatible_gray':d['gray']<=cfg['within_gray'],'distinct_gray_observation':adequate,
              'cross_time_physical_identity':'unknown','interior_coarse_count':len(adjacent_gap)}
        relations.append(link)
        if adjacent and adequate:
            pair_id=f'{a["id"]}_{b["id"]}'
            pairs.append({'id':pair_id,'a':a['prototypes'],'b':b['prototypes'],
                          'scope':'automatic image-state separation; physical continuity unknown'})
    return {'groups':groups,'rejected':rejected,'group_relations':relations,'pairs':pairs}


def direct_group(bank, rank, group):
    pairs=[bank.metrics_on_pair(rank,k) for k in group if k!=rank]
    caps=bank.config['support_caps']
    values={}
    for ch in METRICS:
        distances=[v['metrics'][ch] for v in pairs if v['metrics'][ch] is not None]
        d=median(distances) if len(distances)>=2 else None
        values[ch]={'distance':d,'valid_prototypes':len(distances),'support':
                    None if d is None or caps[ch] is None else d<=caps[ch]}
    confirmed=[v['sift']['distance'] for v in pairs if v['sift']['verified']]
    values['sift']={'distance':None,'verified_prototypes':len(confirmed),
                    'support':median(confirmed)<=caps['sift'] if len(confirmed)>=2 else None}
    values['aligned']={'distance':None,'applicable_prototypes':sum(v['aligned']['status']=='measured_overlap' for v in pairs)}
    return {'channels':values,'evidence_roots':[r['root'] for r in pairs]}


def local_presence(bank, rank, a, b):
    a_refs=[bank.measure_local(rank,p) for p in a if p!=rank]
    b_refs=[bank.measure_local(rank,p) for p in b if p!=rank]
    aa=Counter(x['id'] for r in a_refs for x in r['cells'] if x['state']=='heldout_supported')
    bb=Counter(x['id'] for r in b_refs for x in r['cells'] if x['state']=='heldout_supported')
    minimum=bank.config['point_registration']['minimum_local_prototypes']
    cells={c['id']:c for r in a_refs+b_refs for c in r['cells']}
    apos=[k for k in cells if aa[k]>=minimum and bb[k]<minimum]
    bpos=[k for k in cells if bb[k]>=minimum and aa[k]<minimum]
    coexist=[{'a':k,'b':l} for k,l in product(apos,bpos) if cells[k]['grid']==cells[l]['grid']]
    return {'a_support':dict(aa),'b_support':dict(bb),'different_cells_with_one_group_supported':coexist,
            'status':'two_disjoint_image_support_regions' if coexist else 'not_supported_on_current_cells',
            'cell_boxes_analysis':{k:c['analysis_xywh'] for k,c in cells.items()},
            'roots':[r['root'] for r in a_refs+b_refs],'claim_boundary':'A lack of support for the other group does not establish it is absent; no physical link inferred.'}


def crossing(points, eps, cap, max_gap, separable):
    if not separable:
        return {'state':'anchor_separation_zero_or_unavailable','events':[],'applicable_points':sum(p['balance'] is not None for p in points)}
    runs=[]; current=[]
    for p in points:
        if p['balance'] is None or (current and p['t']-current[-1]['t']>max_gap):
            if current:
                runs.append(current)
            current=[]
        if p['balance'] is not None:
            current.append(p)
    if current:
        runs.append(current)
    events=[]
    def sign(info):
        return 0 if abs(info['balance']-.5)<=eps else (1 if info['balance']>.5 else -1)
    def add(section, kind, direction=None):
        through=[d for p in section for d in (p['da'],p['db'])]
        support='uncalibrated' if cap is None else 'both_anchors_high' if max(through)<=cap else 'both_anchors_low' if min(through)>cap else 'mixed_support'
        events.append({'kind':kind,'from_pts':section[0]['pts'],'to_pts':section[-1]['pts'],
                       'from_seconds':section[0]['t'],'to_seconds':section[-1]['t'],'support':support,
                       'a_b_travel_direction':direction,'raw_distances':[[p['pts'],p['da'],p['db']] for p in section]})
    for run in runs:
        prev=None; equal=[]
        for p in run:
            s=sign(p)
            if s==0:
                equal.append(p)
            elif equal:
                reversal=prev is not None and sign(prev)!=s
                add(equal,'crossing_through_sampled_equality' if reversal else 'sampled_equality_only',f'{sign(prev)}_to_{s}' if reversal else None)
                equal=[]
                prev=p
            else:
                if prev and sign(prev)!=s:
                    add([prev,p],'sign_change_between_samples',f'{sign(prev)}_to_{s}')
                prev=p
        if equal:
            add(equal,'sampled_equality_only')
    real=sum(e['kind']!='sampled_equality_only' for e in events)
    state='unreliable_no_comparable_bracket' if not runs or max(map(len,runs))<2 else 'multiple_crossings' if real>1 else 'one_crossing' if real==1 else 'equality_without_observed_side_change' if events else 'no_crossing'
    return {'state':state,'events':events,'applicable_points':sum(map(len,runs))}


def path_curve(bank, ranks, channel):
    eps=bank.config['numerical_epsilon']; maxgap=bank.config['maximum_gap_seconds']
    t=[bank.rows[q]['seconds'] for q in ranks]
    pts=[bank.rows[q]['pts'] for q in ranks]
    values=[]; complete=True
    for i,(a,b) in enumerate(zip(ranks,ranks[1:])):
        duration=t[i+1]-t[i]
        d=bank.metric(a,b)[channel]
        valid=duration>0 and duration<=maxgap and d is not None
        complete &= valid
        values.append({'from_pts':pts[i],'to_pts':pts[i+1],'width':duration,'time':(t[i]+t[i+1])/2,
                       'distance':d,'applicable':valid,
                       'reason':None if valid else 'missing_channel_or_unobserved_time_gap',
                       'root':f'{bank.rows[a]["id"]}>{bank.rows[b]["id"]}'})
    total=sum(e['distance'] for e in values) if complete else None
    direct=bank.metric(ranks[0],ranks[-1])[channel]
    if total is not None and direct is not None:
        assert total+1e-6 >= direct, (channel,total,direct)
    state='path_incomplete' if total is None else 'zero_change' if total<=eps else 'computed'
    cum=[]
    for i,p in enumerate(pts):
        aa=values[:i];bb=values[i:]
        ca=sum(e['distance'] for e in aa) if all(e['applicable'] for e in aa) else None
        cb=sum(e['distance'] for e in bb) if all(e['applicable'] for e in bb) else None
        cum.append({'pts':p,'C_A':ca,'C_B':cb,'P_path':ca/total if state=='computed' else None})
    smooth={}
    for sigma in bank.config['smooth_seconds']:
        window=[]
        for p,time in zip(pts,t,strict=True):
            nearby=[e for e in values if abs(e['time']-time)<=3*sigma]
            k=[exp(-.5*((e['time']-time)/sigma)**2) for e in nearby]
            ws=sum(w*e['width'] for w,e in zip(k,nearby))
            mass=sum(w*e['width'] for w,e in zip(k,nearby) if e['applicable'])
            coverage=mass/ws if ws else 0.
            local=sum(e['distance'] for e in nearby) if nearby and all(e['applicable'] for e in nearby) else None
            val=sum(w*e['distance'] for w,e in zip(k,nearby) if e['applicable'])/mass if mass and coverage>=bank.config['smooth_minimum_mass'] else None
            window.append({'pts':p,'weighted_distance_rate_per_second':val,'valid_mass_fraction':coverage,
                           'local_path_distance_within_3_sigma':local,'adjacent_roots':[e['root'] for e in nearby]})
        smooth[str(sigma)]=window
    return {'state':state,'adjacent':values,'cumulative':cum,'total':total,'endpoint_direct':direct,
            'path_to_endpoint_direct_ratio':total/direct if total is not None and direct is not None and direct>eps else None,
            'smoothed':smooth}
