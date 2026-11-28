"""Localized selection before identity. Exposure, geometry, relevance and claims stay distinct."""
from collections import defaultdict
from itertools import product
from math import hypot
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import ROOT,digest
from dacf_round3.acquire import Reader
from evidenweft_phase1.source_review import timeline
from partial_modules_stage1.study import prepare, correspond, fit, in_box, box_at, centroid, box_distance, root
from partial_modules_stage2.discovery import proposals,iou

K3=np.ones((3,3),np.uint8)
K5=np.ones((5,5),np.uint8)
ENV='environment';CHANGE='independent_change';NEIGHBOR='relation_participant';UNEXPLAINED='unexplained';GOAL='specified_goal'


def intersect(a,b):
    return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))


def expand(box,scale):
    x,y,w,h=box;dx=w*(scale-1)/2;dy=h*(scale-1)/2
    return [x-dx,y-dy,w+2*dx,h+2*dy]


def overlaps_inside(a,b):
    return intersect(a,b)/max(1,a[2]*a[3])


def box_scope(box, shape):
    h,w=shape[:2]
    x,y,ww,hh=box
    x0=max(0,int(x));y0=max(0,int(y))
    x1=min(w,int(x+ww));y1=min(h,int(y+hh))
    return x0,y0,max(0,x1-x0),max(0,y1-y0)


def contours(mask, least=100):
    shapes,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return sorted((list(cv2.boundingRect(c)) for c in shapes if least<=cv2.contourArea(c)<=.08*mask.size),
                  key=lambda b:(-b[2]*b[3],b[0],b[1]))


def hist(rgb,box):
    x,y,w,h=box
    hsv=cv2.cvtColor(rgb[y:y+h,x:x+w],cv2.COLOR_RGB2HSV)
    bins=cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]).flatten()
    return bins/max(1,bins.sum())


def colordiff(a,b):
    return float(np.abs(a-b).sum()/2)


def warp_box(box,m,shape):
    x,y,w,h=box; pts=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]],float)
    target=pts@m[:,:2].T+m[:,2]; lo=target.min(axis=0); hi=target.max(axis=0)
    return box_scope([*lo,*(hi-lo)],shape)


class Media:
    def __init__(self,spec,output):
        self.path=ROOT/spec['path']; self.sha=digest(self.path)
        assert self.sha==spec['sha256']
        stream,self.clock=timeline(self.path)
        self.ticks=spec['pts']; self.images={}; self.packets={}; self.feature_pairs={};self.chart_cache={}
        self.raw_reader=Reader(self.path,self.sha,stream,self.clock,output/'native')
        self.colors=defaultdict(dict)
        for t in self.ticks:
            f,_=self.raw_reader.read(t)
            with Image.open(f['path']) as im:self.images[t]=np.array(im.convert('RGB'))
            self.packets[t]=prepare(f['path'])
        self.gray={t:cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for t,im in self.images.items()}

    def source_root(self,t):
        return self.raw_reader.receipts[t]['id']

    def feature_pair(self,a,b):
        if (a,b) not in self.feature_pairs:
            p,q=self.packets[a],self.packets[b]
            pairs=correspond(p[3],q[3]);idx=[np.array([d[i] for d in pairs],dtype=int) for i in (0,1)]
            self.feature_pairs[a,b]=(p[2][idx[0]],q[2][idx[1]])
        return self.feature_pairs[a,b]

    def chart(self,a,b,left=(),right=()):
        key=(a,b,tuple(map(tuple,left)),tuple(map(tuple,right)))
        if key in self.chart_cache:return self.chart_cache[key]
        xy,yx=self.feature_pair(a,b); keep=np.ones(len(xy),bool)
        for view,t,boxes in ((xy,a,left),(yx,b,right)):
            for roi in boxes:keep&=~in_box(view,box_at(roi,self.packets[t]))
        raw,m=fit(xy,yx,keep)
        record={'ref':root('geometry',self.sha,key),'pts':[a,b],'source_roots':[self.source_root(a),self.source_root(b)],
                'excluded_rois':[left,right],'scope':'2D background support outside the stated regions',**raw}
        if m is not None:
            scale=np.array(self.packets[a][1])/np.array(self.packets[a][0].shape[::-1])
            native=np.c_[np.diag(scale)@m[:,:2]@np.diag(1/scale),m[:,2]*scale]
            record['native_affine']=native.tolist()
        self.chart_cache[key]=record;return record

    def feature(self,t,box):
        key=tuple(box)
        if key not in self.colors[t]: self.colors[t][key]=hist(self.images[t],box)
        return self.colors[t][key]

    def region(self,t,box,roles,why):
        return {'id':root('observation',self.sha,t,box),'pts':t,'native_xywh':box,'source_sha256':self.sha,
                'source_root':self.source_root(t),'roles':sorted(set(roles)),'basis':why}


def structures(media,t):
    rgb=media.images[t]
    sat=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1]
    chroma=cv2.morphologyEx((sat>=55).astype(np.uint8)*255,cv2.MORPH_CLOSE,K5)
    edge=cv2.morphologyEx(cv2.Canny(media.gray[t],45,110),cv2.MORPH_CLOSE,K5)
    rows=sorted(contours(chroma)+contours(edge),key=lambda box:(-box[2]*box[3],box[0],box[1]))
    unique=[]
    for box in rows:
        if not any(iou(box,b)>=.62 for b in unique):unique.append(box)
    return unique[:8], max(0,len(unique)-8)


def compensated(media,a,b,left,right,structure_boxes):
    geom=media.chart(a,b,left,right)
    if geom['state']!='heldout_supported':
        return [],{'pts':[a,b],'context':geom['ref'],'state':'unassessable_geometry','discarded':0}
    native=np.array(geom['native_affine']); h,w=media.images[b].shape[:2]
    previous=cv2.warpAffine(media.images[a].astype(np.float32),native,(w,h))
    coverage=cv2.warpAffine(np.ones((h,w),np.float32),native,(w,h))>=.999
    if not coverage.any():return [],{'pts':[a,b],'context':geom['ref'],'state':'unassessable_overlap','discarded':0}
    delta=media.images[b].astype(np.float32)-previous
    contextual=coverage.copy()
    for x,y,bw,bh in right:contextual[y:y+bh,x:x+bw]=False
    offset=np.median(delta[contextual] if contextual.any() else delta[coverage],axis=0)
    restored=np.clip(previous+offset,0,255)
    center=np.max(np.abs(media.images[b]-restored),axis=2)
    mask=((center>=26)&coverage).astype(np.uint8)*255
    mask=cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,K3),cv2.MORPH_CLOSE,K5)
    rows=contours(mask); regions=[]
    for bbox in rows[:8]:
        snapped=[box for box in structure_boxes
                 if overlaps_inside(bbox,box)>=.5 and bbox[2]*bbox[3]*.5<=box[2]*box[3]<=4*bbox[2]*bbox[3]]
        chosen=max(snapped,key=lambda box:iou(bbox,box),default=bbox)
        if any(iou(chosen,o['native_xywh'])>.62 for o in regions):continue
        x,y,bw,bh=chosen
        scope='contrast_connected_structure' if snapped else 'difference_region_boundary_unresolved'
        regions.append(media.region(b,chosen,[CHANGE],[{'kind':'compensated_local_difference','pts':[a,b],'calculation_root':geom['ref'],
                    'source_roots':geom['source_roots'],'explicit_roi':bbox,'roi_scope':scope,
                    'target_abs_rgb_mean':float(center[y:y+bh,x:x+bw].mean())}]))
    diag={'pts':[a,b],'state':'measured_overlap','geom':geom['ref'],'brightness_rgb_displacement':offset.tolist(),
          'valid_overlap_fraction':float(coverage.mean()),'residual_pixel_count':int(np.count_nonzero(mask)),
          'discarded':max(0,len(rows)-8),'median_registered_remaining_difference':float(np.median(center[coverage]))}
    return regions,diag


def referents(media,a,b,old,new,context):
    if context['state']!='heldout_supported':return []
    m=np.array(context['native_affine']); shape=media.images[b].shape
    transformed=[warp_box(v,m,shape) for v in old]
    results=[]
    for box in new:
        matches=[i for i,q in enumerate(transformed) if iou(q,box)>=.4 and colordiff(media.feature(a,old[i]),media.feature(b,box))<=.20]
        if len(matches)!=1:continue
        i=matches[0]
        results.append({'old':old[i],'current':box,'warped_old':transformed[i],'pts':[a,b],'reference_chart':context['ref'],'reference_source_roots':context['source_roots']})
    return results


def role_discovery(media,goal_queries):
    median_rows=list(proposals(np.stack([media.gray[t] for t in media.ticks])))
    labs={}; timing=defaultdict(float); diagnostics=[]; close_witnesses=[]
    for t,(rows,discarded) in zip(media.ticks,median_rows):
        start=perf_counter()
        stable,extras=structures(media,t)
        base=[media.region(t,list(box),[CHANGE],[{'kind':'legacy_temporal_median_difference'}]) for box in rows]
        labs[t]={'base':base,'base_boxes':[list(x) for x in rows],'struct':stable,'changes':[],'ref':[],'near':[],
                 'marked':[],'goal':[],'dropped':[discarded,extras]}
        timing['source_structure_seconds']+=perf_counter()-start
    for a,b in zip(media.ticks,media.ticks[1:]):
        s,l=labs[a],labs[b]
        start=perf_counter()
        l['changes'],diag=compensated(media,a,b,s['base_boxes'],l['base_boxes'],l['struct'])
        diagnostics.append(diag);timing['compensated_candidate_seconds']+=perf_counter()-start
        start=perf_counter()
        ctx=media.chart(a,b,s['base_boxes'],l['base_boxes'])
        static=referents(media,a,b,s['struct'],l['struct'],ctx)
        for rr in static:
            box=rr['current']
            if any(iou(box,c['native_xywh'])>.60 for c in l['changes']):continue
            env=media.region(b,box,[ENV],[{'kind':'registered_stable_structure',**rr}])
            l['ref'].append(env)
            for change in l['changes']:
                region=change['native_xywh']; maxrad=hypot(*region[2:]);gap=box_distance(region,box)
                if iou(region,box)>.12 or gap>maxrad:continue
                native=np.array(ctx['native_affine'])
                old_candidates=[v['native_xywh'] for v in s['changes']+s['base']
                         if torchless_match(media,a,b,v['native_xywh'],region,native,maxrad)]
                unique=[]
                for o in old_candidates:
                    if not any(iou(o,f)>.55 for f in unique):unique.append(o)
                if len(unique)!=1:continue
                prev_gap=box_distance(warp_box(unique[0],native,media.images[b].shape),box)
                if abs(gap-prev_gap)<5:continue
                member=media.region(b,box,[ENV,NEIGHBOR],[{'kind':'conditional_region_gap_change',
                        'current_moving_view':change['id'],'previous_roi':unique[0],
                        'delta_projected_native_gap':gap-prev_gap,'identity_assumption':'unconfirmed',**rr}])
                l['near'].append(member);close_witnesses.append(member)
                # Both measured endpoints participate, without promoting the static reference to an object track.
                s['near'].append(media.region(a,rr['old'],[ENV,NEIGHBOR],
                    [{'kind':'previous_static_endpoint','referent_successor':member['id'],**rr}]))
        timing['referent_and_roles_seconds']+=perf_counter()-start
    for t,l in labs.items():
        l['marked']=[r|{'roles':[UNEXPLAINED],'basis':r['basis']+[{'kind':'no_separate_registered_change_support'}]}
                     for r in l['base'] if not any(iou(r['native_xywh'],v['native_xywh'])>.45 for v in l['changes'])]
        for rr in goal_queries:
            if rr['pts']==t:l['goal'].append(media.region(t,rr['native_xywh'],[GOAL],[{'kind':'explicit_observation_target','intent':rr['intent']}]))
    return labs,diagnostics,close_witnesses,dict(timing)


def torchless_match(media,a,b,f,g,m,limit):
    charted=warp_box(f,m,media.images[b].shape)
    return np.linalg.norm(centroid(charted)-centroid(g))<=limit and colordiff(media.feature(a,f),media.feature(b,g))<=.25


def merge(records):
    found={}
    for _,rows in sorted(records.items()):
        for r in rows:
            if r['id'] in found:
                old=found[r['id']]
                found[r['id']]=r|{'roles':sorted(set(r['roles']+old['roles'])),'basis':old['basis']+r['basis']}
            else:found[r['id']]=r
    return list(found.values())


def arms_from(labs):
    arms={k:{} for k in ('A0','A1','A2','A3','A3-no-flow','A3-no-role','A3-asked')}
    for t,r in labs.items():
        for k,features in {'A0':['base'],'A1':['base','changes'],'A2':['base','changes','near'],
                           'A3':['changes','ref','near','marked'],'A3-no-flow':['changes','ref','near','marked'],
                           'A3-no-role':['base','changes','near'],'A3-asked':['changes','ref','near','marked','goal']}.items():
            arms[k][t]=merge({f:r[f] for f in features})
    return arms


def optical_point_support(media,a,b,l,r,matrix):
    x,y,w,h=l; margin_x=max(1,int(.1*w));margin_y=max(1,int(.1*h))
    mask=np.zeros(media.gray[a].shape,np.uint8);mask[y+margin_y:y+h-margin_y,x+margin_x:x+w-margin_x]=255
    p=cv2.goodFeaturesToTrack(media.gray[a],maxCorners=72,qualityLevel=.01,minDistance=3,mask=mask)
    if p is None or len(p)<8:return {'state':'too_few_texture_samples','point_count':0 if p is None else len(p)}
    q1,status,_=cv2.calcOpticalFlowPyrLK(media.gray[a],media.gray[b],p,None,winSize=(21,21),maxLevel=2)
    if q1 is None:return {'state':'flow_not_applicable'}
    q0,check,_=cv2.calcOpticalFlowPyrLK(media.gray[b],media.gray[a],q1,None,winSize=(21,21),maxLevel=2)
    if q0 is None:return {'state':'flow_back_not_applicable'}
    x0=p.reshape(-1,2); y0=q1.reshape(-1,2)
    good=(status.ravel()>0)&(check.ravel()>0)&(np.linalg.norm(x0-q0.reshape(-1,2),axis=1)<=1.5)
    good&=in_box(y0,np.array(r))
    used=np.where(good)[0];measure={'point_count':len(p),'valid_count':len(used),'forward_inverse_fraction':len(used)/len(x0)}
    if len(used)<8 or len(used)/len(x0)<.7:return measure|{'state':'point_agreement_insufficient'}
    src=x0[used]; dst=y0[used]
    quadrants={(bool(q[0]>=x+w/2),bool(q[1]>=y+h/2)) for q in src}
    if len(quadrants)<3:return measure|{'state':'insufficient_interior_spread'}
    displacements=dst-(src@matrix[:,:2].T+matrix[:,2])
    hold=np.arange(len(src))%4==0
    inferred=np.median(displacements[~hold],axis=0)
    error=float(np.quantile(np.linalg.norm(displacements[hold]-inferred,axis=1),.9))
    colour=colordiff(media.feature(a,l),media.feature(b,r))
    passed=np.linalg.norm(inferred)>3 and error<=max(3.,.08*min(w,h)) and colour<=.30
    return measure|{'state':'heldout_local_nonrigid_support' if passed else 'heldout_motion_or_color_inconclusive',
       'parallax_or_deformation_possible':True,'relative_motion_native_xy':inferred.tolist(),'heldout_p90_native':error,'HSV_tv':colour,
       'local_flow_points':np.c_[src,dst].tolist(),'heldout_index':np.where(hold)[0].tolist(),
       'scope':'Native interior points only; box bounds and physical identity require their own review.'}


def compute_pair(media,a,b,left,right,ctx,flow):
    native=np.array(ctx.get('native_affine',[[1,0,0],[0,1,0]]))
    l,r=left['native_xywh'],right['native_xywh']
    scope=float(np.linalg.norm(centroid(warp_box(l,native,media.images[b].shape))-centroid(r)))
    if scope>hypot(*l[2:]):return {'local':'outside_local_scope','flow':None,'passes':False}
    u,v=media.feature_pair(a,b)
    measured,_=fit(u,v,in_box(u,box_at(l,media.packets[a]))&in_box(v,box_at(r,media.packets[b])))
    old=measured['state']=='heldout_supported'
    extra=optical_point_support(media,a,b,l,r,native) if flow and measured['state']=='insufficient_direct_descriptors' and ctx['state']=='heldout_supported' else None
    return {'local':measured['state'],'sift_holdout':measured,'center_domain_residual_native_px':scope,'flow':extra,
            'passes':old or bool(extra and extra['state']=='heldout_local_nonrigid_support')}


def eligible(arm,records):
    if arm not in ('A3','A3-no-flow','A3-asked','A4'):
        return records
    # Stable referents and unresolved diagnostics remain searchable without repeated identity calculation.
    return [r for r in records if GOAL in r['roles'] or CHANGE in r['roles']]


def scoped_identity(media, inventory, arm):
    identifiers={};links=[];positions=[];reviews=[]; counts=defaultdict(int)
    for i,t in enumerate(media.ticks):
        current=eligible(arm,inventory[t])
        for v in current:identifiers[v['id']]=root('local-identity',media.sha,v['id'])
        if i==0:continue
        a=media.ticks[i-1]; prior=eligible(arm,inventory[a])
        ctx=media.chart(a,t,[v['native_xywh'] for v in prior],[v['native_xywh'] for v in current])
        comparing=[]
        flow=arm in ('A3','A3-no-role','A3-asked','A4')
        for u,v in product(prior,current):
            c=compute_pair(media,a,t,u,v,ctx,flow)
            counts[c['local']]+=1
            if c['flow']:counts['flow_tested']+=1
            rec=c|{'pts':[a,t],'views':[u['id'],v['id']],'source_roots':[u['source_root'],v['source_root']],
                   'geometry_calculation_root':ctx['ref'],'calculation_root':root('local-relation',media.sha,u['id'],v['id']),
                   'comparison_boxes':[u['native_xywh'],v['native_xywh']]}
            if c['local']!='outside_local_scope':reviews.append(rec)
            if c['passes']:comparing.append(rec)
        roots=defaultdict(int); ends=defaultdict(int)
        for q in comparing:roots[q['views'][0]]+=1;ends[q['views'][1]]+=1
        for e in comparing:
            u,v=e['views']
            if roots[u]==1 and ends[v]==1:
                identifiers[v]=identifiers[u];links.append(e|{'local_identity':identifiers[v],'physical_identity':'unknown'})
            else:counts['competing_abstained']+=1
        accepted={v['views'][1] for v in links if v['pts'][-1]==t}
        positions.extend({'pts':[a,t],'view':r['id'],'state':'unknown_local_continuation'} for r in current if r['id'] not in accepted)
    projections=[{'pts':t,'views':[u['id'],v['id']],'source_root':u['source_root'],'physical_instance_exclusion':'unknown_reflection_copy_or_distinct'}
                 for t,rows in inventory.items() for i,u in enumerate(eligible(arm,rows)) for v in eligible(arm,rows)[i+1:]
                 if box_distance(u['native_xywh'],v['native_xywh'])>0]
    return {'identifiers':identifiers,'links':links,'unresolved':positions,'attempts_by_reason':dict(counts),'pair_audit':reviews,
            'coappearance_projections':projections,'cross_range_default':'no assertion; optional review needs an explicit objective or independent observations'}
