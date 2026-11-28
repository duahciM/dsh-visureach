"""M4 development entry: use the original residual calculation with an explicit neighborhood."""

from collections import Counter
from math import hypot

import cv2
import numpy as np

from partial_modules_stage1.study import box_distance, centroid, root
from partial_modules_stage6.downstream import blocks, nomination


class Foreground:
    """Factor the existing IF16 foreground into a computation for one observed source PTS."""

    def __init__(self,frames):
        self.frames={r['pts']:r for r in frames}
        gray=frames[0]['gray']
        h,w=gray.shape
        self.size=(w,h)
        self.taper=cv2.createHanningWindow((w,h),cv2.CV_32F)
        self.colors={r['pts']:cv2.cvtColor(r['rgb'],cv2.COLOR_BGR2YCrCb).astype(np.float32)/255 for r in frames}
        self.cfg=nomination.RULES['local']
        self.calls=0

    def measure(self,target,neighbors):
        sample=self.frames[target]
        scope={'pts':target,'comparison_pts_requested':neighbors,'original':sample['id'],
            'rank_independent_root':root('m4-window-residual-v7',sample['id'],[self.frames[p]['id'] for p in neighbors])}
        if sample['sharpness']<8:
            return scope|{'state':'quality_abstention','fit_refs':[],'candidates':[]}
        warped=[]; verified=[]; w,h=self.size
        for ref in neighbors:
            self.calls+=1
            warp=nomination.register(self.frames[ref]['gray'],sample['gray'],self.taper,self.cfg['max_registration_px'],self.cfg)
            if warp is None:
                continue
            value=cv2.warpAffine(self.colors[ref],warp,(w,h),borderMode=cv2.BORDER_REPLICATE)
            valid=cv2.warpAffine(np.ones((h,w),np.float32),warp,(w,h))>.999
            warped.append((value,valid));verified.append({'pts':ref,'root':self.frames[ref]['id'],'matrix':warp.tolist()})
        if len(warped)<self.cfg['minimum_comparisons']:
            return scope|{'state':'registration_abstention','fit_refs':verified,'candidates':[]}
        valid=np.logical_and.reduce([mask for _,mask in warped])
        bg=np.median(np.stack([pixels for pixels,_ in warped]),axis=0)
        difference=np.max(np.abs(self.colors[target]-bg),axis=2);difference[~valid]=0
        candidates,state=nomination.regions(difference,self.cfg)
        stable=np.uint8(np.clip(bg[:,:,0]*255,0,255))
        edges=cv2.Canny(stable,48,124)>0
        for c in candidates:
            x,y,rw,rh=c['bbox']
            pad=self.cfg['context_edge_distance_px']
            ys,xs=np.nonzero(edges[max(0,y-pad):min(h,y+rh+pad),max(0,x-pad):min(w,x+rw+pad)])
            xx=xs+max(0,x-pad);yy=ys+max(0,y-pad)
            # Maintain the old inclusive separation at the outer right/lower side.
            external=(xx<x-3)|(xx>=x+rw+3)|(yy<y-3)|(yy>=y+rh+3)
            if external.any():
                cx,cy=c['center']
                ds=np.hypot(xx[external]-cx,yy[external]-cy)
                nearest=int(np.argmin(ds))
                if ds[nearest]<=pad:
                    c.update(context_edge_xy=[int(xx[external][nearest]),int(yy[external][nearest])],
                             context_image_distance=float(ds[nearest]))
        return scope|{'state':state,'fit_refs':verified,'candidates':candidates,
            'measured_valid_overlap_pixels':int(valid.sum())}


def legal_refs(series,i,choice):
    radius=nomination.RULES['local']['neighbor_offset']
    base=[series[j] for j in range(max(0,i-radius),min(len(series),i+radius+1)) if i!=j]
    k=len(base)
    if choice=='old' or k==0:
        return [base]
    if choice in ('forward','backward'):
        move=2 if choice=='forward' else -2
        start=min(max(0,i-radius+move),len(series)-k-1)
        return [[p for p in series[start:start+k+1] if p!=series[i]] if start<=i<start+k+1 else
                choose_valid(series,i,k,move)]
    if choice=='wide':
        wide=[p for p in series[max(0,i-2*radius):i+2*radius+1] if p!=series[i]]
        return [[wide[j] for j in np.linspace(0,len(wide)-1,k).round().astype(int)]]
    if choice=='split' and k>=6:
        ordered=choose_valid(series,i,k,2)
        return [sorted(ordered[::2]),sorted(ordered[1::2])]
    return []


def choose_valid(series,i,k,move):
    indices=[j for j in range(len(series)) if j!=i]
    chosen=sorted(indices,key=lambda j:(abs(j-(i+move)),abs(j-i),j))[:k]
    return [series[j] for j in sorted(chosen)]


def stable_lookup(geometry):
    by_time={}
    for row in geometry:
        if not row['supported']:
            continue
        for side,ts,other in zip(('left','right'),row['pts'],reversed(row['pts'])):
            cells=set(row[f'pooled_{side}_cells'])|set(row['fixed_local'][side]['supported_cells'])
            by_time.setdefault(ts,{})[other]={'cells':cells,'fit':row['id'],'sources':row['source_roots']}
    return by_time


def measure_reasons(person, result, with_structure=False, geometry=None):
    proof=result['rank_independent_root']
    ts=result['pts']
    output=[]
    for r in result['candidates']:
        output.append({'dimension_id':'D1','role':'local_appearance_residual','pts':ts,
             'applicable_region':r['bbox'],'calculation_id':proof,'source_root':result['original']})
        if 'context_edge_xy' in r:
            output.append({'dimension_id':'D3','role':'component_near_repeatable_edge','pts':ts,
                'applicable_region':r['bbox'],'calculation_id':proof,'source_root':result['original'],
                'projected_reference_point':r['context_edge_xy'],'analysis_distance_px':r['context_image_distance']})
    if not with_structure:
        return output
    by_neighbor=(geometry or {}).get(ts,{})
    direct=[by_neighbor[p] for p in result['comparison_pts_requested'] if p in by_neighbor]
    counts=Counter(c for q in direct for c in q['cells'] if c.startswith('3x2:'))
    w,h=person.size
    for cell in sorted(c for c,n in counts.items() if n>=2):
        i,j=map(int,cell.split(':')[1].split(','))
        bbox=[i*w//3,j*h//2,((i+1)*w//3)-(i*w//3),((j+1)*h//2)-(j*h//2)]
        refs=sorted({q['fit'] for q in direct if cell in q['cells']})
        region={'dimension_id':'structure','role':'local_context_reference','pts':ts,'applicable_region':bbox,
                'calculation_id':root('local-structure-v7',proof,cell,refs),'independent_geometry_roots':refs,
                'source_root':result['original']}
        output.append(region)
        # Image-space relation from a residual footprint to independently measured structural support.
        for r in result['candidates']:
            if box_distance(r['bbox'],bbox)<=1.0 or box_distance(r['bbox'],bbox)>hypot(*r['bbox'][2:]):
                continue
            output.append({'dimension_id':'relation','role':'residual_to_context_projection','pts':ts,
              'analysis_delta_xy':(centroid(r['bbox'])-centroid(bbox)).tolist(),
              'applicable_region':r['bbox'],'reference_analysis_roi':bbox,'source_root':result['original'],
              'calculation_id':root('projection-offset-v7',proof,region['calculation_id'],r['bbox']),
              'requires':[proof,region['calculation_id']],'scope':'same-frame projection; no interpretation of physical instance or contact'})
    if result['state'] in ('registration_abstention','quality_abstention'):
        output.append({'dimension_id':'unknown','role':result['state'],'pts':ts,'applicable_region':None,
                       'source_root':result['original'],'calculation_id':proof})
    return output


def adjacent_reasons(worker,frames,scope):
    output=[]; pairs=0
    for _,items in blocks(frames,scope):
        stamps=[r['id'] for r in items]
        differences={}
        for dimension in ('D2','D4','D5'):
            differences[dimension]=[]
        if len(items)<2:
            continue
        charts=nomination.charts(items)
        pairs+=len(charts)
        nomination.pair_observations(differences,items,stamps,{},charts,worker.cfg)
        for dimension,proposals in differences.items():
            for r in proposals:
                sample=items[r['sample']]
                output.append(r|{'dimension_id':dimension,'pts':sample['pts'],'source_root':sample['id'],
                    'calculation_id':root('independent-pair-v7',r['comparison_frame_ids'],r['role'],r.get('applicable_region'))})
    return output,pairs


def arm(worker,frames,scope,choice,with_structure=False,geometry=None):
    receipts=[];nominations=[]; schedule=Counter();prev=worker.calls
    for sid,rows in blocks(frames,scope):
        stamps=[r['pts'] for r in rows]
        for i,pts in enumerate(stamps):
            wanted=legal_refs(stamps,i,choice)
            for refs in wanted:
                assert pts not in refs and len(set(refs))==len(refs)
                proof=worker.measure(pts,refs)|{'context_id':sid}
                receipts.append(proof)
                schedule[pts]+=len(refs)
                nominations.extend(measure_reasons(worker,proof,with_structure,geometry))
    return {'measurements':receipts,'nominations':nominations,'independent_schedule_per_pts':dict(schedule),
            'executed_phase_calls':worker.calls-prev,'scope_state':'contains_scanned_addresses' if receipts else 'abstained_no_closed_scope'}
