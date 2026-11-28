"""Independent image-region evidence in the unchanged native pixel coordinate system."""

from collections import defaultdict
from itertools import product
from math import hypot

import cv2
import numpy as np

from partial_modules_stage1.study import box_distance, centroid, root
from partial_modules_stage3.observation import (
    CHANGE, ENV, NEIGHBOR, UNEXPLAINED, GOAL, box_scope, iou, merge, overlaps_inside,
    referents, torchless_match, warp_box,
)


def edge_inputs(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,45,110)
    color = np.maximum.reduce([cv2.Canny(plane,45,110) for plane in cv2.split(rgb)])
    return edges, color


def boundary_regions(media, stamp, color_edges):
    candidates = []
    for kernel in (3,7):
        closed = cv2.morphologyEx(color_edges,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
        contours,_ = cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            box = list(cv2.boundingRect(contour))
            area = box[2]*box[3]
            occupancy = cv2.contourArea(contour)/area
            if min(box[2:])<12 or area>.12*closed.size or occupancy<.57:
                continue
            poly = cv2.approxPolyDP(contour,.015*cv2.arcLength(contour,True),True).reshape(-1,2)
            if len(poly)<4 or len(poly)>12:
                continue
            candidates.append(media.region(stamp,box,[],[{'kind':'closed_edge_candidate','closure_kernel_px':kernel,
                'contour_fill_fraction':occupancy,'support_polygon_native':poly.tolist(),
                'boundary_scope':'image contour candidate; complete object extent unverified'}]))
    candidates.sort(key=lambda q:(-q['basis'][0]['contour_fill_fraction'],q['native_xywh']))
    return list({r['id']:r for r in reversed(candidates[:96])}.values())


def native_points(media,a,b):
    old, new = media.feature_pair(a,b)
    def scale(t):
        prep = media.packets[t]
        return np.array(prep[1])/np.array(prep[0].shape[::-1])
    return old*scale(a),new*scale(b)


def points_box(points,shape):
    lo = np.floor(points.min(axis=0)); hi = np.ceil(points.max(axis=0))+1
    return list(box_scope([*lo,*(hi-lo)],shape))


def stable_points(media,a,b,chart):
    old,new = native_points(media,a,b)
    matrix = np.array(chart['native_affine'])
    projected = old@matrix[:,:2].T+matrix[:,2]
    compatible = np.linalg.norm(new-projected,axis=1)<=6
    held_points = np.array(chart['held_points'],dtype=float)
    if held_points.size==0:
        return []
    prep_a,prep_b=media.packets[a],media.packets[b]
    scales = np.r_[np.array(prep_a[1])/np.array(prep_a[0].shape[::-1]), np.array(prep_b[1])/np.array(prep_b[0].shape[::-1])]
    used_in_fit = {tuple(np.round(row*scales,6)) for row in np.asarray(chart['fit_points'],dtype=float)}
    held = {tuple(np.round(row*scales,6)) for row in held_points}-used_in_fit
    examined = np.c_[old,new]
    hold_index = np.array([tuple(np.round(row,6)) in held for row in examined],dtype=bool)
    step = max(16,round(min(media.images[b].shape[:2])*.07))
    cells = defaultdict(list)
    for i,(x,y) in enumerate(new):
        if compatible[i]:
            for u,v in product((0,step//2),repeat=2):
                cells[(u,v,int((x+u)//step),int((y+v)//step))].append(i)
    rows = []
    for group in dict.fromkeys(tuple(i) for i in cells.values()):
        ids = np.array(group); hold = ids[hold_index[ids]]; train = ids[~hold_index[ids]]
        if len(ids)<4 or len(hold)==0 or len(train)<3 or len(np.unique(new[ids].round(1),axis=0))<4:
            continue
        if float(np.linalg.norm(np.ptp(new[ids],axis=0)))<8:
            continue
        pred_error = float(np.quantile(np.linalg.norm(new[hold]-projected[hold],axis=1),.9))
        if pred_error>4:
            continue
        rows.append({'kind':'stable_feature_patch','pts':[a,b], 'source_roots':chart['source_roots'],
            'chart_ref':chart['ref'],'calculation_root':root('stable-patch',chart['ref'],group),
            'rois':[points_box(old[ids],media.images[a].shape),points_box(new[ids],media.images[b].shape)],
            'native_points_left':old[ids].tolist(),'native_points_right':new[ids].tolist(),
            'holdout_raw_match_indices':hold.tolist(),'fit_compatible_raw_match_indices':train.tolist(),
            'p90_held_error_native_px':pred_error,'scope':'local image registration; extent and object identity unknown'})
    return rows


def lines(edges):
    found = cv2.HoughLinesP(edges,1,np.pi/180,35,minLineLength=24,maxLineGap=7)
    if found is None:
        return []
    ordered = sorted((v.reshape(2,2) for v in found),key=lambda v:(-np.linalg.norm(v[1]-v[0]),v.flatten().tolist()))
    return ordered[:128]


def stable_lines(media,a,b,chart,line_inventory):
    matrix = np.array(chart['native_affine'])
    bounds = media.images[b].shape
    proposals = []
    new = np.asarray(line_inventory[b],dtype=float).reshape(-1,2,2)
    if len(new)==0:
        return proposals
    targets = new[:,1]-new[:,0]
    lengths = np.linalg.norm(targets,axis=1)
    directions = targets/lengths[:,None]
    for actual in line_inventory[a]:
        projected = actual@matrix[:,:2].T+matrix[:,2]
        vec = projected[1]-projected[0]
        if np.linalg.norm(vec)<15 or min(projected[:,0])<0 or min(projected[:,1])<0:
            continue
        if max(projected[:,0])>=bounds[1] or max(projected[:,1])>=bounds[0]:
            continue
        relative = projected[None,:,:]-new[:,0,None,:]
        along = np.einsum('ijk,ik->ij',relative,directions)
        perpendicular = np.abs(relative[:,:,0]*directions[:,1,None]-relative[:,:,1]*directions[:,0,None])
        error = perpendicular.max(axis=1)
        inside = (along.min(axis=1)>=-10)&(along.max(axis=1)<=lengths+10)
        matches = np.flatnonzero((error<=6)&inside)
        if len(matches)!=1:
            continue
        index=int(matches[0]); segment=new[index]
        proposals.append({'kind':'stable_edge','pts':[a,b],'source_roots':chart['source_roots'], 'chart_ref':chart['ref'],
            'calculation_root':root('stable-edge',chart['ref'], actual.tolist(),segment.tolist()),
            'segments_native':[actual.tolist(),segment.tolist()], 'projected_left_native':projected.tolist(),
            'max_orthogonal_error_native_px':float(error[index]),
            'rois':[points_box(actual,media.images[a].shape),points_box(segment,bounds)],
            'scope':'line support only; normal direction validated, endpoints do not certify extent'})
    return proposals


def discover_extra(media, legacy):
    atlas={}; calibration=[]; links=[]; patches=[]; edge_links=[]; positions=[]; appearances=[]
    for t in media.ticks:
        edges,color=edge_inputs(media.images[t]); boundaries=boundary_regions(media,t,color)
        atlas[t]={'boundaries':boundaries,'lines':lines(edges),'roles':[],
                  'raw_edge_pixels':int(np.count_nonzero(edges)),'color_edge_pixels':int(np.count_nonzero(color))}
    for a,b in zip(media.ticks,media.ticks[1:]):
        origin,dest=legacy[a],legacy[b]
        chart=media.chart(a,b,origin['base_boxes'],dest['base_boxes'])
        if chart['state']!='heldout_supported':
            calibration.append({'pts':[a,b],'state':'incomparable_environment','chart_ref':chart['ref']})
            continue
        record={'pts':[a,b],'chart_ref':chart['ref'],'state':'chart_supported'}
        point_support=stable_points(media,a,b,chart)
        line_support=stable_lines(media,a,b,chart,{a:atlas[a]['lines'],b:atlas[b]['lines']})
        patches.extend(point_support);edge_links.extend(line_support)
        for region in dest['changes']:
            evidence=region['basis'][0]
            appearances.append({'pts':[a,b],'source_roots':chart['source_roots'],'roi':evidence['explicit_roi'],
                'mean_registered_max_channel_absolute_difference_8bit':evidence['target_abs_rgb_mean'],
                'calculation_root':root('registered-residual',chart['ref'],evidence['explicit_roi']),
                'chart_ref':chart['ref'],'scope':'photometric residual can include motion or misregistration'})
        for target in merge({'base':dest['base'],'compensated':dest['changes']}):
            overlap=target['native_xywh']; radius=hypot(*overlap[2:]); matrix=np.array(chart['native_affine'])
            options=prior_views(media,a,b,origin,overlap,matrix,radius)
            record_pos={'pts':[a,b],'source_roots':chart['source_roots'],'right_roi':overlap,
                        'chart_ref':chart['ref'],'prior_candidates':options,
                        'calculation_root':root('conditional-layout-offset',chart['ref'],target['id'])}
            if len(options)==1:
                displacement=centroid(overlap)-centroid(warp_box(options[0],matrix,media.images[b].shape))
                record_pos.update({'delta_xy_native':displacement.tolist(),'state':'conditional_projection_difference',
                                   'object_continuity':'unobserved'})
            else:
                record_pos['state']='ambiguous_prior' if options else 'prior_not_supported'
            positions.append(record_pos)
        left=[r['native_xywh'] for r in atlas[a]['boundaries']];right=[r['native_xywh'] for r in atlas[b]['boundaries']]
        for comparison in referents(media,a,b,left,right,chart):
            ref=boundary_relation(media,a,b,origin,dest,chart,comparison)
            for t,box in zip((a,b),(comparison['old'],comparison['current'])):
                atlas[t]['roles'].append(media.region(t,box,[ENV],[{'kind':'registered_edge_region',**comparison}]))
            if ref:
                links.extend(ref)
                for endpoint in ref:
                    atlas[endpoint['pts']]['roles'].append(endpoint)
        # Local change and complete boundaries are separate. Promote a boundary candidate only as a candidate.
        for region in dest['changes']:
            raw=region['basis'][0]['explicit_roi']
            fits=[q for q in atlas[b]['boundaries'] if overlaps_inside(raw,q['native_xywh'])>=.65
                  and q['native_xywh'][2]*q['native_xywh'][3]<=4*raw[2]*raw[3]]
            if fits:
                candidate=max(fits,key=lambda q:iou(raw,q['native_xywh']))
                atlas[b]['roles'].append(candidate|{'roles':[CHANGE],'basis':candidate['basis']+[region['basis'][0]]})
        record.update({'feature_patches':len(point_support),'stable_lines':len(line_support)})
        calibration.append(record)
    return atlas,{'calibration':calibration,'feature_patches':patches,'stable_edges':edge_links,'referent_endpoints':links,
                  'position_channel':positions,'registered_appearance_channel':appearances}


def prior_views(media,a,b,old,region,matrix,radius):
    witnesses=[]
    for x in merge({'base':old['base'],'compensated':old['changes']}):
        box=x['native_xywh']
        if torchless_match(media,a,b,box,region,matrix,radius) and not any(iou(box,p)>.55 for p in witnesses):
            witnesses.append(box)
    return witnesses


def boundary_relation(media,a,b,old,new,chart,comparison):
    reference=comparison['current']
    matrix=np.array(chart['native_affine'])
    endpoints=[]
    for target in merge({'base':new['base'],'compensated':new['changes']}):
        region=target['native_xywh'];radius=hypot(*region[2:]);gap=box_distance(region,reference)
        if iou(region,reference)>.12 or gap>radius:
            continue
        witnesses=prior_views(media,a,b,old,region,matrix,radius)
        if len(witnesses)!=1:
            continue
        prior=warp_box(witnesses[0],matrix,media.images[b].shape)
        d=gap-box_distance(prior,reference)
        if abs(d)<5:
            continue
        endpoints.append(media.region(b,reference,[ENV,NEIGHBOR],[{'kind':'conditional_region_gap_change',
            'current_moving_view':target['id'],'previous_roi':witnesses[0], 'delta_projected_native_gap':d,
            'identity_assumption':'unconfirmed','boundary_scope':'candidate rectangles',**comparison}]))
        endpoints.append(media.region(a,comparison['old'],[ENV,NEIGHBOR],[{'kind':'previous_static_endpoint',**comparison}]))
    return endpoints


def policy(legacy,atlas):
    result={k:{} for k in ('R0P0','R0P1','R1P0','R1P1')}
    pools={'R0':{},'R1':{}}
    for t,l in legacy.items():
        old={k:l[k] for k in ('changes','ref','near','marked')}
        base=l['base']
        additions=atlas[t]['roles']
        basis={**old,'original_change':base}
        candidate_pool=merge(basis)
        extended=merge({**basis,'edge_candidates':atlas[t]['boundaries'],'new':additions})
        pools['R0'][t]=candidate_pool; pools['R1'][t]=extended
        paid=merge({'change':l['changes'],'new':additions})
        extra_old=[r for r in additions if CHANGE in r['roles'] or not any(
            iou(r['native_xywh'],s['native_xywh'])>.60 and CHANGE in s['roles'] for s in paid)]
        result['R0P0'][t]=merge(old)
        result['R0P1'][t]=candidate_pool
        result['R1P0'][t]=merge({**old,'new':extra_old})
        unresolved=[r|{'roles':[UNEXPLAINED]} for r in atlas[t]['boundaries']]
        result['R1P1'][t]=merge({**basis,'raw_boundary':unresolved,'new':additions})
        for key,value in result.items():
            assert {r['id'] for r in value[t]} <= {r['id'] for r in pools[key[:2]][t]}
    return pools,result
