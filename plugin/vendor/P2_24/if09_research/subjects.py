"""Track a frozen proposal in original image coordinates and measure relative motion."""
from collections import Counter

import cv2
import numpy as np

from p14_scene_prototypes.features import hull_coverage


def shape_iou(a,b):
    return float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))


def candidates(gray,bg,cfg):
    mask=np.uint8(np.abs(gray-bg)>=cfg['delta'])
    if mask.mean()>cfg['max_component_area_fraction']:return [],'too_much_foreground_for_static_reference_proposal'
    n,ids,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    out=[]
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area<cfg['min_component_pixels'] or min(w,h)<2:continue
        cut=(ids[y:y+h,x:x+w]==k).astype(np.uint8)
        norm=cv2.resize(cut,(24,24),interpolation=cv2.INTER_NEAREST)>0
        appearance=cv2.resize(gray[y:y+h,x:x+w],(24,24))
        out.append({'bbox':[x,y,w,h],'center':centers[k].tolist(),'area':area,'shape':norm,'appearance':appearance})
    return out,'candidate_components' if out else 'no_candidate_component'


def dist_box(a,b):
    x,y,w,h=a; X,Y,W,H=b
    inter=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
    return inter/max(1,min(w*h,W*H))


def public_candidate(k):
    return {x:y for x,y in k.items() if x not in ('shape','appearance')}


def s1(frames,proposal,cfg,budget,motion):
    n=len(frames);a,b=proposal['samples']; ref_box=proposal['bbox_xywh']
    ordered=sorted(range(n),key=lambda i:(min(abs(i-a),abs(i-b)),i))[:min(budget,cfg['max_patch_frames'])]
    gray={i:frames[i]['gray'].astype(np.float32)/255 for i in ordered}
    if a not in gray or b not in gray:
        return {},{'state':'budget_unassessed','examined':len(ordered),'proposal_samples':[a,b],'per_frame':[]}
    clean=[i for i in sorted(ordered) if frames[i]['sharpness']>=8 and not (
        motion.get(i,{}).get('brightness',{}).get('supported') and motion.get(i,{}).get('raw_mad',0)>.035)]
    if len(clean)<3:return {},{'state':'not_evaluable_background_seed','examined':len(ordered),'proposal_samples':[a,b],'per_frame':[]}
    bg=np.median([gray[i] for i in clean],axis=0);pool={}; evidence={}
    for i in sorted(ordered):
        if i not in clean: pool[i]=[];reason='quality_or_photometric_conflict'
        else:pool[i],reason=candidates(gray[i],bg,cfg)
        evidence[i]={'sample':i,'t':frames[i]['t'],'pts':frames[i]['pts'],'candidates':[
            public_candidate(x) for x in pool[i]],'reason':reason}
    seeds=sorted(((dist_box(c['bbox'],ref_box),i,k,c) for i in [b,a] for k,c in enumerate(pool[i])),
                 key=lambda s:(-s[0],s[1]!=b,-s[3]['area'],s[2]))
    chosen={}
    if not seeds or seeds[0][0]<.35:
        return {},{'state':'no_candidate_seed_in_proposal_region','examined':len(ordered),'proposal_samples':[a,b],
                   'unassessed':sorted(set(range(n))-set(ordered)),'per_frame':list(evidence.values())}
    overlap,anchor,_,p=seeds[0];chosen[anchor]=p
    evidence[anchor]['accepted']=public_candidate(p)
    for direction in (-1,1):
        current=p; previous=anchor;misses=0
        for i in range(anchor+direction,n if direction==1 else -1,direction):
            if i not in pool:break
            if abs(frames[i]['t']-frames[previous]['t'])>.62:break
            choices=[]
            for k,c in enumerate(pool[i]):
                distance=float(np.linalg.norm(np.asarray(c['center'])-current['center']))
                iou=shape_iou(c['shape'],current['shape'])
                q=min(c['area'],current['area'])/max(c['area'],current['area'])
                maxstep=cfg['max_image_step_px']*abs(i-previous)
                if distance<=maxstep and iou>=cfg['min_shape_iou'] and q>=.40:
                    sc=iou+.45*q-.75*distance/maxstep
                    choices.append((sc,k,c,iou,distance))
            choices.sort(key=lambda k:(-k[0],k[1]))
            evidence[i]['qualified_count']=len(choices)
            if choices and (len(choices)==1 or choices[0][0]-choices[1][0]>=cfg['unique_score_margin']):
                best=choices[0];current=best[2]; chosen[i]=current;previous=i;misses=0
                evidence[i].update(accepted=public_candidate(current),center_step_px=best[4],shape_overlap=best[3])
            else:
                evidence[i]['association_abstention']='ambiguous_competition' if choices else 'no_continuous_candidate'
                misses+=1
                if misses>=cfg['max_candidate_gap']:break
    return chosen,{'state':'appearance_continuity_candidate_only','anchor':anchor,'reference_box':ref_box,'proposal_samples':[a,b],
        'examined':len(ordered),'unassessed':sorted(set(range(n))-set(ordered)),'trace_indices':sorted(chosen),
        'per_frame':list(evidence.values()),'all_subject_ids_are_hypotheses':True}


def in_box(xy,b,pad=5):
    x,y,w,h=b
    return (xy[:,0]>=x-pad)&(xy[:,1]>=y-pad)&(xy[:,0]<=x+w+pad)&(xy[:,1]<=y+h+pad)


def background(pairs,i,ref,p,r,shape,cfg):
    if i==ref:return np.array([[1.,0,0],[0,1,0]]),{'state':'reference_frame','same_coordinates':True}
    ab=min(i,ref),max(i,ref);root=pairs.get(ab)
    if not root or not root['verified'] or not root['applies'] or not root['quality']:
        return None,{'state':'background_coordinates_unavailable'}
    x,y=root['xy'] if i<ref else root['xy'][::-1]
    good=(~in_box(x,p['bbox']))&(~in_box(y,r['bbox']))
    x,y=x[good],y[good]
    def quadrants(p):
        return len({(u>=shape[1]/2,v>=shape[0]/2) for u,v in p})
    if (len(x)<cfg['min_reference_points'] or min(quadrants(x),quadrants(y))<cfg['min_background_quadrants'] or
        min(hull_coverage(x,shape),hull_coverage(y,shape))<cfg['min_background_coverage']):
        return None,{'state':'background_points_insufficient_or_narrow','root':root['id'],'points':len(x)}
    ix=np.arange(len(x))%4!=0
    cv2.setRNGSeed(i+ref*997)
    H,inliers=cv2.estimateAffinePartial2D(x[ix],y[ix],method=cv2.RANSAC,ransacReprojThreshold=cfg['max_reference_fit_error_px'])
    if H is None or not ix.any() or not (~ix).any():return None,{'state':'background_affine_unavailable','root':root['id']}
    dist=np.linalg.norm(x[~ix]@H[:,:2].T+H[:,2]-y[~ix],axis=1)
    if float(np.median(dist))>cfg['max_reference_fit_error_px'] or float(np.quantile(dist,.9))>2*cfg['max_reference_fit_error_px']:
        return None,{'state':'background_held_point_conflict','root':root['id'],'median_px':float(np.median(dist))}
    return H,{'state':'locally_registered','root':root['id'],'points':len(x),'held_median_error_px':float(np.median(dist)),
              'held_p90_error_px':float(np.quantile(dist,.9)),'matrix_to_reference':H.tolist(),
              'reference_points_xy':y[~ix][:20].tolist(),
              'note':'static background hypothesis; holdout shares original SIFT inlier selection'}


def compare_motion(before,after,background_before,background_after,appearance_distance=None):
    if before is None or after is None or background_before is None or background_after is None:
        return {'raw_image_delta':None,'relative_delta':None,'appearance':appearance_distance,'state':'not_evaluable'}
    a,b=np.asarray(before),np.asarray(after);x,y=np.asarray(background_before),np.asarray(background_after)
    rel=(b-a)-(y-x)
    return {'raw_image_delta':(b-a).tolist(),'relative_delta':rel.tolist(),'appearance':appearance_distance,'state':'relative_2D_only'}


def s2(frames,pairs,path,arm,cfg):
    if not path:return {'state':'not_evaluable_no_S1_path','points':[],'initial_middle_final':None}
    ref=arm['anchor'];anchor=path[ref]; aligned={};log=[]
    for i,p in sorted(path.items()):
        H,ch=background(pairs,i,ref,p,anchor,frames[0]['gray'].shape,cfg)
        result={'sample':i,'t':frames[i]['t'],'pts':frames[i]['pts'],'continuity_candidate':public_candidate(p),'background':ch,
                'raw_center_xy':p['center'],'raw_appearance_mad_to_anchor':float(np.mean(np.abs(p['appearance']-anchor['appearance']))),
                'center_relative_to_reference_xy':None}
        if H is not None:
            z=np.asarray(p['center'])@H[:,:2].T+H[:,2];aligned[i]=z;result['center_relative_to_reference_xy']=z.tolist()
            result['signed_relative_displacement_to_anchor_px']=(z-np.asarray(anchor['center'])).tolist()
        log.append(result)
    plateaus=[];s=cfg['plateau_frames']
    for i in sorted(aligned):
        ks=list(range(i,i+s))
        if not all(k in aligned for k in ks) or frames[ks[-1]]['t']-frames[ks[0]]['t']<.25:continue
        xyz=np.array([aligned[k] for k in ks]); m=np.median(xyz,axis=0); dev=float(np.max(np.linalg.norm(xyz-m,axis=1)))
        if dev<=cfg['max_plateau_jitter_px']:plateaus.append((ks,m))
    accepted=None
    for a,x in plateaus:
        for b,y in reversed(plateaus):
            if b[0]<=a[-1]:break
            if not all(k in aligned for k in range(a[0],b[-1]+1)):continue
            delta=y-x
            if float(np.linalg.norm(delta))<cfg['min_relative_displacement_px']:continue
            midtimes=range(a[-1]+1,b[0])
            mid=min(midtimes,key=lambda k:float(np.linalg.norm(aligned[k]-(x+y)/2)),default=None)
            accepted={'before':a[s//2],'middle':mid,'after':b[s//2],'before_plateau':a,'after_plateau':b,
               'relative_delta_px':delta.tolist(),'reason':'continued_candidate_and_corroborated_background_relative_separation'}
            break
        if accepted:break
    a,b=arm['proposal_samples']
    unit={'frame_a':a,'frame_b':b,'visual_appearance_difference':None,'background_relative_difference':None}
    if a in path and b in path:
        unit['visual_appearance_difference']=float(np.mean(np.abs(path[a]['appearance']-path[b]['appearance'])))
    if a in aligned and b in aligned:unit['background_relative_difference']=(aligned[b]-aligned[a]).tolist()
    return {'state':'supports_specific_relative_image_motion' if accepted else 'unknown_or_no_verified_distinct_plateaus',
            'initial_middle_final':accepted,'reference_sample':ref,'points':log,'proposal_seed_comparison':unit,
            'visibility_without_measurement':'unknown','physical_object_cause_and_metric_pose':None}


def fixtures():
    ok=[
        ('fixed_camera_subject_walks',[10,0],[18,0],[0,0],[0,0],.02,[8,0]),
        ('fixed_subject_camera_pans',[10,0],[15,0],[0,0],[5,0],.02,[0,0]),
        ('both_move',[10,0],[21,0],[0,0],[5,0],.02,[6,0]),
        ('appearance_changes_only',[10,0],[15,0],[0,0],[5,0],.5,[0,0]),
        ('ambiguous_occluded_or_repeated_texture',[10,0],None,[0,0],None,None,None)]
    ret=[]
    for name,a,b,x,y,appearance,target in ok:
        q=compare_motion(a,b,x,y,appearance)
        assert q['relative_delta']==target
        ret.append({'case':name,'results':q,'expected':target,'scope':'logic fixture with supplied subject/background outcomes; not full detector validation'})
    return ret
