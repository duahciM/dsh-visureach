"""Independent local state-scale and original-video query budget ablations."""
import cv2
import numpy as np


def small_states(frames,motion,baseline,old_roles,cfg):
    ts=np.array([f['t'] for f in frames]);g=np.stack([f['gray'] for f in frames]).astype(np.float32)/255.
    ok=np.array([f['sharpness']>=8 for f in frames]);s=cfg['side_s'];metrics=motion['pair_metrics']; audit=[]; qualified=[]
    included_legacy=[i for x in old_roles for i in x['samples']]
    for i in range(1,len(frames)):
        for lag in cfg['lags_s']:
            left=np.flatnonzero(ok&(ts>=ts[i-1]-lag/2-s-1e-7)&(ts<=ts[i-1]-lag/2+1e-7))
            right=np.flatnonzero(ok&(ts>=ts[i]+lag/2-1e-7)&(ts<=ts[i]+lag/2+s+1e-7))
            r={'pivot_sample':i,'lag_s':lag,'left_support':left.tolist(),'right_support':right.tolist()}
            audit.append(r)
            if min(len(left),len(right))<3:
                r['state']='insufficient_persistent_views';continue
            used=[metrics[int(k)] for side in (left,right) for k in side[1:]]
            between=metrics[left[-1]+1:right[0]+1]
            if any(q.get('brightness',{}).get('supported') and q.get('raw_mad',0)>.035 for q in used+between):
                r['state']='photometric_explanation';continue
            if any(q.get('registration_valid') and q.get('motion_supported') and
                max(abs(q.get('dx',0)),abs(q.get('dy',0)))>cfg['max_shift_px'] for q in used+between):
                r['state']='image_registration_shift_outside_static_domain';continue
            a,b=np.median(g[left],axis=0),np.median(g[right],axis=0)
            difference=b-a; delta=np.abs(difference); mask=delta>=cfg['pixel_delta']
            if mask.mean()>cfg['max_changed_fraction'] or float(delta[~mask].mean())>cfg['max_nonlocal_residual']:
                r['state']='distributed_shift_or_change';continue
            pieces=[]
            for sign in (-1,1):
                count,labels,stat,_=cv2.connectedComponentsWithStats(np.uint8(difference*sign>=cfg['pixel_delta']),connectivity=8)
                for c in range(1,count):
                    x,y,w,h,area=map(int,stat[c])
                    if area<cfg['min_component_px']:continue
                    z=labels==c; center=float(delta[z].mean())
                    jitter=max(float(np.max(np.mean(np.abs(g[left][:,z]-a[z]),axis=1))),float(np.max(np.mean(np.abs(g[right][:,z]-b[z]),axis=1))))
                    if jitter>cfg['max_state_mask_jitter'] or center<3*jitter:continue
                    pieces.append((center,jitter,area,[x,y,w,h],sign,z))
            if not pieces:
                r['state']='no_persistent_signed_component';continue
            mean,jitter,area,box,sign,z=max(pieces,key=lambda x:(x[0]*np.sqrt(x[2]),x[2]))
            aa=min(map(int,left),key=lambda k:(round(float(np.abs(g[k][z]-a[z]).mean()),6),abs(ts[k]-np.median(ts[left])),k))
            bb=min(map(int,right),key=lambda k:(round(float(np.abs(g[k][z]-b[z]).mean()),6),abs(ts[k]-np.median(ts[right])),k))
            if ts[bb]-ts[aa]<.6-1e-6:
                r['state']='views_too_close';continue
            r.update(state='qualified',samples=[aa,bb],t=[float(ts[aa]),float(ts[bb])],delta=mean,jitter=jitter,area=area,
                     bbox_xywh=box,polarity=sign,score=mean*np.sqrt(area))
            qualified.append(r)
    chosen=[]
    for r in sorted(qualified,key=lambda r:(-r['score'],r['samples'])):
        lo,hi=r['t']
        if any(max(p['t'][0],lo)<=min(p['t'][1],hi) for p in chosen):
            r['state']='overlapping_investigation_role';continue
        if any(i in included_legacy for i in r['samples']):
            r['state']='overlap_with_frozen_broad_role';continue
        if len(chosen)>=cfg['max_new_pairs'] or len(set(included_legacy+[i for p in chosen+[r] for i in p['samples']]))>len(baseline):
            r['state']='pair_budget';continue
        r['state']='reserved';chosen.append(r)
    priority=list(dict.fromkeys(included_legacy+[i for x in chosen for i in x['samples']]+baseline))
    picks=sorted(priority[:len(baseline)])
    return picks,chosen,audit


def query_plan(frames,source,coarse,refined,motion,cfg):
    # Prioritize interior investigation before reserving the next disjoint window for context.
    stable=[d for d in refined['diagnostics'] if d['transition_accepted']]
    s,e=source['start'],source['end']; half=cfg['query_half_s']
    if stable:
        a,b=stable[0]['bracket']
    elif coarse['boundaries']:
        a,b=coarse['boundaries'][0]['bracket']
    else:
        a,b=s+(e-s)/3,s+2*(e-s)/3
    legacy=[a-.5,b+.5] if coarse['boundaries'] else [a,b]
    legacy=[max(s,legacy[0]),min(e,legacy[1])]
    default=[legacy[0]+(legacy[1]-legacy[0])/4,legacy[1]-(legacy[1]-legacy[0])/4]
    possible=[]; center=(a+b)/2
    for f,m in zip(frames,motion['pair_metrics']):
        t=f['t']
        if not (a<=t<=b and f['sharpness']>=8):continue
        if m.get('brightness',{}).get('supported'):continue
        # Keep away from already summarized sides; favor contrast/change in the body of the bracket.
        contrast=abs(t-center)/max((b-a)/2,1e-6)
        activity=m.get('registered_mad',m.get('raw_mad',0)) or m.get('raw_mad',0)
        if .25*(b-a)+a<=t<=b-.25*(b-a):
            possible.append({'sample':f['sample'],'t':t,'score':(1-contrast)*(.06+activity),'role':'interior_distinct_view'})
    focus=max(possible,key=lambda x:(x['score'],-x['t']),default=None)
    if not focus: return default,{'fallback':'no_interior_quality_view','considered':possible,'target_roles':['two legacy windows']}
    context=[]; anchor=focus['sample']; aimg=frames[anchor]['gray'].astype(np.float32)/255.
    for f in frames:
        t=f['t']
        if not s<=t<=e or abs(t-focus['t'])<cfg['query_min_nonoverlap'] or f['sharpness']<8:continue
        slots=[(abs(t-a),'departure'),(abs(t-b),'arrival')]
        distance,side=min(slots)
        if distance>max(1.1,half):continue
        x=f['gray'].astype(np.float32)/255.
        mad=float(np.abs(x-aimg).mean())
        context.append({'t':t,'sample':f['sample'],'role':side,'score':mad/(1+distance)})
    other=max(context,key=lambda d:(d['score'],-d['t']),default=None)
    if not other:return default,{'fallback':'no_disjoint_context_window','considered':possible,'target_roles':['two legacy windows']}
    chosen=[focus,other]
    return [r['t'] for r in chosen],{'targets':chosen,'uncovered_context_role':'arrival' if other['role']=='departure' else 'departure',
                 'considered':possible+context,'frozen_slice_is_not_truth':[a,b]}
