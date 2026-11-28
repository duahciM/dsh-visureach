"""P2-01 state role selector. Prediction consumes no review labels."""
import numpy as np
from v05.engine import choose


def state_select(frames,coarse,refined,motion,cfg):
    base,old_reasons,v,uniform=choose(frames,refined['segments'],cfg['budget'])
    times=np.asarray([f['t'] for f in frames]); g=np.stack([f['gray'] for f in frames]).astype(np.float32)/255
    sharp=np.asarray([f['sharpness']>=8 for f in frames]); s=cfg['state_support_s']
    pair_metrics=motion['pair_metrics']
    grid=[(y,x) for y in np.array_split(np.arange(g.shape[1]),cfg['tile_grid'][0])
                 for x in np.array_split(np.arange(g.shape[2]),cfg['tile_grid'][1])]
    records=[]
    def compare(a,b):
        raw=np.abs(a-b)
        return float(raw.mean()), [float(raw[np.ix_(y,x)].mean()) for y,x in grid]
    def window(ids):
        ids=np.asarray(ids); ids=ids[sharp[ids]]
        return ids if len(ids)>=3 and times[ids[-1]]-times[ids[0]]>=s*.6 else np.asarray([],dtype=int)
    def propose(a,b,left,right,kind,signal):
        x=window(a);y=window(b)
        row={'kind':kind,'bracket':[left,right],'left_support':x.tolist(),'right_support':y.tolist(),'status':'rejected'}
        records.append(row)
        if not len(x) or not len(y): row['reason']='insufficient_persistent_views';return
        l=np.median(g[x],axis=0);r=np.median(g[y],axis=0); intensity,tiles=compare(l,r)
        top=int(np.argmax(tiles)); yy,xx=grid[top]
        stable=max(float(np.mean(np.abs(g[x][:,yy][:,:,xx]-l[np.ix_(yy,xx)]))),float(np.mean(np.abs(g[y][:,yy][:,:,xx]-r[np.ix_(yy,xx)]))))
        ratio=tiles[top]/max(stable,.01)
        k,lidx=map(int,(min(x,key=lambda i:(round(float(np.abs(g[i]-l).mean()),5),-min(i-x[0],x[-1]-i),i)),
                       min(y,key=lambda i:(round(float(np.abs(g[i]-r).mean()),5),-min(i-y[0],y[-1]-i),i))))
        row.update(samples=[k,lidx],raw_mad=intensity,peak_tile=top,tile_mad=tiles[top],side_jitter=stable,stability_ratio=ratio)
        if times[lidx]-times[k]+1e-9<cfg['minimum_role_gap_s'] or stable>cfg['max_state_jitter_mad']:
            row['reason']='states_too_close_or_unstable';return
        if kind=='local_persistent' and (tiles[top]<cfg['state_min_tile_mad'] or ratio<cfg['state_min_contrast_ratio'] or tiles[top]<1.45*intensity):
            row['reason']='no_isolated_persistent_change';return
        if kind=='frozen_context_change' and intensity<cfg['state_min_raw_mad']:
            row['reason']='stable_contexts_insufficiently_different';return
        row.update(status='eligible',score=tiles[top]-stable+signal,reason='qualified_visual_state_proxy')
    for d in refined.get('diagnostics',[]):
        if not d['transition_accepted']: continue
        left,right=d['bracket']
        a=np.where((times>=left-2*s)&(times<=left-.4*s))[0]; b=np.where((times>=right+.4*s)&(times<=right+2*s))[0]
        propose(a,b,left,right,'frozen_context_change',.18)
    for m in pair_metrics[1:]:
        i=m['sample'];left,right=times[i-1],times[i]
        local_peak=max(compare(g[i-1],g[i])[1])
        if local_peak<cfg['state_min_tile_mad'] or m.get('brightness',{}).get('supported') or (m.get('registration_valid') and max(abs(m.get('dx',0)),abs(m.get('dy',0)))>cfg['max_abs_phase_shift']):
            continue
        a=np.where((times>=left-s-1e-7)&(times<=left))[0];b=np.where((times>=right)&(times<=right+s+1e-7))[0]
        propose(a,b,left,right,'local_persistent',0)
    picks=[]; reasons={}; roles=[]
    for row in sorted((r for r in records if r['status']=='eligible'),key=lambda r:(-r['score'],r['bracket'])):
        pair=row['samples']
        if len(roles)>=cfg['max_state_pairs'] or len(picks)+2>len(base): row['status']='budget';continue
        if any(set(pair)&set(r['samples']) or (row['left_support'][-1]>=r['left_support'][0] and r['left_support'][-1]>=row['left_support'][0] and row['peak_tile']==r['peak_tile']) for r in roles):
            row['status']='redundant_role';continue
        row['status']='reserved'; roles.append(row)
        for name,i in zip(('before','after'),pair):
            picks.append(i);reasons[i]=f"state_pair_{len(roles)}:{name}:{row['kind']}"
    if not picks: return base,old_reasons,roles,records
    for allow in (False,True):
        for i in base:
            if len(picks)>=len(base):break
            if i in picks or (not allow and any(np.abs(g[i]-g[j]).mean()<cfg['near_duplicate_mad'] for j in picks)): continue
            picks.append(i);reasons[i]=old_reasons[i]
    return sorted(picks),reasons,roles,records
