"""Equal-original investigation comparison on frozen visual and independent relation axes."""
from collections import defaultdict
from copy import copy
from time import perf_counter
import sys

import cv2
import numpy as np
from PIL import Image

from dacf_research.field import Bank,METRICS,clip_encoder
from dacf_research.prepare_inputs import ROOT,digest,read,write
from dacf_round3.acquire import Reader
from dacf_round3.sampling import cumulative, ranked_gaps, time_pick
from evidenweft_phase1.source_review import native_stream,timeline
from partial_modules_stage1.study import patch_color,prepare,root
from partial_modules_stage2.scoped import AXES,H,Video,accumulated,episodes,freeze_known,scopes


ROUTES={'time':('time',),**{k:(k,) for k in METRICS},
        **{n:(n,) for n in AXES},'round_robin':('ref_x','gray','ref_gap')}


class Material:
    """Release a dense role annotation only for a verified actual source receipt."""
    def __init__(self,source,annotations):
        self.source=source
        self.sha=digest(source)
        self.annotation_file=annotations

    def release(self,actual):
        assert actual['source_sha256']==self.sha
        assert digest(__import__('pathlib').Path(actual['path']))==actual['png_sha256']
        return [v for v in read(self.annotation_file)['observations'] if v['pts']==actual['pts']]


def clone(common,arm):
    arm.mkdir()
    view=copy(common)
    stream,clock=timeline(common.source)
    view.reader=Reader(common.source,common.sha,stream,clock,arm/'native')
    view.reader.receipts=dict(common.reader.receipts)
    view.views=defaultdict(list,{k:list(v) for k,v in common.views.items()})
    view.groups=defaultdict(list,{k:list(v) for k,v in common.groups.items()})
    view.images=dict(common.images);view.prepared=dict(common.prepared);view.permits=set(common.permits)
    view.comparisons=dict(common.comparisons);view.background=dict(common.background)
    return view


def extend(video, store, tick):
    actual,cold=video.reader.read(tick)
    if tick in video.permits:
        return actual,cold
    rows=store.release(actual)
    with Image.open(actual['path']) as p:
        video.images[tick]=np.array(p.convert('RGB'))
    video.prepared[tick]=prepare(actual['path'])
    for row in rows:
        roi=row['native_xywh']; name=row['episode']; identifier=root('view',actual['id'],roi,name)
        v=row|{'source_root':actual['id'],'source_sha256':video.sha,'view_key':identifier,
               'color':patch_color(video.images[tick],roi)}
        video.views[tick].append(v)
        if row['role']=='candidate':
            video.groups[name].append(v);video.groups[name].sort(key=lambda v:v['pts'])
    video.permits.add(tick)
    return actual,cold


def visual_rows(video,ticks):
    rows={}
    for t in ticks:
        r=video.reader.receipts[t]
        rgb=video.images[t];h,w=rgb.shape[:2]
        raster=Image.fromarray(rgb).resize((320,max(8,round(h*320/w))),Image.Resampling.BILINEAR)
        rows[t]={'id':r['id'],'pts':t,'rgb':np.asarray(raster)}
    return rows


def relation_pick(video, axis, observed, times, native):
    result=scopes(video)
    best=None
    details=[]
    for ep,edges in episodes(video,result).items():
        coordinate=accumulated(edges)[axis]
        pos=coordinate['positions']; ticks=sorted(pos)
        if not ticks:continue
        # No axis crosses an unknown component. Endpoints and existing observations define coverage.
        selected,all_=ranked_gaps(pos,ticks,observed,times,native,[(ep,ticks[0],ticks[-1])],None,False)
        if selected and selected['pts'] not in observed:
            if best is None or (selected['coverage'],selected['gap'],-selected['pts'])>(best['coverage'],best['gap'],-best['pts']):
                best=selected
        details.append({'episode':ep,'positions':pos,'edges':[
          {'pts':e['pts'],'delta':e['axes'][axis],'source_roots':e['source_roots'],
           'identity_dependency':e['scope_dependency'],'measurement_ref':e['measurement_ref']} for e in edges],
          'eligible_candidates':len(all_),'units':coordinate['units']})
    return best,details


def visual_pick(bank,axis,observed,times,native):
    grid=sorted(bank.rows)
    pos,edges,_=cumulative(grid,times,lambda a,b:bank.metric(a,b)[axis],bank.config['maximum_gap_seconds'])
    selected,rows=ranked_gaps(pos,grid,observed,times,native,[('source',grid[0],grid[-1])],None,False)
    return selected,{'positions':pos,'edges':edges,'eligible_candidates':len(rows)}


def run_arm(name,cycle,base,visual,encoder,store,anchors,dest):
    started=perf_counter()
    arm=dest/name; video=clone(base,arm); bank=copy(visual)
    bank.rows=dict(visual.rows);bank.matrix=dict(visual.matrix)
    ticks=[r['pts'] for r in video.clock];times={r['pts']:r['seconds'] for r in video.clock}
    observed=set(anchors); trace=[]; spent=[]
    for step in range(3):
        axis=cycle[step%len(cycle)]
        t=perf_counter()
        if axis in AXES:selection,details=relation_pick(video,axis,observed,times,ticks)
        elif axis in METRICS:selection,details=visual_pick(bank,axis,observed,times,ticks)
        else:selection,gap=time_pick(ticks,observed,times,[],False);details={'time_gap_seconds':gap}
        used=axis
        if not selection or selection['pts'] in observed:
            used='time_fallback';selection,_=time_pick(ticks,observed,times,[],False)
        pt=selection['pts']; actual,cold=extend(video,store,pt)
        observed.add(pt);spent.append(actual)
        if cold and any(a in METRICS for a in cycle):
            added=Bank(visual_rows(video,[pt]),bank.config,encoder);bank.rows.update(added.rows)
        trace.append({'step':step+1,'requested_axis':axis,'used_axis':used,'selected':selection,'original':actual,
                      'axis_before_request':details,'new_exact_decode':cold,'wall_seconds':perf_counter()-t})
    post=scopes(video)
    final={ep:[r for r in rs if any(t in {v['pts'] for v in spent} for t in r['pts'])] for ep,rs in episodes(video,post).items()}
    write(arm/'result.json',{'route':name,'source_sha256':video.sha,'initial_observations':anchors,'requests':trace,
          'revised_relation_edges_incident_to_supplement':final,'runtime_seconds':perf_counter()-started,
          'caution':'Cross-axis quantities are not pooled. A source sample does not imply any continuous path between endpoints.'})


if __name__=='__main__':
    dest=H/'sampling';dest.mkdir()
    plan=read(H/'plan/plan_receipt.json');source=H/'inputs/control/source.mkv'
    anno=H/'inputs/control/observations-dense.json';store=Material(source,anno)
    manifest=read(anno);grid=set(plan['pool_pts'])
    available=[v for v in manifest['observations'] if v['pts'] in grid]
    folder=dest/'coarse_input'; freeze_known(manifest|{'sampled_pts':sorted(grid)},folder,source,available)
    t=perf_counter(); cv2.setNumThreads(1);cv2.setRNGSeed(628)
    base=Video(folder,dest);encoder=clip_encoder()
    bank=Bank(visual_rows(base,sorted(grid)),read(ROOT/'dacf_research/protocol.json'),encoder)
    provided=plan['nomination_pts']
    anchors=[provided[round((len(provided)-1)*a/4)] for a in range(5)]
    assert len(set(anchors))==5
    write(dest/'run_lock.json',{'source_sha256':store.sha,'role_inputs_sha256':digest(anno),
          'catalog_sha256':plan['catalog_sha256'],'protocol_sha256':digest(H/'PROTOCOL.md'),
          'source_code_sha256':digest(H/'sampling.py'),'routes':ROUTES,'anchors':anchors,
          'original_coarse_grid':sorted(grid),'original_nominees':provided, 'clip':bank.clip_state,
          'coarse_setup_seconds':perf_counter()-t})
    for arm,steps in ROUTES.items():run_arm(arm,steps,base,bank,encoder,store,anchors,dest)
    write(dest/'coarse_receipts.json',list(base.reader.receipts.values()))
