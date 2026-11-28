"""Execute selection, scoped association, and independent relation evidence in that order."""
from argparse import ArgumentParser
from collections import defaultdict
from time import perf_counter

import cv2
import numpy as np

from dacf_research.prepare_inputs import ROOT,digest,read,write
from partial_modules_stage1.study import centroid,box_distance,root
from partial_modules_stage3.material import H
from partial_modules_stage3.observation import (
    Media,role_discovery,arms_from,scoped_identity,ENV,NEIGHBOR,CHANGE,GOAL,iou,
)


def diff(a,b,c,d):
    a,b,c,d=(r['native_xywh'] if isinstance(r,dict) else r for r in (a,b,c,d))
    return (centroid(b)-centroid(d)-centroid(a)+centroid(c)).tolist(),box_distance(b,d)-box_distance(a,c)


def m5(media,inventory,identify,manual=False):
    linkset={(r['views'][0],r['views'][1]):r for r in identify['links']}
    by_id={r['id']:r for rows in inventory.values() for r in rows}
    projected=[];time_diffs=[]
    for t, items in inventory.items():
        candidates=[r for r in items if any(k in r['roles'] for k in (CHANGE,GOAL))]
        partners=[r for r in items if any(k in r['roles'] for k in (NEIGHBOR,ENV))]
        for x in candidates:
            for y in partners:
                distance=box_distance(x['native_xywh'],y['native_xywh'])
                if x['id']==y['id'] or distance>np.hypot(*x['native_xywh'][2:]) or iou(x['native_xywh'],y['native_xywh'])>.12:
                    continue
                projected.append({'pts':t,'objects':[x['id'],y['id']],'native_roi_boxes':[x['native_xywh'],y['native_xywh']],
                    'offset_native_xy':(centroid(x['native_xywh'])-centroid(y['native_xywh'])).tolist(),'gap_native_px':distance,
                    'source_root':media.source_root(t),'observation_ids_without_identity_requirement':True})
    if manual:
        for a,b in zip(media.ticks,media.ticks[1:]):
            left={r['episode']:r for r in inventory[a]};right={r['episode']:r for r in inventory[b]}
            for ep,x in left.items():
                cur=right.get(ep)
                if not cur or CHANGE not in x['roles'] or CHANGE not in cur['roles']:continue
                for key,ref in left.items():
                    rr=right.get(key)
                    if not rr or NEIGHBOR not in ref['roles'] or NEIGHBOR not in rr['roles']:continue
                    xy,gap=diff(x,cur,ref,rr)
                    created=(x['id'],cur['id'])
                    time_diffs.append(make_time(media,a,b,x['native_xywh'],cur['native_xywh'],
                         ref['native_xywh'],rr['native_xywh'],xy,gap,linkset.get(created),'assessor_supplied_endpoint_roles'))
        return {'same_tick_projected':projected,'raw_cross_time':time_diffs}
    for t,items in inventory.items():
        for rr in items:
            for basis in rr['basis']:
                if basis['kind']!='conditional_region_gap_change':continue
                a,b=basis['pts'];before=basis['previous_roi']; actor=by_id.get(basis['current_moving_view'])
                if actor is None:continue
                prev=media.region(a,before,[CHANGE],[])
                local=linkset.get((prev['id'],actor['id']))
                xy,gap=diff(before,actor['native_xywh'],basis['old'],rr['native_xywh'])
                time_diffs.append(make_time(media,a,b,before,actor['native_xywh'],basis['old'],rr['native_xywh'],xy,gap,
                                    local,basis['reference_chart']))
    return {'same_tick_projected':projected,'raw_cross_time':time_diffs}


def make_time(media,a,b,u,v,x,y,xy,gap,identity,context):
    scopes='identity_supported_local_interpretation' if identity else 'conditional_identity_only'
    return {'pts':[a,b],'source_roots':[media.source_root(a),media.source_root(b)],'object_rois':[u,v],'reference_rois':[x,y],
            'measurement_ref':root('raw-offset',media.sha,a,b,u,v,x,y),'native_projection_delta_xy':xy,
            'native_image_gap_delta_px':gap,'reference_geometry':context,'interpretation_scope':scopes,
            'identity_dependency':identity['calculation_root'] if identity else None,
            'physical_identity_and_contact':'unknown','interval_between_samples':'unobserved'}


def cohort(case,spec,out,mode):
    dest=out/case;dest.mkdir()
    start=perf_counter();media=Media(spec,dest);prepared=perf_counter()-start
    if mode=='predict':
        t=perf_counter()
        labs,diag,witness,prep_cost=role_discovery(media,spec['explicit_goal_queries'])
        prep_cost['selection_wall_seconds']=perf_counter()-t
        arms=arms_from(labs)
    else:
        refs=read(H/'references/sealed.json')['targets'][case]
        manual={t:[] for t in media.ticks}
        for row in refs:
            rr=media.region(row['pts'],row['box'],[row['role']],[{'kind':'assessor_supplied_diagnostic_only'}])
            manual[row['pts']].append(rr|{'episode':row['episode']})
        arms={'A4':manual,'A4-no-flow':manual}; diag=[];witness=[];prep_cost={}
    for arm, inventory in arms.items():
        t=perf_counter();identity=scoped_identity(media,inventory,arm);elapsed=perf_counter()-t
        t=perf_counter();relation=m5(media,inventory,identity,manual=mode=='manual');rels=perf_counter()-t
        result={'source_sha256':media.sha,'arm':arm,'regions':list(r for rows in inventory.values() for r in rows),
                'identity':identity,'relation':relation,'seconds_identity':elapsed,'seconds_relation':rels}
        write(dest/f'{arm}.json',result)
        print(case,arm,len(result['regions']),len(identity['links']),len(relation['raw_cross_time']),flush=True)
    write(dest/'base_receipts.json',{'source_sha256':media.sha,'source':str(media.path),'receipts':list(media.raw_reader.receipts.values()),
           'media_preparation_seconds':prepared,'role_preparation_cost':prep_cost, 'pair_contexts':list(media.chart_cache.values()),
           'compensated_change_diagnostics':diag,'relationship_role_nominations':witness,
           'discovery_candidate_discarded':{k:v['dropped'] for k,v in labs.items()} if mode=='predict' else None})


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--mode',choices=['predict','manual'],default='predict')
    parser.add_argument('--out');p=parser.parse_args()
    name=p.out or ('predictions' if p.mode=='predict' else 'diagnostic_manual')
    out=H/name
    if out.parent!=H:raise ValueError('Use a study-local output.')
    lock=read(H/'lock.json')
    assert all(digest(ROOT/key)==value for key,value in lock['prediction'].items())
    if p.mode=='manual':
        assert (H/'predictions/run_complete.json').is_file()
        assert digest(H/'references/sealed.json')==lock['assessment_sha256']
    spec=read(H/'inputs/source_inventory.json');out.mkdir()
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    for name,setting in spec.items():cohort(name,setting,out,p.mode)
    write(out/'run_complete.json',{'mode':p.mode,'sources':sorted(spec)})
