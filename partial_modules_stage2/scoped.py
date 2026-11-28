"""M2 scoped view continuity and M5 independent measurement coordinates."""
from collections import defaultdict
from itertools import combinations,product
from argparse import ArgumentParser
import math
import shutil
from time import perf_counter

import cv2
import numpy as np

from dacf_research.prepare_inputs import ROOT,digest,read,write
from partial_modules_stage1.study import Video,box_distance,centroid,difference,root,snapshot,support
from partial_modules_stage2.inputs import H
from partial_modules_stage2.ledger import Ledger


AXES={
    'ref_x':'native px in held-out 2D background chart','ref_y':'native px in held-out 2D background chart',
    'peer_x':'native px in held-out 2D background chart','peer_y':'native px in held-out 2D background chart',
    'ref_gap':'native axis-aligned bbox gap px','peer_gap':'native axis-aligned bbox gap px',
    'own_luma':'arithmetic mean delta 8bit luma','relative_luma':'object-minus-reference delta 8bit luma',
    'own_color':'HSV H12S4 histogram total variation or contrast delta',
    'relative_color':'change of object-to-reference HSV H12S4 histogram TV contrast',
}


def compare_local(video, a,b,ctx, use_gate):
    fit=video.comparison(a,b)
    ar,br=a['native_xywh'],b['native_xywh']
    av,bv=centroid(ar),centroid(br)
    if ctx['state']=='heldout_supported':
        m=np.array(ctx['affine']); analysis=np.array(video.prepared[a['pts']][0].shape[::-1])
        scale=analysis/np.array(video.prepared[a['pts']][1]); av=(m[:,:2]@(av*scale)+m[:,2])/scale
    distance=float(np.linalg.norm(av-bv)); limit=math.hypot(*ar[2:])
    positive=support(fit); nearby=distance<=limit
    return {'views':[a['view_key'],b['view_key']], 'analysis_fit':fit['comparison_ref'],
            'local_state':fit['local']['state'],'background_geometry':ctx['calculation_root'],'background_state':ctx['state'],
            'center_residual_native_px':distance,'spatial_domain_limit_native_px':limit,
            'eligible':(positive and nearby) if use_gate else a['episode']==b['episode'],
            'explanations':['local appearance and context measured separately',
                       'movement, viewpoint, weak texture, projection copies or substitution remain possible']}


def scopes(video, use_gate=True, supplied_labels=True):
    ticks=sorted(video.permits)
    ids={}; members=defaultdict(list); inquiries=[]; confirmed={}
    by_pts={k:[v for v in video.views[k] if v['role']=='candidate'] for k in ticks}
    considered=[]; coappearance=[]
    def assign(view,sid):
        ids[view['view_key']]=sid;members[sid].append(view)
    def open_at(view):
        return root('local-scope',video.sha,view['view_key'])
    for j,t in enumerate(ticks):
        right=by_pts[t]
        for a,b in combinations(right,2):
            gap=box_distance(a['native_xywh'],b['native_xywh'])
            coappearance.append({'pts':t,'view_ids':[a['view_key'],b['view_key']],'gap_native_px':gap,
                'source_root':a['source_root'],'status':'separate_nonoverlapping_projections' if gap>0 else 'overlap_unknown',
                'physical_instance_exclusion':'unknown_unless_reflections_screens_and_duplication_are_ruled_out'})
        if j==0 or not right or not by_pts[ticks[j-1]]:
            for b in right:assign(b,open_at(b))
            if j and (right or by_pts[ticks[j-1]]):
                inquiries.append({'pts':[ticks[j-1],t],'state':'missing_actual_view','physical_absence':None})
            continue
        left=by_pts[ticks[j-1]]
        ctx=video.context(left[0]['pts'],t); variants={}; incoming=defaultdict(list); outgoing=defaultdict(list)
        for a,b in product(left,right):
            va,vb=a['view_key'],b['view_key']
            if supplied_labels and a['episode']!=b['episode']:
                continue
            cmp=compare_local(video,a,b,ctx,use_gate and j>0)
            variants[va,vb]=cmp
            considered.append(cmp|{'pts':[a['pts'],b['pts']]})
            if cmp['eligible']:
                incoming[vb].append(va);outgoing[va].append(vb)
        for b in right:
            k=b['view_key']; contenders=incoming[k]
            if len(contenders)==1 and len(outgoing[contenders[0]])==1:
                parent=contenders[0];sid=ids[parent];assign(b,sid)
                confirmed[parent,k]=variants[parent,k]|{'local_scope_id':sid,'identity_kind':'scoped_support','physical_identity':'unknown'}
            else:
                assign(b,open_at(b))
                inquiries.append({'pts':[ticks[j-1],t], 'view':k, 'contenders':contenders,
                   'state':'ambiguity' if contenders else 'not_supported_or_inapplicable','physical_absence':None})
    return {'observation_to_scope':ids,'scopes':{k:{'pts':[r['pts'] for r in rows],'view_ids':[r['view_key'] for r in rows],
                                                  'scope_kind':'local_projected_view','physical_identity':'unknown'} for k,rows in members.items()},
            'relations':confirmed,'unknown_transitions':inquiries,'considered':considered,'coappearance':coappearance}


def tv(a,b):
    return float(np.abs(np.array(a)-np.array(b)).sum()/2)


def relation(video,a,b,identity):
    t0,t1=a['pts'],b['pts']
    snapshots=[snapshot(video,p) for p in (a,b)]; raw=difference(video,*snapshots)
    left,right=(v['values'] for v in snapshots)
    ax={n:None for n in AXES}
    raw_appearance={'own_luma':right['object_luma_8bit']-left['object_luma_8bit'],
                    'own_color':tv(left['object_hsv_h12s4'],right['object_hsv_h12s4'])}
    for role,key in [('reference','ref'),('peer','peer')]:
        measured=raw['channels'][role]
        common=measured.get('delta_offset_common_native_xy')
        if common is not None and identity:
            ax[key+'_x'],ax[key+'_y']=common
            ax[key+'_gap']=measured['delta_bbox_gap_native_px']
    refs=[next((v for v in video.views[p] if v['role']=='reference'),None) for p in (t0,t1)]
    if all(refs) and refs[0]['episode']==refs[1]['episode'] and raw['context_applicability']=='heldout_supported':
        ra,rb=(v['color'] for v in refs)
        raw_appearance['relative_luma']=raw_appearance['own_luma']-(rb['mean_luma_8bit']-ra['mean_luma_8bit'])
        raw_appearance['relative_color']=(tv(right['object_hsv_h12s4'],rb['hsv_h12s4'])-tv(left['object_hsv_h12s4'],ra['hsv_h12s4']))
    if identity:
        ax.update(raw_appearance)
    source_roots=[a['source_root'],b['source_root']]
    return {'measurement_ref':root('raw-relation',video.sha,a['view_key'],b['view_key']),'pts':[t0,t1],'source_roots':source_roots,
          'views':[a['view_key'],b['view_key']],'scope_dependency':identity['local_scope_id'] if identity else None,
          'identity_pair_ref':identity['analysis_fit'] if identity else None,'raw':raw,'raw_appearance':raw_appearance,'axes':ax,
          'axis_units':AXES,'comparison_domain':'supplied native ROIs at separate source samples',
          'interval_state':'unobserved_between_endpoints','physical_contact_or_continuity':'unknown'}


def episodes(video,scoped):
    out={}
    for name,rows in video.groups.items():
        out[name]=[relation(video,a,b,scoped['relations'].get((a['view_key'],b['view_key']))) for a,b in zip(rows,rows[1:])]
    return out


def accumulated(edges):
    axes={}
    for name in AXES:
        component=0; coordinate=0.; points={}; previous=None
        for row in edges:
            a,b=row['pts']; val=row['axes'][name]
            if a!=previous:
                component+=1;coordinate=0.
            points[a]=[component,coordinate]
            if val is None:
                component+=1;coordinate=0.
            else:
                coordinate+=abs(val)
            points[b]=[component,coordinate]
            previous=b
        axes[name]={'units':AXES[name],'positions':points,'unknown_edges':[r['pts'] for r in edges if r['axes'][name] is None]}
    return axes


def cross_query(video,episode_a,episode_b):
    left,right=video.groups[episode_a],video.groups[episode_b]
    pairs=[video.comparison(a,b) for a,b in product(left[:2],right[:2])]
    overlapping_times=sorted({a['pts'] for a in left}&{p['pts'] for p in right})
    return {'episodes':[episode_a,episode_b],'mode':'optional_scoped_cross_range_investigation',
            'appearance_comparisons':[q['comparison_ref'] for q in pairs],
            'local_appearance_support':[q['comparison_ref'] for q in pairs if support(q)],
            'same_time_proj_ticks':overlapping_times,'alternatives':['same after interruption','substituted lookalike','projection or unrelated candidate'],
            'physical_identity':'unknown','promoted_into_local_scopes':False}


def ledger_trial(video, scoped, path):
    # Retrospective engineering check: add independently supplied two-projection evidence after a provisional single-view-chain claim.
    observed=next(q for q in scoped['coappearance'] if q['status']=='separate_nonoverlapping_projections')
    pts=observed['pts']; view=[v for v in video.views[pts] if v['view_key'] in observed['view_ids']]
    sample=[v for v in video.groups[view[1]['episode']] if v['pts']!=pts][0]
    comparison=video.comparison(view[0],sample);ctx=video.context(pts,sample['pts']);roots=comparison['source_roots']
    raw=relation(video,view[0],sample,None)
    book=Ledger(path,video.sha)
    book.add('p-appearance','appearance',roots,calculation=comparison['comparison_ref'],fit_status=comparison['local']['state'])
    book.add('p-background','background_geometry',roots,calculation=ctx['calculation_root'],fit_status=ctx['state'])
    book.add('p-identity','identity_hypothesis',roots,['p-appearance'],status='hypothesis_only',
             views=[view[0]['view_key'],sample['view_key']],episodes=[view[0]['episode'],sample['episode']],
             statement='Provisional claim that these two projected roles belong to one uninterrupted visible ROI chain.')
    book.add('p-raw','raw_relation',roots,['p-background'],measurement=raw)
    book.add('p-dependent','identity_dependent_relation',roots,['p-raw','p-identity'],
             statement='Interpret this projected difference as a change along one uninterrupted visible ROI chain.')
    book.add('p-unknown','coappearance_projection',[observed['source_root']],status='unassessable',scope='Unknown projection cause.')
    try:book.withdraw('p-identity','p-unknown','Rejected test: missing scope information.')
    except ValueError:blocked=True
    else:raise AssertionError('Unknown accepted as counterevidence.')
    before=list(book.query());book.add('p-conflict','coappearance_projection',[observed['source_root']],status='counter_evidence_in_scope',
        scope='Two supported disjoint image ROIs contradict a single visible ROI chain, not a physical-instance assertion.',view_ids=observed['view_ids'],pts=pts)
    book.withdraw('p-identity','p-conflict','Revise the uninterrupted projected-chain interpretation after observing both roles together.')
    fresh=Ledger(path,video.sha);after=list(fresh.query()); by_id={r['id']:r for r in after}
    assert by_id['p-raw']['effective_status']=='recorded' and by_id['p-dependent']['effective_status']=='dependency_withdrawn'
    joined=list(fresh.query(root=observed['source_root']))
    return {'before':before,'after_reload':after,'root_query':joined,'unknown_not_counterevidence':blocked}


def freeze_known(data,folder,video,obs):
    folder.mkdir(parents=True)
    name=video.name
    if not (folder/name).exists():
        try:
            (folder/name).hardlink_to(video)
        except OSError:
            shutil.copyfile(video,folder/name)
    write(folder/'observations.json',data|{'source':name,'observations':obs})


def perturb(rows,size):
    res=[]; mw,mh=size
    for v in rows:
        u=dict(v);x,y,w,h=u['native_xywh']
        if v['role']=='candidate':
            direction=-1 if v['frame']%2 else 1
            x=max(0,min(mw-w,x+4+direction*6));y=max(0,min(mh-h,y+4-direction*4))
            u['native_xywh']=[x,y,w-8,h-8]
        res.append(u)
    return res


def save_case(input_, dest, stress=False, old=False, auto=False):
    dest.mkdir(parents=True)
    start=perf_counter();video=Video(input_,dest); decode=perf_counter()-start
    t=perf_counter(); scoped=scopes(video,supplied_labels=not auto); relations=episodes(video,scoped); scoped_sec=perf_counter()-t
    t=perf_counter(); free=scopes(video,use_gate=False,supplied_labels=not auto); free_m5=episodes(video,free); free_sec=perf_counter()-t
    serial=lambda m:m|{'relations':list(m['relations'].values())}
    out={'source_sha256':video.sha,'scoped':serial(scoped),'ungated_edges':list(free['relations'].values()),
         'relations':relations,'ungated_relations':free_m5,'axes':{n:accumulated(e) for n,e in relations.items()},'read_failures':video.failures,
         'receipts':list(video.reader.receipts.values()), 'computed_pairs':list(video.comparisons.values()),
         'context_calculations':list(video.background.values()),'view_inventory':dict(video.views),
         'runtime':{'decode_and_preparation_seconds':decode,'local_and_m5_seconds':scoped_sec,'no_gate_ablation_shared_cache_seconds':free_sec}}
    if len(video.groups)>=2 and all(len(video.groups[k]) for k in ('g0','g1')):
        out['requested_cross_control']=cross_query(video,'g0','g2' if old else 'g1')
    if old:
        out['requested_replacement_control']=cross_query(video,'g2','g3')
        out['ledger_revisions']=ledger_trial(video,scoped,dest/'evidence.jsonl')
    for n,rows in relations.items():
        if rows:rows[-1]['validation_role']='last observed edge; never used to tune gates or thresholds'
    write(dest/'result.json',out)


def main():
    p=ArgumentParser();p.add_argument('--out',default='scope_run');p.add_argument('--skip-at', nargs='*',default=[]);a=p.parse_args()
    dst=H/a.out
    if dst.parent!=H:raise ValueError('Use a directory inside this study.')
    lock=read(H/'lock.json')
    assert all(digest(ROOT/file)==val for file,val in lock['prediction_hashes'].items())
    dst.mkdir();cv2.setNumThreads(1);cv2.setRNGSeed(628)
    for name in ('paired','panning'):
        if name not in a.skip_at:save_case(ROOT/'partial_modules_stage1/inputs'/name,dst/name,old=True)
    control=H/'inputs/control'; data=read(control/'observations-dense.json'); grid=data['sampled_pts'][::4]
    fixed=data|{'sampled_pts':grid}; known=[v for v in data['observations'] if v['pts'] in grid]
    if 'control' not in a.skip_at:
        stage=dst/'control-input';freeze_known(fixed,stage,control/'source.mkv',known)
        save_case(stage,dst/'control')
        noisy=dst/'control-jitter-input';freeze_known(fixed,noisy,control/'source.mkv',perturb(known,data['native_size']))
        save_case(noisy,dst/'control-jitter',stress=True)
    if 'street' not in a.skip_at:save_case(H/'inputs/street',dst/'street')


if __name__=='__main__':main()
