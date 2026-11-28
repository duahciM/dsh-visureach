"""Assess the locked original-address cohorts and decide the preregistered AMR gate."""
from collections import Counter,defaultdict
from statistics import median
from pathlib import Path
import numpy as np

from dacf_research.prepare_inputs import ROOT,read,write,digest
from partial_modules_stage2.evaluate import source_audit
from partial_modules_stage3.material import H
from partial_modules_stage3.observation import iou,ENV,CHANGE,NEIGHBOR,GOAL
from partial_modules_stage3.run import diff
from partial_modules_stage1.study import centroid


def best(r,given):
    returned=max(((iou(r,t['box']),i) for i,t in enumerate(given)),default=(0.,None))
    return given[returned[1]] if returned[1] is not None and returned[0]>=.5 else None


def summarize(d,refs):
    by_time=defaultdict(list);observed=defaultdict(list)
    for r in refs:by_time[r['pts']].append(r)
    for r in d['regions']:observed[r['pts']].append(r)
    hits=Counter();total=Counter(t['role'] for t in refs);centers=[];boundary=[];match={};known_identity_env=set()
    distinct_goal_hits=[]
    for pts,rs in observed.items():
        for row in rs:
            f=best(row['native_xywh'],by_time[pts]); 
            if f:match[row['id']]=f
            if f and f['role']==ENV and row['id'] in d['identity']['identifiers']:known_identity_env.add(row['id'])
    for target in refs:
        candidates=[r for r in observed[target['pts']] if target['role'] in r['roles']]
        strongest=max(candidates,key=lambda c:iou(c['native_xywh'],target['box']),default=None)
        if strongest is None:continue
        overlap=iou(strongest['native_xywh'],target['box'])
        if overlap<.5:continue
        hits[target['role']]+=1
        if target['role'] in (CHANGE,NEIGHBOR):
            u=np.array(target['box']);v=np.array(strongest['native_xywh'])
            centers.append(float(np.linalg.norm(centroid(u)-centroid(v))))
            boundary.append(float(np.abs(np.r_[u[:2],u[:2]+u[2:]]-np.r_[v[:2],v[:2]+v[2:]]).mean()))
        if target['role']==GOAL:distinct_goal_hits.append(strongest['id'])
    links=Counter();link_details=[]
    for row in d['identity']['links']:
        a,b=(match.get(i) for i in row['views'])
        kind='unassessed_label_coverage' if not a or not b else 'same_local_episode' if a['episode']==b['episode'] else 'crosses_labelled_local_episodes'
        ch='flow' if row.get('flow',{} )else 'sift'
        links[kind]+=1;links[ch+'_'+kind]+=1
        link_details.append({'pts':row['pts'],'source_roots':row['source_roots'],'kind':kind,'support':ch})
    roles_required={CHANGE,NEIGHBOR}
    n=sum(total[k] for k in roles_required);found=sum(hits[k] for k in roles_required)
    geometry_diagnostics=[];correct=0;errorful=0;unknown=0
    all_compare=d['relation']['raw_cross_time']
    for x in all_compare:
        a,b=x['pts']
        loc=[best(u,by_time[t]) for u,t in zip(x['object_rois']+x['reference_rois'],[a,b,a,b])]
        if any(v is None for v in loc):
            unknown+=1;continue
        u,v,s,t=loc
        if u['episode']!=v['episode'] or s['episode']!=t['episode'] or not {NEIGHBOR,ENV}&{s['role'],t['role']}:
            unknown+=1;continue
        delta,_=diff(u['box'],v['box'],s['box'],t['box'])
        abs_err=float(np.max(np.abs(np.array(delta)-x['native_projection_delta_xy'])))
        correct+=abs_err<=8;errorful+=abs_err>8
        geometry_diagnostics.append({'pts':x['pts'],'max_abs_error_native_px':abs_err,'measurement_ref':x['measurement_ref'],
            'state':x['interpretation_scope'],'actor_episode':u['episode'],'referent_episode':s['episode']})
    return {'source_sha256':d['source_sha256'],'arm':d['arm'],'all_proposals':len(d['regions']),'role_positive_counts':dict(total),
       'role_matched':dict(hits),'required_value_regions':n,'matched_value_regions':found,'coverage':found/n if n else None,
       'median_center_abs_error_native_px':median(centers) if centers else None,'median_boundary_absolute_error_px':median(boundary) if boundary else None,
       'confirmed_links':dict(links),'identity_positive_link_detail':link_details,'identity_probes':d['identity']['attempts_by_reason'],
       'tagged_background_identity_work':len(known_identity_env),'identified_projection_views':len(d['identity']['identifiers']),
       'single_time_relations':len(d['relation']['same_tick_projected']),'raw_two_time_relations':len(all_compare),
       'within_8px_real_projected_endpoint_differences':correct,'outside_8px_real_projected_endpoint_differences':errorful,
       'unassessed_two_time_relations':unknown,'assessable_two_time_rows':geometry_diagnostics,'explicit_goal_hits':distinct_goal_hits,
       'compute_seconds':{'identity':d['seconds_identity'],'relation':d['seconds_relation']}}


def scoped_gate(rows):
    checks={}
    a3={k:next(x for x in lines if x['arm']=='A3') for k,lines in rows.items()}
    for k in ('external-1','external-2'):
        r=a3[k]
        checks[k+'_coverage']=bool(r['coverage'] is not None and r['coverage']>=.5)
        checks[k+'_median_center']=bool(r['median_center_abs_error_native_px'] is not None and r['median_center_abs_error_native_px']<=8)
    real=[a3[k] for k in ('external-1','external-2','street-repeat')]
    checks['real_two_labelled_accepted_links_no_labelled_wrong_links']=(sum(r['confirmed_links'].get('same_local_episode',0) for r in real)>=2
        and sum(r['confirmed_links'].get('crosses_labelled_local_episodes',0) for r in real)==0)
    r=a3['roles'];checks['static_referent_recall']=r['role_matched'].get(NEIGHBOR,0)>=r['role_positive_counts'][NEIGHBOR]/2
    checks['projected_relation_endpoints_at_least_two']=sum(r['within_8px_real_projected_endpoint_differences'] for r in a3.values())>=2
    return {'checks':checks,'accepted':all(checks.values()),'disposition':'perform equal-original AMR arms' if all(checks.values()) else 'new AMR evaluation deferred per protocol'}


if __name__=='__main__':
    out=H/'evaluation';out.mkdir()
    ground=read(H/'references/sealed.json')['targets']
    # Predictions and hand-assisted diagnostic files are complete before opening any labels here.
    assert all((H/p/'run_complete.json').is_file() for p in ('predictions','diagnostic_manual'))
    sources=read(H/'inputs/source_inventory.json'); results={};bundles=[];cost={}
    for case,ref in ground.items():
        lines=[]
        base=read(H/'predictions'/case/'base_receipts.json')
        bundles.append(base['receipts'])
        costs={'media_preparation_seconds':base['media_preparation_seconds'],'role_preparation':base['role_preparation_cost'],
            'exact_original_count':len(base['receipts']),'native_original_png_bytes':sum(f['png_bytes'] for f in base['receipts']),
            'input_filter_frames':sum(a['frames_at_input_filter'] or 0 for f in base['receipts'] for a in f['decode_attempts'])}
        for label in ('predictions','diagnostic_manual'):
            for p in sorted((H/label/case).glob('A*.json')):lines.append(summarize(read(p),ref))
        results[case]=lines;cost[case]=costs
    evidence=source_audit(bundles,[ROOT/spec['path'] for spec in sources.values()])
    metrics={'cases':results,'source_pixel_audit':evidence,'AMR_gate':scoped_gate(results),'common_input_cost':cost,
         'note':'Role correct is separate from locator overlap, from mechanical computed relation, and from physical identity.'}
    write(out/'result.json',metrics)
