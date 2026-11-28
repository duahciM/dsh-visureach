"""Open withheld roles after both prediction arms; audit original samples and controls."""
import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from dacf_research.field import crossing, path_curve
from dacf_research.prepare_inputs import HERE, ROOT, digest, read, write
from dacf_research.freeze import validate
from evidenweft_phase1.prepare import select_time


def in_role(p, role):
    a,b=role['pts']
    return p is not None and a<=p<=b


def validate_controls(ro, protocol):
    r=ro['manual']
    gray=lambda name:r[name][0]['path']['gray']
    keys=lambda name:r[name][0]
    assert gray('static')['state']=='zero_change' and keys('static')['selections']['gray_anchor_balance'] is None
    assert gray('linear')['state']=='computed' and abs(gray('linear')['path_to_endpoint_direct_ratio']-1)<.005
    assert keys('detour')['observed_direction_reversals']>=2 and gray('detour')['path_to_endpoint_direct_ratio']>2.9
    hard=keys('cut')['intersections']['gray']['events']
    assert len(hard)==1 and hard[0]['from_pts']==2800 and hard[0]['to_pts']==3000 and hard[0]['kind']=='sign_change_between_samples'
    assert all(abs(row['anchor_balance']['gray']-.5)>.49 for row in keys('cut')['rows'])
    bridge=keys('local_bridge')['selections']['disjoint_local_support']
    assert bridge is not None and 2400<=bridge<=3200
    rows={0:{'seconds':0.,'pts':11,'id':'f11'},1:{'seconds':.2,'pts':22,'id':'f22'},2:{'seconds':.8,'pts':33,'id':'f33'}}
    bank=SimpleNamespace(rows=rows,config=protocol,metric=lambda a,b:{'gray':1.0})
    gap=path_curve(bank,[0,1,2],'gray')
    assert gap['state']=='path_incomplete' and all(v['P_path'] is None for v in gap['cumulative'])
    assert [v['C_A'] for v in gap['cumulative']]==[0,1,None]
    points=[{'pts':i,'t':i*.2,'da':p,'db':1-p,'balance':p} for i,p in enumerate((.2,.3,.4))]
    assert crossing(points,1e-7,.15,.32,True)['state']=='no_crossing'
    points[1]['balance']=None; points[2]['balance']=.6
    assert crossing(points,1e-7,.15,.32,True)['state']=='unreliable_no_comparable_bracket'
    return {'actual_video':{'static':'zero_change','linear':'distance_ratio_near_one','detour':'path_ratio_near_three_and_reversals',
              'cut':'adjacent_bracket_without_interpolated_frame','local_bridge':'declared spatial mix selected'},
            'math_input_edge_checks':{'missing_gap':gap,'no_crossing':'passed','unreliable_gap_between_observed_sides':'passed'}}


def baseline_paths(case):
    if case in {'doorway','bench'}:
        return ROOT/'evidenweft_phase1/inventory'/case/'index'
    return HERE/'shared_catalogues'/case


def record_predictions(arm, case, pairs, roles, manual, ledger):
    index=baseline_paths(case)
    m=read(index/'manifest.json')
    catalogue=[r for r in m['pool'] if r['in_catalog']]
    original={r['pts']:r for r in ledger}
    selection=[]
    if arm=='manual':
        for p in pairs:
            pid=p['pair']['id']
            role_id=manual.get(case,{}).get(pid)
            if not role_id:
                continue
            role=next(x for x in roles if x['id']==role_id)
            a,b=p['boundary_pts']
            existing=[r for r in catalogue if a<r['pts']<b]
            time=(original[a]['seconds']+original[b]['seconds'])/2
            choice=min(existing,key=lambda x:(abs(x['seconds']-time),x['pts']))['pts'] if existing else None
            predictions=p['selections']|{'frozen_catalogue_nearest_midpoint':choice}
            for name,pt in predictions.items():
                selection.append({'case':case,'pair':pid,'reference_role':role_id,'route':name,'count':int(pt is not None),
                                  'selected_pts':pt,'hit':in_role(pt,role)})
            gray=p['intersections']['gray']['events']
            crossing_pts=sorted({x[k] for x in gray for k in ('from_pts','to_pts')})
            selection.append({'case':case,'pair':pid,'reference_role':role_id,'route':'gray_crossing_endpoints_not_equal_count',
                'count':len(crossing_pts),'selected_pts':crossing_pts,'hit':any(in_role(pt,role) for pt in crossing_pts)})
        return selection
    eligible=[p for p in pairs if p['rows']]
    for name in ('time_midpoint','gray_anchor_balance','gray_path_midpoint','gray_multiscale','disjoint_local_support'):
        selected=sorted({p['selections'][name] for p in eligible if p['selections'][name] is not None})
        count=len(selected)
        coarse=select_time(m['pool'],count) if count else []
        uniform=select_time(catalogue,count) if 0<count<=len(catalogue) else []
        evaluate=lambda pts:[r['id'] for r in roles if any(in_role(p,r) for p in pts)]
        selection.append({'case':case,'route':name,'count':count,'selected_pts':selected,'roles':evaluate(selected),
              'equal_coarse_uniform_pts':[p['pts'] for p in coarse],'equal_coarse_uniform_roles':evaluate([p['pts'] for p in coarse]),
              'equal_frozen_catalogue_pts':[p['pts'] for p in uniform],'equal_frozen_catalogue_roles':evaluate([p['pts'] for p in uniform]),
              'role_in_any_predicted_bracket':{r['id']:any(any(k['eligible_interior'] and in_role(k['pts'],r) for k in p['rows']) for p in eligible) for r in roles}})
    return selection


def chart(report, label, folder):
    path=report['path']['gray']
    rows=report['rows']
    times=[f['seconds'] for f in rows]
    fig, axes = plt.subplots(3,1,sharex=True,figsize=(9.8,7.0),layout='constrained')
    axes[0].plot(times,[r['a']['channels']['gray']['distance'] for r in rows],marker='.',label='Gray distance to A')
    axes[0].plot(times,[r['b']['channels']['gray']['distance'] for r in rows],marker='.',label='Gray distance to B')
    axes[0].axhline(.15,color='.55',ls=':',label='Frozen gray support cap')
    axes[0].set_ylabel('Normalized luma L1'); axes[0].legend(loc='best')
    axes[1].plot(times,[r['anchor_balance']['gray'] for r in rows],marker='.',label='Anchor balance')
    axes[1].plot(times,[x['P_path'] for x in path['cumulative']],marker='.',label='Sampled path fraction')
    axes[1].axhline(.5,color='.55',ls=':')
    axes[1].set_ylim(-.05,1.05); axes[1].set_ylabel('Fraction'); axes[1].legend(loc='best')
    for sigma,values in path['smoothed'].items():
        axes[2].plot(times,[x['weighted_distance_rate_per_second'] for x in values],marker='.',label=f'Gaussian σ={sigma}s')
    axes[2].set_ylabel('Luma L1 per second'); axes[2].set_xlabel('Source PTS seconds'); axes[2].legend(loc='best')
    fig.suptitle(label)
    fig.savefig(folder/f'{label}.png', dpi=145)
    plt.close(fig)


def panel(arm, case, report, roles, original, folder, case_dir):
    by_pts={r['pts']:r for r in report['rows']}
    wanted={}
    for kind,pts in report['selections'].items():
        if pts is not None:
            wanted.setdefault(pts,[]).append(kind)
    for event in report['intersections']['gray']['events']:
        for p in (event['from_pts'],event['to_pts']):
            wanted.setdefault(p,[]).append('crossing endpoint')
    for p in by_pts:
        if any(in_role(p,r) for r in roles):
            wanted.setdefault(p,[]).append('reference interval')
    if not wanted:
        return
    out=Image.new('RGB',(930,((len(wanted)+2)//3)*215), '#151d28')
    ink=ImageDraw.Draw(out)
    for i,(pts,labels) in enumerate(sorted(wanted.items())):
        rec=original[pts]; detail=by_pts[pts]['local']
        im=Image.open(case_dir/rec['png']).convert('RGB')
        assert sha256(im.tobytes()).hexdigest()==rec['rgb_sha256']
        im.thumbnail((300,170))
        x,y=i%3*310,i//3*215
        out.paste(im,(x+5,y+4))
        if labels and 'disjoint_local_support' in labels:
            aw,ah=rec['analysis_size']
            first=detail['different_cells_with_one_group_supported'][0]
            for g,color in [('a','#75c7ff'),('b','#ffb56a')]:
                xx,yy,w,h=detail['cell_boxes_analysis'][first[g]]
                sx,sy=im.width/aw,im.height/ah
                ink.rectangle((x+5+xx*sx,y+4+yy*sy,x+5+(xx+w)*sx-1,y+4+(yy+h)*sy-1),outline=color,width=3)
                ink.text((x+9+xx*sx,y+8+yy*sy),g.upper(),fill=color)
        ink.text((x+5,y+177),f'{pts} / {rec["seconds"]:.6f}s',fill='white')
        ink.text((x+5,y+191),' · '.join(dict.fromkeys(labels))[:58],fill='#bac8dc')
    out.save(folder/f'{arm}-{case}-{report["pair"]["id"]}.png')


def main():
    lock=validate('manual')
    assert all(digest(HERE/p)==v for p,v in lock['evaluation_hashes'].items())
    cfg=read(HERE/'protocol.json'); labels=read(HERE/'evaluation_only/roles.json')
    root=HERE/'evaluation'
    root.mkdir()
    plots=root/'plots'; plots.mkdir()
    images=root/'contact_sheets'; images.mkdir()
    all_pairs={}; paired=[]; nominations=[]; globals_=[]
    for arm in ('manual','automatic'):
        all_pairs[arm]={}
        for case,roles in labels['roles'].items():
            case_dir=HERE/f'run_{arm}'/case
            pairs=read(case_dir/'pair_results.json')
            all_pairs[arm][case]=pairs
            ledger=read(case_dir/'frame_receipts.json')
            original={r['pts']:r for r in ledger}
            ref=read(HERE/'evaluation_only/original_reviews'/f'{case}.json')
            locator={r['pts']:r for r in ref['frames']}
            assert all(r['frame']==locator[r['pts']]['frame'] for r in ledger)
            assert all(digest(case_dir/r['png'])==r['png_sha256'] for r in ledger)
            rows=record_predictions(arm,case,pairs,roles,labels['manual_scoring_roles'],ledger)
            (paired if arm=='manual' else globals_).extend(rows)
            for r in pairs:
                reports=r['path']
                for ch,f in reports.items():
                    if f['state']=='computed':
                        cum=f['cumulative']
                        assert all(abs(p['C_A']+p['C_B']-f['total'])<1e-7 for p in cum)
                        assert all(a['P_path']<=b['P_path'] for a,b in zip(cum,cum[1:]))
                    nominations.append({'arm':arm,'case':case,'pair':r['pair']['id'],'channel':ch,'sample_count':len(r['rows']),
                             'a_distinct_rgb':len({original[p]['rgb_sha256'] for p in r['anchor_pts']['a']}),
                             'b_distinct_rgb':len({original[p]['rgb_sha256'] for p in r['anchor_pts']['b']}),
                             'distance_state':r['intersections'][ch]['state'],'event_supports':dict(Counter(k['support'] for k in r['intersections'][ch]['events'])),
                             'path_state':f['state'],'path_sum':f['total'],'endpoint_distance':f['endpoint_direct'],
                             'path_over_endpoint':f['path_to_endpoint_direct_ratio'],'max_abs_anchor_path_delta':r['anchor_minus_path'][ch]['max_abs']})
                if arm=='manual':
                    chart(r,f'{case}-{r["pair"]["id"]}',plots)
                    panel(arm,case,r,roles,original,images,case_dir)
    checks=validate_controls(all_pairs,cfg)
    write(root/'control_checks.json',checks)
    write(root/'oracle_equal_count.json',paired)
    write(root/'automatic_equal_count.json',globals_)
    with (root/'curves_summary.csv').open('x',newline='',encoding='utf-8-sig') as out:
        w=csv.DictWriter(out,fieldnames=list(nominations[0]))
        w.writeheader()
        w.writerows((r|{'event_supports':json.dumps(r['event_supports'])}) for r in nominations)
    print('checked controls and original receipts; manual comparisons',len(paired),'; automatic comparisons',len(globals_),flush=True)


if __name__=='__main__':
    main()
