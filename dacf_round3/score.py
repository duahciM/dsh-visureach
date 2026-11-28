"""Post-prediction scoring, common-axis metrics, and independent source verification."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.field import METRICS, clip_encoder
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round2.regions import try_middle
from dacf_round3.run import Experiment, max_gap
from dacf_round3.sampling import gaussian_profile
from dacf_round3.setup import HERE, R2, verify
from evidenweft_phase1.source_review import native_stream, timeline


def references(case, task, added):
    local = read(R2/'references/roles.json')
    legacy = read(ROOT/'dacf_research/evaluation_only/roles.json')['roles']
    phase1 = read(ROOT/'evidenweft_phase1/evaluation_only/rubric.json')['cases']
    items = local['independent'].get(case, legacy.get(case,[])).copy()
    items += [{'id':key,'pts':phase1[case]['slots'][key]['pts']} for key in local['frozen_catalogue_context'].get(case,[])]
    for role in added.get(case,[]):
        field = 'seconds' if 'seconds' in role else 'frame'
        values = task.times if field=='seconds' else task.ranks
        points = [p for p in task.native if role[field][0]<=values[p]<=role[field][1]]
        assert points, role
        items.append(role | {'pts':[min(points),max(points)]})
    relations = added['relations'].get(case,local['relation_sets'].get(case,[]))
    return items, relations


def assign(roles, groups, observed):
    matches = {role['id']: sorted(p for p in observed if role['pts'][0]<=p<=role['pts'][1]) for role in roles}
    return {'hits':matches,'complete':[g for g in groups if all(matches[key] for key in g)]}


def reconstruct(task, selected, channel):
    return task.axis(sorted(set(task.coarse)|set(selected)),task.initial|set(selected),channel)


def score_one(task, route, count, roles, groups, receipts, initial):
    selected = [step['pts'] for step in route['steps'][:count]]
    union = task.initial|set(selected)
    own = assign(roles,groups,union)
    native = [task.times[p] for p in task.native]
    observed_times = {p:(0,task.times[p]) for p in union|{task.native[0],task.native[-1]}}
    mapped = {ch:reconstruct(task,selected,ch) for ch in METRICS}
    initial_hashes = {task.coarse_receipts[p]['rgb_sha256'] for p in task.initial}
    chosen_hashes = {receipts[str(p)]['rgb_sha256'] for p in selected}
    return {'new_originals':len(selected),'selected_pts':selected,'held_catalogue':len(task.initial),
            'different_rgb_not_in_catalogue':len(chosen_hashes-initial_hashes),'different_rgb_count_not_semantic_score':True,
            'new_roles':[k for k,v in own['hits'].items() if v and not initial['hits'][k]],'role_pts':own['hits'],
            'complete_combinations':own['complete'],'new_combinations':[g for g in own['complete'] if g not in initial['complete']],
            'max_time_gap_seconds':max_gap(observed_times,union), 'source_duration_seconds':native[-1]-native[0],
            'change_gap':{ch:{'maximum':a['maximum_observed_gap'],'path_total':a['total'],'state':a['state'],
                              'unknown_edges':sum(not edge['applicable'] for edge in a['edges']),
                              'zero_edges':sum(edge['state']=='zero' for edge in a['edges'])} for ch,a in mapped.items()},
            'cold_proxy_filter_frames':sum(a['frames_at_input_filter'] or 0 for p in selected for a in receipts[str(p)]['decode_attempts']),
            'cold_proxy_wall_seconds':round(sum(receipts[str(p)]['wall_seconds'] for p in selected),6),
            'original_png_bytes':sum(receipts[str(p)]['png_bytes'] for p in selected),
            'ranking_seconds':sum(step['selection_seconds'] for step in route['steps'][:count])}


def verify_native(task, receipts):
    started = perf_counter()
    stream,clock = timeline(task.reader.source)
    seen = set()
    for row,bits in native_stream(task.reader.source,stream,clock):
        item = receipts.get(str(row['pts']))
        if not item:
            continue
        assert digest(Path(item['path'])) == item['png_sha256']
        assert row['frame']==item['source_frame'] and stream['time_base']==item['time_base']
        assert item['source_sha256']==task.reader.signature and sha256(bits).hexdigest()==item['rgb_sha256']
        with Image.open(item['path']) as image:
            assert sha256(image.tobytes()).hexdigest()==item['rgb_sha256']
        if row['pts'] in task.coarse_receipts:
            assert item['rgb_sha256']==task.coarse_receipts[row['pts']]['rgb_sha256']
        seen.add(row['pts'])
    assert seen==set(map(int,receipts))
    return {'matched_original_pts':len(seen),'audit_all_source_frames':len(clock),'verification_only_seconds':round(perf_counter()-started,6)}


def materialize(task, receipts):
    needed = {}
    for p,item in receipts.items():
        p = int(p)
        if p in task.coarse_receipts:
            continue
        with Image.open(item['path']) as image:
            rgb = np.asarray(image.resize((320,max(8,round(image.height*320/image.width))),Image.Resampling.BILINEAR))
        needed[task.ranks[p]] = item | {'rgb':rgb}
    from dacf_research.field import Bank
    task.bank.rows.update(Bank(needed,task.bank.config,task.encoder).rows)


def coverage_density(task, cfg):
    result = {}
    for ch in ('time',*METRICS):
        if ch=='time':
            positions = {p:(0,task.times[p]) for p in task.coarse}
            span = task.times[task.native[-1]]-task.times[task.native[0]]
        else:
            axis = task.axis(task.coarse,task.initial,ch)
            positions,span = axis['positions'],axis['total']
        coords = {c:np.asarray([x for p,(k,x) in positions.items() if c==k and p in task.initial]) for c in {c for c,_ in positions.values()}}
        result[ch] = [{'pts':p,'component':c,'coord':x,'nomination':p in task.initial,
                       'at_three_scales':gaussian_profile(x,coords[c],[span*s for s in cfg['kde_relative_scales']]) if span>0 else None}
                      for p,(c,x) in positions.items()]
    return result


def middle_candidates(task, routes):
    out = []
    for route in routes:
        if route['arm'] not in {'E5','E6'}:
            continue
        acquired = {s['pts'] for s in route['steps']}
        for root in {s['root'] for s in route['steps']}:
            region = task.plan[root]
            lo,hi = region['bounds_pts']
            inner = sorted(p for p in set(task.coarse)|acquired if lo<p<hi)
            if len(inner)<5:
                continue
            result = try_middle(task.bank,[task.ranks[p] for p in inner],region['left'],region['right'],
                                task.catalogue,read(R2/'protocol.json'))
            out.append({'arm':route['arm'],'channel':route['channel'],'root':root,'actual_interior_pts':inner,
                        'qualified_middle':result[0] if result else None,'channels':result[1] if result else None})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir()
    lock = read(HERE/'lock.json')
    verify(lock)
    cv2.setNumThreads(1)
    encoder = clip_encoder()
    cfg = read(HERE/'protocol.json')
    source_roles = read(HERE/'independent_roles.json')
    r2 = {r['case']:r for r in read(R2/'evaluation/summary.json')}
    summary = []
    for name, registration in lock['cases'].items():
        directory = args.run/name
        if not (directory/'routes.json').is_file():
            continue
        out = args.out/name
        out.mkdir()
        review_start = perf_counter()
        task = Experiment(name,registration,encoder,read(ROOT/'dacf_research/protocol.json'),out)
        receipts = read(directory/'receipts.json')
        materialize(task,receipts)
        routes = read(directory/'routes.json')
        assert all(r['initial_preserved'] and all(str(step['pts']) in receipts for step in r['steps']) for r in routes)
        roles, groups = references(name,task,source_roles)
        initial = assign(roles,groups,task.initial)
        scored = []
        for route in routes:
            equal = score_one(task,route,cfg['equal_new_originals'],roles,groups,receipts,initial)
            natural = score_one(task,route,route['natural_stop']['after'],roles,groups,receipts,initial)
            scored.append({'arm':route['arm'],'channel':route['channel'],'equal':equal,'natural':natural,
                           'stop':route['natural_stop'],'equal_comparable':equal['new_originals']==cfg['equal_new_originals']})
        two = {r['id'] for r in task.plan.values() if r['kind']=='paired_investigation' and not r['coarse_interior_pts']}
        investigations = [step for route in routes if route['arm'] in {'E0','E5','E6'} for step in route['steps']]
        opened = {step['root'] for step in investigations if step['root'] in two}
        snap = {'case':name,'cohort':registration['cohort'],'initial_catalogue':len(task.initial),'registered_roles':roles,
                'source_upper_bound':assign(roles,groups,set(task.native)), 'coarse':assign(roles,groups,set(task.coarse)),
                'initial':initial,'routes':scored,'two_boundary_regions':len(two),'two_boundary_touched':len(opened),
                'two_boundary_unopened':[key for key in sorted(two-opened)],
                'joint_mechanism_operator_counts':Counter(step['used_operator'] for route in routes if route['arm']=='E6' for step in route['steps']),
                'middle_state_checks':middle_candidates(task,routes),'r2_full_reference':r2.get(name,{}).get('actual_unique_native_reread'),
                'prediction_physical_cache_reads':len(receipts),'prediction_filter_frames':sum(a['frames_at_input_filter'] or 0 for v in receipts.values() for a in v['decode_attempts']),
                'new_computations_postprediction_seconds':round(perf_counter()-review_start,6)}
        snap['native_audit'] = verify_native(task,receipts)
        write(out/'density.json',coverage_density(task,cfg))
        write(out/'score.json',snap)
        summary.append(snap)
        print(name,len(receipts),'actual RGB joins',len(two),len(opened),'two-endpoint intervals',flush=True)
    verify(lock)
    write(args.out/'summary.json',summary)


if __name__ == '__main__':
    main()
