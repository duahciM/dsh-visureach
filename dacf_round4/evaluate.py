"""Read role files only after predictions and verify exact acquisition receipts."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter

from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round4.offline import HERE,R3,verify
from evidenweft_phase1.source_review import timeline,native_stream


def evidence(roles, points):
    return {r['id']:sorted(p for p in points if r['pts'][0]<=p<=r['pts'][1]) for r in roles}


def score(route,n,roles,groups,initial,bins):
    original=evidence(roles,initial)
    steps=route['steps'][:n]
    chosen={s['pts'] for s in steps}
    expanded=evidence(roles,set(initial)|chosen)
    changes=[]
    for step in steps:
        ch=route['channel'] or 'gray'
        band=next((b for b in bins[ch]['bins'] if b['interval_seconds'][0]<=step['source_seconds']<b['interval_seconds'][1]),None)
        prior=step.get('original_edges',[])
        revised=step.get('revised_edges',[])
        inserted=step['pts'] not in {p for edge in prior for p in edge['pts']}
        valid=all(e['applicable'] for e in prior+revised)
        measured=(sum(e['value'] for e in revised)-sum(e['value'] for e in prior)) if prior and revised and inserted and valid else None
        local_cov=step.get('coverage_gap_before')
        a,b=step['bounds']; p=step['pts']
        before=step.get('chord_terms')
        chord_delta=step.get('path_increment')
        if before and None not in before and chord_delta is not None:
            assert abs((before[0]+before[1]-before[2])-chord_delta)<1e-9
        changes.append({'pts':p,'interval':[a,b],'source_seconds':step['source_seconds'],'axis':ch,
                        'region_class':[band['density_band'],band['change_band']] if band else None,
                        'chord_terms':before,'direct_chord_delta':chord_delta,'new_sampling_path_delta':measured,
                        'before_gap':local_cov,'after_chord_half_max':max(before[:2]) if before and None not in before else None,
                        'new_registered_roles_at_this_image':[key for key in expanded if p in expanded[key] and not original[key]],
                        'is_probe_beyond_coarse_path':inserted and bool(prior)})
    return {'images':len(chosen),'points':sorted(chosen),'new_roles':[k for k,v in expanded.items() if v and not original[k]],
            'new_combinations':[g for g in groups if all(expanded[k] for k in g) and not all(original[k] for k in g)],
            'cold_filter_frames':sum(s['cold_filter_frames'] for s in steps),'cold_read_seconds':sum(s['cold_read_seconds'] for s in steps),
            'transport_png_bytes':sum(s['png_bytes'] for s in steps),'ranking_seconds':sum(s['seconds_selection'] for s in steps), 'refinements':changes}


def source_check(c,receipts):
    start=perf_counter()
    p=ROOT/c['source']; stream,clock=timeline(p)
    checked=set()
    for r, pixels in native_stream(p,stream,clock):
        item=receipts.get(str(r['pts']))
        if item is not None:
            assert r['frame']==item['source_frame'] and stream['time_base']==item['time_base']
            assert sha256(pixels).hexdigest()==item['rgb_sha256'] and digest(Path(item['path']))==item['png_sha256']
            checked.add(r['pts'])
    assert checked==set(map(int,receipts))
    return {'audited_native_frames':len(clock),'rgb_verified':len(checked),'seconds':perf_counter()-start}


def main():
    lock=read(HERE/'lock.json'); verify(lock)
    dest=HERE/'evaluation'; dest.mkdir()
    former={r['case']:r for r in read(R3/'evaluation/summary.json')}
    known_new=read(HERE/'independent_roles.json')
    meta=read(ROOT/'dacf_round2/references/roles.json')
    more_groups=read(R3/'independent_roles.json')['relations']
    summary=[]
    for case,c in lock['cases'].items():
        p=HERE/'offline'/case
        route=read(p/'routes.json'); bins=read(p/'density.json'); receipts=read(p/'receipts.json')
        registry=read(ROOT/c['index']/'manifest.json')
        nominal={f['pts'] for f in registry['pool'] if f['in_catalog']}
        roles=former[case]['registered_roles'] if case in former else known_new['roles']
        groups=meta['relation_sets'].get(case,more_groups.get(case,known_new['relation_sets'] if case=='independent_pan' else []))
        match_count=0
        old_route=read(R3/'run'/case/'routes.json') if case in former else read(p/'extension_r3_arms.json')
        for old in old_route:
            if old['arm']!='E5': continue
            new=next(z for z in route if z['strategy']=='S0' and z['channel']==old['channel'])
            match_count+=[r['pts'] for r in old['steps']]==[r['pts'] for r in new['steps']]
        rows=[]
        for r in route:
            assert r['initial_nominees_retained']==len(nominal)
            rows.append({'strategy':r['strategy'],'axis':r['channel'],'stop':r['natural'],
                         'equal':score(r,3,roles,groups,nominal,bins),
                         'natural':score(r,r['natural']['after'],roles,groups,nominal,bins)})
        quad={ch:dict(Counter(r['density_band']+'/'+r['change_band'] for r in info['bins'])) for ch,info in bins.items()}
        operator_summary=Counter(s['used_operator'] for r in old_route if r['arm']=='E6' for s in r['steps'])
        clip={'case':case,'initial':evidence(roles,nominal),'nomination_count':len(nominal),'roles':roles,'density_quadrants':quad,
              'coarse':evidence(roles,{f['pts'] for f in registry['pool']}),'S0_matches_R3_E5_trajectories':match_count,
              'r3_secondary_operators':operator_summary,'routes':rows,'physically_retrieved':len(receipts),
              'physical_filter_frames':sum(a['frames_at_input_filter'] or 0 for v in receipts.values() for a in v['decode_attempts']),
              'physically_retrieved_png_bytes':sum(v['png_bytes'] for v in receipts.values()),'physical_read_seconds':sum(v['wall_seconds'] for v in receipts.values()),
              'native_verification':source_check(c,receipts)}
        write(dest/f'{case}.json',clip)
        summary.append(clip)
        print(case,match_count,'of 4 original trajectories match; verified',len(receipts),'originals',flush=True)
    write(dest/'summary.json',summary)
    verify(lock)


if __name__=='__main__': main()
