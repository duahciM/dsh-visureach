"""Re-run the old measurements while observing cross-module output invariants."""

from collections import Counter, defaultdict
from copy import deepcopy

import cv2

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage3.observation import Media, iou
from partial_modules_stage3.run import m5
from partial_modules_stage5.review_sources import H
from partial_modules_stage5.scopes import context, derive, priority


def raw_measurements(relationship):
    return [tuple((field,entry[field]) for field in entry if field not in ('identity_dependency','interpretation_scope'))
            for entry in relationship['raw_cross_time']]


def stable_pool_superset(before,after):
    entries={row['id']:row for rows in after.values() for row in rows}
    for rows in before.values():
        for accepted in rows:
            newer=entries[accepted['id']]
            assert all(accepted[k]==newer[k] for k in ('id','native_xywh','source_root','pts'))
            assert all(item in newer['basis'] for item in accepted['basis'])


def influence_graph(case,spec,out):
    frozen=ROOT/'partial_modules_stage4/predictions'/case
    arms={arm:read(frozen/f'{arm}.json') for arm in ('R0P0','R0P1','R1P0','R1P1')}
    old,new=(read(frozen/f'{arm}-ledger.json')['candidate_rows'] for arm in ('R0','R1'))
    stable_pool_superset(old,new)
    baseline_roots={r['id'] for r in arms['R0P1']['regions']}
    assert baseline_roots=={r['id'] for rows in old.values() for r in rows}
    common={}
    for region, extra in (('R0','R0'),('R1','R1')):
        limited,full=arms[region+'P0'],arms[extra+'P1']
        limited_raw=raw_measurements(limited['m5']); full_raw=raw_measurements(full['m5'])
        left={dict(x)['measurement_ref']:dict(x) for x in limited_raw}
        right={dict(x)['measurement_ref']:dict(x) for x in full_raw}
        assert all(left[k]==right[k] for k in left.keys()&right.keys())
        common[region]={'same_measurement_ids_and_values':len(left.keys()&right.keys()),
                         'p0_only_measurement_ids':len(left.keys()-right.keys()),
                         'p1_only_measurement_ids':len(right.keys()-left.keys())}
    folder=out/case;folder.mkdir()
    media=Media(spec,folder)
    full=arms['R1P1']; inventory={t:[] for t in media.ticks}
    for region in full['regions']:inventory[region['pts']].append(region)
    before=deepcopy(inventory)
    replay=m5(media,inventory,full['identity_follows_role'])
    fallback=m5(media,inventory,{'links':[]})
    assert inventory==before and replay==full['m5']
    assert replay['same_tick_projected']==fallback['same_tick_projected']
    assert raw_measurements(replay)==raw_measurements(fallback)
    changed=sum(a['interpretation_scope']!=b['interpretation_scope'] for a,b in
                zip(replay['raw_cross_time'],fallback['raw_cross_time']))
    # Explicitly separate same-image overlap from different candidate views competing for an endpoint.
    support=defaultdict(list)
    for r in full['regions']:support[r['source_root']].append(r)
    overlapping=[]
    for same_root,rows in support.items():
        for i,r in enumerate(rows):
            for s in rows[i+1:]:
                metric=iou(r['native_xywh'],s['native_xywh'])
                if metric>=.55:
                    overlapping.append({'views':[r['id'],s['id']],'same_original':same_root,'iou':metric})
    real_competition=defaultdict(set)
    for row in full['identity_follows_role']['pair_audit']:
        if row['passes']:
            a,b=row['views'];real_competition[(row['pts'][0],a)].add(b)
            real_competition[(row['pts'][1],b)].add(a)
    write(folder/'result.json',{'source_sha256':media.sha,'candidate_preservation':{'R0_count':len(baseline_roots),
           'R1_retained_exact_R0_count':len(baseline_roots),'digest_of_read_only_baseline':digest(frozen/'R0-ledger.json')},
        'role_switch':common,'m5_direct_actual_replay':{'raw_cross_time_unchanged':len(replay['raw_cross_time']),
        'same_tick_unchanged':len(replay['same_tick_projected']),'identity_interpretations_converted_to_unknown':changed},
        'overlap_same_source_raw_views':overlapping,
        'qualified_competition_endpoints':[{'pts':t,'view':r,'alternatives':sorted(q)} for (t,r),q in real_competition.items() if len(q)>1],
        'pending_same_source_views_merged':0,'source_receipts':list(media.raw_reader.receipts.values())})
    return {'case':case,'original_count':len(media.ticks),'raw_count':len(replay['raw_cross_time']),
            'converted_to_unknown':changed,'explicit_competition_endpoints':sum(len(q)>1 for q in real_competition.values()),
            'overlapping_pairs':len(overlapping)}


def boundary_contract():
    def item(label,limits):
        return {'id':label,'pts':limits,'kind':'hard_cut_visual_regime','source_sha256':'test'}
    base=[0,100]
    one=derive('test',base,[item('first',[10,20])])
    assert one['closed']==[] and len(one['open'])==2
    assert derive('test',base,[])['open']==[{'kind':'both_open_unknown','pts':base}]
    linked=derive('test',base,[item('first',[10,20]),item('second',[40,50]),item('third',[70,80])])
    assert (context(20,linked),context(40,linked),context(50,linked),context(70,linked))==(None,)*4
    assert priority(25,60,linked)=='cross_bounded_context_deferred'
    assert priority(25,60,linked,True)=='explicit_cross_context_allowed'
    for c in ([18,21],[20,21]):
        overlap=derive('test',base,[item('first',[10,20]),item('second',c),item('third',[40,50])])
        assert overlap['closed']==[] and len(overlap['conflicts'])==2
    try:derive('cross-source',base,[item('first',[10,20])])
    except ValueError:pass
    else:raise AssertionError('Cross-source bracket scope must be rejected.')
    return ['no supported transition','one transition with open sides','strict interior','overlap','touch',
            'optional explicitly selected cross-context inspection','cross-source rejection']


if __name__=='__main__':
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    out=H/'isolation';out.mkdir()
    cases=read(ROOT/'partial_modules_stage4/inputs/source_inventory.json')
    wrap=[influence_graph(k,cases[k],out) for k in ('roles','camera','street-repeat')]
    write(out/'summary.json',{'cases':wrap,'scope_contract_assertions':boundary_contract()})
