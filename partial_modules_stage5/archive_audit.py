"""Summarize actual stored DACF artefacts without relabeling unexplored intervals."""

from collections import Counter
from pathlib import Path
import gzip
import json

from dacf_research.prepare_inputs import ROOT, read, write
from partial_modules_stage5.review_sources import H


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return json.load(stream)


if __name__=='__main__':
    rounds={}
    for mode in ('manual','automatic'):
        cases={}
        for path in (ROOT/f'dacf_research/run_{mode}').glob('*/pair_results.json'):
            pairs=read(path)
            cases[path.parent.name]={
                'count':len(pairs),'anchor_brackets':[p['boundary_pts'] for p in pairs],
                'choice_family':dict(Counter(k for r in pairs for k,v in r['selections'].items() if v is not None))}
        rounds['r1_'+mode]=cases
    r2={}
    for branch in ('dacf_round2/replay','dacf_round2/independent_execution/independent','partial_modules_stage5'):
        matched=(ROOT/branch).glob('*/regions.json')
        for path in matched:
            r=read(path)
            signals=read_gzip(path.with_name('region_signals.json.gz'))
            r2[path.parent.name]={'source_sha256':r['source_sha256'],'nominations':len(r['nomination_pts']),
                'regions':len(r['regions']),'by_type':dict(Counter(u['kind'] for u in r['regions'])),
                'triggers':dict(Counter(u['trigger'] for u in r['regions'])),
                'without_interior_sample':sum(not u.get('coarse_interior_pts') for u in r['regions']),
                'nomination_gap_not_admitted':len(r['omitted']),
                'regions_with_prototypes':sum(bool(u.get('left') and u.get('right')) for u in r['regions']),
                'observed_middle_regions':sum(bool(u.get('middle')) for u in r['regions']),
                'appearance_recurrences':len(r['recurrence_candidates']),
                'channel_evidence_rows':sum(len(u['rows']) for u in signals),
                'crossing_state_by_channel':{ch:dict(Counter(v['crossings'][ch].get('status',v['crossings'][ch].get('state','unlabeled'))
                    for v in signals if ch in v['crossings'])) for ch in ('gray','hsv','edge','clip')},
                'scene_confirmation_field_count':sum(u.get('scene_boundary_verification') is not None for u in r['regions'])}
    rounds['r2']=r2
    r3={}
    for path in (ROOT/'dacf_round3/run').glob('*/tree-v2.json'):
        t=read(path); nodes=t['nodes']
        if isinstance(nodes,dict):nodes=list(nodes.values())
        r3[path.parent.name]={'nodes':len(nodes),'reported_kinds':dict(Counter(n['kind'] for n in nodes)),
                              'graph_links':len(t['nonhierarchical_links'])}
    rounds['r3']=r3
    r4=read(ROOT/'dacf_round4/evaluation/summary.json')
    rounds['r4']={'sources':len(r4),'base_nominations':sum(x['nomination_count'] for x in r4),
                  'replay S0 matches R3 E5':{x['case']:x['S0_matches_R3_E5_trajectories'] for x in r4}}
    r5=read(ROOT/'dacf_round5/preset-S0/summary.json')
    rounds['r5']={'S0_preset_sources':list(r5['cases']), 'persistent_prior':'per-session',
                 'installed_acceptance':'dacf_round5/release-fixed/verification.json'}
    r6=read(ROOT/'dacf_round6/dynamic-gain/result.json')
    rounds['r6']={k:r6[k] for k in ('not_in_coarse_pool','old_node','new_child','actual_originals')}
    write(H/'transition_archive.json',rounds)
