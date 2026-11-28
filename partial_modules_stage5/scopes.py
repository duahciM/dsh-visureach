"""Read-only evidence plumbing for scene context. No scene detector or identity propagation."""

from collections import Counter
from itertools import combinations
import gzip
import json
from pathlib import Path

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage1.study import root
from partial_modules_stage5.review_sources import H

ADMITTED = {'dissolve_visual_regime','hard_cut_visual_regime','observed_spatial_context_change'}


def overlaps(a,b):
    return max(a[0],b[0]) <= min(a[1],b[1])


def derive(sha,extent,verified):
    ordered=sorted(verified,key=lambda r:r['pts'])
    if not ordered:
        return {'closed':[],'open':[{'kind':'both_open_unknown','pts':extent}], 'conflicts':[]}
    groups=[]
    conflicts=[]
    for row in ordered:
        lo,hi=row['pts']
        if not extent[0]<=lo<=hi<=extent[1] or row['source_sha256']!=sha or row['kind'] not in ADMITTED:
            raise ValueError('Expected a reviewed and source-scoped transition bracket.')
        if groups and lo<=groups[-1]['pts'][1]:
            group=groups[-1];group['pts'][1]=max(group['pts'][1],hi);group['ids'].append(row['id'])
        else:
            groups.append({'pts':[lo,hi],'ids':[row['id']]})
    output=[]
    for group in groups:
        if len(group['ids'])>1:conflicts.append(group|{'state':'overlapping_or_touching_boundary'})
    for left,right in zip(groups,groups[1:]):
        if len(left['ids'])>1 or len(right['ids'])>1:
            conflicts.append({'pts':[left['pts'][1],right['pts'][0]],'ids':left['ids']+right['ids'],
                              'state':'adjacent_gap_ambiguous_no_context'})
            continue
        limits=[left['pts'][1],right['pts'][0]]
        output.append({'id':root('appearance-episode',sha,limits,left['ids'],right['ids']),
                       'pts':limits,'endpoint_policy':'strictly_interior_only',
                       'left_transition':left['ids'][0],'right_transition':right['ids'][0],
                       'scope':'local visual context only','physical_scene_identity':'unknown'})
    edges=[{'kind':'open_start','pts':[extent[0],groups[0]['pts'][0]]},
           {'kind':'open_end','pts':[groups[-1]['pts'][1],extent[1]]}]
    return {'closed':output,'open':edges,'conflicts':conflicts}


def context(stamp,index):
    return next((s['id'] for s in index['closed'] if s['pts'][0]<stamp<s['pts'][1]),None)


def priority(a,b,scope,explicit=False):
    if explicit:return 'explicit_cross_context_allowed'
    x,y=context(a,scope),context(b,scope)
    if x is None or y is None:return 'ordinary_unknown_or_open_context'
    return 'same_context_priority' if x==y else 'cross_bounded_context_deferred'


def compare_jobs(pairs,scope,explicit=False):
    bins=Counter(priority(a,b,scope,explicit) for a,b in pairs)
    return {'total':len(pairs),'priority':dict(bins),'removed_jobs':0}


def compressed_signal(path):
    if not path.is_file():return {}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        signals=json.load(f)
    return {r['region_id']:{'crossings':r['crossings'],'path_channels':list(r['paths']),
            'direct_group_roots':r['direct_group_pair_roots']} for r in signals}


def review_case(case,reviews):
    folder=H/'source_review'/case
    raw=read(folder/'receipts.json')
    clock=raw['clock'];extent=[clock[0]['pts'],clock[-1]['pts']]
    sha=digest(ROOT/raw['source'])
    assert sha==raw['sha256']
    if case=='real_return':
        history=H/'real_round2'
    elif case=='camera':
        history=None
    else:
        history=ROOT/'dacf_round2/replay'/case
    plans=read(history/'regions.json') if history else {'regions':[],'nomination_pts':[],'source_sha256':sha}
    assert sha==plans['source_sha256']
    signals=compressed_signal(history/'region_signals.json.gz') if history else {}
    tree=ROOT/f'dacf_round3/run/{case}/tree-v2.json'
    nodes=read(tree)['nodes'] if tree.is_file() else []
    if isinstance(nodes,dict):nodes=list(nodes.values())
    bank={r['pts']:r for r in raw['receipt_rows']}
    admitted=[]; all_reviewed=[]
    for row in reviews:
        refs=[bank[p] for p in row['witness_pts']]
        for ref in refs:assert digest(Path(ref['path']))==ref['png_sha256']
        linked=[r['id'] for r in plans['regions'] if overlaps(r['bounds_pts'],row['pts'])]
        kind=row['kind']
        reviewed=row|{'id':root('reviewed-boundary',sha,row['pts'],kind),'confirmed_by':'manual_visual_review',
            'source_sha256':sha,'source_roots':[r['id'] for r in refs],
            'dacf_candidates_intersecting':linked,'reviewed_matches_automated_boundary_claim':False}
        all_reviewed.append(reviewed)
        if kind in ADMITTED:
            admitted.append(reviewed)
    inventory=[]
    for r in plans['regions']:
        inventory.append({'id':r['id'],'pts':r['bounds_pts'],'trigger':r['trigger'],'kind':r['kind'],
            'appearance_comparison':r.get('distinction',{}).get('supported_by',[]),
            'anchor_support_windows':[{k:a[k] for k in ('pts','prototype_pts','prototype_roots')} if a else None
                                      for a in (r.get('left'),r.get('right'))],
            'weak_cues':r.get('weak_cues',[]),'intermediate_children':r.get('children',[]),
            'signals':signals.get(r['id']),'automatic_boundary':'unknown_without_independent_scene_evidence'})
    # Legacy investigation files have no typed independently verified scene boundary contracts.
    automatic=derive(sha,extent,[])
    reviewed=derive(sha,extent,admitted)
    nominees=plans['nomination_pts']
    frozen_clock=[r['pts'] for r in clock]
    contiguous=list(zip(frozen_clock,frozen_clock[1:]))
    # Explicit retrospective opportunities use the exact same first nominations in each arm.
    all_anchor_jobs=list(combinations(nominees,2))
    return {'source_sha256':sha,'source_extent_pts':extent,'manual_review':all_reviewed,
            'dacf_inventory':inventory,'r3_node_count':len(nodes),
            'first_nomination_pts':nominees,'automatic_scope':automatic,'reviewed_scope':reviewed,
            'posthoc':{'already_available_exact_original_count':len(raw['receipt_rows']),
                       'context_nominee_pts':{s['id']:[p for p in nominees if context(p,reviewed)==s['id']] for s in reviewed['closed']},
                       'native_adjacent_address_jobs':{label:compare_jobs(contiguous,idx) for label,idx in (
                            ('ordinary_address_scope',automatic),('reviewed_scope',reviewed))},
                       'optional_nominee_pairs':{label:compare_jobs(all_anchor_jobs,idx,explicit) for label,idx,explicit in (
                            ('ordinary_address_scope',automatic,False),('reviewed_scope',reviewed,False),
                            ('explicit_question_scope',reviewed,True))}}}


if __name__=='__main__':
    frozen=read(H/'freeze.json')
    for name,signature in frozen['protocol_code_and_review'].items():assert digest(H/name)==signature
    review=read(H/'reviewed_bounds.json')
    cases={name:review_case(name,review[name]) for name in ('detour','cut','camera','real_return')}
    write(H/'scene_scope.json',{'cases':cases,'algorithm_state':'diagnostic only; scene scope never changes eligibility for source reads'})
