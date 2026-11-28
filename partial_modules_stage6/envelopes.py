"""Apply a fixed, conservative qualification to saved DACF investigation evidence."""

from collections import Counter
from itertools import pairwise

from dacf_research.field import detect_groups, local_presence
from dacf_round2.investigate import evidence
from dacf_round2.regions import anchor, distinguish
from partial_modules_stage1.study import root
from partial_modules_stage5.scopes import derive


def investigation_chains(regions):
    """Join navigationally adjacent windows while retaining the factual member edges."""
    pending=[]
    for row in sorted(regions,key=lambda r:r['bounds_pts']):
        left,right=row.get('left'),row.get('right')
        roots={k for side in (left,right) if side for k in side['prototype_roots']}
        lo,hi=row['bounds_pts']
        if pending and (lo<pending[-1]['pts'][1] or
                        lo==pending[-1]['pts'][1] and roots & set(pending[-1]['shared_support_roots'])):
            prev=pending[-1]
            prev['pts'][1]=max(prev['pts'][1],hi)
            prev['members'].append(row['id'])
            prev['shared_support_roots']=sorted(roots | set(prev['shared_support_roots']))
        else:
            pending.append({'pts':[lo,hi],'members':[row['id']],'shared_support_roots':sorted(roots),
                            'boundary_status':'investigation_only'})
    return pending


def distinctive(support, own, minimum):
    other='b' if own=='a' else 'a'
    local={cell for cell,n in support[f'{own}_support'].items() if n>=minimum and
           support[f'{other}_support'].get(cell,0)<minimum}
    grid=Counter(c.split(':',1)[0] for c in local)
    return max(grid.values(),default=0)>=2, sorted(local)


def qualify(bank,pool,catalogue,regions,preset,sha,extent):
    accepted=[]; audit=[]; groups=[]
    measured_chains=investigation_chains(regions)
    for ranks in pool:
        discovery=detect_groups(bank,ranks)
        groups.append(discovery)
        for g0,g1 in pairwise(discovery['groups']):
            a,b=g0['prototypes'],g1['prototypes']
            lo,hi=bank.rows[a[-1]]['pts'],bank.rows[b[0]]['pts']
            span=[x for x in ranks if lo<=bank.rows[x]['pts']<=hi]
            ref=root('automatic-context',sha,lo,hi)
            hits=[r['id'] for r in regions if max(lo,r['bounds_pts'][0])<min(hi,r['bounds_pts'][1])]
            bridges=[i for i,c in enumerate(measured_chains) if set(hits)&set(c['members'])]
            left,right=anchor(bank,a[-1],a),anchor(bank,b[0],b)
            row={'id':ref,'bounds_pts':[lo,hi],'investigation_members':hits,'investigation_chains':bridges,
                 'left_anchor':left,'right_anchor':right}
            if not hits:
                audit.append(row|{'state':'unknown','reasons':['no intersecting existing DACF investigation']})
                continue
            difference=distinguish(bank,left,right,catalogue,preset)
            row['prototype_comparison']=difference
            if not {'hsv','clip'}&set(difference['supported_by']):
                audit.append(row|{'state':'unknown','reasons':['no corroborated appearance-group distinction']})
                continue
            probes=[]
            minimum=bank.config['point_registration']['minimum_local_prototypes']
            for rank in sorted(set(a[-2:]+b[:2]+span)):
                support=local_presence(bank,rank,a,b)
                ls,lc=distinctive(support,'a',minimum); rs,rc=distinctive(support,'b',minimum)
                probes.append({'rank':rank,'pts':bank.rows[rank]['pts'],'source_root':bank.rows[rank]['id'],
                   'left_support':ls,'right_support':rs, 'left_cells':lc,'right_cells':rc,'points':support})
            byrank={x['rank']:x for x in probes}
            reasons=[]
            left_good=all(byrank[x]['left_support'] and not byrank[x]['right_support'] for x in a[-2:])
            right_good=all(byrank[x]['right_support'] and not byrank[x]['left_support'] for x in b[:2])
            if not left_good or not right_good:
                reasons.append('two separate locally supported prototypes per side not established')
            adjacency=[bank.metrics_on_pair(x,y) for x,y in pairwise(span)]
            trace=[]
            for x,y,pair in zip(span,span[1:],adjacency):
                if bank.rows[y]['seconds']-bank.rows[x]['seconds']>bank.config['maximum_gap_seconds']:
                    if 'missing interior samples' not in reasons:
                        reasons.append('missing interior samples')
                trace.append({'root':pair['root'],'pts':pair['pts'],'phase':pair['phase'],
                              'photometry':pair['photometry'],'aligned_state':pair['aligned']['status']})
            if any(p['phase']['motion_supported'] for p in adjacency):
                reasons.append('measured camera or image-motion competition')
            if any(p['photometry']['supported'] for p in adjacency):
                reasons.append('measured photometric competition')
            signal=evidence(bank,{'id':ref,'bounds_pts':[lo,hi],'left':left,'right':right},ranks,catalogue)
            list_state={'left':[], 'right':[], 'both':[], 'neither':[]}
            for p in probes:
                l,r=p['left_support'],p['right_support']
                list_state['both' if l and r else 'left' if l else 'right' if r else 'neither'].append(p['pts'])
            if list_state['left'] and list_state['right'] and any(
                y<x for x in list_state['left'] for y in list_state['right']):
                reasons.append('local support direction reverses')
            row.update({'source_probes':probes,'support_by_state':list_state,'source_edge_evidence':trace,
                        'r2_pair_signal':signal,'reasons':reasons,
                        'state':'unknown' if reasons else 'visual_context_change_candidate'})
            if not reasons:
                row['qualified_type']='typed_visual_context_candidate'
                accepted.append({'id':ref,'pts':[lo,hi],'source_sha256':sha,'kind':'observed_spatial_context_change',
                    'scope':'automatic visual support; physical separation, mechanism and exact boundaries remain unverified'})
            audit.append(row)
    return {'accepted':accepted,'all_adjacent_groups':groups,'all_candidates':audit,
        'appearance_scope':derive(sha,extent,accepted),'investigation_chains':measured_chains,
        'recurrence_is_physical_identity':False}
