"""Development DACF qualification with independent structure and photometric scopes."""

from collections import Counter
from itertools import pairwise, product

from dacf_research.field import local_presence
from dacf_round2.regions import anchor, distinguish
from partial_modules_stage1.study import root
from partial_modules_stage5.scopes import derive
from partial_modules_stage6.envelopes import distinctive, investigation_chains


def independent_cell_votes(profile,other):
    support=set(profile['positive_cells'])-set(other['positive_cells'])
    grids=Counter(c.split(':')[0] for c in support)
    return max(grids.values(),default=0)>=2,sorted(support)


def support_at(bank,layout,stamp,a,b,local_mode):
    if local_mode:
        left=layout.group_profile(stamp,a)
        right=layout.group_profile(stamp,b)
        shared=sorted(set(left['positive_cells']) & set(right['positive_cells']))
        ls,lc=independent_cell_votes(left,right);rs,rc=independent_cell_votes(right,left)
        scope={'left':left,'right':right,'shared_supported_cells':shared}
    else:
        old=local_presence(bank,stamp,a,b)
        minimum=bank.config['point_registration']['minimum_local_prototypes']
        ls,lc=distinctive(old,'a',minimum);rs,rc=distinctive(old,'b',minimum)
        scope=old
    return {'rank':stamp,'pts':bank.rows[stamp]['pts'],'root':bank.rows[stamp]['id'],
            'left':ls,'right':rs,'left_exclusive_cells':lc,'right_exclusive_cells':rc,'evidence':scope}


def competition(bank,layout,a,b,edges,new):
    adjacent=[]
    for item in edges:
        aligned=item['aligned']
        adjacent.append({'pts':item['pts'],'root':item['root'],'phase':item['phase'],'photometry':item['photometry'],
          'aligned_scope':{k:aligned.get(k) for k in ('status','mad','overlap_pixels','gain_offset_applied','region_candidates')},
          'unexplained_difference_invalidated':False})
    photo=[p['root'] for p in edges if p['photometry']['supported']]
    motion=[p['root'] for p in edges if p['phase']['motion_supported']]
    cross=[]
    if new:
        for former,later in product(a[-2:],b[:2]):
            measured=layout.compare(former,later)
            ps=bank.metrics_on_pair(former,later)
            cross.append({'root':measured['id'],'source_pair_root':ps['root'],'pts':ps['pts'],
              'scope_supported':measured['supported'],'phase_supported':ps['phase']['motion_supported'],
              'photometry_supported':ps['photometry']['supported'],'aligned':ps['aligned']['status']})
    explanation=(sum(q['scope_supported'] for q in cross)>=3 and (photo or motion or any(
        q['scope_supported'] and (q['phase_supported'] or q['photometry_supported']) for q in cross)))
    blocks=bool(explanation) if new else bool(photo or motion)
    return {'adjacent_measured_photometry':photo,'adjacent_measured_view_motion':motion,
            'cross_observed_structure':cross,'adjacent_measurements':adjacent,
            'withheld_context_envelope':blocks,
            'decision_scope':'same-domain geometry and competing effect' if new else 'legacy any measured competitor',
            'keeps_local_and_photometric_observations':True}


def qualify(bank,layout,windows,group_sets,catalogue,regions,r2,sha,extent,local_mode,competition_mode):
    accepted=[];audited=[]
    chains=investigation_chains(regions)
    for pool,groups in zip(windows,group_sets):
        for left,right in pairwise(groups['groups']):
            a=left['exit_prototypes'] if local_mode else left['prototypes']
            b=right['entry_prototypes'] if local_mode else right['prototypes']
            lo,hi=bank.rows[a[-1]]['pts'],bank.rows[b[0]]['pts']
            query=[p for p in pool if lo<=bank.rows[p]['pts']<=hi]
            linked=[r['id'] for r in regions if max(lo,r['bounds_pts'][0])<min(hi,r['bounds_pts'][1])]
            ref=root('context-envelope-v7',sha,[lo,hi],local_mode,competition_mode)
            info={'id':ref,'pts':[lo,hi],'matched_R2':linked,'groups':[left['id'],right['id']]}
            def abstain(reason):
                audited.append(info|{'state':'unknown','cause':[reason]})
            if not linked:
                abstain('no intersecting existing DACF investigation')
                continue
            first,last=anchor(bank,a[-1],a),anchor(bank,b[0],b)
            comparison=distinguish(bank,first,last,catalogue,r2)
            info.update({'first':first,'last':last,'comparison':comparison})
            if not {'hsv','clip'}&set(comparison['supported_by']):
                abstain('no corroborating contrast in existing appearance channels')
                continue
            probes=[support_at(bank,layout,p,a,b,local_mode) for p in sorted(set(a[-2:]+b[:2]+query))]
            by_rank={r['rank']:r for r in probes}
            causes=[]
            if not (all(by_rank[p]['left'] and not by_rank[p]['right'] for p in a[-2:]) and
                    all(by_rank[p]['right'] and not by_rank[p]['left'] for p in b[:2])):
                causes.append('two separate locally supported prototypes per side not established')
            adjacent=[bank.metrics_on_pair(i,j) for i,j in pairwise(query)]
            if any(bank.rows[j]['seconds']-bank.rows[i]['seconds']>bank.config['maximum_gap_seconds'] for i,j in pairwise(query)):
                causes.append('missing coarse observations within the span')
            state={k:[] for k in ('left','right','both','neither')}
            for p in probes:
                k='both' if p['left'] and p['right'] else 'left' if p['left'] else 'right' if p['right'] else 'neither'
                state[k].append(p['pts'])
            if state['left'] and state['right'] and max(state['left'])>min(state['right']):
                causes.append('local support reverses along the sampled timeline')
            competing=competition(bank,layout,a,b,adjacent,competition_mode)
            if competing['withheld_context_envelope']:
                causes.append('old any photo or motion' if not competition_mode else 'cross-side same-structure with measured view/light alternative')
            info.update({'support_profiles':probes,'support_states':state,'competition':competing,'cause':causes,
                         'state':'unknown' if causes else 'visual_context_change_candidate'})
            if not causes:
                accepted.append({'id':ref,'pts':[lo,hi],'kind':'observed_spatial_context_change','source_sha256':sha,
                    'scope':'observed projection context only; mechanism and physical scene identity remain unknown'})
            audited.append(info)
    return {'accepted':accepted,'audit':audited,'closed_scope':derive(sha,extent,accepted),'R2_chains':chains}
