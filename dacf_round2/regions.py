"""Nomination-led discovery. Channel-specific evidence and unknowns stay separate."""
from itertools import combinations, product
from statistics import median


def make_anchor(bank, pool, seed, side, policy):
    t=bank.rows[seed]['seconds']
    selected=[q for q in pool if abs(bank.rows[q]['seconds']-t)<=policy['prototype_neighborhood']['radius_s']
              and (q<=seed if side=='left' else q>=seed)]
    limit=policy['prototype_neighborhood']['samples']
    p=selected[-limit:] if side=='left' else selected[:limit]
    return anchor(bank,seed,p)


def anchor(bank, seed, prototypes):
    return {'seed':seed,'pts':bank.rows[seed]['pts'], 'root':bank.rows[seed]['id'],'prototypes':prototypes,
            'prototype_pts':[bank.rows[q]['pts'] for q in prototypes],
            'prototype_roots':[bank.rows[q]['id'] for q in prototypes],
            'distinct_rgb':len({bank.rows[q]['rgb_sha256'] for q in prototypes})}


def local_links(left, right, catalogue):
    recorded=[]
    for first,second in ((left,right),(right,left)):
        row=catalogue.get(first['pts'],{})
        for v in row.get('nominations',[]):
            roots=set(v.get('comparison_frame_ids',[]))
            if v['dimension_id']=='D2' and {first['root'],second['root']} <= roots:
                recorded.append({'reported_at_pts':first['pts'],'measurement_scope':v.get('confidence_scope'),
                                 'roi':v.get('applicable_region'),'roots':v['evidence_root_ids'],'role':v['role']})
    return recorded


def distinguish(bank,left,right,catalogue,policy):
    if max(left['prototypes']) >= min(right['prototypes']):
        return {'channels':{},'local_d2':[],'state':'unknown_overlapping_reference_windows','supported_by':[]}
    def summary(pairs, ch):
        values=[bank.metric(a,b)[ch] for a,b in pairs]
        return {'values':values,'median':median(values) if values and all(v is not None for v in values) else None}
    channels={}
    passed=[]
    for ch, floor in policy['appearance_margin'].items():
        a=summary(list(combinations(left['prototypes'],2)),ch)
        b=summary(list(combinations(right['prototypes'],2)),ch)
        ab=summary(list(product(left['prototypes'],right['prototypes'])),ch)
        identity=bank.metric(left['seed'],right['seed'])[ch]
        medians=(a['median'],b['median'],ab['median'],identity)
        margin=None if any(x is None for x in medians) else ab['median']-max(a['median'],b['median'])
        support=margin is not None and margin>=floor and identity>=0.5*ab['median']
        channels[ch]={'a_internal':a,'b_internal':b,'a_b':ab,'seed_distance':identity,
                      'relative_margin':margin,'locked_minimum':floor,'state':'different_measured_appearance' if support else
                      'unassessable' if margin is None else 'insufficient_distinction'}
        if support:
            passed.append(ch)
    d2=local_links(left,right,catalogue)
    if d2:
        passed.append('frozen_registered_D2')
    return {'channels':channels,'local_d2':d2,'supported_by':passed,
            'state':'distinguishable_sampled_appearance' if passed else 'unknown_insufficient_distinction'}


def window_checks(bank, span):
    raw=[bank.metrics_on_pair(a,b) for a,b in zip(span,span[1:])]
    flags=[]
    for a,b,edge in zip(span,span[1:],raw):
        if edge['metrics']['gray']<=bank.params.activity_mad:
            continue
        reasons=[]
        if min(bank.rows[a]['sharpness'],bank.rows[b]['sharpness'])<bank.params.min_sharpness:
            reasons.append('low_sharpness_with_observed_pixel_change')
        if not edge['phase']['registration_valid']:
            reasons.append('phase_unreliable_with_observed_pixel_change')
        if reasons:
            flags.append({'source_pair_pts':edge['pts'],'reason':reasons,'root':edge['root'],'gray_mad':edge['metrics']['gray']})
    return flags


def try_middle(bank, inside, left, right, catalogue, policy):
    if len(inside)<5:
        return None
    for three in zip(inside,inside[1:],inside[2:]):
        if three[0]<=max(left['prototypes']) or three[-1]>=min(right['prototypes']):
            continue
        coherent=[ch for ch,floor in policy['appearance_margin'].items()
                  if all((v:=bank.metric(a,b)[ch]) is not None and v<=floor/2 for a,b in combinations(three,2))]
        if not coherent:
            continue
        middle=anchor(bank,three[1],list(three))
        ac=distinguish(bank,left,middle,catalogue,policy)
        cb=distinguish(bank,middle,right,catalogue,policy)
        if ac['supported_by'] and cb['supported_by']:
            return middle,coherent
    return None


def discover(bank,pool,catalogue,policy):
    seeds=[q for q in pool if bank.rows[q]['pts'] in catalogue]
    nominated={q:i for i,q in enumerate(seeds)}
    regions=[]
    omitted=[]
    edges=[]
    anchors=[]

    def area(left,right,trigger,depth=0):
        lo,hi=(r['seed'] for r in (left,right))
        span=[q for q in pool if lo<=q<=hi]
        comparison=distinguish(bank,left,right,catalogue,policy)
        weak=window_checks(bank,span)
        # A parent always retains the measured comparison and the sample ranges.
        base={'id':f'rg{len(regions)+1:04d}','bounds_pts':[left['pts'],right['pts']],'trigger':trigger,
              'coarse_interior_pts':[bank.rows[q]['pts'] for q in span[1:-1]], 'left':left,'right':right,
              'distinction':comparison,'weak_cues':weak,'depth':depth,'continuity':'unknown','children':[]}
        middle=try_middle(bank,span[1:-1],left,right,catalogue,policy) if depth<2 else None
        if middle:
            pivot,similar=middle
            base.update(kind='nested_parent',middle=pivot,middle_local_similarity_channels=similar)
            regions.append(base)
            for a,b in [(left,pivot),(pivot,right)]:
                child=area(a,b,'witnessed_intermediate_state',depth+1)
                assert child is not None
                base['children'].append(child['id'])
            return base
        if comparison['supported_by']:
            base['kind']='paired_investigation'
            regions.append(base)
            return base
        if weak and trigger!='registered_D2_adjacency':
            base['kind']='unknown_investigation'
            regions.append(base)
            return base
        omitted.append(base|{'reason':'no_distinct_pair_or_supported_unknown_cue'})
        return None

    for a,b in zip(seeds,seeds[1:]):
        left=make_anchor(bank,pool,a,'left',policy); right=make_anchor(bank,pool,b,'right',policy)
        d2=local_links(left,right,catalogue)
        interior=any(a<q<b for q in pool)
        if interior or d2:
            item=area(left,right,'catalogue_gap' if interior else 'registered_D2_adjacency')
            if item and item['kind'] in {'paired_investigation','nested_parent'}:
                edges.append(item)
                anchors.extend([left,right])
    if seeds:
        for head,a,b in [(True,pool[0],seeds[0]),(False,seeds[-1],pool[-1])]:
            if a==b: continue
            span=[q for q in pool if a<=q<=b]
            weak=window_checks(bank,span)
            fixed=make_anchor(bank,pool,b if head else a,'right' if head else 'left',policy)
            regions.append({'id':f'rg{len(regions)+1:04d}','bounds_pts':[bank.rows[q]['pts'] for q in (a,b)],
                'kind':'one_sided_unknown','trigger':'missing_preceding_reference' if head else 'missing_following_reference',
                'left':None if head else fixed,'right':fixed if head else None,'weak_cues':weak,
                'coarse_interior_pts':[bank.rows[q]['pts'] for q in span[1:-1]],'continuity':'unknown','children':[]})
    elif pool:
        regions.append({'id':'rg0001','bounds_pts':[bank.rows[pool[0]]['pts'],bank.rows[pool[-1]]['pts']],
                        'kind':'two_sides_unknown','trigger':'no_initial_nominations','continuity':'unknown','children':[]})
    parents=[]
    for l,r in zip(edges,edges[1:]):
        if l['right']['seed']!=r['left']['seed']:
            continue
        parents.append({'id':f'chain{len(parents)+1:04d}','kind':'adjacent_state_chain', 'bounds_pts':[l['bounds_pts'][0],r['bounds_pts'][1]],
          'middle_pts':l['right']['pts'],'children':[l['id'],r['id']],'continuity':'unknown',
          'note':'A to middle and middle to B have separate comparisons. No direct A-to-B continuity assumption.'})
    states={a['seed']:a for a in anchors}
    recurrences=[]
    adjacency={(x['left']['seed'],x['right']['seed']) for x in edges}
    ordered=sorted(states)
    for q,r in combinations(ordered,2):
        if nominated[r]-nominated[q]<2 or not any(q<=a<b<=r and (a,b)!=(q,r) for a,b in adjacency):
            continue
        left,right=states[q],states[r]
        channels={}
        for ch,limit in policy['appearance_margin'].items():
            direct=bank.metric(q,r)[ch]
            group=[bank.metric(a,b)[ch] for a,b in product(left['prototypes'],right['prototypes']) if a!=b]
            if not group or direct is None or not all(d is not None for d in group) or direct>limit or median(group)>limit:
                continue
            witness=next((s for s in ordered if q<s<r and
                (distance_a:=bank.metric(q,s)[ch]) is not None and distance_a>limit and
                (distance_b:=bank.metric(s,r)[ch]) is not None and distance_b>limit),None)
            if witness is not None:
                channels[ch]={'direct':direct,'group_median':median(group),'limit':limit,
                              'intervening_contrast_pts':bank.rows[witness]['pts'],'contrast_root':bank.rows[witness]['id']}
        if channels:
            recurrences.append({'a_pts':left['pts'],'return_pts':right['pts'],'anchors':[left,right],'channels':channels,
                                'candidate_type':'similar_appearance_returns','physical_identity':'unknown'})
    return {'regions':regions,'region_chains':parents,'omitted':omitted,'recurrence_candidates':recurrences,
            'nomination_pts':[bank.rows[q]['pts'] for q in seeds]}
