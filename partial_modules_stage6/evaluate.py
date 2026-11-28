"""Join frozen assessment roles only after the actual blind and reference replays."""

from collections import Counter
from statistics import median

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage3.observation import iou
from partial_modules_stage6.material import H


def boundary_iou(a,b):
    width=max(0,min(a[1],b[1])-max(a[0],b[0]))
    return width/max(1,max(a[1],b[1])-min(a[0],b[0]))


def match(predicted,allowed):
    candidates=sorted((boundary_iou(u,v),i,j) for i,u in enumerate(predicted) for j,v in enumerate(allowed))
    paired={}; seen=set()
    for s,i,j in reversed(candidates):
        if s>=.30 and i not in paired and j not in seen:
            paired[i]=(j,s);seen.add(j)
    return {'matches':[{'predicted':predicted[i],'reference':allowed[j],'iou':s,
                'reference_fraction_covered':max(0,min(predicted[i][1],allowed[j][1])-max(predicted[i][0],allowed[j][0]))/
                   max(1,allowed[j][1]-allowed[j][0]),
                'ticks_outside_reference':max(0,allowed[j][0]-predicted[i][0])+max(0,predicted[i][1]-allowed[j][1])}
                 for i,(j,s) in paired.items()],
            'unmatched_outputs':[u for i,u in enumerate(predicted) if i not in paired],
            'missed_reference':[v for j,v in enumerate(allowed) if j not in seen],
            'sum_output_width_ticks':sum(b-a for a,b in predicted)}


def nominations_for_first(prediction):
    catalogue=read(ROOT/prediction['initial_index']/'catalog.json')
    return [r|{'pts':frame['pts'],'source_root':frame['frame_id']} for frame in catalogue['frames'] for r in frame['nominations']]


def nominations_for(arm):
    return [r for g in arm['nomination_groups'] for r in g['nominations']]


def role_rows(arm,gold,dimensions):
    native,scaled=dimensions
    w,h=native
    result={}
    for r in gold['roles']:
        lo,hi=r['pts']; roi=r['native_roi']
        circle=[row for row in arm if lo<=row['pts']<=hi]
        meaningful=[row for row in circle if row['dimension_id']!='D6']
        region_proofs=[]
        if roi:
            target=[roi[0]*scaled[0]/w,roi[1]*scaled[1]/h,roi[2]*scaled[0]/w,roi[3]*scaled[1]/h]
            region_proofs=[row for row in meaningful if row.get('applicable_region') and iou(row['applicable_region'],target)>=.2]
        ids=lambda qs:sorted({row['pts'] for row in qs})
        result[r['id']]={'candidate_sources':ids(circle),'non_D6_source_evidence':ids(meaningful),
            'assessed_native_ROI_support_pts':ids(region_proofs) if roi else None,
            'roi_support_channels':sorted({p['dimension_id'] for p in region_proofs})}
    return result


def summarized_roles(roles,gold):
    measures=('candidate_sources','non_D6_source_evidence','assessed_native_ROI_support_pts')
    source=gold['roles']
    declared_roi={x['id'] for x in source if x['native_roi']}
    summaries={k:{'covered':sum(bool(v[k]) for v in roles.values()),'assessable':sum(v[k] is not None for v in roles.values())}
               for k in measures}
    def combinable(combo,pointkey):
        previous=None
        for name in combo:
            address=next((p for p in roles[name][pointkey] if previous is None or p>previous),None)
            if address is None:
                return False
            previous=address
        return True
    summaries['complete_assessed_combinations']={k:{'covered':sum(combinable(chain,k) for chain in gold['relations']
            if k!=measures[2] or set(chain)<=declared_roi),
        'assessable':sum(k!=measures[2] or set(chain)<=declared_roi for chain in gold['relations'])} for k in measures}
    return summaries


def process_local(study,arm,cohort,gold):
    folder=study/cohort; file=folder/f'{arm}-local.json'
    if not file.is_file():return None
    item=read(file); blocks=item['partitions']; elapsed=item['seconds']
    relations=[r for g in blocks for r in g['raw_m5_without_identity']['raw_cross_time']]
    evidence=[r for g in blocks for r in g['raw_m5_without_identity']['same_tick_projected']]
    links=[r for g in blocks for r in g['identity_work_and_competition']['links']]
    attempts=Counter()
    for context in blocks:
        attempts.update(context['identity_work_and_competition']['attempts_by_reason'])
    unresolved=sum(len(g['identity_work_and_competition']['unresolved']) for g in blocks)
    supported_context=sum(c['state']=='heldout_supported' for g in blocks for c in g['recomputed_background_charts'])
    projection=[r for g in blocks for r in g['m5_with_identity']['raw_cross_time'] if r['identity_dependency']]
    # Independently count source endpoints visible in actual evaluated windows and avoid using a semantic identity label.
    role_lookup={r['id']:r for r in gold['roles']}
    actual_combinations=[]
    for chain in gold['relations']:
        if len(chain)!=2 or any(not role_lookup[n]['native_roi'] for n in chain):
            continue
        first,last=(role_lookup[n] for n in chain)
        for r in relations:
            a,b=r['pts']; u,v=r['object_rois']
            if first['pts'][0]<=a<=first['pts'][1] and last['pts'][0]<=b<=last['pts'][1] and \
               iou(first['native_roi'],u)>=.2 and iou(last['native_roi'],v)>=.2:
                actual_combinations.append({'roles':chain,'measure':r['measurement_ref'],'pts':[a,b],
                    'metric_scope':'two ROI endpoints; no physical contact or same instance assertion'})
    return {'seconds':elapsed,'scanned_frames':sum(len(g['pts']) for g in blocks),
        'measured_identity_attempts_by_reason':dict(attempts),'identity_supports':len(links),'unresolved_scopes':unresolved,
        'background_charts_executed':sum(len(g['recomputed_background_charts']) for g in blocks),
        'applicable_heldout_background_charts':supported_context,'direct_feature_pair_builds':sum(g['actual_feature_pair_builds'] for g in blocks),
        'raw_m5_two_endpoint_count':len(relations),'raw_m5_single_endpoint_count':len(evidence),
        'conditional_interpretations_which_use_identity_count':len(projection),'true_task_assessed_M5_combinations':actual_combinations}


def metric_keys(rows):
    def signature(r):
        return r['pts'],r['dimension_id'],r['role'],tuple(r.get('applicable_region') or [])
    return set(map(signature,rows))


def case_result(name,spec,ref):
    base=H/'automatic'/name
    auto=read(base/'auto.json')
    metadata=read(base/'frame_receipts.json')[0]
    dims=metadata['native_size'],metadata['analysis_size']
    first=nominations_for_first(auto)
    arms={'FIRST':first}; artifacts={}
    groups={'GLOBAL':'automatic','AUTO':'automatic','MANUAL':'reference','WRONG':'reference'}
    for arm,kind in groups.items():
        item=read(H/'observations'/kind/name/f'{arm}.json')
        assert item['input_hashes']==auto['index_sha']
        arms[arm]=nominations_for(item)
        artifacts[arm]=item
    scores={}
    cases_eval={arm:role_rows(rows,ref,dims) for arm,rows in arms.items()}
    base_candidate_set=metric_keys(arms['GLOBAL'])
    for arm,rows in arms.items():
        detail=cases_eval[arm]
        supplemental=role_rows(arms['FIRST']+rows,ref,dims)
        if arm!='FIRST':
            runs=process_local(H/'observations'/groups[arm],arm,name,ref)
        else:
            runs=None
        comparison=cases_eval['GLOBAL']
        def missing(axis):
            return [k for k,v in comparison.items() if v[axis] and not detail[k][axis]]
        scores[arm]={'nominations':len(rows),'unique_pts':len({r['pts'] for r in rows}), 'per_reason':dict(Counter(r['dimension_id'] for r in rows)),
            'independent_evidence_assessment':summarized_roles(detail,ref),'role_detail':detail,
            'keep_first_plus_this_arm':summarized_roles(supplemental,ref),
            'new_reason_region_signatures_relative_to_global':len(metric_keys(rows)-base_candidate_set),
            'region_signatures_absent_from_supplement_relative_to_global':len(base_candidate_set-metric_keys(rows)),
            'global_non_D6_roles_absent_from_supplement':missing('non_D6_source_evidence'),
            'actual_nomination_cost':{k:artifacts[arm][k] for k in ('actual_operations','seconds','additional_source_decodes')} if arm!='FIRST' else None,
            'local_measured':runs}
    preflight={arm:read(H/'observations'/arm/name/'run.json') for arm in ('automatic','reference')}
    allowed=ref['bounds']; expected_roi=ref.get('subcoarse_insert')
    potential=[q['pts'] for q in auto['accepted']]
    wrong=artifacts['WRONG']['reference_envelopes']
    return {'source_sha256':spec['sha256'],'measured_coverage_windows':auto['evaluated_windows'],
            'automatic_boundary_score':match(potential,allowed),'wrong_control_boundary_score':match(wrong,allowed),
            'envelope_counts':{'reference':len(allowed),'auto':len(potential),'wrong':len(wrong),
                'auto_closed':len(auto['appearance_scope']['closed']),
                'manual_closed':len(artifacts['MANUAL']['scope']['closed']),
                'wrong_closed':len(artifacts['WRONG']['scope']['closed'])},
            'known_short_insert':expected_roi|{'available_in_initial_pool':expected_roi['visible_only_at_pts'] in auto['pool_pts']}
                                  if expected_roi else None,
            'all_source_arms':scores,'preparation_costs':preflight,'automatic_envelope_cost':auto['cost']}


if __name__=='__main__':
    lock=read(H/'freeze.json');path=ROOT/lock['assessor']['file']
    assert digest(path)==lock['assessor']['sha256']
    blind=read(H/'observations/automatic/run_complete.json')
    manual=read(H/'observations/reference/run_complete.json')
    assert not blind['read_assessor'] and manual['read_assessor'] and blind['cases']==manual['cases']
    assessments=read(path);spec=read(H/'inputs/cases.json')
    write(H/'evaluation.json',{'cases':{k:case_result(k,spec[k],assessments[k]) for k in blind['cases']},
         'limits':'Local ROI threshold IoU 0.20; D6 is temporal access only; no physical conclusion or global exhaustive negative.'})
