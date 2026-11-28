"""Read separately sealed labels, score all four factors and compare equal-volume windows."""

from collections import Counter
from pathlib import Path

from dacf_research.prepare_inputs import ROOT,digest,read,write
from partial_modules_stage6.evaluate import match, nominations_for_first, role_rows, summarized_roles
from partial_modules_stage7.inputs import H


def verdicts(cases,truth):
    output={}
    for name in cases:
        d=read(H/'automatic'/name/'results.json')
        comparisons=[r for window in d['adjacent_group_diagnostic'] for r in window]
        condition={}
        for branch,arm in d['arms'].items():
            report=match([q['pts'] for q in arm['accepted']],truth[name]['bounds'])
            crossing=[]
            for scope in arm['closed_scope']['closed']:
                lo,hi=scope['pts']
                for a,b in truth[name]['bounds']:
                    if lo<a and b<hi:
                        crossing.append({'scope':scope['pts'],'enclosed_missing_reference_boundary':[a,b]})
            condition[branch]=report|{'closed_scopes':arm['closed_scope']['closed'],'internal_reference_boundaries_swallowed':crossing,
                'qualification_causes':dict(Counter(c for r in arm['audit'] for c in r['cause']))}
        output[name]={'answers':condition,'manual_bounds':truth[name]['bounds'],
            'old_stable_groups':[len(x['groups']) for x in d['old_groups_scope']],
            'local_verified_groups':[len(x['groups']) for x in d['development_local_groups']],
            'old_R1_rejection_reasons':dict(Counter(r['reason'] for w in d['old_groups_scope'] for r in w['rejected'])),
            'adjacent_measured':len(comparisons),'gray_within_threshold':sum(not r['above_old_within_gray'] for r in comparisons),
            'geometry_verified':sum(q['local_direct_supported'] for q in comparisons),
            'gray_split_but_local_geometry_verified':sum(q['local_direct_supported'] and q['above_old_within_gray'] for q in comparisons),
            'cost':d['cost']}
    return output


def window_case(name,reference,spec):
    blind=H/'windows/blind'/name
    review=H/'windows/reference'/name
    options={}
    for file in [*blind.glob('*.json'),*review.glob('*.json')]:
        if file.name not in ('summary.json','common_pair_facts.json'):
            options[file.stem]=(read(file),file.parent)
    if spec.get('stage6'):
        old=read(ROOT/spec['stage6']/'auto.json')
        first=nominations_for_first(old)
        image=read(ROOT/spec['stage6']/'frame_receipts.json')[0]
    else:
        first=[r for item in read(H/'automatic'/name/'first_control_index.json') for r in item['nominations']]
        image=read(H/'automatic'/name/'frame_receipts.json')[0]
    dims=(image['native_size'],image['analysis_size'])
    def compute(rows):
        values=role_rows(rows,reference,dims)
        return {'by_role':values,'numbers':summarized_roles(values,reference)}
    def coverage(x):
        return {k for k,v in x['by_role'].items() if v['assessed_native_ROI_support_pts']}
    scores={'FIRST':compute(first)}
    for key,(report,folder) in sorted(options.items()):
        shared=read(folder/'common_pair_facts.json')[report['pair_channel']]
        primary=[r for r in report['nominations'] if r['dimension_id'] in ('D1','D3')]
        positive=primary+shared['rows']
        total=compute(positive)
        joined=compute(first+positive)
        types=Counter(r['dimension_id'] for r in report['nominations'])
        scores[key]={'measurements_processed':len(report['measurements']),'executed_phase_calls':report['executed_phase_calls'],
            'same_channel_secondary_relations':len(shared['rows']),'shared_adjacent_pair_calls':shared['actual_pair_calls'],
            'independent_reason_dimensions':dict(types),'foreground_only':compute(primary),
            'with_shared_pair_channels':total,'first_preserved_plus_branch':joined,
            'new_assessed_ROI_roles_over_first':sorted(coverage(joined)-coverage(scores['FIRST'])),
            'structure_supported_roots':len({r['calculation_id'] for r in report['nominations'] if r['dimension_id']=='structure'}),
            'real_2D_reference_relations':types['relation'],
            'closed_scopes':report['scope']['closed'] if report['scope'] else [],
            'unexplained_records':types['unknown']}
    targets={}
    for t in (4400,9200,9400) if name=='reel' else ():
        def vectors(key):
            r,_=options[key]
            return [{'pts':q['pts'],'refs_wanted':q['comparison_pts_requested'],
                'registered_refs':[z['pts'] for z in q['fit_refs']],
                'roots':[z['root'] for z in q['fit_refs']],
                'projected_M4_candidate_bboxes':[z['bbox'] for z in q['candidates']],
                'comparison_root':q['rank_independent_root'],'state':q['state']}
                    for q in r['measurements'] if t==q['pts']]
        targets[t]={label:vectors(label) for label in ('A','B','B_back','B_wide','B_split','C','D','E','WRONG')}
    return {'scores':scores,'budgets':read(blind/'summary.json')['checks'],
       'execution_speed':{'blind':read(blind/'summary.json'),'reference':read(review/'summary.json')},
       'targeted_replay_after_scoring':targets}


if __name__=='__main__':
    frozen=read(H/'freeze.json')
    assert all(digest(ROOT/p)==v for p,v in frozen['assessor_files'].items())
    assert not read(H/'windows/blind/completed.json')['assessor_opened']
    assert read(H/'windows/reference/completed.json')['assessor_opened']
    specs=read(H/'inputs/cases.json')
    truth=read(ROOT/'partial_modules_stage6/assessment/sealed.json')
    supplements=read(H/'assessment_extra.json')
    truth.update({k:supplements[k] for k in ('new_recording','reverse_transfer')})
    contour=verdicts(specs,truth)
    timings={k:window_case(k,truth[k],specs[k]) for k in ('reel','real_return','new_recording')}
    output={'boundary_cases':contour,'window_cases':timings,'heldout_sources':['new_recording','reverse_transfer'],
       'boundary_IoU_policy':.30,'local_ROI_IoU_policy':.20,
       'independent_R1_and_photometry_ablations_keep_old_reference':True}
    write(H/'evaluation.json',output)
