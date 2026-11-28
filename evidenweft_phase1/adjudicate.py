"""Combine a blinded content review with immutable provenance and process receipts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from evidenweft_phase1.measure import covered, csv_rows
from evidenweft_phase1.prepare import HERE, read, verify, write

FACT_TYPES = {'supported_direct','supported_cross_frame','reasonable_inference','insufficient',
              'wrong_strong','correct_citation_wrong_reading','correct_content_citation_incomplete'}


def metrics(item):
    return Counter(v for v in item.values())


def text_trace(packet, judgements, rule):
    turns = {}
    for r in packet['rounds']:
        n = str(r['turn'])
        if not r['answer_available']:
            turns[n] = {'answer_available':False,'stop':r['stop']}
            continue
        d = judgements.get(n, {})
        facts = {str(f['id']):f for f in r['facts']}
        classified = d.get('fact_categories',{})
        assert set(classified) == set(facts), (packet['review_id'], n, sorted(facts), sorted(classified))
        assert set(classified.values()) <= FACT_TYPES
        claims = d.get('slots',{})
        chains = d.get('compounds',{})
        for key, fact_ids in claims.items():
            assert key in rule['slots'] and fact_ids
            assert any(key in facts[str(j)]['candidate_roles_from_images']['slots'] for j in fact_ids), (key,fact_ids)
        for key, ids in chains.items():
            assert key in rule['compounds'] and ids
            pts = [p for j in ids for p in facts[str(j)]['cited_pts']]
            assert key in covered(pts,rule)['complete_compounds'], (key,ids,pts)
            assert set(rule['compounds'][key]['slots']) <= set(claims), (key,claims)
        turns[n] = {'answer_available':True,'facts':classified,'fact_counts':dict(metrics(classified)),'verified_roles':claims,'verified_compounds':chains,
                    'first_visible_statement_notes':d.get('notes',[])}
    if all(turns.get(n,{}).get('answer_available') for n in ('1','2')):
        old = {v['id'] for v in packet['rounds'][0].get('questions') or [] if isinstance(v,dict) and v.get('id')}
        q = judgements.get('prior_questions',{})
        assert set(q) == old, (packet['review_id'], q, old)
        assert set(q.values()) <= {'resolved','partial','open'}
        end = {v['id'] for v in packet['rounds'][1].get('questions') or [] if isinstance(v,dict) and v.get('id')}
        aliases = judgements.get('repeat_question_aliases',{})
        assert set(aliases) <= end and set(aliases.values()) <= old
        first, final = turns['1'], turns['2']
        return {'turns':turns,'retained_roles':sorted(first['verified_roles'].keys() & final['verified_roles'].keys()),
            'lost_roles':sorted(first['verified_roles'].keys() - final['verified_roles'].keys()),
            'new_roles':sorted(final['verified_roles'].keys() - first['verified_roles'].keys()),
            'retained_compounds':sorted(first['verified_compounds'].keys() & final['verified_compounds'].keys()),
            'lost_compounds':sorted(first['verified_compounds'].keys() - final['verified_compounds'].keys()),
            'new_compounds':sorted(final['verified_compounds'].keys() - first['verified_compounds'].keys()),
            'prior_question_evaluations':q,'reworded_prior_questions':aliases,'new_questions':sorted(end-old-set(aliases))}
    old = [v['id'] for v in packet['rounds'][0].get('questions') or [] if isinstance(v,dict) and v.get('id')]
    return {'turns':turns, 'prior_questions_without_assessable_followup':old, 'retention':None}


def explain_presence(run, scores, rule):
    result = []
    rounds = {int(n):v for n,v in run['stages'].items()}
    descriptions = rule['slots'] | rule['compounds']
    for key, ref in descriptions.items():
        compound = key in rule['compounds']
        bucket = 'complete_compounds' if compound else 'slots'
        has = lambda data: key in data[bucket]
        present = has(run['source_upper'])
        in_pool = has(run['coarse'])
        selected_by_algorithm = has(run['algorithm_inventory'])
        searchable = has(run['input_catalogue'])
        queried_pts = {p for r in rounds.values() for p in r['metadata_pts'] + r['explicit_requested_pts']}
        query_coverage = covered(queried_pts,rule)
        queried = has(query_coverage)
        first_original = {}
        for n, stage in rounds.items():
            for r in stage['first_new_originals']:
                first_original[r['pts']] = r | {'round':n}
        receipt_pts = set(first_original)
        delivery_coverage = covered(receipt_pts,rule)
        received = has(delivery_coverage)
        first_round = next((n for n,r in sorted(rounds.items()) if has(r['image_evidence_through_round'])), None)
        catalogue_read = has(covered([p for p,r in first_original.items() if r['catalogue']],rule))
        off_catalogue_reread = has(covered([p for p,r in first_original.items() if r['catalogue'] or r['via']=='source_resample'],rule))
        recovered_by_resample = received and not catalogue_read and off_catalogue_reread
        judge_group = 'verified_compounds' if compound else 'verified_roles'
        correct = [int(n) for n,r in scores['turns'].items() if key in r.get(judge_group,{})]
        final = correct and max(correct) == 2
        answer_available = any(r.get('answer_available') for r in scores['turns'].values())
        if not present:
            status = 'source_unclassified'
        elif final:
            status = 'kept_in_final'
        elif correct and scores['turns'].get('2',{}).get('answer_available'):
            status = 'lost_after_correct_round'
        elif correct:
            status = 'correct_initial_final_unavailable'
        elif received and not answer_available:
            status = 'received_answer_unavailable'
        elif received:
            status = 'received_not_correctly_expressed'
        elif run['status'] == 'technical_failure':
            status = 'technical_failure_before_verified_role_receipt'
        elif not in_pool:
            status = 'outside_coarse_pool_unrecovered'
        elif not searchable:
            status = 'absent_from_experimental_catalogue_unrecovered'
        elif not queried:
            status = 'searchable_not_observed_in_queries_or_reads'
        else:
            status = 'queried_no_complete_original_evidence'
        result.append({'case':run['case'],'arm':run['arm'],'repeat':run['repeat'],'blind_id':run['blind_id'],'role':key,
            'role_type':ref.get('type','single_state'),'source':present,'coarse':in_pool,'nominated':selected_by_algorithm,
            'searchable_in_assigned_view':searchable,'metadata_or_explicit_pts_queried':queried,
            'matched_originals_received':received,'first_complete_read_round':first_round,'correct_rounds':correct,
            'kept_in_final':bool(final),'outside_catalogue_source_resample_needed_for_read':recovered_by_resample,'last_observed_stage':status,
            'source_reference_pts': [rule['slots'][p]['witness'] for p in ref['slots']] if compound else [ref['witness']],
            'coarse_witness_pts': run['coarse'][bucket].get(key,[]),
            'nominee_witness_pts':run['algorithm_inventory'][bucket].get(key,[]),
            'searchable_witness_pts':run['input_catalogue'][bucket].get(key,[]),
            'queried_witness_pts':query_coverage[bucket].get(key,[]),
            'received_witness_refs':[{'pts':p,'receipt':first_original[p]['evidence_id'],'round':first_original[p]['round'],
                'tool':first_original[p]['via'],'in_catalogue':first_original[p]['catalogue']} for p in delivery_coverage[bucket].get(key,[])]})
    return result


def main(in_folder, reviewed_file, out):
    verify()
    out.mkdir(parents=True, exist_ok=False)
    judgement = read(reviewed_file)
    reference = read(HERE/'evaluation_only/rubric.json')['cases']
    evidence = read(in_folder/'attribution_inputs.json')
    receipt = {(r['blind_id'],r['attempt']):r for r in evidence}
    after = []
    stages = []
    for case, rule in reference.items():
        packet = read(in_folder/f'{case}-blind-review.json')
        mapping = read(in_folder/f'{case}-review-key.json')
        assert set(judgement[case]) == set(mapping)
        for item in packet:
            blind,n = mapping[item['review_id']].rsplit('-t',1)
            run = receipt[blind,int(n)]
            manual = text_trace(item,judgement[case][item['review_id']],rule)
            manual.update(case=case,arm=run['arm'],repeat=run['repeat'],key=item['review_id'],blind_id=blind,attempt=int(n),status=run['status'])
            after.append(manual)
            stages.extend(explain_presence(run,manual,rule))
    rows = []
    for arm in ('A0','A1','A2'):
        members = [r for r in after if r['arm'] == arm and r['status'] == 'completed']
        if not members:
            initial = [r for r in after if r['arm'] == arm and r['turns'].get('1',{}).get('answer_available')]
            rows.append({'arm':arm,'sessions':0,'scoreable_first_rounds':len(initial),
                'observed_first_round_roles':sum(len(r['turns']['1']['verified_roles']) for r in initial),
                'observed_first_round_compounds':sum(len(r['turns']['1']['verified_compounds']) for r in initial),
                'prior_questions_without_followup':sum(len(r['prior_questions_without_assessable_followup']) for r in initial),
                'causal_comparison':'unavailable; no completed pairs','final_retention':None})
            continue
        summed = defaultdict(int)
        errors = Counter()
        questions = Counter()
        for r in members:
            for n,t in r['turns'].items():
                for channel, name in [('verified_roles','roles'),('verified_compounds','compounds')]:
                    summed[f'round_{n}_{name}'] += len(t[channel])
            for k in ('retained_roles','lost_roles','new_roles','retained_compounds','lost_compounds','new_compounds','new_questions'):
                summed[k] += len(r.get(k,[]))
            errors.update(r['turns']['2']['fact_counts'])
            questions.update(r.get('prior_question_evaluations',{}).values())
        q = {'total':sum(questions.values()), **dict(questions)}
        rows.append({'arm':arm,'sessions':len(members),**summed,'final_fact_labels':dict(errors),'old_questions':q})
    write(out/'scored_sessions.json',after)
    write(out/'agent_summary.json',rows)
    csv_rows(out/'role_stage_attribution.csv',stages)
    answered = sum(any(t.get('answer_available') for t in r['turns'].values()) for r in after)
    print(f'{len(after)} audited sessions; {answered} with adjudicable answers; {len(stages)} traced evidence roles')


if __name__ == '__main__':
    cmd = argparse.ArgumentParser()
    for parameter in ('data','judgements','output'):
        cmd.add_argument(parameter,type=Path)
    a = cmd.parse_args()
    main(a.data,a.judgements,a.output)
