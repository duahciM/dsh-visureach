"""Aggregate actual costs and compare deterministic decisions across two host runs."""
from collections import Counter
from dacf_research.prepare_inputs import read,write
from partial_modules_stage2.inputs import H


def exact_cost(receipts):
    return {'exact_reads':len(receipts), 'retained_original_png_bytes':sum(r['png_bytes'] for r in receipts),
            'filter_inputs_counted':sum(a['frames_at_input_filter'] or 0 for r in receipts for a in r['decode_attempts']),
            'exact_decode_wall_seconds':sum(r['wall_seconds'] for r in receipts)}


if __name__=='__main__':
    stable={}
    scope_cost={}
    for p in sorted((H/'scope_run').glob('*/result.json')):
        old=read(p);new=read(H/'replay_verification'/p.parent.name/'result.json')
        compare=['scoped','ungated_edges','relations','ungated_relations','axes','read_failures']
        assert all(old[k]==new[k] for k in compare),(p.parent.name,[k for k in compare if old[k]!=new[k]])
        stable[p.parent.name]=compare
        scope_cost[p.parent.name]=old['runtime']|exact_cost(old['receipts'])|{'stored_result_bytes':p.stat().st_size}
    dsh=[]
    totals=Counter()
    for p in sorted(H.glob('agent*/native/result.json')):
        r=read(p);totals.update(r['usage']);dsh.append({'run':p.relative_to(H).as_posix(),'wall_seconds':r['seconds'],'usage':r['usage']})
    arm=read(H/'evaluation/result.json')['equal_originals']['arms']
    cost={'scope_cases':scope_cost,'common_sampling':exact_cost(read(H/'sampling/coarse_receipts.json')),
          'sampling_arms':[{k:r[k] for k in ('route','cold_exact_reads','cold_decode_filter_inputs','bytes_of_selected_originals','wall_seconds')} for r in arm],
          'stage1_A0_A1_A2_A3_preserved':True,'dsh':dsh,'dsh_total_usage':dict(totals),
          'study_directory_bytes_at_summary':sum(p.stat().st_size for p in H.rglob('*') if p.is_file()),
          'scope':'Source decode filters measured per exact read. Common preparation and repeated arm wall times are separate. Allocated volume may differ because input copies use hard links. Provider currency was not audited.'}
    write(H/'replay_verification/verification.json',{'same_semantic_fields':stable,'status':'passed',
         'scope':'Timings, absolute paths and receipt/container representation outside the selected deterministic fields are not compared.'})
    write(H/'evaluation/costs.json',cost)
