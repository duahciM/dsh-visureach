"""Rebuild stop-display metadata without rewriting the first native experiment."""
from dacf_research.prepare_inputs import read, write
from dacf_round4.offline import HERE


def main():
    base=HERE/'agent'
    saved=read(base/'public/regions.json')
    state=read(HERE/'offline/rolling/initial_stops.json')
    for n in saved['nodes']:
        for channel,s in n['stop'].items():
            if n['id']=='source':
                s['S1_state']='not_evaluated_on_aggregate_consult_paired_leaves'
                s['eligible_descendant_leaf_count']=len({row['root'] for row in state[channel]['candidates'] if row['local']['eligible']})
                s.pop('eligible_local_count',None)
            if n['kind']=='route_view' or (n.get('parent_id') or '').startswith('E5-'):
                s['S1_state_scope']='parent_initial_leaf_only; this cached branch was not re-evaluated'
    assert any(s.get('eligible_descendant_leaf_count',0)>0 for n in saved['nodes'] if n['id']=='source' for s in n['stop'].values())
    write(base/'regions_scope_corrected.json',saved)


if __name__=='__main__': main()
