"""Small native DSH comprehension check with a real output sample and no answer key."""
import json

from dacf_round7.run import launch,PROTOCOL
from dacf_research.prepare_inputs import read,write
from partial_modules_stage2.inputs import H

if __name__=='__main__':
    data=read(H/'scope_run/control/result.json')
    edge=next(x for rows in data['relations'].values() for x in rows if x['axes']['ref_x'] is not None)
    projection=data['scoped']['coappearance'][0]
    rev=read(H/'scope_run/paired/result.json')['ledger_revisions']['after_reload']
    packet={'scoped_relation':{k:edge[k] for k in ('pts','source_roots','scope_dependency','axes','axis_units','interval_state')},
            'same_time_projection':projection,
            'cross_range_query':data['requested_cross_control'],
            'ledger_after_update':[{k:r[k] for k in ('kind','requires','effective_status','affected_by')} for r in rev]}
    question=('You are evaluating tools for local video observation. In your own words distinguish a one-frame observation ID, a local supported identity scope, '
              'and an optional cross-scope hypothesis. Using this actual tool packet, explain how you would validate the source and a local relation. '
              'Give one genuinely new scenario where it helps, one realistic misuse, and a case where you would skip it. '
              'What do the simultaneous regions, unknown gaps and withdrawn dependency imply, and what remains unknowable? '
              'Give one executable type of follow-up with real parameter source (exact source ticks or interval seconds, and ROI as appropriate). '
              'Use brief JSON with summary and answers. Do not treat the task as a physical identity recognition problem.\n'+json.dumps(packet,ensure_ascii=False))
    out=H/'agent';out.mkdir()
    write(out/'lock.json',{'model':PROTOCOL['route'],'actual_case_source':data['source_sha256'],'rubric':[
      'three identity layers and scoped endpoints','coappearance only projection unless independent support',
      'unknown not absence','measurement unchanged after withdrawal','new use, misuse, reasonable skip','legal source follow-up']})
    launch(out/'native',question,plain=True,watchdog=200)
