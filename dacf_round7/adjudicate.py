"""Recheck disputed capability claims independently, blinded to the version."""
import json
from random import Random

from dacf_research.prepare_inputs import write
from dacf_round7.compact_review import selected,patches
from dacf_round7.run import H,launch

if __name__=='__main__':
    data=selected(); refs=patches(); rows=[]
    for phase in ('original','revised'):
        tools={r['tool']:r for r in data[phase,'tools-2']['answer']['tools']}
        for t in ('amr_query','amr_refine','change_recheck'):
            r=tools[t]
            rows.append((phase,t,data[phase,'tools-2']['path'],
                         r.get('limit',''),r.get('supports',''),r['scenarios'][0]['fit']))
    Random(7025).shuffle(rows)
    base=H/'adjudication';base.mkdir()
    write(base/'mapping.json',{str(n):row[:3] for n,row in enumerate(rows,1)})
    questions=[{'id':n,'tool':row[1],'answer_fields':row[3:]} for n,row in enumerate(rows,1)]
    p=('Answer only six compact marks. Using the tool facts below, classify whether the observer answer states an actual capability falsehood '
       'or leaves a material output unspecified. Mark X for explicit false, U for unspecified, S for sufficiently correct. '
       'An unmentioned output is not an explicit rejection. Assess whether a node ID is needed for node detail and whether query/refine/recheck return actual pictures. '
       'Return {"summary":"short","scores":{"1":"X|U|S",...},"notes":{"1":"at most one short reason",...}}. Each of six IDs must be represented.\n'
       +json.dumps({t:refs[t] for t in ('amr_query','amr_refine','change_recheck')})+'\n'+json.dumps(questions))
    launch(base/'masked',p,plain=True,watchdog=170)
