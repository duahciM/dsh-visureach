"""Small independent comprehension test of the researchers' actual optional output."""
import json
from dacf_research.prepare_inputs import read,write
from dacf_round7.run import launch,PROTOCOL
from partial_modules_stage1.fixtures import HERE

if __name__=='__main__':
    a=read(HERE/'first_run/paired/A3.json')
    conflict=next(row for row in a['m2'] if row['outcome']=='coappearance_conflict')
    accepted=next(row for row in a['m2'] if row['outcome']=='supported_visual_candidate')
    missing=next(edge for ep in a['m5'] for edge in ep['edges'] if edge['channels']['peer']['state']=='unknown')
    packet={'candidate_examples':[conflict,accepted],'latest_relation_example':missing}
    guide=('A candidate compares previously supplied local episode views with direct multi-reference ROI measurements. '
           'Check actual source roots and coappearance rows. No candidate link assigns physical identity. The relation records are '
           'separate sampled 2D coordinates. A common-coordinate result is conditional on a held-out background fit. '
           'Missing referents split a relation coordinate. Actual source ticks stay available independently of these records.')
    prompt=('You are seeing a new research output. In your own words, give one useful new use, one realistic misuse, and one case where '
            'existing evidence is enough without this output. Based on the real sample packet, explain what would reconcile a repeat-versus-replacement '
            'question, whether either record proves physical sameness, and whether a missing relationship can be interpolated. '
            'Name one actual follow-up operation, its required inputs and what it could establish. '
            'Return JSON with summary and answers, short. Do not copy the guide.\n'+guide+'\n'+json.dumps(packet,ensure_ascii=False))
    out=HERE/'module_agent';out.mkdir()
    write(out/'protocol.json',{'guide':guide,'route':PROTOCOL['route'],'expected':['keep support separate from coappearance','physical identity unknown','missing relation unknown','valid original/ROI comparison follow-up']})
    launch(out/'independent',prompt,plain=True,watchdog=210)
