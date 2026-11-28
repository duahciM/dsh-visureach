"""After a demonstrated packet-interpretation error, test explicit source labels in one new DSH session."""
import json
from dacf_round7.run import launch,PROTOCOL
from dacf_research.prepare_inputs import read,write
from partial_modules_stage2.inputs import H

if __name__=='__main__':
    out=H/'agent_scope_retry';out.mkdir()
    control=read(H/'scope_run/control/result.json')
    paired=read(H/'scope_run/paired/result.json')
    row=next(r for e in control['relations'].values() for r in e if r['scope_dependency'])
    records=[{k:v[k] for k in ('id','kind','source_sha256','effective_status')} for v in paired['ledger_revisions']['after_reload']
             if v['kind'] in ('identity_hypothesis','raw_relation','identity_dependent_relation')]
    obj={'control_source_scoped_edge':{k:row[k] for k in ('pts','source_roots','scope_dependency','interval_state')},
         'separate_source_ledger_rows':records}
    question=('These are two real records from a scoped offline observer. What actually changed after the ledger revision, which source is affected, '
              'and should any raw endpoint measurement be discarded? Could the separately cited local scope or a physical track be inferred from this revision? '
              'Give one source-correct next operation if further visual proof is needed. Return short JSON summary and answers.\n'+json.dumps(obj))
    write(out/'lock.json',{'route':PROTOCOL['route'],'reason':'First output confused one source scope with another source withdrawal; restore source SHA on every excerpt.',
     'rubric':['cross-source scope cannot be affected','only dependent interpretation changes','raw endpoints remain','physically unobserved stays unknown']})
    launch(out/'native',question,plain=True,watchdog=185)
