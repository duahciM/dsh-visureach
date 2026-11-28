"""Produce the separate repaired research surface from the pinned r5.1 entry."""
from pathlib import Path
from dacf_research.prepare_inputs import ROOT,read,digest

H=Path(__file__).resolve().parent
BASE=ROOT/'dacf_round5/release-fixed/installed/package'


def main():
    data=(BASE/'entry.mjs').read_text('utf8')
    def sub(a,b):
        nonlocal data
        assert data.count(a)==1,(a,data.count(a))
        data=data.replace(a,b,1)
    sub("import { createHash", "import { createHash") if "import { createHash" in data else None
    sub("import { randomUUID } from 'node:crypto'", "import { randomUUID, createHash } from 'node:crypto'\nimport {join} from 'node:path'\nimport {treeJournal} from './journal.mjs'")
    sub("from './baseline/index.js'","from '../dacf_round5/release-fixed/installed/package/baseline/index.js'")
    sub("  const clock=new Map","  const journal=hierarchical&&config.cacheDir?treeJournal(join(config.cacheDir,'amr-v1.sqlite')):null\n  const planHash=plan?createHash('sha256').update(readFileSync(config.planPath)).digest('hex'):null\n  const scopeKey=exec=>String(exec.agent.id)\n  const baseMap=()=>new Map((plan?.nodes??[]).map(n=>[n.id,structuredClone(n)]))\n  const verifySource=()=>{if(plan&&createHash('sha256').update(readFileSync(config.authorizedSource)).digest('hex')!==plan.sha256)throw new Error('STALE_SOURCE. Restore original bytes or prepare a new plan.')}\n  const clock=new Map")
    sub("    const binding=bindings.get(exec.agent.session)","    verifySource()\n    const binding=bindings.get(exec.agent.session)")
    sub("          const opened=await inherited(tool.name,args,exec,'model')","          verifySource()\n          const opened=await inherited(tool.name,args,exec,'model')")
    sub("          if(!bindings.has(exec.agent.session)) bindings.set(exec.agent.session,new Map((plan?.nodes??[]).map(n=>[n.id,structuredClone(n)])))","          if(!bindings.has(exec.agent.session)) {\n            const map=baseMap();if(journal)journal.restore(scopeKey(exec),plan.sha256,planHash,map)\n            bindings.set(exec.agent.session,map)\n          }")
    sub("      const ids=[]","      const ids=[], durableChildren=[]")
    sub("        map.set(id,n);ids.push(brief(n));node.children.push(id)","        durableChildren.push(n);ids.push(brief(n))")
    sub("      node.calculation_refs=[...new Set([...(node.calculation_refs??[]),...refs])]",
        "      const version=journal?.add(scopeKey(exec),plan.sha256,planHash,{parent:node.id,children:durableChildren,refs})??null\n"
        "      for(const n of durableChildren){map.set(n.id,n);node.children.push(n.id)}\n"
        "      node.calculation_refs=[...new Set([...(node.calculation_refs??[]),...refs])]")
    sub("host_status:faults.length?", "version,host_status:faults.length?")
    with (H/'persistent.mjs').open('x',encoding='utf-8') as f:f.write(data)


if __name__=='__main__':main()
