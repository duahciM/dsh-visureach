import assert from 'node:assert/strict'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { performance } from 'node:perf_hooks'
import { installResearch } from './probe.mjs'
const dir='dacf_round4/index_benchmark'
await mkdir(dir)
const prior=JSON.parse(await readFile('dacf_round4/agent/engineering/config.json','utf8'))
const regionData=JSON.parse(await readFile(prior.regionPlan,'utf8'))
const requests=regionData.nodes.filter(n=>n.kind==='frozen_r2_region').slice(0,12).map(n=>({node_id:n.id,page_size:12}))
const arms={}
const med=values=>values.sort((a,b)=>a-b)[Math.floor(values.length/2)]
for(const arm of ['A1','A2']) {
  const registered=new Map()
  const start=performance.now()
  installResearch({tools:{register:t=>registered.set(t.name,t)}},{...prior,arm,controlLog:join(dir,arm+'.jsonl')})
  const setup=performance.now()-start
  const tool=registered.get(arm==='A1'?'investigation_regions':'amr_query')
  const metrics=[]
  for(const request of requests) {
    const from=performance.now()
    const v=await tool.execute(request)
    metrics.push({id:request.node_id,pts:v.originals.items.map(r=>r.pts),children:v.child_ids??null,ms:performance.now()-from,
      encoded_bytes:Buffer.byteLength(JSON.stringify(v))})
  }
  arms[arm]={setup_ms:setup,lookup_ms_including_research_event_write:med(metrics.map(r=>r.ms)),source_decodes:0,
    direct_id_lookup:true,metrics}
}
assert.deepEqual(arms.A1.metrics.map(x=>x.pts),arms.A2.metrics.map(x=>x.pts))
await writeFile(join(dir,'result.json'),JSON.stringify({same_bottom_layer_evidence:true,scope:'Saved node ID lookup, not an Agent or search-quality test.',arms},null,2),{flag:'wx'})
