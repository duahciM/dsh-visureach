import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {mkdir,readFile,writeFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {treeJournal} from './journal.mjs'
import {apply} from './discovery_corrected.mjs'
const d=resolve('dacf_round6/index-compare');await mkdir(d)
const prepared=resolve('dacf_round5/inputs/fixture'),p=await readFile(resolve(prepared,'plan.json')),plan=JSON.parse(p)
const nodes=new Map(plan.nodes.map(n=>[n.id,n]))
const seq=treeJournal(resolve('dacf_round6/engineering/cache/amr-v1.sqlite')).restore('r6-durable-session',plan.sha256,
 createHash('sha256').update(p).digest('hex'),nodes)
assert.equal(seq,2)
const snapshot=resolve(d,'shared_snapshot.json')
await writeFile(snapshot,JSON.stringify({...plan,nodes:[...nodes.values()]}))
const source=resolve(prepared,'source.mp4')
const bind=arm=>{
 const tools=new Map(),exec={agent:{id:'index-'+arm,session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
 apply({effect:f=>f(),get:()=>null,tools:{register:t=>tools.set(t.name,t)},deepseekLlmApiExtensions:{register:()=>{}}},{
  arm,authorizedSource:source,planPath:snapshot,cacheDir:resolve(d,arm),pythonPath:resolve('.venv/Scripts/python.exe'),
  controlLog:resolve(d,arm+'.jsonl'),requestAudit:resolve(d,arm+'-request.jsonl')})
 return {arm,call:(n,a)=>tools.get(n).execute(a,exec)}
}
const arms=[bind('A1'),bind('A2')]
const outcomes=[]
for(const {arm,call} of arms){
 await call('video_open',{path:source})
 const method=arm==='A1'?'investigation_regions':'amr_query'
 // No supplied node ID. Start with source-wide provider inventory and the same page size.
 const visible=await call(method,{start:0,end:14,page_size:8})
 const first=visible.items.find(r=>r.kind==='source')
 const scopes=[]
 if(arm==='A2'){
  const r=await call(method,{node_id:first.id,detail:'node'})
  for(const id of r.children)if(nodes.get(id).kind==='session_temporal')
   scopes.push(await call(method,{node_id:id,detail:'node'}))
 }else{
  let page=visible
  const ids=[]
  while(true){
   ids.push(...page.items.filter(r=>r.kind==='session_temporal').map(r=>r.id))
   if(page.next_cursor===null)break
   page=await call(method,{start:0,end:14,page_size:8,cursor:page.next_cursor})
  }
  for(const id of ids)scopes.push(await call(method,{node_id:id,detail:'node'}))
 }
 const log=(await readFile(resolve(d,arm+'.jsonl'),'utf8')).trim().split('\n').map(JSON.parse)
 outcomes.push({arm,branch_refs:scopes.map(r=>({id:r.id,pts:r.original_addresses.items,known_refs:r.refs})),
   metadata_queries:log.filter(r=>r.event==='amr_query').length,original_decoder_calls:log.filter(r=>r.tool==='delivery_request'||r.tool==='source_resample').length})
}
assert.deepEqual(outcomes[0].branch_refs,outcomes[1].branch_refs)
await writeFile(resolve(d,'result.json'),JSON.stringify({scope:'Deterministic retrieval of previously computed temporal branches from a common saved snapshot; no model or task truth.',outcomes},
 null,2))
