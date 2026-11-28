import assert from 'node:assert/strict'
import {mkdir,readFile,writeFile} from 'node:fs/promises'
import {createHash} from 'node:crypto'
import {resolve} from 'node:path'
import {apply} from './persistent.mjs'
const d=resolve('dacf_round6/dynamic-gain');await mkdir(d)
const root=resolve('dacf_round6/independent'), planPath=resolve(root,'plan.json'), path=resolve(root,'source.mp4')
const plan=JSON.parse(await readFile(planPath,'utf8')), coarse=JSON.parse(await readFile(resolve(root,'index/manifest.json'),'utf8'))
const known=new Set(coarse.pool.map(x=>x.pts))
const tools=new Map(), exec={agent:{id:'gain-check',session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
const service={llm:{resolveModelInfo:async()=>({inputModalities:['image']})},attachments:{saveImage:async({data,mediaType})=>({
  attachmentId:'sha256:'+createHash('sha256').update(data).digest('hex'),mediaType,width:data.readUInt32BE(16),
  height:data.readUInt32BE(20),bytes:data.length})}}
apply({effect:f=>f(),tools:{register:t=>tools.set(t.name,t)},get:key=>service[key]},
  {arm:'A2',authorizedSource:path,planPath,cacheDir:resolve(d,'cache'),controlLog:resolve(d,'calls.jsonl'),pythonPath:resolve('.venv/Scripts/python.exe')})
const call=(n,a)=>tools.get(n).execute(a,exec)
await call('video_open',{path});await call('observation_goal',{goal:'Engineering check of an actual supplementary original.'})
const first=plan.nodes.find(n=>n.kind==='paired_investigation'&&plan.clock.some(r=>!known.has(r.pts)&&r.pts>n.bounds_pts[0]&&r.pts<n.bounds_pts[1]))
const middle=plan.clock.find(r=>!known.has(r.pts)&&r.pts>first.bounds_pts[0]&&r.pts<first.bounds_pts[1])
const original=await call('amr_query',{node_id:first.id})
const after=await call('amr_refine',{node_id:first.id,mode:'time',pts:[middle.pts],reason:'Sample a source time with no coarse index address.'})
const child=await call('amr_query',{node_id:after.summary.children[0].id,detail:'proof'})
const binding=await call('evidence_routes',{amr_node_id:child.id,start:middle.seconds-.000001,end:middle.seconds+.000001,channel:'original'})
assert.ok(!original.original_addresses.items.includes(middle.pts))
assert.ok(child.original_addresses.items.includes(middle.pts)&&after.images.length===3)
assert.ok(binding.items.some(r=>r.pts===middle.pts&&r.citation===after.summary.originals[middle.pts]))
await writeFile(resolve(d,'result.json'),JSON.stringify({not_in_coarse_pool:middle,old_node:first.id,
 new_child:child.id,source_dimensions:[plan.width,plan.height],source_receipt:after.summary.originals[middle.pts],
 unrelated_semantic_gain_unscored:true,actual_originals:after.images.length,linked:binding.items,
 geometry_scope:child.proof.local_measurements.temporal},null,2))
