import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { join } from 'node:path'
import { installResearch } from './probe.mjs'

const root='dacf_round4/agent/engineering'
const cfg=JSON.parse(await readFile(join(root,'config.json'),'utf8'))
const tools=new Map()
const ctx={tools:{register:t=>tools.set(t.name,t)},get:n=>{
  assert.equal(n,'attachments')
  return {saveImage:async({data})=>({attachmentId:'sha256:'+createHash('sha256').update(data).digest('hex'),
    width:data.readUInt32BE(16),height:data.readUInt32BE(20)})}
}}
installResearch(ctx,cfg)
const call=(name,args)=>tools.get(name).execute(args)
const guide=await call('amr_query',{detail:'guide'})
const listed=await call('amr_query',{start:0,end:2,page_size:3})
assert.ok(listed.items.length&&guide.recovery)
const source=await call('amr_query',{node_id:'source',detail:'node'})
assert.ok(source.originals.total>0)
const sourcePts=source.originals.items.slice(0,3).map(p=>p.pts)
await assert.rejects(call('amr_query',{node_id:'absent'}),/List nearby times/)
const targeted=(await call('amr_query',{start:4.3,end:4.7,page_size:20})).items.find(n=>n.kind==='frozen_r2_region')
assert.ok(targeted)
const timed=await call('amr_refine',{node_id:targeted.id,mode:'time'})
assert.ok(timed.summary.new_probe_pts!=null&&timed.summary.children.length===2)
const timedProof=await call('amr_query',{node_id:timed.summary.children[0].id,detail:'proof'})
assert.ok(timedProof.proof.comparison.root)
const spatial=await call('amr_refine',{node_id:'source',mode:'space',pts:sourcePts,native_xywh:[180,125,700,470],crop_pts:sourcePts[1]})
assert.equal(spatial.images.length,1)
assert.equal(spatial.summary.children.length,4)
const spatialProof=await call('amr_query',{node_id:spatial.summary.children[0].id,detail:'proof'})
assert.ok(spatialProof.proof.calculation_roots.length)
const flat=new Map()
installResearch({tools:{register:t=>flat.set(t.name,t)}},{...cfg,arm:'A1'})
const flatQuery=await flat.get('investigation_regions').execute({node_id:targeted.id})
const treeQuery=await call('amr_query',{node_id:targeted.id})
assert.deepEqual(flatQuery.originals,treeQuery.originals)
await writeFile(join(root,'acceptance.json'),JSON.stringify({status:'passed',time:timed.summary,spatial:spatial.summary,spatial_proof:spatialProof.proof,
  same_source_evidence_with_flat:true,route:'real Python exact source; mocked attachment host for model-free engineering validation'},null,2),{flag:'wx'})
