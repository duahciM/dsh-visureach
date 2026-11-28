import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {mkdir,readFile,writeFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {apply} from '../dacf_round7/surface.mjs'
import {relationEvidenceAtNode} from './amr_evidence.mjs'

const here=resolve('partial_modules_stage2'), out=resolve(here,process.argv[2]??'engineering')
await mkdir(out)
const json=async path=>JSON.parse(await readFile(resolve(here,path),'utf8'))
const sha=data=>createHash('sha256').update(data).digest('hex')
const images=new Map()
const host={llm:{resolveModelInfo:async()=>({inputModalities:['image']})},attachments:{async saveImage({data,mediaType}){
  const hash=sha(data);images.set(hash,data)
  return {attachmentId:'sha256:'+hash,mediaType,bytes:data.length,width:data.readUInt32BE(16),height:data.readUInt32BE(20)}
}}}
const tools=new Map(),exec={agent:{id:'scoped-m2-m5-accept',session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
const source=resolve(here,'inputs/control/source.mkv'),planPath=resolve(here,'plan/tree.json')
apply({tools:{register:t=>tools.set(t.name,t)},get:key=>host[key],effect:f=>f(),deepseekLlmApiExtensions:{register(){}}},{
  authorizedSource:source,planPath,cacheDir:resolve(out,'cache'),controlLog:resolve(out,'calls.jsonl'),
  pythonPath:resolve('.venv/Scripts/python.exe'),overviewInSystem:false
})
const call=(n,a)=>tools.get(n).execute(a,exec)
const plan=await json('plan/tree.json'), early=await json('scope_run/control/result.json')
const supplemental=await json('sampling/ref_x/result.json')
const candidate=supplemental.requests.find(x=>x.new_exact_decode && x.used_axis==='ref_x')
const pt=candidate.original.pts
const outReceipt={}
try{
  const opened=await call('video_open',{path:source})
  assert.equal(opened.video_sha256,early.source_sha256)
  await call('observation_goal',{goal:'Review exact endpoint and relationship evidence from a local investigation candidate.'})
  const original=await call('amr_query',{node_id:'source',detail:'proof'})
  const layer=await call('amr_refine',{node_id:'source',mode:'time',pts:[pt],reason:'Verify actual relation support around the selected source tick.'})
  assert.ok(layer.images.length===3 && layer.summary.originals[pt])
  const scope=await call('amr_query',{node_id:layer.summary.children[0].id,detail:'proof'})
  const edgeSet=Object.values(supplemental.revised_relation_edges_incident_to_supplement).flat()
  const pkg={sourceSha:opened.video_sha256,edges:edgeSet}
  const lookup=relationEvidenceAtNode(pkg,scope,{sourceSha256:plan.sha256,enabled:true})
  assert.ok(lookup.rows.some(r=>r.pts.includes(pt)&&r.scope_dependency&&r.axes.ref_x!==null))
  assert.deepEqual(lookup.original_stop_states,scope.stops)
  assert.equal(relationEvidenceAtNode(pkg,scope,{sourceSha256:plan.sha256}).rows.length,0)
  assert.throws(()=>relationEvidenceAtNode(pkg,scope,{sourceSha256:'other',enabled:true}),/SOURCE/)
  const row=early.view_inventory[String(candidate.selected.bounds[0])]?.find(v=>v.role==='candidate')
  const roi=row?.native_xywh??[170,110,120,100]
  const crop=await call('evidence_crop',{evidence_id:layer.summary.originals[pt],native_xywh:roi,reason:'Inspect the local original scope.'})
  const cropId=JSON.parse(crop.description).derived_id
  const refs=await call('evidence_reference',{ids:[cropId,layer.summary.originals[pt]]})
  assert.equal(refs.items.find(v=>v.type==='derived').parent_evidence_id,layer.summary.originals[pt])
  const joint=await call('evidence_routes',{amr_node_id:scope.id,channel:'derived',page_size:8})
  assert.ok(joint.items.some(v=>v.citation===cropId))
  const oldRef=scope.refs[0]
  const details=await call('evidence_detail',{result_ref:oldRef,path:[]})
  assert.equal(details.result_ref,oldRef)
  const hashes={}
  for (const [id,data] of images){
    const filename=id+'.png';await writeFile(resolve(out,filename),data,{flag:'wx'});hashes[id]=filename
  }
  Object.assign(outReceipt,{passed:true,source_sha256:opened.video_sha256,new_actual_pts:pt,
     source_parent:original.id,original_stops:original.stops,child:scope.id,lookup,actual_originals:layer.summary.originals,
     detail_ref:details.result_ref,crop:refs.items.find(v=>v.type==='derived'),computed_receipts:layer.summary,
     transport_files:hashes,novel_native_tick:!plan.nodes[0].known_pts.includes(pt)})
}catch(error){outReceipt.error=error.stack;process.exitCode=1}
finally{await writeFile(resolve(out,'result.json'),JSON.stringify(outReceipt,null,2),{flag:'wx'});console.log(outReceipt.passed??outReceipt.error)}
