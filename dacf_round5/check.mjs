import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const {apply}=await import(pathToFileURL(resolve(process.argv[2]??'dacf_round5/release/installed/package/entry.mjs')).href)
const dest=resolve(process.argv[3]??'dacf_round5/engineering')
await mkdir(dest)
const case_=resolve('dacf_round5/inputs/fixture'), source=resolve(case_,'source.mp4')
const plan=JSON.parse(await readFile(resolve(case_,'plan.json'),'utf8'))
const services={llm:{resolveModelInfo:async()=>({inputModalities:['image']})},attachments:{saveImage:async({data,mediaType})=>({
  attachmentId:'sha256:'+createHash('sha256').update(data).digest('hex'),mediaType,bytes:data.length,width:data.readUInt32BE(16),height:data.readUInt32BE(20)})}}
const open=(arm,label,automaticPolicy='S0')=>{
  const tools=new Map()
  const exec={agent:{id:label,session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
  apply({effect:f=>f(),get:k=>services[k],tools:{register:t=>tools.set(t.name,t)}},{arm,automaticPolicy,
    authorizedSource:source,planPath:resolve(case_,'plan.json'),cacheDir:resolve(dest,`cache-${label}`),controlLog:resolve(dest,`${label}.jsonl`),
    pythonPath:resolve('.venv/Scripts/python.exe'),compactEvidence:false})
  return {tools,call:async(name,args)=>tools.get(name).execute(args,exec)}
}
const a2=open('A2','main'), a0=open('A0','baseline'), a1=open('A1','flat')
const {call}=a2
await call('video_open',{path:source})
await call('observation_goal',{goal:'Trace a verified local tree branch, projected context and the original receipt.'})
assert.deepEqual([...a0.tools.keys()].sort(),[...a1.tools.keys()].filter(k=>k!=='investigation_regions').sort())
assert.deepEqual([...a0.tools.keys()].sort(),[...a2.tools.keys()].filter(k=>!['amr_refine','amr_query'].includes(k)).sort())
const sourceView=await call('amr_query',{node_id:'source',detail:'wave',page_size:4})
assert.ok(sourceView.wave)
assert.ok(Object.values(sourceView.stops).every(s=>s.S1==='not_evaluated'))
await assert.rejects(call('amr_query',{node_id:'missing'}),/UNKNOWN_NODE/)
await assert.rejects(call('amr_refine',{node_id:'source',mode:'time',pts:[999999],reason:'invalid test'}),/INVALID_PTS/)
await assert.rejects(call('amr_refine',{node_id:'source',mode:'space',pts:[0,5120],native_xywh:[0,0,9999,9999],reason:'invalid test'}),/INVALID_ROI/)
const real=plan.nodes.find(n=>n.kind==='paired_investigation'&&plan.clock.some(r=>r.pts>n.bounds_pts[0]&&r.pts<n.bounds_pts[1]&&!n.known_pts.includes(r.pts)))
const newTime=plan.clock.find(r=>r.pts>real.bounds_pts[0]&&r.pts<real.bounds_pts[1]&&!real.known_pts.includes(r.pts))
const time=await call('amr_refine',{node_id:real.id,mode:'time',pts:[newTime.pts],reason:'Check the actual interior before interpreting the interval.'})
assert.equal(time.summary.children.length,2)
assert.ok(time.summary.refs.length>=3&&time.images.length===3)
assert.equal(time.summary.host_status,'computed')
const receipt=time.summary.originals[newTime.pts]
assert.ok(receipt)
const space=await call('amr_refine',{node_id:'source',mode:'space',pts:[0,newTime.pts],native_xywh:[48,56,180,135],crop_pts:newTime.pts,
  reason:'Check actual local ROI and preserve traceable parent.'})
assert.equal(space.images.length,3)
assert.equal(space.summary.children.length,4)
assert.equal(space.summary.host_status,'computed')
const crop=space.summary.derived
assert.ok(crop)
const reference=await call('evidence_reference',{ids:[crop]})
const verified=await call('claim_verify',{evidence_ids:[crop,receipt]})
assert.equal(verified.items[0].parent_verification.integrity,'verified')
const node=time.summary.children[0].id
const data=await call('amr_query',{node_id:node,detail:'proof'})
assert.ok(data.proof.local_measurements.temporal.measurements.length===3)
const proof=await call('evidence_detail',{result_ref:space.summary.geometry.proof,path:['rows']})
assert.ok(proof.result_ref)
const obj=await call('object_candidate',{action:'add',label:'local visible test box',role:'observed_view',view:{pts:newTime.pts,native_xywh:[48,56,180,135]},evidence_ids:[receipt]})
assert.ok(obj.candidate_id)
const first=await call('observation_record',{action:'create',kind:'direct',content:'A fixture is visible within this exact sampled region.',evidence_ids:[receipt]})
const state=await call('state_chain',{action:'add',candidate_id:obj.candidate_id,pts:newTime.pts,phase:'observation',content:'Record this local view only.',evidence_ids:[receipt],revision_ids:[first.revision_id]})
const revised=await call('observation_record',{action:'revise',kind:'direct',content:'A fixture is visible at this PTS, with the named scope confined to the sample.',
  evidence_ids:[receipt],observation_id:first.observation_id,supersedes_revision:first.revision_id,reason:'Restrict scope to the actual source sample.'})
const review=await call('observation_review',{observation_id:first.observation_id})
const joins={}
for(const channel of ['catalogue','original','computed','object','state','observation','derived']) {
  const linked=await call('evidence_routes',{amr_node_id:'source',channel,start:newTime.seconds-.000001,end:newTime.seconds+.000001,page_size:16})
  joins[channel]={total:linked.total,refs:linked.items.map(x=>x.result_ref??x.citation??x.state_id??x.identity_id??x.frame_id)}
  if(channel!=='catalogue') assert.ok(linked.total>0,channel)
}
const whole=await call('evidence_routes',{amr_node_id:'source',channel:'catalogue'})
assert.ok(whole.items.length)
const json=JSON.parse(JSON.stringify(space))
assert.deepEqual(space,json)
const cross=open('A2','local','S1')
await cross.call('video_open',{path:source})
const root=await cross.call('amr_query',{node_id:'source'})
assert.ok(Object.values(root.policy_hint.statuses).every(s=>s.state==='not_evaluated'))
await assert.rejects(cross.call('amr_query',{node_id:node}),/UNKNOWN_NODE/)
await writeFile(resolve(dest,'result.json'),JSON.stringify({status:'passed',common_tools:[...a0.tools.keys()],time:time.summary,space:space.summary,
  refs:reference,verified,object:obj.candidate_id,state:state.state_id,history:{revisions:[first.revision_id,revised.revision_id],review},joins,
  isolated_session:true,attachments:'in-process mock attachment host with real native PNGs; real DSH separate'},null,2),{flag:'wx'})
