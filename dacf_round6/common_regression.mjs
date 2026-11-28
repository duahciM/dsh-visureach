import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtemp, readdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const here=dirname(fileURLToPath(import.meta.url)), project=resolve(here,'..')
const entry=resolve(process.argv[2]??'D:/LLM/dsh-video-observer-0.5.0/index.js')
const dir=await mkdtemp(join(here,'guide-acceptance-'))
const {apply}=await import(pathToFileURL(entry).href)
const {GUIDE}=await import(pathToFileURL(resolve(project,'dacf_round5/release-fixed/installed/package/baseline/guide.js')).href)
const tools=new Map(), payloads=new Map()
const services={
  llm:{resolveModelInfo:async()=>({inputModalities:['image']})},
  attachments:{async saveImage({data,mediaType}) {
    const hash=createHash('sha256').update(data).digest('hex'),attachmentId=`sha256:${hash}`
    payloads.set(hash,data)
    return {attachmentId,mediaType,bytes:data.length,width:data.readUInt32BE(16),height:data.readUInt32BE(20)}
  }},
}
apply({tools:{register:t=>tools.set(t.name,t)},effect:f=>f(),get:k=>services[k]}, {
  arm:'A0',cacheDir:dir,pythonPath:join(project,'.venv/Scripts/python.exe'),whisperModel:join(project,'plugin_integration_20261124/dependencies/whisper.bin'),
  semanticAssets:join(project,'p215_discovery_20261121'),semanticPythonPaths:[join(project,'if13_research_20261120/runtime_onnx')],
})
const exec={agent:{id:'guide-check',session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
const call=(n,a)=>tools.get(n).execute(a,exec)
const out={entry,cache:dir,tools:{}}
const v={video:join(project,'plugin_v040/inputs/scene04.mp4'),p0:0,p1:2560,region:[58,117,120,142]}
const substitute=x=> typeof x==='string'&&x.startsWith('$') ? (assert.ok(Object.hasOwn(v,x.slice(1)),x),v[x.slice(1)])
  : Array.isArray(x)?x.map(substitute):x!=null&&typeof x==='object'?Object.fromEntries(Object.entries(x).map(([k,z])=>[k,substitute(z)])):x
const image=(s)=>[...s.matchAll(/pts=(-?\d+)(?: time_base=\S+)? source_frame=(\d+).*?evidence_id=(ev-[a-f0-9]+)/g)].map(m=>({pts:Number(m[1]),index:Number(m[2]),id:m[3]}))
function typecheck(value,schema) {
  for (const key of schema.required??[]) assert.ok(Object.hasOwn(value,key),`Missing required ${key}`)
  for (const [k,d] of Object.entries(value)) {
    const p=schema.properties[k]
    assert.ok(p,`Unknown schema parameter ${k}`)
    if (p.type==='integer') assert.ok(Number.isSafeInteger(d))
    if (p.type==='string') assert.equal(typeof d,'string')
    if (p.type==='number') assert.equal(typeof d,'number')
    if (p.type==='boolean') assert.equal(typeof d,'boolean')
    if (p.type==='array') assert.ok(Array.isArray(d))
    if (p.enum) assert.ok(p.enum.includes(d))
  }
}
async function example(name) {
  const a=substitute(GUIDE[name][1]);typecheck(a,tools.get(name).parameters)
  const result=await call(name,a)
  assert.deepEqual(result,JSON.parse(JSON.stringify(result)),`lossless JSON: ${name}`)
  const rendered=tools.get(name).output?.render?.(a,result)
  if(rendered)assert.deepEqual(rendered,JSON.parse(JSON.stringify(rendered)),`render JSON: ${name}`)
  assert.ok(!['failed','invalid_parameters','dependency_missing','original_required'].includes(result.status),JSON.stringify(result).slice(0,1000))
  out.tools[name]={arguments:a,status:result.status??'completed',keys:Object.keys(result),images:result.images?.length??0}
  return result
}

try {
  const announced=await call('observer_guide',{})
  assert.equal(announced.available.length,tools.size)
  for (const t of tools.keys()) assert.ok(GUIDE[t],t)
  await example('observer_guide')
  await example('video_open')
  await example('observation_goal')
  const cat=await example('catalog_search')
  v.catalog_id=cat.frames.split(' | ')[0]
  for(const n of ['wave_query','observation_coverage','external_source','evidence_search','evidence_graph','change_search','semantic_search','frame_ocr','appearance_search']) await example(n)
  const supplied=await example('delivery_request')
  v.ev0=supplied.envelope.match(/evidence_id=(ev-[a-f0-9]+)/)[1]
  const r=await example('source_resample')
  const sample=image(r.envelope).find((s)=>s.pts%2458!==0&&s.index===25)??image(r.envelope)[1]
  v.reread_pts=sample.pts
  const negative=await call('compare_object_views',{mode:'ab',samples:[{pts:sample.pts,native_xywh:v.region}]})
  assert.equal(negative.code,'MISSING_AB_REFERENCES')
  const positive=await example('compare_object_views')
  assert.equal(positive.mode,'ab')
  const graph=await example('relation_figure2')
  assert.equal(graph.index_layer,'session_incremental')
  assert.ok(graph.summary.regions>0)
  const detailGraph=await call('evidence_detail',{result_ref:graph.result_ref,path:['figure2','nodes']})
  assert.ok(detailGraph.total>0)
  const mixed=await call('evidence_routes',{anchor_pts:sample.pts,hops:1,channel:'graph',page_size:4})
  assert.ok(mixed.total>0)
  v.pair=(await example('visual_compare')).result_ref
  await example('evidence_detail')
  v.crop=JSON.parse((await example('evidence_crop')).description).derived_id
  await example('evidence_reference')
  const t=await example('change_triggers')
  const item=t.items.find((x)=>x.end>x.start)
  Object.assign(v,{triggers:t.result_ref,trigger:item.trigger_id,change_start:item.start,change_end:item.end})
  await example('change_recheck')
  await example('evidence_list')
  const obs=await call('observation_record',{action:'create',content:'A colored rectangle in the opening source view.',kind:'direct',evidence_ids:[v.ev0]})
  v.revision=obs.revision_id
  await example('observation_record')
  await example('claim_verify')
  await example('observation_dispute')
  const candidate=await example('object_candidate')
  Object.assign(v,{candidate:candidate.candidate_id,view_id:candidate.observation_id})
  const aliases=['observation_id','appearance_id','slot_id','context_id','instance_candidate_id','track_id']
  for (const fld of aliases) assert.equal((await call('object_candidate',{action:'list',identity_id:candidate[fld]})).total,1)
  assert.equal((await call('compare_object_views',{mode:'direct',reference_view_ids:[candidate.track_id],samples:[{pts:v.reread_pts,native_xywh:v.region}]})).mode,'direct')
  await example('state_chain')
  await example('observation_review')
  const rev2=await call('observation_record',{action:'create',content:'Local condition assessed as inference.',kind:'inference',evidence_ids:[v.ev0],depends_on_revisions:[v.revision]})
  await call('observation_record',{action:'revise',observation_id:obs.observation_id,supersedes_revision:v.revision,
    content:'A left colored region is visible.',kind:'direct',evidence_ids:[v.ev0],reason:'Clarify color and location after reread.'})
  const check=await call('observation_review',{revision_id:rev2.revision_id,final_claims:[{content:'A marker.',stated_details:['left'],evidence_ids:[v.ev0,v.crop],revision_ids:[rev2.revision_id]}],omitted_observation_ids:[obs.observation_id]})
  assert.deepEqual(check.items[0].needs_review_upstream,[v.revision])
  assert.deepEqual(check.final_claim_checks[0].declared_details_not_written_verbatim,['left'])
  assert.equal(check.final_claim_checks[0].same_source_groups[0].length,2)
  await example('evidence_routes')
  const opened=await call('video_open',{path:join(project,'plugin_integration_20261124/dependencies/text-channel-verified.mkv')})
  assert.ok(opened.video_sha256)
  assert.equal((await example('video_text_search')).status,'completed')
  assert.equal(Object.keys(out.tools).length,tools.size)
  out.pass=true
  out.recovery=negative
  out.evidence={graph:graph.summary,result_ref:graph.result_ref,source_pts:v.reread_pts,check,aliases}
  out.ledgers=await readdir(join(dir,'sessions'))
} catch(error) {out.error=error.stack;process.exitCode=1}
finally {await writeFile(join(dir,'result.json'),JSON.stringify(out,null,2));console.log(dir,out.pass??out.error)}
