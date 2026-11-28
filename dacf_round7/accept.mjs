import assert from 'node:assert/strict'
import {mkdir, readFile, writeFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {createHash} from 'node:crypto'
import {apply} from './surface.mjs'

const out = resolve(process.argv[2] ?? 'dacf_round7/engineering')
await mkdir(out)
const source = resolve('dacf_round6/independent/source.mp4')
const planPath = resolve('dacf_round6/independent/plan.json')
const guide = JSON.parse(await readFile(resolve('dacf_round7/revised_surface.json'),'utf8'))
const shape = schema => JSON.parse(JSON.stringify(schema, (k,v) => k==='description' ? undefined : v))
const sha = bits => createHash('sha256').update(bits).digest('hex')
const media = new Map()
const services = {llm:{resolveModelInfo:async()=>({inputModalities:['image']})}, attachments:{saveImage:async({data,mediaType})=>{
  const digest=sha(data); media.set(digest,data)
  return {attachmentId:`sha256:${digest}`,mediaType,bytes:data.length,width:data.readUInt32BE(16),height:data.readUInt32BE(20)}
}}}
const register = revised => {
  const tools=new Map()
  const context={tools:{register:t=>tools.set(t.name,t)},get:k=>services[k],effect:f=>f(),deepseekLlmApiExtensions:{register(){}}}
  apply(context,{authorizedSource:source,planPath,docsPath:revised?resolve('dacf_round7/revised_surface.json'):undefined,
    overviewInSystem:false,cacheDir:resolve(out,revised?'cache':'base'),controlLog:resolve(out,revised?'new.jsonl':'base.jsonl'),
    requestAudit:resolve(out,'unused.jsonl'),pythonPath:resolve('.venv/Scripts/python.exe')})
  return tools
}
const tools=register(true), baseline=register(false)
assert.deepEqual([...tools.keys()],[...baseline.keys()])
for (const [name,t] of tools) {
  assert.deepEqual(shape(t.parameters),shape(baseline.get(name).parameters),name)
  assert.deepEqual(t.output?.schema,baseline.get(name).output?.schema,name)
}
const exec={agent:{id:'round7-engineering',session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
const result={source,same_tool_and_validation_shapes:true,tool_results:{},recovery:[],optional:[]}
const fill=(v,x)=>typeof v==='string'&&v.startsWith('$')?(assert.ok(Object.hasOwn(x,v.slice(1)),v),x[v.slice(1)]):
  Array.isArray(v)?v.map(z=>fill(z,x)):v&&typeof v==='object'?Object.fromEntries(Object.entries(v).map(([k,z])=>[k,fill(z,x)])):v
const names=['failed','invalid_parameters','original_required']
const x={video:source,p0:0,p1:5120,region:[45,40,105,98],reread_pts:38912}
async function call(name, args) {
  const tool=tools.get(name)
  args ??= fill(guide.tools[name].example,x)
  const begin=Date.now()
  const value=await tool.execute(args,exec)
  assert.deepEqual(value,JSON.parse(JSON.stringify(value)),name)
  const rendered=tool.output?.render?.(args,value)
  if(rendered) assert.deepEqual(rendered,JSON.parse(JSON.stringify(rendered)),name)
  assert.ok(!names.includes(value.status),name+': '+JSON.stringify(value).slice(0,800))
  result.tool_results[name]={args,keys:Object.keys(value),status:value.status??'returned',attachments:value.images?.length??0,ms:Date.now()-begin,
    image_digests:(value.images??[]).map(v=>v.attachmentId)}
  return value
}

try {
  const docs=await call('observer_guide',{})
  assert.equal(docs.overview,guide.overview)
  assert.deepEqual((await call('observer_guide',{tool:'compare_object_views'})).example,guide.tools.compare_object_views.example)
  await call('video_open')
  await call('observation_goal')
  const catalog=await call('catalog_search'); x.catalog_id=catalog.frames.split(' | ')[0]
  for(const n of ['wave_query','observation_coverage','external_source','evidence_search','evidence_graph','change_search','appearance_search','semantic_search','frame_ocr','video_text_search']) await call(n)
  const sent=await call('delivery_request'); x.ev0=sent.envelope.match(/evidence_id=(ev-[a-f0-9]+)/)[1]
  await call('source_resample')
  await call('delivery_request',{pts:[x.reread_pts],reason:'Bind an independent original to this non-catalogue figure request.'})
  // Keep the listed A/B guide example tied to the published source-address placeholders.
  const primary=await call('compare_object_views'); assert.equal(primary.mode,'ab')
  const figure=await call('relation_figure2'); assert.equal(figure.index_layer,'session_incremental')
  x.pair=(await call('visual_compare')).result_ref
  await call('evidence_detail')
  const crop=await call('evidence_crop'); x.crop=JSON.parse(crop.description).derived_id
  await call('evidence_reference')
  const trigger=await call('change_triggers')
  const item=trigger.items.find(v=>v.end>v.start)
  Object.assign(x,{triggers:trigger.result_ref,trigger:item.trigger_id,change_start:item.start,change_end:item.end})
  const check=await call('change_recheck'); assert.equal(check.images,undefined)
  await call('evidence_list')
  const obs=await call('observation_record',{action:'create',kind:'direct',content:'The opening view contains a local image feature.',evidence_ids:[x.ev0]})
  x.revision=obs.revision_id; await call('observation_record')
  await call('claim_verify'); await call('observation_dispute')
  const candidate=await call('object_candidate'); x.candidate=candidate.candidate_id; x.view_id=candidate.observation_id
  await call('state_chain'); await call('observation_review'); await call('evidence_routes')
  const root=await call('amr_query'); assert.equal(root.id,'source')
  const invalid=[['amr_query',{node_id:'unknown-round7'},/UNKNOWN_NODE.*time-range query/],
    ['amr_refine',{node_id:'source',mode:'time',pts:[12],reason:'Verify an invalid tick is reported.'},/INVALID_PTS/],
    ['evidence_detail',{result_ref:x.ev0,path:[]},/result_ref/]]
  for(const [n,a,e] of invalid){
    try {await tools.get(n).execute(a,exec); assert.fail(`${n} did not reject`)}
    catch(error){ assert.match(error.message,e); result.recovery.push({tool:n,arguments:a,error:error.message}) }
  }
  const time=await call('amr_refine'); assert.equal(time.images.length,3)
  const space=await call('amr_refine',{node_id:'source',mode:'space',pts:[x.p0,x.p1],native_xywh:x.region,crop_pts:x.p1,reason:'Recover original local views and a separate crop.'})
  assert.equal(space.images.length,3);assert.ok(space.summary.derived)
  const node=await call('amr_query',{node_id:time.summary.children[0].id,detail:'proof'}); assert.ok(node.refs.length)
  result.operations={temporal:time.summary,spatial:space.summary,crop_parent:JSON.parse(crop.description),change_recheck_attachments:0}
  for(const n of ['semantic_search','frame_ocr','video_text_search']) result.optional.push({tool:n,status:result.tool_results[n].status})
  assert.equal(Object.keys(result.tool_results).length,tools.size)
  result.pass=true
} catch(error) {result.failure=error.stack;process.exitCode=1}
finally {await writeFile(resolve(out,'result.json'),JSON.stringify(result,null,2),{flag:'wx'}); console.log(result.pass??result.failure)}
