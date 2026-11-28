import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {mkdir,readFile,writeFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {apply} from '../dacf_round7/surface.mjs'
import {queryEvidence} from './query.mjs'

const home=resolve('partial_modules_stage1'), out=resolve(home,process.argv[2]??'engineering')
await mkdir(out)
const digest=bytes=>createHash('sha256').update(bytes).digest('hex')
const json=async path=>JSON.parse(await readFile(path,'utf8'))
const assets=new Map()
const host={llm:{resolveModelInfo:async()=>({inputModalities:['image']})},attachments:{async saveImage({data,mediaType}){
  const hash=digest(data);assets.set(hash,data)
  return {attachmentId:'sha256:'+hash,mediaType,bytes:data.length,width:data.readUInt32BE(16),height:data.readUInt32BE(20)}
}}}
const config=(source,planPath,id)=>{
  const tools=new Map(),exec={agent:{id,session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
  apply({tools:{register:t=>tools.set(t.name,t)},get:k=>host[k],effect:f=>f(),deepseekLlmApiExtensions:{register(){}}},
    {authorizedSource:source,planPath,cacheDir:resolve(out,id),controlLog:resolve(out,id+'.jsonl'),
      pythonPath:resolve('.venv/Scripts/python.exe'),overviewInSystem:false})
  return (name,args)=>tools.get(name).execute(args,exec)
}

const getId=(envelope,pts)=>{
  const result=[...envelope.matchAll(/pts=(-?\d+).*?evidence_id=(ev-[a-f0-9]+)\b/g)].find(v=>Number(v[1])===pts)
  assert.ok(result,`Missing actual receipt for tick ${pts}`)
  return result[2]
}

async function repeatUnits(){
  const source=resolve('dacf_round6/independent/source.mp4');const call=config(source,resolve('dacf_round6/independent/plan.json'),'unit-retest')
  await call('video_open',{path:source});await call('observation_goal',{goal:'Verify independent exact and interval retrieval and projected scope.'})
  const catalogue=await call('catalog_search',{start:0,end:2.4,dimension:'D6',mode:'earliest',page_size:4})
  const frame_id=catalogue.frames.split('\n').find(line=>line.includes('[D6]')).split(' | ')[0]
  const graph=await call('evidence_graph',{frame_id,page_size:4});assert.equal(graph.anchor.frame_id,frame_id)
  const exact=await call('delivery_request',{pts:[25600],reason:'Get known original tick.'})
  const ev=getId(exact.envelope,25600)
  const interval=await call('source_resample',{start:2,end:2.4,fps:8,page_size:4,reason:'Locate other exact samples in the span.'})
  assert.ok(interval.images.length>1)
  const another=[...interval.envelope.matchAll(/pts=(-?\d+).*?evidence_id=(ev-[a-f0-9]+)\b/g)].map(v=>Number(v[1])).filter(p=>p!==25600)[0]
  assert.ok(Number.isInteger(another))
  const crop=await call('evidence_crop',{evidence_id:ev,native_xywh:[30,35,45,38],reason:'Review local pixels at one known original.'})
  const cropId=JSON.parse(crop.description).derived_id
  await assert.rejects(call('evidence_graph',{frame_id:cropId}),/frame_id is not in this catalogue/)
  const refs=await call('evidence_reference',{ids:[cropId,ev]});assert.equal(refs.items.find(r=>r.type==='derived').parent_evidence_id,ev)
  const matched=await call('evidence_routes',{start:2,end:2.4,channel:'derived',page_size:8});assert.ok(matched.items.some(row=>row.citation===cropId))
  const pair=await call('visual_compare',{pts_a:25600,pts_b:another});assert.ok(pair.result_ref)
  const stored=await call('evidence_detail',{result_ref:pair.result_ref,path:[]});assert.equal(stored.result_ref,pair.result_ref)
  return {exact_pts:25600,actual_other_interval_pts:another,exact_images:exact.images.length,interval_images:interval.images.length,
    derived_parent:refs.items.find(r=>r.type==='derived'),graph_scope:graph.scope,derived_routes:matched.items,stored_comparison:stored.result_ref}
}

async function linkAMR(){
  const source=resolve(home,'inputs/panning/source.mkv'),planPath=resolve(home,'amr_plan_retry/plan.json')
  const plan=await json(planPath), baseline=await json(resolve(home,'first_run/panning/A0.json'))
  const arms={...(await json(resolve(home,'first_run/panning/A3.json'))),roi_states:baseline.roi_state_samples}
  const call=config(source,planPath,'amr-module-join')
  const opened=await call('video_open',{path:source});assert.equal(opened.video_sha256,arms.source_sha256)
  await call('observation_goal',{goal:'Check original object and referent measurements in a saved investigation node.'})
  const initial=await call('amr_query',{node_id:'source',detail:'links'})
  const choice=arms.roi_states[Object.keys(arms.roi_states)[0]].slice(0,2)
  const [t0,t1]=choice.map(x=>x.pts)
  const roi=choice[1].target_region
  const refinement=await call('amr_refine',{node_id:initial.id,mode:'space',pts:[t0,t1],native_xywh:roi,crop_pts:t1,reason:'Check existing source data and new local crop.'})
  assert.equal(refinement.images.length,3)
  assert.ok(refinement.summary.derived)
  const evidenceIds=choice.map(x=>refinement.summary.originals[x.pts]);assert.ok(evidenceIds.every(Boolean))
  const views=[]
  for(let i=0;i<choice.length;i++) {
    views.push(await call('object_candidate',{action:'add',...(i?{candidate_id:views[0].candidate_id}:{}),
      label:'Two supplied local views',role:'prototype',view:{pts:choice[i].pts,native_xywh:choice[i].target_region},evidence_ids:[evidenceIds[i]]}))
  }
  for(let i=0;i<choice.length;i++)await call('state_chain',{action:'add',candidate_id:views[0].candidate_id,
      pts:choice[i].pts,identity_id:views[i].observation_id,phase:'observation',content:'A supplied local ROI and its 2D referent are in this original.',
      relationship:'projected-local-offset',evidence_ids:[evidenceIds[i]]})
  const child=await call('amr_query',{node_id:refinement.summary.children[0].id,detail:'proof'});assert.ok(child.refs.length)
  const joint=await call('evidence_routes',{amr_node_id:child.id,candidate_id:views[0].candidate_id,channel:'state',page_size:8});assert.equal(joint.items.length,2)
  const reference=await call('evidence_reference',{ids:[refinement.summary.derived,...evidenceIds]});assert.equal(reference.items.find(v=>v.type==='derived').parent_evidence_id,evidenceIds[1])
  const match=queryEvidence(arms,{source_sha256:plan.sha256,node_id:child.id,bounds_pts:[t0,t1]},{m2:true,m5:true})
  assert.ok(match.sampled_relation_edges.some(s=>s.edges.some(e=>e.pts[0]===t0&&e.pts[1]===t1)))
  assert.ok(match.candidate_hypotheses.length)
  const noExtras=queryEvidence(arms,{source_sha256:plan.sha256,node_id:child.id,bounds_pts:[t0,t1]})
  assert.equal(noExtras.candidate_hypotheses,undefined);assert.equal(noExtras.sampled_relation_edges,undefined)
  assert.throws(()=>queryEvidence(arms,{source_sha256:'different',node_id:child.id,bounds_pts:[t0,t1]},{m2:true}),/SOURCE/)
  const stored=await call('evidence_detail',{result_ref:child.refs[0],path:[]});assert.equal(stored.result_ref,child.refs[0])
  return {source_sha256:opened.video_sha256,old_node:initial.id,actual_child:child,original_evidence_ids:evidenceIds,
    images:refinement.images,reference:reference.items,indexed_state_items:joint.items,joined_study_records:match,
    logical_module_switches_and_cross_source_rejection:true,'new_pts_outside_catalogue':0}
}

const result={}
try {result.round7_units=await repeatUnits();result.amr_link=await linkAMR();result.passed=true}
catch(error){result.failure=error.stack;process.exitCode=1}
finally{await writeFile(resolve(out,'result.json'),JSON.stringify(result,null,2),{flag:'wx'}); console.log(result.passed??result.failure)}
