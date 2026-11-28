import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {mkdir, readFile, writeFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {apply} from '../video_observer_offline060/index.js'

const home=resolve('partial_modules_stage4')
const out=resolve(home,'host')
await mkdir(out)
const hash=data=>createHash('sha256').update(data).digest('hex')
const screenshots=new Map(), tools=new Map()
const host={llm:{resolveModelInfo:async()=>({inputModalities:['image']})}, attachments:{async saveImage({data,mediaType}){
  const sha=hash(data)
  screenshots.set(sha,data)
  return {attachmentId:'sha256:'+sha,mediaType,bytes:data.length,width:data.readUInt32BE(16),height:data.readUInt32BE(20)}
}}}
apply({tools:{register:t=>tools.set(t.name,t)},get:k=>host[k],effect:f=>f()}, {cacheDir:resolve(out,'cache'),
  pythonPath:resolve('.venv/Scripts/python.exe'),integration:true})
const exec={agent:{id:'stage4-local-audit',session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
const call=(name,params)=>tools.get(name).execute(params,exec)
const read=async file=>JSON.parse(await readFile(resolve(home,file),'utf8'))
const all=await read('evaluation.json')
const p=all.cohorts['new-local'].arms.R1P1.reference_scores.feature_support_detail[0]
const features=await read('predictions/new-local/region_features.json')
const patch=features.extra_geometry.feature_patches.find(r=>r.calculation_root===p.calculation_root)
const spec=(await read('inputs/source_inventory.json'))['new-local']
let receipt={}
try {
  const source=resolve(spec.path)
  const opened=await call('video_open',{path:source})
  assert.equal(opened.video_sha256,spec.sha256)
  assert.equal(hash(await readFile(source)),spec.sha256)
  await call('observation_goal',{goal:'Verify a bounded image reference from exact originals.'})
  const original=await call('delivery_request',{pts:patch.pts,reason:'Audit exact original source endpoints for a local reference.'})
  const ids=[...original.envelope.matchAll(/evidence_id=(ev-[a-f0-9]+)\b/g)].map(m=>m[1])
  assert.equal(ids.length,2)
  assert.equal(original.images.length,2)
  const crop=await call('evidence_crop',{evidence_id:ids[1],native_xywh:patch.rois[1],reason:'Inspect the small region supporting a reference.'})
  const derived=JSON.parse(crop.description).derived_id
  const annotation=await call('evidence_reference',{ids:[...ids,derived]})
  assert.equal(annotation.items.find(x=>x.type==='derived').parent_evidence_id,ids[1])
  const geometry=await call('compare_object_views',{mode:'direct',prototypes:[{pts:patch.pts[0],native_xywh:patch.rois[0]}],
    samples:[{pts:patch.pts[1],native_xywh:patch.rois[1]}]})
  assert.ok(geometry.result_ref)
  const details=await call('evidence_detail',{result_ref:geometry.result_ref,path:[]})
  assert.equal(details.result_ref,geometry.result_ref)
  for(const [sha,data] of screenshots) await writeFile(resolve(out,sha+'.png'),data,{flag:'wx'})
  receipt={passed:true,original_envelope:original.envelope,roi_comparison_request_scope:'published compare_object_views',analysis_ref:patch.calculation_root,
     source_sha256:opened.video_sha256,pts:patch.pts,roi:patch.rois,source_roots:patch.source_roots,refs:annotation.items,result:geometry,
     typed_png_sha256:[...screenshots.keys()]}
} catch(err){receipt={passed:false,error:err.stack};process.exitCode=1}
finally {await writeFile(resolve(out,'result.json'),JSON.stringify(receipt,null,2),{flag:'wx'});console.log(receipt.passed,receipt.error??'')}
