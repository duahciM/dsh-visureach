import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {mkdir,readFile,writeFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {apply} from '../video_observer_offline060/index.js'

const home=resolve('partial_modules_stage3')
const out=resolve(home,process.argv[2]??'engineering')
if (!out.startsWith(home+'/'.replace('/',process.platform==='win32'?'\\':'/'))) throw new Error('Use a study-local result.')
await mkdir(out)
const sha=x=>createHash('sha256').update(x).digest('hex')
const digests=new Map(),tools=new Map()
const host={llm:{resolveModelInfo:async()=>({inputModalities:['image']})},attachments:{async saveImage({data,mediaType}){
  const h=sha(data);digests.set(h,data)
  return {attachmentId:'sha256:'+h,mediaType,bytes:data.length,width:data.readUInt32BE(16),height:data.readUInt32BE(20)}
}}}
const ctx={tools:{register:t=>tools.set(t.name,t)},get:k=>host[k],effect:f=>f()}
apply(ctx,{cacheDir:resolve(out,'cache'),pythonPath:resolve('.venv/Scripts/python.exe'),integration:true})
const exec={agent:{id:'stage3-offline',session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
const call=(n,a)=>tools.get(n).execute(a,exec)
const locked=JSON.parse(await readFile(resolve(home,'references/sealed.json'),'utf8')).targets.roles
const offset=Number(process.argv[3]??0)
const rows=locked.filter(r=>r.role==='relation_participant').slice(offset,offset+2)
const a=rows[0].pts,b=rows[1].pts
const actors=[a,b].map(p=>locked.find(r=>r.pts===p&&r.episode==='target_before'))
const source=resolve(home,'inputs/roles/source.mkv')
const receipt={}
try{
  const opened=await call('video_open',{path:source})
  assert.equal(opened.video_sha256,sha(await readFile(source)))
  await call('observation_goal',{goal:'Compare a moving candidate and static spatial reference at real sample ticks.'})
  const original=await call('delivery_request',{pts:[a,b],reason:'Verify original two-dimensional endpoints.'})
  const ids=[...original.envelope.matchAll(/evidence_id=(ev-[a-f0-9]+)\b/g)].map(m=>m[1])
  assert.ok(original.images.length===2 && ids.length===2)
  const [ea,eb]=ids
  const f=await call('relation_figure2',{pts:b,incremental:true,context_pts:[a],regions_native:[actors[1].box,rows[1].box]})
  assert.ok(f.result_ref)
  if(offset===2)assert.ok(f.summary.measured_pairs>0)
  const stored=await call('evidence_detail',{result_ref:f.result_ref,path:[]})
  assert.equal(stored.result_ref,f.result_ref)
  const candidate=await call('object_candidate',{action:'add',label:'moving visible patch',role:'prototype',view:{pts:b,native_xywh:actors[1].box},evidence_ids:[eb]})
  const match=await call('compare_object_views',{mode:'direct',reference_view_ids:[candidate.observation_id],samples:[{pts:a,native_xywh:actors[0].box}]})
  assert.ok(match.result_ref)
  const state=await call('state_chain',{action:'add',candidate_id:candidate.candidate_id,pts:b,identity_id:candidate.observation_id,
        phase:'observation',relationship:'bbox-projection-to-reference',content:'Two finite image regions at the exact original PTS.',
        evidence_ids:[eb]})
  assert.ok(state.state_id)
  const spatial=await call('evidence_routes',{start:0,end:3,channel:'state',candidate_id:candidate.candidate_id,page_size:8})
  assert.ok(spatial.items.some(r=>r.state_id===state.state_id))
  const provisional=await call('observation_record',{action:'create',content:'The stable local reference initially serves registration.',kind:'direct',evidence_ids:[ea]})
  const dependent=await call('observation_record',{action:'create',content:'The role summary depends on the initial registration-only interpretation.',
        kind:'inference',evidence_ids:[eb],depends_on_revisions:[provisional.revision_id]})
  const newRole=await call('observation_record',{action:'revise',observation_id:provisional.observation_id,
       supersedes_revision:provisional.revision_id,reason:'Review the new actual frame and the two projected ROIs.',
       content:'The stable reference also participates in an image-plane relation; physical contact is not inferred.',
       kind:'inference',evidence_ids:[ea,eb]})
  const re=await call('observation_review',{revision_id:dependent.revision_id})
  assert.deepEqual(re.items[0].needs_review_upstream,[provisional.revision_id])
  assert.equal((await call('evidence_detail',{result_ref:f.result_ref,path:[]})).result_ref,f.result_ref)
  const crop=await call('evidence_crop',{evidence_id:eb,native_xywh:rows[1].box,reason:'Review the stationary participant on its original parent.'})
  const derived=JSON.parse(crop.description).derived_id
  const refs=await call('evidence_reference',{ids:[eb,derived]})
  assert.equal(refs.items.find(v=>v.type==='derived').parent_evidence_id,eb)
  const geo=await call('visual_compare',{pts_a:a,pts_b:b})
  assert.ok(geo.result_ref)
  for(const [h,data] of digests)await writeFile(resolve(out,h+'.png'),data,{flag:'wx'})
  Object.assign(receipt,{passed:true,source_sha256:opened.video_sha256,pts:[a,b],source_power:ids,role_roi:rows,
     region_figure:f,stored_relationship_detail:stored,matched_candidate:match.result_ref,spatial_query:spatial,
     heldout_full_pair_ref:geo.result_ref,revision_before:provisional.revision_id,revision_after:newRole.revision_id,
     dependents:re.items,reference_and_crop:refs.items,typed_png_sha256:[...digests.keys()]})
}catch(e){receipt.failure=e.stack;process.exitCode=1}
finally{await writeFile(resolve(out,'result.json'),JSON.stringify(receipt,null,2),{flag:'wx'});console.log(receipt.passed??receipt.failure)}
