import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {mkdir,readFile,readdir,writeFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {apply as official} from '../video_observer_offline060/index.js'
import {apply as amr} from '../dacf_round6/persistent.mjs'

const study=resolve('partial_modules_stage6')
const out=resolve(study,process.argv[2]??'host')
assert.equal(resolve(out,'..'),study)
await mkdir(out)
const input=resolve('dacf_round6/independent')
const source=resolve(input,'source.mp4')
const plan=JSON.parse(await readFile(resolve(input,'plan.json'),'utf8'))
const sha=data=>createHash('sha256').update(data).digest('hex')
const attachments=new Map()
const services={
  llm:{resolveModelInfo:async()=>({inputModalities:['image']})},
  attachments:{saveImage:async({data,mediaType})=>{
    const hash=sha(data)
    attachments.set(hash,data)
    return {attachmentId:'sha256:'+hash,mediaType,width:data.readUInt32BE(16),height:data.readUInt32BE(20),bytes:data.length}
  }}
}
const config={integration:true,cacheDir:resolve(out,'cache'),pythonPath:resolve('.venv/Scripts/python.exe')}
function install(name,apply,settings){
  const tools=new Map()
  const exec={agent:{id:name,session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
  apply({effect:f=>f(),tools:{register:t=>tools.set(t.name,t)},get:k=>services[k]},settings)
  return {tools,call:(tool,params)=>tools.get(tool).execute(params,exec)}
}
const baseline=install('stage5-official',official,config)
const b=baseline.call
const opened=await b('video_open',{path:source})
assert.equal(opened.video_sha256,plan.sha256)
assert.ok(!baseline.tools.has('amr_query')&&!baseline.tools.has('amr_refine'))
await b('observation_goal',{goal:'Verify that explicit source access works with no scene index.'})
const candidate=plan.nodes.find(n=>n.id!=='source'&&Object.values(n.stops??{}).length===4&&
  Object.values(n.stops).every(s=>s.S0==='stopped')&&plan.clock.some(r=>r.pts>n.bounds_pts[0]&&r.pts<n.bounds_pts[1]))
assert.ok(candidate)
const pool=new Set(plan.nodes.find(n=>n.id==='source').known_pts)
const probe=plan.clock.find(r=>r.pts>candidate.bounds_pts[0]&&r.pts<candidate.bounds_pts[1]&&!pool.has(r.pts))??
  plan.clock.find(r=>r.pts>candidate.bounds_pts[0]&&r.pts<candidate.bounds_pts[1])
const ordinary=await b('delivery_request',{pts:[probe.pts],reason:'Check an original with no scene or AMR filtering.'})
assert.ok(ordinary.images.length===1)
const ordinaryDigest=ordinary.images[0].attachmentId
const scanRoot=resolve(config.cacheDir,'observer-index-v5')
const index=resolve(scanRoot,(await readdir(scanRoot)).find(n=>n.startsWith(plan.sha256)))
async function compareIndex(){
  return Object.fromEntries(await Promise.all(['manifest.json','catalog.json','wave.json'].map(async f=>[f,sha(await readFile(resolve(index,f)))])))
}
const before=await compareIndex()
const live=install('stage5-research',amr,{...config,arm:'A2',authorizedSource:source,planPath:resolve(input,'plan.json'),
  automaticPolicy:'S0',controlLog:resolve(out,'calls.jsonl')}).call
await live('video_open',{path:source})
await live('observation_goal',{goal:'Continue a stopped branch with a source address.'})
const shown=await live('amr_query',{node_id:candidate.id,detail:'proof'})
assert.ok(shown.can_request_time_refinement)
assert.ok(Object.values(shown.policy_hint.statuses).every(s=>s.state==='stopped'))
const refined=await live('amr_refine',{node_id:candidate.id,mode:'time',pts:[probe.pts],reason:'Verify autonomous refinement eligibility after automatic stop.'})
assert.equal(refined.images.length,3)
assert.ok(refined.images.some(im=>im.attachmentId===ordinaryDigest))
const child=await live('amr_query',{node_id:refined.summary.children[0].id,detail:'proof'})
assert.ok(child.original_addresses.items.includes(probe.pts))
assert.deepEqual(before,await compareIndex())
const hashes=[]
for(const [hash,data] of attachments){
  hashes.push(hash)
  await writeFile(resolve(out,hash+'.png'),data,{flag:'wx'})
}
const result={source_sha256:plan.sha256,no_scene_or_amr_tools_in_official:true,original_without_scene:ordinary.envelope,
  original_attachment_without_scene:ordinaryDigest,probe,catalogue_and_pool_hashes_unchanged:before,
  checked_stopped_node:shown, refinement:refined.summary,new_child:child.id, native_attachment_digests:hashes,
  two_tool_paths_used_same_pixel_payload:true, scope:'real source and native worker, harness attachment service supplied by the test; no model call'}
await writeFile(resolve(out,'result.json'),JSON.stringify(result,null,2),{flag:'wx'})
console.log(candidate.id,probe.pts,hashes.length)
