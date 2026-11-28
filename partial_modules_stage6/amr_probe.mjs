import assert from 'node:assert/strict'
import {execFileSync} from 'node:child_process'
import {createHash} from 'node:crypto'
import {mkdir,readFile,writeFile} from 'node:fs/promises'
import {join,resolve} from 'node:path'
import {apply} from '../dacf_round6/persistent.mjs'

const study=resolve('partial_modules_stage6')
const dir=join(study,'one-boundary-probe')
await mkdir(dir)
const spec=JSON.parse(await readFile(join(study,'inputs/cases.json'),'utf8')).reel
const autoFile=join(study,'automatic/reel/auto.json')
const original=await readFile(autoFile)
const prediction=JSON.parse(original)
const candidate=prediction.accepted[0]
assert.ok(candidate)
const source=resolve(spec.source)
const p=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_frames','-show_streams',
  '-show_entries','frame=best_effort_timestamp:stream=time_base,width,height','-of','json',source],{encoding:'utf8'}))
const {time_base,width,height}=p.streams[0]
const [num,den]=time_base.split('/').map(Number)
const clock=p.frames.map((r,frame)=>({frame,pts:r.best_effort_timestamp,seconds:r.best_effort_timestamp*num/den}))
const [left,right]=candidate.pts
const before=clock.find(r=>r.pts>left&&r.pts<right&&!prediction.pool_pts.includes(r.pts))
assert.ok(before)
const node=(id,bounds,nominees,parent_id,children,known_pts)=>({id,kind:parent_id?'frozen_r2_region':'source',
  parent_id,children,bounds_pts:bounds,start:bounds[0]*num/den,end:bounds[1]*num/den,nominees,known_pts,stops:{}})
const plan={sha256:spec.sha256,time_base,width,height,clock,
  nodes:[node('source',spec.extent,prediction.first_pts,null,[candidate.id],prediction.pool_pts),
    node(candidate.id,candidate.pts,prediction.first_pts.filter(p=>p>=left&&p<=right),'source',[],[left,right])],edges:[]}
const planPath=join(dir,'supplied_plan.json')
await writeFile(planPath,JSON.stringify(plan,null,2),{flag:'wx'})
const tools=new Map(), saved=new Map(), sha=data=>createHash('sha256').update(data).digest('hex')
const services={llm:{resolveModelInfo:async()=>({inputModalities:['image']})},attachments:{saveImage:async({data,mediaType})=>{
  const digest=sha(data);saved.set(digest,data)
  return {attachmentId:'sha256:'+digest,mediaType,width:data.readUInt32BE(16),height:data.readUInt32BE(20),bytes:data.length}
}}}
apply({effect:f=>f(),tools:{register:t=>tools.set(t.name,t)},get:k=>services[k]},
  {arm:'A2',automaticPolicy:'S0',authorizedSource:source,planPath,cacheDir:join(dir,'cache'),
    pythonPath:resolve('.venv/Scripts/python.exe'),controlLog:join(dir,'calls.jsonl')})
const exec={agent:{id:'stage6-single-scope-refinement',session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
const call=(key,params)=>tools.get(key).execute(params,exec)
const start=performance.now()
await call('video_open',{path:source})
await call('observation_goal',{goal:'Check one existing candidate bracket at an actual source address.'})
const refined=await call('amr_refine',{node_id:candidate.id,mode:'time',pts:[before.pts],reason:'Independent spot check without revising the frozen bracket.'})
assert.equal(refined.images.length,3)
assert.ok(refined.summary.originals[before.pts])
const after=await call('amr_query',{node_id:refined.summary.children[0].id,detail:'proof'})
assert.ok(after.original_addresses.items.includes(before.pts))
assert.equal(sha(original),sha(await readFile(autoFile)))
const digests=[]
for(const [key,data] of saved){
  digests.push(key)
  await writeFile(join(dir,key+'.png'),data,{flag:'wx'})
}
const receipt={source_sha256:spec.sha256,candidate,coarse_pool_did_not_have:before,
  summary:refined.summary,child_id:after.id,proof:after.proof,actual_image_payload_digests:digests,
  envelope_sha_unchanged:sha(original),elapsed_ms:performance.now()-start}
await writeFile(join(dir,'result.json'),JSON.stringify(receipt,null,2),{flag:'wx'})
console.log(before.pts,refined.summary.host_status,saved.size)
