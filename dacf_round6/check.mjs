import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {copyFile,mkdir,readFile,writeFile,appendFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {apply} from './persistent.mjs'
const out=resolve(process.argv[2]??'dacf_round6/engineering');await mkdir(out)
const location=resolve('dacf_round5/inputs/fixture')
const plan=JSON.parse(await readFile(resolve(location,'plan.json'),'utf8'))
const source=resolve(out,'source.mp4');await copyFile(resolve(location,'source.mp4'),source)
const services={llm:{resolveModelInfo:async()=>({inputModalities:['image']})},attachments:{saveImage:async({data,mediaType})=>({
  attachmentId:'sha256:'+createHash('sha256').update(data).digest('hex'),mediaType,bytes:data.length,width:data.readUInt32BE(16),height:data.readUInt32BE(20)})}}
const bind=(id)=>{
 const tools=new Map(),exec={agent:{id,session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
 apply({effect:f=>f(),get:k=>services[k],tools:{register:t=>tools.set(t.name,t)}},{arm:'A2',authorizedSource:source,
   planPath:resolve(location,'plan.json'),cacheDir:resolve(out,'cache'),pythonPath:resolve('.venv/Scripts/python.exe')})
 return {exec,tools,call:(n,a)=>tools.get(n).execute(a,exec)}
}
const {call,tools}=bind('r6-durable-session')
await call('video_open',{path:source})
await call('observation_goal',{goal:'Verify references, tree recovery and exact native data.'})
await assert.rejects(call('amr_query',{node_id:'missing'}),/UNKNOWN_NODE.*time-range query/)
await assert.rejects(call('amr_refine',{node_id:'source',mode:'time',pts:[999999],reason:'Check invalid timestamps.'}),/INVALID_PTS.*interior PTS/)
await assert.rejects(call('amr_refine',{node_id:'source',mode:'space',pts:[0,999999],native_xywh:[60,60,160,120],reason:'Check invalid timestamps.'}),/INVALID_PTS.*exact source PTS/)
const candidates=await call('amr_query',{start:0,end:2,page_size:4})
const wave=await call('amr_query',{node_id:'source',detail:'wave'})
const links=await call('amr_query',{node_id:'source',detail:'links'})
assert.ok(candidates.total>0&&wave.wave&&links.links)
const time=await call('amr_refine',{node_id:'source',mode:'time',reason:'Compare the original samples between the endpoints'})
assert.ok(time.images.length===3&&time.summary.version===1)
const pts=time.summary.pts
const space=await call('amr_refine',{node_id:'source',mode:'space',pts:[pts[0],pts[1]],native_xywh:[60,60,160,120],crop_pts:pts[1],reason:'Test an ordinary local comparison.'})
assert.ok(space.images.length===3&&space.summary.version===2)
const n=time.summary.children[0].id
const child=await call('amr_query',{node_id:n,detail:'proof'})
assert.ok(child.proof.local_measurements.temporal.measurements.length===3)
const detail=await call('evidence_detail',{result_ref:child.refs[0],path:[]})
assert.ok(detail.result_ref)
const fresh=bind('r6-durable-session')
await fresh.call('video_open',{path:source})
const recovered=await fresh.call('amr_query',{node_id:n,detail:'proof'})
assert.deepEqual(child,recovered)
const receipt=await fresh.call('evidence_reference',{ids:[space.summary.derived]})
assert.equal(receipt.items[0].type,'derived')
const another=bind('r6-other-session');await another.call('video_open',{path:source})
await assert.rejects(another.call('amr_query',{node_id:n}),/UNKNOWN_NODE/)
await appendFile(source,Buffer.from([0]))
await assert.rejects(fresh.call('amr_query',{node_id:n}),/STALE_SOURCE/)
await copyFile(resolve(location,'source.mp4'),source)
const normal=await fresh.call('amr_query',{node_id:n,detail:'proof'})
assert.deepEqual(normal,recovered)
await writeFile(resolve(out,'result.json'),JSON.stringify({status:'passed',time:time.summary,space:space.summary,recovered,receipt,
 scopes:['same stable session restore','other session denied','changed source denied then identical source restored'],
 recovery_messages:['UNKNOWN_NODE','INVALID_PTS (time)','INVALID_PTS (space)'],
 guide_examples:['range','node waveform','node links','time refine','space refine','proof handle'],
 registered_tool_count:tools.size,image_host:'mock transport with actual source and native worker'},null,2))
