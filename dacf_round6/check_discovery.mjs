import assert from 'node:assert/strict'
import {mkdir,readFile,writeFile} from 'node:fs/promises'
import {resolve} from 'node:path'
import {apply} from './discovery_corrected.mjs'
const dest=resolve('dacf_round6/discovery-engineering');await mkdir(dest)
const base=resolve('dacf_round6/independent'),source=resolve(base,'source.mp4')
const bind=arm=>{
 const tools=new Map(),exec={agent:{id:arm,session:{requestHeader:()=>({config:{provider:'test',model:'test'}})}}}
 apply({effect:f=>f(),get:()=>null,tools:{register:t=>tools.set(t.name,t)},deepseekLlmApiExtensions:{register:()=>{}}},{arm,
  authorizedSource:source,planPath:resolve(base,'plan.json'),cacheDir:resolve(dest,arm),pythonPath:resolve('.venv/Scripts/python.exe'),
  controlLog:resolve(dest,arm+'.jsonl'),requestAudit:resolve(dest,arm+'-wire.jsonl')})
 return {tools,call:(n,a)=>tools.get(n).execute(a,exec)}
}
const sides=[bind('A2'),bind('A3')];const [a,b]=sides
await Promise.all(sides.map(x=>x.call('video_open',{path:source})))
assert.deepEqual([...a.tools].map(([n,t])=>[n,t.description,t.parameters]),[...b.tools].map(([n,t])=>[n,t.description,t.parameters]))
const args={page_size:14}, findings=[]
for(const tool of ['catalog_search','wave_query']){
  const [before,after]=await Promise.all(sides.map(x=>x.call(tool,args)))
  const parsed=after.note.match(/\nOptional related evidence: (.+)$/)
  assert.ok(parsed,tool)
  assert.equal(after.note.slice(0,parsed.index),before.note)
  assert.deepEqual({...after,note:before.note},before)
  const rendered=b.tools.get(tool).output.render(args,after)
  assert.ok(rendered.some(x=>x.text.includes(parsed[1])))
  const hint=JSON.parse(parsed[1]);const links=await Promise.all(sides.map(x=>x.call('amr_query',hint.query.arguments)))
  assert.deepEqual(links[0],links[1])
  findings.push({tool,hint,ref_equal:true,wire_text:rendered})
}
await writeFile(resolve(dest,'result.json'),JSON.stringify({same_tools:true,only_note_added:true,findings},null,2),{flag:'wx'})
