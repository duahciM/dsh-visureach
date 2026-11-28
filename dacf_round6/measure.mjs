import {readFileSync,appendFileSync} from 'node:fs'
import {createHash} from 'node:crypto'
import {apply as r51} from '../dacf_round5/release-fixed/installed/package/transport.mjs'
export const name='amr-round6-study'
export const inject=['tools','systemPrompt','deepseekLlmApiExtensions']
const checksum=v=>createHash('sha256').update(JSON.stringify(v)).digest('hex')
export function apply(ctx,config){
  r51(ctx,config)
  const guide=JSON.parse(readFileSync(config.guidePath,'utf8'))
  const parts=({P0:[],P1:['overview'],P2:['examples'],P3:['overview','examples','advantages']})[config.condition]
  if(parts.length)ctx.systemPrompt.section({name:'amr:usage-note',order:5050,text:parts.map(k=>guide[k]).join('\n\n'),interpolate:false})
  let count=0
  const log=row=>appendFileSync(config.requestAudit,JSON.stringify(row)+'\n','utf8')
  ctx.deepseekLlmApiExtensions.register('r6_local_observation',{
    prepare(request){
      if(request.purpose) return
      const {messages=[],tools=[]}=request.body
      const system=messages.filter(m=>m.role==='system')
      const toolHash=checksum(tools), systemHash=checksum(system)
      const ix=++count
      log({ix,systemHash,toolHash,purpose:request.purpose??null,
        present:parts.map(k=>({section:k,present:JSON.stringify(system).includes(guide[k].replaceAll('"','\\"'))||system.some(m=>typeof m.content==='string'&&m.content.includes(guide[k]))})),
        ...(ix===1?{system,tools}:{}), message_count:messages.length})
      return undefined
    }
  })
}
