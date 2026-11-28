import {apply as original} from './measure.mjs'
import {appendFileSync} from 'node:fs'
export const name='amr-wire-proof'
export const inject=['tools','systemPrompt','deepseekLlmApiExtensions']
export function apply(ctx,cfg){
 original(ctx,cfg)
 let first=true
 ctx.deepseekLlmApiExtensions.register('r6_wire_system_proof',{prepare(request){
  if(!first||request.purpose)return
  first=false
  const {system,tools=[],messages=[]}=request.body
  appendFileSync(cfg.requestAudit+'.effective',JSON.stringify({system,tools,
   role_counts:Object.fromEntries([...new Set(messages.map(m=>m.role))].map(role=>[role,messages.filter(m=>m.role===role).length]))})+'\n')
 }})
}
