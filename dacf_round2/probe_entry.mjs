import { appendFile } from 'node:fs/promises'
import { readFileSync } from 'node:fs'
import { apply as shared } from '../evidenweft_phase1/control_plugin.mjs'

export const name = 'dacf-investigation-probe'
export const inject = ['tools']

export function apply(ctx, config) {
  shared(ctx, config)
  if (!config.regionPlan) return
  const plan = JSON.parse(readFileSync(config.regionPlan, 'utf8'))
  ctx.tools.register({
    name: 'investigation_regions',
    description: 'Find time ranges between existing visual observations that may justify a closer look. Metadata only; use source_resample or delivery_request to choose native originals yourself.',
    parameters: {type:'object',properties:{
      start:{type:'number',description:'earliest source seconds'},end:{type:'number',description:'latest source seconds'},
      cursor:{type:'integer'},page_size:{type:'integer'},
    }},
    output:{schema:{type:'object',additionalProperties:true},render:(_a,v)=>[{type:'text',text:JSON.stringify(v)}]},
    async execute(args) {
      const values=plan.filter(r=>r.end>= (args.start??-Infinity) && r.start <= (args.end??Infinity))
      const cursor=Math.max(0,args.cursor??0), page=Math.max(1,Math.min(args.page_size??12,30))
      const items=values.slice(cursor,cursor+page)
      await appendFile(config.controlLog, JSON.stringify({event:'investigation_query',args,regionIds:items.map(r=>r.id)})+'\n')
      return {items,total:values.length,next_cursor:cursor+page<values.length?cursor+page:null,
        scope:'Candidate investigation windows. The comparison supports visual difference only. Interior visibility and physical continuity remain unknown.'}
    },
  })
  void appendFile(config.controlLog, JSON.stringify({event:'investigation_tool_registered'})+'\n')
}
