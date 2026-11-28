import {appendFileSync,readFileSync} from 'node:fs'
import {createHash} from 'node:crypto'
import {apply as persistent} from './persistent.mjs'
export const name='amr-round6-discovery'
export const inject=['tools','deepseekLlmApiExtensions']
const JSONcopy=x=>JSON.parse(JSON.stringify(x))
const relevant=new Set(['catalog_search','wave_query','relation_figure2','visual_compare','evidence_routes','evidence_search'])
export function apply(ctx,config){
  const show=config.arm==='A3'
  const plan=JSON.parse(readFileSync(config.planPath,'utf8'))
  const sessions=new WeakMap()
  const log=(row)=>appendFileSync(config.controlLog,JSON.stringify(row)+'\n')
  const settings={...config,arm:show?'A2':config.arm}
  const bridge=new Proxy(ctx,{get(target,k){
    if(k==='tools')return {register(tool){ target.tools.register({...tool,async execute(a,e){
      const result=JSONcopy(await tool.execute(a,e))
      if(!show)return result
      if(tool.name==='video_open') sessions.set(e.agent.session,[])
      const layers=sessions.get(e.agent.session)
      if(tool.name==='amr_refine'&&layers) layers.push(...(result.summary.children??[]))
      if(!layers||!relevant.has(tool.name)||result.status==='failed'||result.status==='original_required')return result
      const pts=new Set([a.pts,a.pts_a,a.pts_b,a.anchor_pts,a.compare_to_pts,
         ...(result.items??[]).slice(0,8).map(r=>r.pts),
         ...[...String(result.frames??'').matchAll(/pts=(-?\d+)/g)].slice(0,8).map(m=>Number(m[1]))].filter(Number.isSafeInteger))
      const intervals=tool.name==='wave_query'?[...String(result.cues??'').matchAll(/^\d+\.\s+(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)s/gm)].slice(0,3).map(m=>[Number(m[1]),Number(m[2])]):[]
      const eligible=[...layers,...plan.nodes.filter(n=>n.id!=='source')].filter(n=>
        [...pts].some(p=>p>=n.bounds_pts[0]&&p<=n.bounds_pts[1])||intervals.some(([l,h])=>n.start<=h&&n.end>=l))
      const n=eligible.sort((a,b)=>a.end-a.start-(b.end-b.start)||a.start-b.start||a.id.localeCompare(b.id))[0]
      if(!n)return result
      const hint={node_id:n.id,kind:n.kind,bounds_seconds:[n.start,n.end],
        status:n.policy_hint??(n.stops?{S0:n.stops.edge?.S0,scope:n.stops.edge?.S0_scope}:null),
        provenance:pts.size?'overlap with an exact result or requested source PTS':'overlap with a returned waveform cue',
        query:{tool:'amr_query',arguments:{node_id:n.id,detail:'node'}}}
      if(typeof result.note==='string') result.note+='\nOptional related evidence: '+JSON.stringify(hint)
      else result.optional_related_evidence=hint
      log({event:'related_hint',tool:tool.name,node:n.id,model_visible_channel:typeof result.note==='string'?'renderer_note':'plain_result'})
      return result
    }})}}
    const v=Reflect.get(target,k,target);return typeof v==='function'?v.bind(target):v
  }})
  persistent(bridge,settings)
  let index=0
  const hash=v=>createHash('sha256').update(JSON.stringify(v)).digest('hex')
  ctx.deepseekLlmApiExtensions.register('r6_discovery_local_audit',{prepare(request){
    if(request.purpose)return
    const {tools=[],system=''}=request.body
    const ix=++index
    appendFileSync(config.requestAudit,JSON.stringify({ix,toolHash:hash(tools),systemHash:hash(system),...(ix===1?{tools,system}:{})})+'\n')
  }})
}
