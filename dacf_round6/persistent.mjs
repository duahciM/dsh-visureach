import { appendFile, realpath } from 'node:fs/promises'
import { readFileSync } from 'node:fs'
import { randomUUID, createHash } from 'node:crypto'
import {join} from 'node:path'
import {treeJournal} from './journal.mjs'
import { apply as visureach } from '../dacf_round5/release-fixed/installed/package/baseline/index.js'

export const name='visureach-amr-research'
export const inject=['tools']
const str={type:'string'}, num={type:'number'}, integer={type:'integer'}
const plain={schema:{type:'object',additionalProperties:true},render:(_,value)=>[{type:'text',text:JSON.stringify(value)}]}
const imagesSchema={type:'array',items:{type:'object',additionalProperties:true}}
const lossless=value=>JSON.parse(JSON.stringify(value))
const cite=(v)=>Object.fromEntries([...v.envelope.matchAll(/pts=(-?\d+).*?evidence_id=(ev-[a-f0-9]+)/g)].map(m=>[Number(m[1]),m[2]]))
const page=(rows,a)=>{
  const from=Math.max(0,a.cursor??0), size=Math.min(24,Math.max(1,a.page_size??8))
  return {items:rows.slice(from,from+size),total:rows.length,next_cursor:from+size<rows.length?from+size:null}
}
const AMR_GUIDE='The tree indexes computed evidence and actual sample addresses. Stop hints describe the policy and scope only. Querying a node does not deliver an image. Use delivery_request or source_resample at any legal time; refinement is optional.'

export function apply(ctx,config={}) {
  const arm=config.arm??'A2', hierarchical=arm==='A2'
  const policy=config.automaticPolicy??'S0'
  if(!['A0','A1','A2'].includes(arm)||!['S0','S1'].includes(policy)) throw new Error('Use A0/A1/A2 and S0 or S1.')
  if(arm!=='A0'&&(!config.planPath||!config.authorizedSource)) throw new Error('Study modes A1/A2 need a source and prepared plan. See TOOLS.md.')
  const plan=config.planPath?JSON.parse(readFileSync(config.planPath,'utf8')):null
  const journal=hierarchical&&config.cacheDir?treeJournal(join(config.cacheDir,'amr-v1.sqlite')):null
  const planHash=plan?createHash('sha256').update(readFileSync(config.planPath)).digest('hex'):null
  const scopeKey=exec=>String(exec.agent.id)
  const baseMap=()=>new Map((plan?.nodes??[]).map(n=>[n.id,structuredClone(n)]))
  const verifySource=()=>{if(plan&&createHash('sha256').update(readFileSync(config.authorizedSource)).digest('hex')!==plan.sha256)throw new Error('STALE_SOURCE. Restore original bytes or prepare a new plan.')}
  const clock=new Map((plan?.clock??[]).map(r=>[r.pts,r]))
  const tools=new Map(), bindings=new WeakMap()
  const log=async row=>{if(config.controlLog) await appendFile(config.controlLog,JSON.stringify({at:new Date().toISOString(),...row})+'\n','utf8')}
  const session=(exec)=>{
    verifySource()
    const binding=bindings.get(exec.agent.session)
    if(!binding) throw new Error('Open the assigned source with video_open before asking about its evidence.')
    return binding
  }
  const lookup=(map,key)=>{
    const n=map.get(key)
    if(!n) throw new Error('UNKNOWN_NODE. First use a time-range query, then pass one of its exact node IDs.')
    return n
  }
  const inherited=async(name,args,exec,reason)=>{
    const started=Date.now()
    try {
      const result=await tools.get(name).execute(args,exec)
      await log({event:'tool_compute',tool:name,reason,args,elapsed_ms:Date.now()-started,status:result.status??'computed',
        output_image_count:result.images?.length??0,result_ref:result.result_ref??null})
      return result
    } catch(e) {
      await log({event:'tool_compute',tool:name,reason,args,elapsed_ms:Date.now()-started,status:'failed',error:e.message})
      throw e
    }
  }
  function brief(n) {
    const {id,start,end,bounds_pts,kind,nominees,known_pts,children,parent_id,stops}=n
    const statuses=Object.fromEntries(Object.entries(stops??{}).map(([key,r])=>[key,{state:r[policy],scope:r[policy+'_scope']}]))
    return {id,start,end,bounds_pts,kind,nomination_count:nominees.length,known_original_address_count:known_pts.length,
      policy_hint:{policy,statuses},can_request_time_refinement:hierarchical&&plan.clock.some(r=>r.pts>bounds_pts[0]&&r.pts<bounds_pts[1]),
      ...(hierarchical?{parent_id,children}:{}),physical_continuity:'unknown'}
  }
  const chosenArg={type:'object',properties:{start:num,end:num,cursor:integer,page_size:integer,node_id:str,
    detail:{type:'string',enum:['node','proof','wave','links','guide']}}}

  const registerBaseline=tool=>{
    tools.set(tool.name,tool)
    const extended=hierarchical&&tool.name==='evidence_routes'
    ctx.tools.register({...tool,...(extended?{parameters:{...tool.parameters,
      properties:{...tool.parameters.properties,amr_node_id:str}},description:tool.description+' Use optional amr_node_id to focus on a known tree node.'}:{}),
      async execute(args,exec){
        if(tool.name==='video_open') {
          if(config.authorizedSource && await realpath(args.path)!==await realpath(config.authorizedSource)) throw new Error('Only the assigned study source can be opened.')
          verifySource()
          const opened=await inherited(tool.name,args,exec,'model')
          if(plan&&opened.video_sha256!==plan.sha256) throw new Error('Source changed; create a fresh plan for this source.')
          if(!bindings.has(exec.agent.session)) {
            const map=baseMap();if(journal)journal.restore(scopeKey(exec),plan.sha256,planHash,map)
            bindings.set(exec.agent.session,map)
          }
          return opened
        }
        if(extended&&args.amr_node_id) {
          const n=lookup(session(exec),args.amr_node_id)
          const {amr_node_id,start,end,...rest}=args
          const lo=Math.max(start??n.start,n.start), hi=Math.min(end??n.end,n.end)
          if(lo>hi) throw new Error('Requested range does not intersect the node.')
          const linked=await inherited(tool.name,{...rest,start:lo,end:hi},exec,'model_node_join')
          return lossless({...linked,amr_node:brief(n),known_calculation_refs:n.calculation_refs??[]})
        }
        return inherited(tool.name,args,exec,'model')
      }})
  }
  visureach(new Proxy(ctx,{get(t,k){
    if(k==='tools') return {register:registerBaseline}
    const v=Reflect.get(t,k,t);return typeof v==='function'?v.bind(t):v
  }}),{...config,integration:true})
  const additional=[]
  const register=tool=>{additional.push(tool.name);ctx.tools.register(tool)}
  if(arm!=='A0') register({name:hierarchical?'amr_query':'investigation_regions',parameters:chosenArg,output:plain,
    description:hierarchical?'Explore a source time range or exact node ID. Optional detail proof, wave, links, or guide. '+AMR_GUIDE:
      'Query the same saved investigation ranges and PTS in a flat list. Use independent original-image tools.',
    async execute(a,exec){
      if(a.detail==='guide') return {purpose:AMR_GUIDE,examples:hierarchical?['amr_query({start:0,end:2})',
        'amr_query({node_id:"source",detail:"wave"})','amr_refine({node_id:"source",mode:"time",reason:"check an uncertain interval"})']:['investigation_regions({start:0,end:2})'],
        recovery:'Use a found ID and actual integer PTS; image-space ROI is [x,y,width,height] in the original.'}
      const map=session(exec)
      await log({event:'amr_query',arm,arguments:a})
      if(!a.node_id) {
        if((a.start??0)>(a.end??Infinity)) throw new Error('Give start <= end in source seconds.')
        return lossless(page([...map.values()].filter(n=>n.start<=(a.end??Infinity)&&n.end>=(a.start??-Infinity)).map(brief),a))
      }
      const n=lookup(map,a.node_id)
      const value={...brief(n),nominees:page(n.nominees,a),original_addresses:page(n.known_pts,a),anchor_pts:n.anchor_pts??[],time_base:plan.time_base,
        unknown_interiors:page(n.unknown_interiors??[],a),native_roi:n.spatial_native,resolution_max_s:n.resolution_max_s,
        stops:n.stops,refs:n.calculation_refs??[],original_receipt_ids:n.original_receipt_ids??{},derived_receipt_ids:n.derived_receipt_ids??[]}
      if(a.detail==='wave') value.wave=await inherited('wave_query',{start:n.start,end:n.end,page_size:a.page_size??4},exec,'amr_query')
      if(a.detail==='links') value.links=await inherited('evidence_routes',{start:n.start,end:n.end,page_size:a.page_size??8},exec,'amr_query')
      if(a.detail==='proof') value.proof={anchor_and_signal:n.anchor_and_signal??null,local_measurements:n.local_measurements??null,
        nonhierarchical:page(plan.edges.filter(e=>e.nodes.includes(n.id)),a)}
      return lossless(value)
    }})
  if(hierarchical) register({name:'amr_refine',description:'Continue any node, including automatically stopped nodes. Time refinement uses an exact interior PTS or probes the nearest midpoint. Space refinement accepts existing exact PTS and native ROI; optional crop_pts requests a derived crop. Uses the shared original, Figure 1/2, and comparison chain. Choose scope and stop freely.',
    parameters:{type:'object',properties:{node_id:str,mode:{type:'string',enum:['time','space']},pts:{type:'array',items:integer},
      native_xywh:{type:'array',items:integer},crop_pts:integer,reason:str},required:['node_id','mode','reason']},
    output:{schema:{type:'object',properties:{summary:{type:'object',additionalProperties:true},envelope:str,images:imagesSchema},required:['summary','envelope','images']},
      render:(_,v)=>[{type:'text',text:v.envelope+'\n'+JSON.stringify(v.summary)},...v.images.map(attachment=>({type:'image',attachment}))]},
    async execute(a,exec){
      const map=session(exec), node=lookup(map,a.node_id)
      const [lo,hi]=node.bounds_pts
      const space=a.mode==='space'
      if(!['time','space'].includes(a.mode)||!a.reason?.trim()) throw new Error('Choose time or space and explain the observation purpose.')
      const legal=p=>Number.isSafeInteger(p)&&clock.has(p)&&p>=lo&&p<=hi
      let pts=[...new Set(a.pts??[])].sort((a,b)=>a-b)
      let probe
      let regions=[]
      if(space) {
        const roi=a.native_xywh
        if(pts.length<2||!pts.every(legal)) throw new Error('INVALID_PTS. Supply at least two exact source PTS inside this node.')
        if(!roi||roi.length!==4||!roi.every(Number.isSafeInteger)||roi[0]<0||roi[1]<0||roi[2]<2||roi[3]<2||roi[0]+roi[2]>plan.width||roi[1]+roi[3]>plan.height)
          throw new Error('INVALID_ROI. Use a native rectangle of at least 2x2 within the original.')
        if(a.crop_pts!=null&&!pts.includes(a.crop_pts)) throw new Error('crop_pts must be one of the supplied actual PTS.')
        const [x,y,w,h]=roi, aW=Math.floor(w/2), aH=Math.floor(h/2)
        regions=[[x,y,aW,aH],[x+aW,y,w-aW,aH],[x,y+aH,aW,h-aH],[x+aW,y+aH,w-aW,h-aH]]
      } else {
        const room=plan.clock.filter(r=>r.pts>lo&&r.pts<hi)
        if(!room.length) throw new Error('NO_INTERNAL_PTS. Query a wider parent or choose a legal source interval independently.')
        if(pts.length>1||pts[0]!=null&&(!legal(pts[0])||pts[0]===lo||pts[0]===hi)) throw new Error('INVALID_PTS. Supply one actual interior PTS or omit pts for a midpoint probe.')
        probe=pts[0]??room.reduce((best,row)=>Math.abs(row.seconds-(node.start+node.end)/2)<Math.abs(best.seconds-(node.start+node.end)/2)?row:best).pts
        pts=[lo,probe,hi]
      }
      const ref=space?pts.at(-1):probe
      const delivered=await inherited('delivery_request',{pts,reason:a.reason},exec,'amr_refine')
      const originals=cite(delivered)
      const refs=[], images=[...delivered.images], faults=[]
      let derived, comparison, geometry, temporal
      const attempt=async(name,args)=>{
        try {
          const r=await inherited(name,args,exec,'amr_refine')
          if(r.status&&r.status!=='completed'&&r.status!=='supported') faults.push({tool:name,status:r.status})
          if(r.result_ref) refs.push(r.result_ref)
          return r
        } catch(e) {faults.push({tool:name,reason:e.message});return null}
      }
      const figure=await attempt('relation_figure2',{pts:ref,incremental:true,context_pts:pts.filter(p=>p!==ref),regions_native:regions})
      if(space) {
        comparison=await attempt('compare_object_views',{mode:'direct',prototypes:[{pts:pts[0],native_xywh:a.native_xywh}],
          samples:pts.slice(1).map(p=>({pts:p,native_xywh:a.native_xywh}))})
        geometry={source:'compare_object_views',proof:comparison?.result_ref??null,state:comparison?'inspect_stored_heldout_scope':'unknown'}
        if(a.crop_pts!=null&&originals[a.crop_pts]) {
          const result=await attempt('evidence_crop',{evidence_id:originals[a.crop_pts],native_xywh:a.native_xywh,reason:a.reason})
          if(result) {derived=JSON.parse(result.description);images.push(...result.images)}
        }
      } else {
        const measurements=[]
        for(const [a,b] of [[lo,hi],[lo,probe],[probe,hi]]) {
          const row=await attempt('visual_compare',{pts_a:a,pts_b:b})
          measurements.push({pts:[a,b],ref:row?.result_ref,root:row?.root,edge:row?.channels?.edge_xor_fraction_unregistered??null,
            hsv_total_variation:row?.channels?.hsv_total_variation??null})
        }
        temporal={measurements,scope:'Whole image grid and HSV total variation measured from exact frames, not the registered residual; other path channels not recomputed here.',
          new_minus_old:Object.fromEntries(['edge','hsv_total_variation'].map(k=>[k,measurements.every(r=>r[k]!=null)?measurements[1][k]+measurements[2][k]-measurements[0][k]:null]))}
      }
      const ids=[], durableChildren=[]
      const scope=space?regions.map(roi=>({bounds:[lo,hi],roi})): [[lo,probe],[probe,hi]].map(bounds=>({bounds,roi:node.spatial_native}))
      const batch=randomUUID().slice(0,8)
      for(const [i,child] of scope.entries()) {
        const [a,b]=child.bounds
        const id=`${node.id}/${space?'R':'T'}${i+1}-${batch}`
        const n={id,parent_id:node.id,children:[],kind:space?'session_spatial':'session_temporal',start:clock.get(a).seconds,end:clock.get(b).seconds,bounds_pts:child.bounds,
          nominees:node.nominees.filter(p=>p>=a&&p<=b),known_pts:pts.filter(p=>p>=a&&p<=b),spatial_native:child.roi,
          original_receipt_ids:originals,calculation_refs:refs,derived_receipt_ids:derived?[derived.derived_id]:[],
          unknown_interiors:pts.slice(1).map((p,i)=>[pts[i],p]).filter(([p,q])=>p>=a&&q<=b&&clock.get(q).frame-clock.get(p).frame>1),
          stops:Object.fromEntries(['gray','hsv','edge','clip'].map(ch=>[ch,{S0:'not_evaluated',S1:'not_evaluated',S0_scope:'session_child',S1_scope:'session_child'}])),
          local_measurements:{geometry,temporal,figure:figure?.result_ref??null}}
        n.resolution_max_s=Math.max(0,...n.known_pts.slice(1).map((p,i)=>clock.get(p).seconds-clock.get(n.known_pts[i]).seconds))
        durableChildren.push(n);ids.push(brief(n))
      }
      const version=journal?.add(scopeKey(exec),plan.sha256,planHash,{parent:node.id,children:durableChildren,refs})??null
      for(const n of durableChildren){map.set(n.id,n);node.children.push(n.id)}
      node.calculation_refs=[...new Set([...(node.calculation_refs??[]),...refs])]
      const record={event:'amr_refinement',node:node.id,mode:a.mode,pts,originals,refs,transport_candidate_images:images.length,
        version,host_status:faults.length?'partial':'computed',faults,children:ids,derived:derived?.derived_id??null,prior_stop:node.stops}
      await log(record)
      return lossless({summary:{...record,figure:figure?.summary??null,geometry,temporal},images,envelope:delivered.envelope
        +(derived?`\nDerived crop ${derived.derived_id}, parent ${derived.parent_evidence_id}, PTS ${derived.pts}, ROI ${derived.native_xywh.join(',')}.`:'')})
    }})
  void log({event:'tool_surface',tools:[...tools.keys()],additional,arm,automaticPolicy:policy})
}
