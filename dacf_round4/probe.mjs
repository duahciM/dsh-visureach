import { appendFile, mkdtemp, readFile, realpath, writeFile } from 'node:fs/promises'
import { readFileSync } from 'node:fs'
import { createHash, randomUUID } from 'node:crypto'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { execFile } from 'node:child_process'
import { apply as common } from '../evidenweft_phase1/control_plugin.mjs'

export const name = 'amr-research'
export const inject = ['tools']
const run = promisify(execFile)
const text = {type:'string'}, number={type:'number'}, integer={type:'integer'}
const plain={schema:{type:'object',additionalProperties:true},render:(_,v)=>[{type:'text',text:JSON.stringify(v)}]}
const fields={start:number,end:number,cursor:integer,page_size:integer}
const guide = {
  purpose:'This is an index of already computed evidence and proposed investigations. Automatic stops mark a policy boundary, and originals and further refinement remain available. Depth only describes subdivision.',
  examples:['amr_query({start:0,end:2})','amr_query({node_id:"source",detail:"node"})',
    'amr_query({node_id:"source",detail:"proof"})','amr_refine({node_id:"source",mode:"time"})',
    'amr_refine({node_id:"source",mode:"space",pts:[0,120],native_xywh:[10,10,120,80],crop_pts:120})'],
  recovery:'Use an actual ID returned by amr_query. Time bounds use source seconds, points use integer source ticks. For spatial work choose two or more observed PTS within the node and an in-bounds native [x,y,width,height] rectangle. When no interior frame exists try a wider node or use source_resample independently.',
}

function projected(node, hierarchical) {
  const {id, bounds_pts, bounds_seconds, kind, source_points, nominated_pts, applicability, level, parent_id, children, stop} = node
  return {id,bounds_pts,bounds_seconds,kind,applicability,nominated_count:nominated_pts?.length,known_pts_count:source_points?.length,stop,
    ...(hierarchical?{level,parent_id,child_ids:children}:{}),can_request_refinement:hierarchical}
}
function page(items,args) {
  const cursor=Math.max(0,args.cursor??0), n=Math.max(1,Math.min(20,args.page_size??8))
  return {items:items.slice(cursor,cursor+n),total:items.length,next_cursor:cursor+n<items.length?cursor+n:null}
}
function compactProof(node) {
  return {coordinate_system:{bounds_pts:node.bounds_pts,time_base:node.time_base,native_xywh:node.space_native_xywh},
    calculation_roots:node.comparisons?.map(r=>({root:r.root,pts:r.pts,registration:r.comparison.excluded_context.state,
      common_coordinates:r.common_coordinates?.native_affine,overlap:r.common_coordinates?.derived_overlap_mask,
      holdout_error_px:r.comparison.excluded_context.median_px,roi_residual:r.comparison.residual})),
    independent_relative_state_axis:node.state_roi_coordinate_mad_luma,
    independent_projection_axis:node.projected_relation_coordinate_px,
    comparison:node.comparison?{root:node.comparison.root,metrics:node.comparison.metrics,aligned:node.comparison.aligned}:undefined,
    graph:node.graph,original_comparison:node.evidence,refinements:node.refinements?.slice(-3).map(r=>({actual_pts:r.actual_pts,
      root:r.source_root,path_increment:r.path_increment,coverage_gap:r.trace?.coverage_gap_before??r.trace?.gap,
      original_edges:r.trace?.original_edges,revised_edges:r.trace?.revised_edges,
      anchor_distances:r.trace?.dual_anchor?{a:r.trace.dual_anchor.a,b:r.trace.dual_anchor.b}:undefined,
      convolution:Object.keys(r.trace?.signal_convolution??{})}))}
}

export function installResearch(ctx, config) {
  if (config.arm==='A0') return
  const hierarchical=config.arm==='A2'
  const plan=JSON.parse(readFileSync(config.regionPlan,'utf8'))
  const nodes=new Map(plan.nodes.map(n=>[n.id,n]))
  async function log(value) {
    await appendFile(config.controlLog,JSON.stringify({at:new Date().toISOString(),...value})+'\n','utf8')
  }
  const lookup=(id)=>{
    const node=nodes.get(id)
    if (!node) throw new Error(`Unknown node ${id}. List nearby times without node_id and use an ID from the result.`)
    return node
  }
  const queryName=hierarchical?'amr_query':'investigation_regions'
  ctx.tools.register({name:queryName,description:hierarchical?
    'Query adaptive evidence nodes by time or exact node_id. detail=node shows original PTS, children, unknowns and stop status; detail=proof shows scoped measurements; detail=guide shows examples. Original images remain accessible using shared source tools.':
    'Query a flat list of measured investigation intervals by source seconds or ID. Metadata and actual available PTS; use the shared original-image tools independently.',
    parameters:{type:'object',properties:{...fields,node_id:text,detail:{type:'string',enum:['node','proof','guide']}}},output:plain,
    async execute(a) {
      if (a.detail==='guide') return hierarchical?guide:{purpose:guide.purpose,example:'investigation_regions({start:0,end:2})'}
      if (a.node_id){
        const node=lookup(a.node_id)
        await log({event:'research_query',tool:queryName,id:node.id,detail:a.detail??'node',stop:node.stop})
        return {...projected(node,hierarchical),scope:'Existing analysis and image addresses; unobserved content remains unknown.',
          originals:page(node.source_points??[],a),space_native_xywh:node.space_native_xywh,time_base:node.time_base,
          temporal_resolution_seconds:node.time_resolution_max_sample_gap_seconds,nomination_pts:node.nominated_pts,
          unknown_ranges_pts:node.unexplained_time_ranges_pts,
          ...(a.detail==='proof'?{proof:compactProof(node)}:{}),
          ...(hierarchical?{relations:page(plan.edges.filter(r=>r.nodes.includes(node.id)),a)}:{})}
      }
      if ((a.start??0)>(a.end??Infinity)) throw new Error('Give start <= end in source seconds.')
      const values=[...nodes.values()].filter(n=>n.bounds_seconds[0]<=(a.end??Infinity)&&n.bounds_seconds[1]>=(a.start??-Infinity))
      const result=page(values.map(n=>projected(n,hierarchical)),a)
      await log({event:'research_query',tool:queryName,ids:result.items.map(x=>x.id),range:[a.start,a.end]})
      return result
    }})
  if (hierarchical) ctx.tools.register({name:'amr_refine',description:
    'Continue an exact investigation by node ID even after automatic stop. Choose time (reads one actual interior sample and measures both sides) or space (two or more integer PTS and an original-pixel ROI; adds ROI children and supported pixel mappings or unknowns). Optional crop_pts returns a labeled derived crop. Use delivery_request for full exact source originals.',
    parameters:{type:'object',properties:{node_id:text,mode:{type:'string',enum:['time','space']},
      pts:{type:'array',items:integer}, native_xywh:{type:'array',items:integer},crop_pts:integer},required:['node_id','mode']},
    output:{schema:plain.schema,render:(_,v)=>[{type:'text',text:JSON.stringify(v.summary)},...v.images.map(attachment=>({type:'image',attachment}))]},
    async execute(a) {
      const node=lookup(a.node_id)
      if (!['time','space'].includes(a.mode)) throw new Error('Use mode time or space.')
      if (a.mode==='space'&&(!Array.isArray(a.pts)||new Set(a.pts).size<2||!a.pts.every(Number.isSafeInteger)||
          a.native_xywh?.length!==4||!a.native_xywh.every(Number.isSafeInteger)|| (a.crop_pts!=null&&!a.pts.includes(a.crop_pts))))
        throw new Error(guide.recovery)
      const scratch=await mkdtemp(join(config.refinementDir,'attempt-'))
      const packet=join(scratch,'request.json')
      await writeFile(packet,JSON.stringify({...a,node,authorized:await realpath(config.authorizedSource),case:config.case,out:join(scratch,'output')}))
      let parsed
      try {
        const {stdout}=await run(config.pythonPath,['-B','-X','utf8','-m','dacf_round4.bridge',packet],
          {cwd:config.project,timeout:90000,maxBuffer:16*1024*1024,windowsHide:true})
        parsed=JSON.parse(stdout)
      } catch(e) {
        await log({event:'research_refinement',status:'error',node:node.id,mode:a.mode,reason:String(e.stderr??e.message).slice(-4000)})
        throw new Error(`Refinement did not return verified results. ${guide.recovery}`)
      }
      const requestId=randomUUID().slice(0,8)
      const children=[]
      const images=[]
      const pts=parsed.source_originals.map(({pts,time_base,id})=>({pts,time_base,root:id}))
      for (const item of parsed.children) {
        const id=`${item.id}-${requestId}`
        const created={...item,id,children:[],stop:{state:'manual_refinement',automatic_stop:'unchanged'},graph:parsed.figure1_2?.summary}
        nodes.set(id,created)
        node.children=[...new Set([...(node.children??[]),id])]
        children.push(projected(created,true))
      }
      let crop
      if (parsed.crop) {
        const {path,...rest}=parsed.crop
        const bytes=await readFile(path)
        if(createHash('sha256').update(bytes).digest('hex')!==rest.sha256) throw new Error('Derived crop integrity check failed.')
        const attachment=await ctx.get('attachments').saveImage({data:bytes,mediaType:'image/png',name:`refine-crop-${requestId}.png`})
        if(attachment.width!==rest.width||attachment.height!==rest.height) throw new Error('Unexpected crop dimensions.')
        images.push(attachment)
        crop={...rest,attachment_id:attachment.attachmentId}
      }
      await log({event:'research_refinement',status:'completed',node:node.id,mode:a.mode,children:children.map(x=>x.id),pts,
        originally_stopped:node.stop,crop,receipt:join(scratch,'output','bridge_result.json')})
      return {summary:{node:node.id,children,originals:pts,new_probe_pts:parsed.new_probe_pts,
        graph:parsed.figure1_2?.summary,crop,lookup:parsed.lookup},images}
    }})
  void log({event:'research_registered',tools:hierarchical?['amr_query','amr_refine']:[queryName]})
}

export function apply(ctx,config) {
  common(ctx,config)
  installResearch(ctx,config)
}
