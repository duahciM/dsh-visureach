export const OVERVIEW = 'Use the catalogue, wave and unviewed axis to navigate. Figure 1 summarizes observed regions; Figure 2 measures projected relationships. Exact original reads can use any valid PTS or your chosen interval and density. Candidate views, state chains and revision review connect receipts without proving physical identity. Originals, crops, calculations and text clues have different support. Choose your own evidence, tools and stopping point. Call observer_guide with a tool name or topic for examples and recovery.'

// Bracketed values name prior tool outputs. Acceptance resolves them against
// a real source and executes every documented example using the same schemas.
export const GUIDE = {
  observer_guide: ['Understand the whole system or a particular tool before unfamiliar work.', { tool: 'compare_object_views' }, 'Use the listed name or topic; no open source is required.', 'video_open', 'Explains the interface, without observing any image.'],
  video_open: ['Bind a local source and construct or resume its source-versioned catalogue.', { path:'$video' }, 'Check the path and decoder. If source bytes changed, open the source again.', 'catalog_search', 'Hash, PTS range, coarse sampling and candidate counts are metadata.'],
  observation_goal: ['Name the evidence question or read the active purpose with empty arguments.', {goal:'Locate a visible state change and check its source timing.'}, 'If enabled, record a concise goal before reading original images.', 'source_resample', 'A goal changes neither the pixels nor what they justify.'],
  catalog_search: ['Find real coarse frame IDs and PTS, then narrow by dimension, time or page.', {start:0,end:8,mode:'earliest',page_size:4}, 'Seconds are range inputs. Pass an actual frame ID or integer PTS to the delivery tool.', 'delivery_request', 'Nomination labels are measurements. Stale versions refer only to the prior source.'],
  wave_query: ['Look for discrete changes or candidate gaps to decide where to inspect.', {start:0,end:12,signal:'all',page_size:4}, 'Use one of the listed signals. A missing cue does not exclude a change.', 'source_resample', 'Wave heights count region and relation tokens; cues do not gate original access.'],
  observation_coverage: ['Distinguish sampled, nominated and actually delivered time for a question.', {zoom:'windows',start:0,end:10}, 'Use zoom=frames to inspect exact receipts rather than interval summaries.', 'evidence_list', 'Coverage tracks receipt, not understanding or a prescribed target.'],
  delivery_request: ['Inspect specific original frames when you already have exact PTS or known frame IDs.', {pts:['$p0','$p1'],reason:'Inspect the selected source views.'}, 'PTS are source ticks, not seconds. For a time interval use source_resample.', 'evidence_crop', 'Native image receipt IDs support the pixels of particular exact samples.'],
  source_resample: ['Reread the source at your chosen precision when coarse timing or coverage is insufficient.', {start:1,end:1.5,fps:24,page_size:4,reason:'Read local original frames to compare their order.'}, 'Choose fps or count. Continue next_cursor with unchanged range and sampling fields.', 'relation_figure2', 'Originals attach at exact PTS; open intervals remain unobserved.'],
  evidence_list: ['Recover IDs of actual original receipts before making a citation.', {start:0,end:8}, 'If absent, request the original and use its returned evidence ID.', 'claim_verify', 'Unique received originals; does not enumerate claims or derived imagery.'],
  external_source: ['Preserve a declared measurement from another tool when it matters to a claim.', {tool:'ffprobe',purpose:'Check the container time base',supports:'Container metadata only'}, 'Supply all three strings and describe the claim scope narrowly.', 'evidence_reference', 'Declared external provenance does not establish that the computation was run here.'],
  evidence_detail: ['Get saved full calculation data when a compact result omits coordinates or comparisons.', {result_ref:'$pair',path:['channels','sift','inlier_pairs'],page_size:4}, 'Use the complete vo- result_ref from the current source/session. Start with path=[] if lost.', 'observation_review', 'Reads stored fields, possibly in pages. Calculations remain distinct from originals.'],
  evidence_reference: ['Clarify original, derived, text, computed or external citation type and supported scope.', {ids:['$ev0','$crop']}, 'Use exact same-source IDs. For a crop, follow parent_evidence_id to the original.', 'claim_verify', 'The allowed scope and integrity receipt do not decide semantic support.'],
  relation_figure2: ['Measure Figure 1 region information and Figure 2 2D relationships at a nominated or reread PTS.', {pts:'$reread_pts',incremental:true,context_pts:['$p0','$p1'],regions_native:['$region']}, 'For a new PTS read the original first; context_pts are exact ticks, regions are native x,y,width,height.', 'evidence_routes', 'Pixel regions, projected distances and evidence roots. User-chosen ROIs are labeled; contact and physical identity remain unknown.'],
  change_triggers: ['Find P2-17 candidate brackets and abstentions for a local investigation.', {start:0,end:18,page_size:4}, 'Use the full result_ref and the trigger_id on the chosen item. Empty results can be investigated directly.', 'change_recheck', 'A comparison trigger proposes a window; it does not assert an event.'],
  change_recheck: ['Compare new source samples around a measured trigger at caller-chosen density.', {trigger_result_ref:'$triggers',trigger_id:'$trigger',start:'$change_start',end:'$change_end',fps:24,page_size:4}, 'Use one trigger and a positive interval. Match the two IDs to the same saved result.', 'source_resample', 'Measurement at actual samples. Use originals for the visual claim; unknown remains unknown.'],
  appearance_search: ['Search for direct 2D correspondences to an anchor across the coarse catalogue.', {pts:'$p0',start:0,end:18,page_size:4}, 'Pick a known PTS. If fixed background dominates, compare smaller actual object regions.', 'compare_object_views', 'Direct similarity only; a miss does not establish a different object.'],
  evidence_search: ['Find nominated roots, regions, dimension roles or measurement relations in a range.', {dimension:'D3',start:0,end:18,page_size:4}, 'For boxes here use analysis pixels; for native object crops use original pixels.', 'evidence_graph', 'Returns candidate metadata. Use evidence_routes for accumulated reread and text evidence.'],
  evidence_graph: ['Follow direct comparison and common evidence-root links from a catalogue frame.', {frame_id:'$catalog_id',page_size:4}, 'Use a full catalogue frame ID. For a reread-only PTS, use evidence_routes with anchor_pts.', 'evidence_routes', 'Navigation edge provenance; common roots are dependent evidence.'],
  change_search: ['Screen local change, relation, view-shift or ambiguity candidates by dimension.', {dimension:'D2',start:0,end:18,page_size:4}, 'Use D2 through D5. Read the original comparison endpoints or reread another span.', 'visual_compare', 'Sampled measuring brackets with uncertainties, not confirmed state transitions.'],
  visual_compare: ['Compare two originals for concurrent hypotheses: phase, SIFT, photometry and registered residual.', {pts_a:'$p0',pts_b:'$p1'}, 'Pass two different exact original PTS; evidence_detail exposes full coordinates.', 'compare_object_views', 'Image-plane measurements, conditional on registration and overlap.'],
  semantic_search: ['Retrieve local CLIP suggestions from text when optional assets are configured.', {query:'colored marker on a dark surface',start:0,end:1,page_size:4}, 'If assets are unavailable, find visual candidates another way and read originals.', 'delivery_request', 'Ranked similarity clues, not proof of object, action, identity or absence.'],
  frame_ocr: ['Read small printed labels at a particular original PTS; use a crop for visual validation.', {pts:'$p0'}, 'Use one exact frame PTS. OCR dependencies or language packs may be unavailable.', 'evidence_crop', 'Text and OCR boxes are clues; quote only what appropriately verified source supports.'],
  video_text_search: ['Search embedded subtitle or optional ASR tracks by source time and text.', {channel:'subtitle',start:0,end:8}, 'Choose subtitle or asr. If missing, a track status is still separate from visible text.', 'evidence_routes', 'Timed transcript text and track metadata, checks do not prove an image action.'],
  observation_record: ['Create, list or revise a typed observation; append revisions when a claim changes.', {action:'list',page_size:4}, 'A revise call needs observation_id, current supersedes_revision, and a reason. Use same-source evidence.', 'observation_review', 'Preserves links, alternatives and original revisions; truth still needs review.'],
  claim_verify: ['Independently re-decode original citations or crop parents to check source, dimensions and pixels.', {evidence_ids:['$ev0']}, 'Pass an original or crop ID, or a revision with citations. Use evidence_reference to inspect mixed types.', 'observation_review', 'Reports original integrity and separately identifies crops. Crop pixels and semantic support need their own check.'],
  evidence_crop: ['Magnify a native rectangle inside an already delivered original to check localized details.', {evidence_id:'$ev0',native_xywh:'$region',scale:2,reason:'Inspect local visible structure.'}, 'Use the full original ID and an in-bounds native rectangle. Reduce payload by narrowing the rectangle.', 'evidence_reference', 'A derived view of its known parent region, not an independent original.'],
  observation_dispute: ['Read or register competing evidence for a particular immutable observation revision.', {revision_id:'$revision'}, 'To add a dispute provide a reason and at least one same-source counter-evidence ID.', 'observation_review', 'An open dispute does not automatically overwrite an interpretation.'],
  object_candidate: ['Save source-stamped views, recurrence hypotheses or conflicting evidence under a local candidate.', {action:'add',label:'selected local marker',role:'prototype',view:{pts:'$p0',native_xywh:'$region'},evidence_ids:['$ev0']}, 'A candidate view needs a delivered original at the same PTS. List by candidate_id or any saved view identity ID.', 'compare_object_views', 'Identity IDs name candidate views; never infer physical continuity from matching color or position.'],
  compare_object_views: ['Use direct mode to compare object ROIs. Use A/B only with distinct reference_a and reference_b views, plus separate samples.', {mode:'ab',reference_a:{pts:'$p0',native_xywh:'$region'},reference_b:{pts:'$p1',native_xywh:'$region'},samples:[{pts:'$reread_pts',native_xywh:'$region'}]}, 'MISSING_AB_REFERENCES means add both reference fields. DUPLICATE_AB_REFERENCES means change one reference, or use direct mode.', 'object_candidate', 'Foreground and excluded-context fits, holdout and residual. MAD fields are arithmetic means in 8-bit luma units. Median reprojection error is measured in analysis pixels. XB/XD are sampled candidates.'],
  state_chain: ['Record or find source-cited states and relations across a candidate history.', {action:'add',candidate_id:'$candidate',identity_id:'$view_id',phase:'initial',pts:'$p0',content:'Source-stamped initial view.',evidence_ids:['$ev0'],revision_ids:['$revision']}, 'Adding a visible state requires an original at that PTS. Use identity_id only for its recorded exact view.', 'evidence_routes', 'Recorded claims and open gaps, not interpolation.'],
  observation_review: ['Check a draft for stale revisions, open disputes, duplicated source roots and citation scopes.', {final_claims:[{content:'A local visible marker.',kind:'direct',evidence_ids:['$ev0'],revision_ids:['$revision'],stated_details:['marker']}]}, 'Use exact IDs. You can list omissions explicitly; literal omissions or scope flags are not semantic judgments.', 'claim_verify', 'Mechanically checks draft attribution and declared details; use originals for correctness.'],
  evidence_routes: ['Find original and crop receipts, candidate views, saved relations, states, observations and timed clues.', {anchor_pts:'$reread_pts',compare_to_pts:'$p0',start:0,end:18,page_size:4}, 'Choose two different indexed or delivered PTS for a nomination comparison. Omit compare_to_pts for graph hops. Filter by candidate_id or revision_id where present; use result_ref for details.', 'relation_figure2', 'Nomination roles and shared roots help navigate. Same roles do not establish semantic redundancy or physical identity.'],
  live_status: ['Inspect committed watermark, time base, and explicit gaps in an independent stream.', {page_size:4}, 'Start the separate operator producer if the database is absent. Continue with next_after_sequence.', 'live_catalog', 'Only exposes known committed progress, no future source information.'],
  live_catalog: ['Search already committed native image receipts by time and page.', {start_pts:0,page_size:4}, 'PTS inputs are ticks; call live_status for the time base.', 'live_read', 'No image bytes are delivered by the catalogue.'],
  live_wave: ['Inspect incremental brightness, edge density and adjacent received-frame differences.', {start_pts:0,page_size:4}, 'Null residual means there was no comparable immediate predecessor.', 'live_catalog', 'Navigation numbers from received frames; no event or identity proof.'],
  live_read: ['Read exact committed native frames, or choose an interval and density independently.', {pts_list:['$live_pts'],page_size:4}, 'If unavailable, inspect committed watermark and known gaps; request again only once arrival is possible.', 'live_status', 'Committed images only, with native PTS and stream identity.'],
}

const TOPICS = {
  short_event: 'A wave cue, sparse catalogue, or coverage gap may merit source_resample at a denser interval. Compare visible neighboring originals and keep unobserved time explicit.',
  recurrence: 'object_candidate can preserve multiple reference views. Use appearance_search to navigate or compare_object_views to check exact local evidence. Consider repeat appearance and replacement separately.',
  motion: 'visual_compare separates phase and photometry. compare_object_views can test target and excluded context separately. Read original views before claiming movement or an edit.',
  state: 'relation_figure2 gives projected relationships. state_chain connects source-stamped states and leaves time between observations open. evidence_routes can retrieve linked records by candidate_id, revision_id or source root.',
  comparison: 'evidence_routes can compare nomination roles at two known PTS and show saved Figure 1/2 results. Distinct source times remain separately available. Use exact originals before deciding whether observations are complementary for the goal.',
  contradiction: 'observation_dispute can point to counter-evidence. observation_review finds dependencies and flags reuse of the same original. Compare original images before assigning meaning.',
  final_check: 'Check that important color, location and time are explicit and cited. Separate established observation, inference and unknown. observation_review checks attribution, while claim_verify rereads original pixels.',
}

export function guideReply(args = {}, names = Object.keys(GUIDE)) {
  if (args.tool) {
    const g = GUIDE[args.tool]
    if (!g || !names.includes(args.tool)) return {status:'unknown_tool', available:names}
    const [when,example,recovery,related,boundary] = g
    return {tool:args.tool,when,example,recovery,related,boundary,
      example_parameters:'$-prefixed values stand for actual source paths, PTS or IDs from earlier tool results. The tool schema specifies the required arguments.'}
  }
  if (args.topic) return TOPICS[args.topic] ? {topic:args.topic, guide:TOPICS[args.topic], workflow:'optional'} : {status:'unknown_topic',topics:Object.keys(TOPICS)}
  return {overview:OVERVIEW,available:names,topics:Object.keys(TOPICS)}
}

export function registerGuided(ctx, enabled, guideSummary = OVERVIEW) {
  const names = []
  const register = (tool) => {
    names.push(tool.name)
    const g = GUIDE[tool.name]
    if (!enabled || !g) return ctx.tools.register(tool)
    ctx.tools.register({ ...tool, description: `${tool.description} ${g[0]} Recovery and example: observer_guide(${tool.name}).`,
      async execute(args, exec) {
        try { return await tool.execute(args, exec) } catch (error) {
          throw new Error(`${error.message} Recovery: ${g[2]} For an example, call observer_guide with tool=${tool.name}.`, {cause:error})
        }
      },
    })
  }
  if (enabled) register({name:'observer_guide',description:'Read the short observer overview, an optional workflow topic, or one tool example. No image or source access.',
    parameters:{type:'object',properties:{tool:{type:'string'},topic:{type:'string'}}},
    output:{schema:{type:'object',additionalProperties:true},render:(_,v)=>[{type:'text',text:JSON.stringify(v)}]},
    async execute(args) { const r=guideReply(args,names); return r.overview?{...r,overview:guideSummary}:r },
  })
  return register
}
