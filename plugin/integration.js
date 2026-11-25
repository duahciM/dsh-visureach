import { createHash, randomUUID } from 'node:crypto'
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { saveResult, loadResult, resultPage, reference } from './proof.js'
import { installObservationTools, validateRevisionDependencies } from './observation_tools.js'
import { installNavigationLinks } from './navigation_links.js'

const text = { type: 'string' }
const number = { type: 'number' }
const integer = { type: 'integer' }
const list = (items) => ({ type: 'array', items })
const box = { ...list(number), minItems: 4, maxItems: 4 }
const range = { start: number, end: number, cursor: integer, page_size: integer }

function page(items, args) {
  const start = Math.max(0, args.cursor ?? 0)
  const count = Math.min(40, Math.max(1, args.page_size ?? 12))
  return { items: items.slice(start, start + count), cursor: start, total: items.length,
    next_cursor: start + count < items.length ? start + count : null }
}

const rootsOf = (row) => [...new Set(row.observations.flatMap((n) => n.evidence_root_ids))]

function brief(row) {
  return { frame_id: row.frame_id, pts: row.pts, seconds: row.t, time_base: row.time_base, source_frame: row.source_frame }
}

function sameRegion(a, b) {
  return a != null && b != null && a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]
}

const digest = (buffer) => createHash('sha256').update(buffer).digest('hex')

function preview(value) {
  const summary = { ...value }
  for (const [name, field] of Object.entries(value)) {
    if (Array.isArray(field) && field.length > 4) summary[name] = { count: field.length, preview: field.slice(0, 4) }
  }
  for (const name of ['candidates', 'comparisons', 'outside_catalogue', 'unfiltered_items', 'all_scores', 'compared_regions', 'all_verified_matches']) {
    if (value[name]) summary[name] = { count: value[name].length }
  }
  if (value.figure2) {
    delete summary.figure2
    delete summary.figure1
  }
  if (value.recheck?.observations) summary.recheck = { ...value.recheck, observations: { count: value.recheck.observations.length } }
  if (value.channels) {
    const { sift, local, phase, photometry, hsv_total_variation, edge_xor_fraction_unregistered } = value.channels
    summary.channels = { sift: { verified: sift.verified, inlier_count: sift.inlier_count, reason: sift.reason },
      local: { state: local.state, region_count: local.regions.length, first_region: local.regions[0]?.bbox },
      phase: { registration_valid: phase.registration_valid, dx: phase.dx, dy: phase.dy, residual_mad: phase.residual_mad },
      photometry: { supported: photometry.supported, reason: photometry.reason }, hsv_total_variation, edge_xor_fraction_unregistered }
  }
  if (Array.isArray(value.items)) {
    summary.items = value.items.slice(0, 4).map((row) => {
      if (!row || typeof row !== 'object') return row
      const select = { ...row }
      if (Array.isArray(row.evidence)) select.evidence = row.evidence.slice(0, 2).map((item) => ({ dimension_id: item.dimension_id,
        role: item.role, region_candidate: item.region_candidate, region_xywh_analysis: item.region_xywh_analysis,
        evidence_root_ids: item.evidence_root_ids.slice(0, 2), status: item.status }))
      if (select.evidence_ids?.length > 3) select.evidence_ids = { count: select.evidence_ids.length, first: select.evidence_ids.slice(0, 3) }
      return select
    })
    summary.returned_in_full_result = value.items.length
    if (Number.isSafeInteger(value.cursor) && value.items.length > summary.items.length) summary.next_cursor = value.cursor + summary.items.length
  }
  if (value.rows?.length && value.prototypes) {
    summary.rows = value.rows.slice(0, 4).map(({ sample, comparisons }) => ({ sample, comparisons: comparisons.map((r) => ({
      source_root: r.source_root, reference_index: r.reference_index, local: r.local.state, excluded_context: r.excluded_context.state,
      residual: r.residual, explanations: r.explanations })) }))
    summary.source_originals = { count: value.source_originals.length }
  }
  return summary
}

export function installExtensions(ctx, config, backend) {
  if (config.integration === false) return
  const proofLocks = new Map()
  function tool(name, description, properties, required, execute, output) {
    ctx.tools.register({ name, description, parameters: { type: 'object', properties, required },
      output: output ?? { schema: { type: 'object', additionalProperties: true }, render: (_, value) => [{type: 'text', text: JSON.stringify(value)}] },
      async execute(args, exec) {
        const binding = backend.bindingFor(exec)
        const started = performance.now()
        const at = new Date().toISOString()
        const receipt = { at, event: 'extension_call', tool: name, video_sha256: binding.manifest.video_sha256, task_id: binding.taskId }
        try {
          await backend.assertSource(binding)
          const value = await execute(binding, args ?? {}, exec)
          if (!output && name !== 'evidence_detail' && name !== 'evidence_reference') {
            const kind = name === 'frame_ocr' && value.status === 'completed' ? 'ocr'
              : name === 'video_text_search' && value.status === 'completed' ? value.channel : 'computed'
            const full = await saveResult(binding, kind, value)
            value.result_ref = full
            receipt.result_ref = full
            receipt.result_channel = kind
            if (name === 'video_text_search' && ['completed', 'unavailable'].includes(value.status)) {
              value.track_ref = await saveResult(binding, 'track_metadata', { channel: value.channel, present: value.status === 'completed', stream: value.stream, reason: value.reason })
            }
          }
          await binding.ledger.append({ ...receipt, status: value.status ?? 'completed', duration_ms: Math.round(performance.now() - started) })
          return !output && config.compactEvidence !== false && !['evidence_detail', 'evidence_reference'].includes(name)
            ? { ...preview(value), scope: value.scope ?? 'Source-bound tool result. Read full fields via evidence_detail.' } : value
        } catch (error) {
          await binding.ledger.append({ ...receipt, status: 'failed', error: String(error.message), duration_ms: Math.round(performance.now() - started) })
          throw error
        }
      },
    })
  }

  function span(binding, args) {
    const { lo, hi } = backend.selection(binding, args.start, args.end)
    return binding.catalog.frames.filter((row) => lo <= row.t && row.t <= hi)
  }

  async function compute(binding, exec, operation, args) {
    const folder = await mkdtemp(join(tmpdir(), 'video_observer_'))
    try {
      const request = join(folder, 'request.json')
      await writeFile(request, JSON.stringify(args))
      return await backend.worker(operation, request, exec.signal)
    } finally {
      await rm(folder, { recursive: true, force: true })
    }
  }

  async function exact(binding, exec, pts) {
    if (!Number.isSafeInteger(pts)) throw new Error('Pass an exact integer source PTS.')
    const timeline = await backend.timeline(binding, exec.signal)
    const matches = timeline.filter((row) => row.pts === pts)
    if (matches.length !== 1) throw new Error('PTS is absent or ambiguous in this source.')
    const row = matches[0]
    return { pts, seconds: row.seconds, time_base: binding.manifest.time_base, source_frame: row.index }
  }

  tool('evidence_detail', 'Read any field of a saved tool result, losslessly and in pages. Start at path=[] and follow field names or array indices. No cumulative query limit.',
    { result_ref: text, path: list(text), cursor: integer, page_size: integer }, ['result_ref'], async (b, a) => {
      const data = await loadResult(b, a.result_ref)
      return { result_ref: a.result_ref, source_sha256: data.source_sha256, channel: data.channel, ...resultPage(data.data, a) }
    })

  async function checkRefs(b, records, ids) {
    const evidence = new Map(backend.evidence(records, b.manifest.video_sha256).map((row) => [row.evidence_id, row]))
    return Promise.all(ids.map((id) => reference(b, records, evidence, id)))
  }
  tool('evidence_reference', 'Inspect citation type, source time, parent original and allowed support scope. Citation membership alone does not establish any claim.',
    { ids: list(text) }, ['ids'], async (b, a) => ({ items: await checkRefs(b, await b.ledger.read(), a.ids) }))

  async function research(b, a, exec) {
    if (!b.manifest.analysis_path) return { status: 'dependency_missing', reason: 'Source index has no analysis raster.' }
    return compute(b, exec, 'research', { catalog: b.manifest.catalog_path, analysis: b.manifest.analysis_path, ...a })
  }

  if (config.researchNavigation !== false) {
    tool('relation_figure2', 'Inspect the P2-23 graph at a catalogue PTS, including measured image-region pairs and degree. Full graphs are stored for evidence_detail.',
      { pts: integer }, ['pts'], (b, a, exec) => research(b, { action: 'figure2', pts: a.pts }, exec))
    tool('change_triggers', 'Find source-linked change prompts from the P2-17 comparison chain. Shows investigated time brackets and abstentions without delivering images.',
      range, [], async (b, a, exec) => {
        const { lo, hi } = backend.selection(b, a.start, a.end)
        const data = await research(b, { action: 'triggers', start: lo, end: hi }, exec)
        return data.items ? { ...data, ...page(data.items, a) } : data
      })
    tool('change_recheck', 'Recheck one P2-17 trigger against actual original source frames at a range and density you choose. Returns local measurements; use source_resample independently for images.',
      { trigger_result_ref: text, trigger_id: text, ...range, fps: number }, ['trigger_result_ref', 'trigger_id', 'start', 'end', 'fps'], async (b, a, exec) => {
        if (!Number.isFinite(a.fps) || a.fps <= 0) throw new Error('Provide a positive finite fps.')
        const { lo, hi } = backend.selection(b, a.start, a.end)
        if (hi <= lo) throw new Error('Choose an interval within the source.')
        const stored = await loadResult(b, a.trigger_result_ref)
        const trigger = stored.data.candidates?.find((v) => v.id === a.trigger_id)
        if (!trigger) throw new Error('Choose a trigger_id from the saved trigger query.')
        const all = await backend.timeline(b, exec.signal)
        const excluded = new Set([trigger.observation_a.pts, trigger.observation_c.pts])
        let next = lo
        const candidates = []
        for (const f of all) {
          if (f.seconds < lo || f.seconds > hi || excluded.has(f.pts) || f.seconds + 1e-9 < next) continue
          candidates.push(f)
          next = lo + (Math.floor((f.seconds - lo + 1e-9) * a.fps) + 1) / a.fps
        }
        const window = page(candidates, a)
        const frames = []
        for (const f of window.items) {
          const path = await backend.native(b, { source_frame: f.index }, exec.signal)
          frames.push({ path, source_frame: f.index, seconds: f.seconds, pts: f.pts, time_base: b.manifest.time_base })
        }
        const result = await research(b, { action: 'feedback', trigger, frames }, exec)
        return { ...result, next_cursor: window.next_cursor, eligible_sampled_frames: window.total, requested_range: [lo, hi], requested_fps: a.fps }
      })
    tool('appearance_search', 'Compare one exact source anchor across the catalogue. Searches the whole requested source range by direct spatial correspondence, with no identity closure.',
      { pts: integer, ...range }, ['pts'], async (b, a, exec) => {
        const f = await exact(b, exec, a.pts)
        const path = await backend.native(b, f, exec.signal)
        const { lo, hi } = backend.selection(b, a.start, a.end)
        const result = await research(b, { action: 'appearance', anchor: { ...f, path, png_sha256: digest(await readFile(path)) }, start: lo, end: hi }, exec)
        return result.items ? { ...result, all_verified_matches: result.items, ...page(result.items, a) } : result
      })
  }

  if (config.evidenceSearch !== false) tool('evidence_search',
    'Search nominated evidence by root, dimension, region, image-space relationship or role. A metadata match is not a verified object/action. Use delivery_request to see originals.',
    { ...range, dimension: text, root: text, region: text, relationship: text, role: text,
      xywh_analysis: { ...box, description: 'Region of interest in analysis image pixels, as x,y,width,height.' } }, [], (b, a) => {
      const items = span(b, a).map((row) => ({ ...brief(row), evidence: row.observations.map((o, i) => {
        const raw = row.nominations[i]
        const relationship_measurement = o.dimension_id === 'D3' && raw?.context_edge_xy
          ? { context_edge_xy_analysis: raw.context_edge_xy, component_distance_px: raw.distance_px } : null
        return relationship_measurement ? { ...o, relationship_measurement } : o
      }).filter((o) =>
        (!a.dimension || o.dimension_id === a.dimension) && (!a.root || o.evidence_root_ids.includes(a.root))
        && (!a.region || a.region === o.region_candidate) && (!a.role || a.role === o.role)
        && (!a.relationship || (o.dimension_id === 'D3' && a.relationship === o.role))
        && (!a.xywh_analysis || sameRegion(a.xywh_analysis, o.region_xywh_analysis))) })).filter((row) => row.evidence.length)
      return { ...page(items, a), coordinates: b.catalog.coordinate_contract, metadata_only: true }
    })

  if (config.graphNavigation !== false) tool('evidence_graph',
    'Follow a catalogue frame through directly recorded comparisons and shared evidence roots. Edges carry scope and are only navigation. No images.',
    { frame_id: text, cursor: integer, page_size: integer }, ['frame_id'], (b, a) => {
      const focal = b.catalog.frames.find((row) => row.frame_id === a.frame_id)
      if (!focal) throw new Error('frame_id is not in this catalogue')
      const observed = new Map(b.catalog.frames.map((row) => [row.frame_id, brief(row)]))
      const seen = new Map()
      const emit = (id, type, roots, scopes) => {
        if (id === focal.frame_id) return
        const key = `${type}:${id}`
        const edge = seen.get(key) ?? { to: id, edge_type: type, roots: [], roles: [] }
        edge.roots = [...new Set([...edge.roots, ...roots])]
        edge.roles = [...new Set([...edge.roles, ...scopes])]
        seen.set(key, edge)
      }
      const focalRoots = new Set(rootsOf(focal))
      for (const observation of focal.observations) {
        for (const to of observation.source_comparison_frames) emit(to, 'direct_comparison', observation.evidence_root_ids, [observation.role])
      }
      for (const row of b.catalog.frames) {
        for (const o of row.observations) {
          if (o.source_comparison_frames.includes(focal.frame_id)) emit(row.frame_id, 'direct_comparison_incoming', o.evidence_root_ids, [o.role])
        }
        const roots = rootsOf(row).filter((root) => focalRoots.has(root))
        if (roots.length) emit(row.frame_id, 'shared_root', roots, [])
      }
      const items = [...seen.values()].map((edge) => ({ ...edge, endpoint: observed.get(edge.to) ?? null }))
      return { ...page(items, a), anchor: brief(focal), metadata_only: true, scope: 'Measured 2D comparisons and shared roots, no physical identity.' }
    })

  if (config.changeNavigation !== false) tool('change_search',
    'Find measured local changes, 2D relations/view shifts or unresolved evidence. Returns candidate intervals and roots for optional follow-up, without images.',
    { ...range, dimension: { type: 'string', enum: ['D2', 'D3', 'D4', 'D5'] } }, [], (b, a) => {
      const frames = new Map(b.manifest.pool.map((row) => [row.frame_id, row]))
      const items = span(b, a).flatMap((row) => row.observations.filter((o) => ['D2', 'D3', 'D4', 'D5'].includes(o.dimension_id)
        && (!a.dimension || o.dimension_id === a.dimension)).map((o) => {
        const endpoints = o.source_comparison_frames.flatMap((id) => frames.has(id) ? [frames.get(id)] : [])
        const times = [row.t, ...endpoints.map((f) => f.seconds)]
        return { ...brief(row), window: [Math.min(...times), Math.max(...times)], dimension: o.dimension_id,
          observation: o.role, state: o.status, region: o.region_xywh_analysis, evidence_roots: o.evidence_root_ids,
          comparison_pts: endpoints.map((f) => f.pts), uncertainty: o.uncertainty }
      }))
      return { ...page(items, a), metadata_only: true, interpretation: 'Candidate windows do not establish an event. Reread any range independently.' }
    })

  if (config.visualCompare !== false) tool('visual_compare',
    'Compute independent 2D geometry, registered local residual, image quality and photometry for two exact PTS. Candidate correspondence does not establish identity, and no match leaves identity unknown.',
    { pts_a: integer, pts_b: integer, include_matches: { type: 'boolean' } }, ['pts_a', 'pts_b'], async (b, a, exec) => {
      if (a.pts_a === a.pts_b) throw new Error('Compare two distinct source frames.')
      const frames = await Promise.all([exact(b, exec, a.pts_a), exact(b, exec, a.pts_b)])
      const paths = await Promise.all(frames.map((f) => backend.native(b, f, exec.signal)))
      const original = await Promise.all(paths.map(async (p, i) => ({ ...frames[i], png_sha256: digest(await readFile(p)) })))
      const root = 'pair-' + digest(JSON.stringify([b.manifest.video_sha256, ...original])).slice(0, 20)
      const result = await compute(b, exec, 'pair', { paths, root })
      if (result.channels == null) return result
      const output = { ...result, root, source_sha256: b.manifest.video_sha256, original, status: 'completed' }
      await b.ledger.append({ event: 'visual_pair', ...output, video_sha256: b.manifest.video_sha256, at: new Date().toISOString() })
      return output
    })

  if (config.semanticSearch !== false) tool('semantic_search',
    'Search measured catalogue regions through the optional local CLIP model. Query in words. Returns text similarities and original PTS, without sending or changing the catalogue.',
    { ...range, query: text }, ['query'], async (b, a, exec) => {
      if (!config.semanticAssets) return { status: 'dependency_missing', reason: 'No local semanticAssets configured.' }
      const rows = span(b, a)
      const base = b.manifest.analysis_path
      if (rows.length && !base) return { status: 'dependency_missing', reason: 'This source index has no retained analysis images. Reopen with semanticAssets configured.' }
      if (!rows.length) return { status: 'completed', items: [], total_scored: 0, next_cursor: null }
      const options = { query: a.query, assets: config.semanticAssets, python_paths: config.semanticPythonPaths ?? [],
        coordinates: b.catalog.coordinate_contract, cursor: Math.max(0, a.cursor ?? 0), page_size: Math.min(40, Math.max(1, a.page_size ?? 12)),
        rows: rows.map((row) => ({ ...brief(row), t: row.t, roots: rootsOf(row), observations: row.observations,
          path: join(base, `${row.source_frame}.png`) })) }
      return { ...(await compute(b, exec, 'semantic', options)), source_sha256: b.manifest.video_sha256 }
    })

  if (config.ocr !== false) tool('frame_ocr',
    'Read printed text from one exact source frame. OCR output is timed metadata with native coordinates, and may be wrong. Use original images for visual claims.',
    { pts: integer, language: text, query: text }, ['pts'], async (b, a, exec) => {
      const frame = await exact(b, exec, a.pts)
      const path = await backend.native(b, frame, exec.signal)
      const result = await compute(b, exec, 'ocr', { path, executable: config.tesseractPath, language: a.language })
      if (a.query && result.items) result.matches = result.items.filter((row) => row.text.toLowerCase().includes(a.query.toLowerCase()))
      return { ...result, video_sha256: b.manifest.video_sha256, source_frame: frame, png_sha256: digest(await readFile(path)), metadata_only: true }
    })

  if (config.textTrack !== false) tool('video_text_search',
    'Find embedded subtitle text or optional local ASR in the video time range. Source type is labelled separately from visual evidence.',
    { ...range, channel: { type: 'string', enum: ['subtitle', 'asr'] }, query: text }, ['channel'], async (b, a, exec) => {
      const time = backend.selection(b, a.start, a.end)
      const result = await compute(b, exec, 'text', { source: b.manifest.video_path, channel: a.channel,
        ffmpeg: backend.ffmpeg, ffprobe: backend.ffprobe, model: config.whisperModel, start: time.lo, end: time.hi })
      const items = result.items?.filter((r) => !a.query || r.text.toLowerCase().includes(a.query.toLowerCase()))
      return { ...result, ...(items ? page(items, a) : {}), unfiltered_items: result.items, source_sha256: b.manifest.video_sha256, metadata_only: true }
    })

  if (config.observationRecords !== false) tool('observation_record',
    'List, create or revise an observation. Direct claims cite evidence IDs delivered in this session and video. Revisions preserve the old record and require its current revision ID. Add records when they help the task, with no fixed observation workflow.',
    { action: { type: 'string', enum: ['list', 'create', 'revise'] }, observation_id: text, supersedes_revision: text, content: text,
      kind: { type: 'string', enum: ['direct', 'inference', 'unknown', 'withdrawn'] }, evidence_ids: list(text),
      claim_scope: { type: 'string', enum: ['visual', 'visual_region', 'ocr_text', 'transcribed_audio', 'subtitle_text', 'stream_metadata', 'measurement', 'external_statement'] },
      reason: text, depends_on_revisions: list(text), cursor: integer, page_size: integer }, ['action'], async (b, a) => {
      const key = b.ledger.file
      const previous = proofLocks.get(key) ?? Promise.resolve()
      const next = previous.catch(() => undefined).then(async () => {
        const records = await b.ledger.read()
        const history = records.filter((row) => row.event === 'observation_revision' && row.video_sha256 === b.manifest.video_sha256)
        const latest = new Map(history.map((row) => [row.observation_id, row]))
        if (a.action === 'list') return { ...page(a.observation_id ? history.filter((row) => row.observation_id === a.observation_id) : [...latest.values()], a) }
        const old = latest.get(a.observation_id)
        if (a.action === 'revise' && (!old || old.revision_id !== a.supersedes_revision || !a.reason?.trim())) {
          throw new Error('Revision requires the current revision ID and a reason. Read the record history first.')
        }
        if (!a.content?.trim() || !['direct', 'inference', 'unknown', 'withdrawn'].includes(a.kind)) throw new Error('A typed observation needs content and a valid kind.')
        const citations = [...new Set(a.evidence_ids ?? [])]
        const refs = await checkRefs(b, records, citations)
        const claim_scope = a.claim_scope ?? 'visual'
        if (a.kind === 'direct' && (!refs.length || refs.some((r) => !r.allowed_scopes.includes(claim_scope)))) {
          throw new Error(`Cited evidence type cannot directly support scope ${claim_scope}. Inspect evidence_reference or label an inference.`)
        }
        const entry = { event: 'observation_revision', observation_id: a.action === 'revise' ? old.observation_id : randomUUID(),
          revision_id: randomUUID(), supersedes_revision: a.action === 'revise' ? old.revision_id : null,
          kind: a.kind, content: a.content, claim_scope, evidence_ids: citations, evidence_refs: refs, reason: a.reason ?? null, task_id: b.taskId,
          depends_on_revisions: validateRevisionDependencies(history, a.depends_on_revisions ?? old?.depends_on_revisions ?? []),
          video_sha256: b.manifest.video_sha256, at: new Date().toISOString() }
        await b.ledger.append(entry, (current) => {
          if (a.action !== 'revise') return
          const newest = current.findLast((r) => r.event === 'observation_revision' && r.video_sha256 === entry.video_sha256 && r.observation_id === entry.observation_id)
          if (newest?.revision_id !== a.supersedes_revision) throw new Error('The observation was revised concurrently. Read its latest revision before proceeding.')
        })
        return entry
      })
      proofLocks.set(key, next)
      try { return await next } finally { if (proofLocks.get(key) === next) proofLocks.delete(key) }
    })

  if (config.claimVerification !== false) tool('claim_verify',
    'Independently re-decode cited source frames and compare their PNG SHA, PTS and dimensions. Reports provenance, and leaves the semantic support decision to human/independent review.',
    { revision_id: text, evidence_ids: list(text) }, [], async (b, a, exec) => {
      const records = await b.ledger.read()
      const revision = a.revision_id && records.find((row) => row.event === 'observation_revision'
        && row.revision_id === a.revision_id && row.video_sha256 === b.manifest.video_sha256)
      if (a.revision_id && !revision) throw new Error('Unknown revision for this session and source.')
      const ids = [...new Set(revision ? revision.evidence_ids : a.evidence_ids ?? [])]
      if (!ids.length) throw new Error('Pass a revision with citations or one or more evidence IDs.')
      const evidence = new Map(backend.evidence(records, b.manifest.video_sha256).map((row) => [row.evidence_id, row]))
      const temp = await mkdtemp(join(tmpdir(), 'verify_observation_'))
      try {
        const items = []
        for (const [index, id] of ids.entries()) {
          const cite = evidence.get(id)
          if (!cite) {
            try {
              const typed = await reference(b, records, evidence, id)
              items.push({ evidence_id: id, ...typed, integrity: typed.integrity, original_verified: false })
            } catch (error) { items.push({ evidence_id: id, integrity: 'unknown_or_invalid_evidence_id', reason: error.message }) }
            continue
          }
          const frame = await exact(b, exec, cite.pts)
          const out = join(temp, `${index}.png`)
          await backend.decode(b, frame.source_frame, out, exec.signal)
          const bytes = await readFile(out)
          const got = digest(bytes)
          const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20)
          items.push({ evidence_id: id, frame, original_sha256: cite.png_sha256, redecoded_sha256: got,
            width, height, integrity: got === cite.png_sha256 && cite.source_frame === frame.source_frame
              && width === b.manifest.probe.width && height === b.manifest.probe.height ? 'verified' : 'mismatch' })
        }
        const receipt = { event: 'claim_verification', revision_id: revision?.revision_id ?? null, items,
          source_integrity: items.every((r) => r.integrity === 'verified') ? 'verified'
            : items.some((r) => r.integrity === 'mismatch' || r.integrity === 'unknown_or_invalid_evidence_id') ? 'failed' : 'mixed_evidence_scope',
          semantic_support: 'requires_independent_review', video_sha256: b.manifest.video_sha256, at: new Date().toISOString() }
        await b.ledger.append(receipt)
        return receipt
      } finally {
        await rm(temp, { recursive: true, force: true })
      }
    })

  if (config.derivedImages !== false) tool('evidence_crop',
    'Crop or magnify a region of a previously received original. The derived PNG has a separate ID, coordinates and parent evidence ID. It is not a newly watched original.',
    { evidence_id: text, native_xywh: box, scale: number, reason: text }, ['evidence_id', 'native_xywh', 'reason'], async (b, a, exec) => {
      await backend.assertImageRoute(exec)
      const parent = backend.evidence(await b.ledger.read(), b.manifest.video_sha256).find((r) => r.evidence_id === a.evidence_id)
      if (!parent) throw new Error('Use a same-session original evidence ID.')
      if (!Array.isArray(a.native_xywh) || a.native_xywh.length !== 4 || !a.native_xywh.every(Number.isSafeInteger)) throw new Error('Use integer native crop coordinates.')
      const scale = a.scale ?? 1
    if (!Number.isFinite(scale) || scale < 1 || !a.reason?.trim()) throw new Error('Provide a reason and a finite scale of at least 1.')
    const [, , w, h] = a.native_xywh
    if (w * scale > 4096 || h * scale > 4096 || w * h * scale * scale > 12_000_000) {
      throw new Error('Derived output exceeds the attachment envelope. Use a smaller crop or lower scale; original access is unchanged.')
    }
      const frame = await exact(b, exec, parent.pts)
      const path = await backend.native(b, frame, exec.signal)
      if (digest(await readFile(path)) !== parent.png_sha256) throw new Error('Original does not match the parent image hash.')
      const temp = await mkdtemp(join(tmpdir(), 'observer_crop_'))
      try {
        const out = join(temp, 'crop.png')
        const status = await compute(b, exec, 'crop', { path, out, scale, native_xywh: a.native_xywh })
        if (status.status !== 'completed') throw new Error(status.reason ?? 'Crop did not complete.')
        const bytes = await readFile(out)
        const ref = await backend.attach(bytes, `crop_${frame.pts}.png`)
        const receipt = { event: 'derived_image', derived_id: 'derived-' + randomUUID(), parent_evidence_id: a.evidence_id,
          parent_png_sha256: parent.png_sha256, attachment_sha256: ref.attachmentId.slice('sha256:'.length),
          video_sha256: b.manifest.video_sha256, ...frame, native_xywh: a.native_xywh, scale,
          presentation: 'derived_crop_png', width: ref.width, height: ref.height, reason: a.reason, at: new Date().toISOString() }
        await b.ledger.append(receipt)
        return { description: JSON.stringify(receipt), images: [ref] }
      } finally {
        await rm(temp, { recursive: true, force: true })
      }
    }, { schema: { type: 'object', properties: {description: text, images: backend.imageArraySchema}, required: ['description', 'images'] },
      render: (_, value) => [{ type: 'text', text: value.description }, ...value.images.map(backend.imageBlock)] })

  if (config.observationRecords !== false) tool('observation_dispute',
    'List or append a competing evidence interpretation on an immutable observation revision. It does not rewrite a finding. Use observation_record to revise after checking the alternatives.',
    { revision_id: text, evidence_ids: list(text), reason: text, cursor: integer, page_size: integer }, ['revision_id'], async (b, a) => {
      const records = (await b.ledger.read()).filter((r) => r.video_sha256 === b.manifest.video_sha256)
      if (!records.some((row) => row.event === 'observation_revision' && row.revision_id === a.revision_id)) throw new Error('Unknown same-source revision.')
      if (!a.reason) return page(records.filter((r) => r.event === 'observation_dispute' && r.revision_id === a.revision_id), a)
      if (!a.evidence_ids?.length) throw new Error('Record what actual observations conflict with the revision.')
      const refs = await checkRefs(b, records, [...new Set(a.evidence_ids)])
      const row = { event: 'observation_dispute', dispute_id: randomUUID(), revision_id: a.revision_id,
        state: 'unresolved', counter_evidence: refs, reason: a.reason, video_sha256: b.manifest.video_sha256,
        task_id: b.taskId, at: new Date().toISOString() }
      await b.ledger.append(row)
      return row
    })
  installObservationTools(tool, config, { backend, checkRefs, exact, compute, page })
  installNavigationLinks(tool, config, { backend, page, loadResult })
}
