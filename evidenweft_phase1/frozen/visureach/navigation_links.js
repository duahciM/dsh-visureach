import { frameRoot, sourceMoments } from './proof.js'

const text = { type: 'string' }
const number = { type: 'number' }
const integer = { type: 'integer' }
const roleKey = (o) => `${o.dimension_id}:${o.role}`

function catalogueFrontier(b, anchor, hops, receipts) {
  const pool = new Map(b.manifest.pool.map((row) => [row.frame_id, row]))
  for (const row of receipts.values()) pool.set(row.frame_id, row)
  const byFrame = new Map(b.catalog.frames.map((row) => [row.frame_id, row]))
  const adjacency = new Map()
  const rootMembers = new Map()
  const byWindow = new Map()
  const add = (from, to, kind, roots) => {
    if (from === to) return
    if (!adjacency.has(from)) adjacency.set(from, new Map())
    const edges = adjacency.get(from)
    const key = `${kind}:${to}`
    const previous = edges.get(key)
    const x = byFrame.get(from)?.observations.map((o) => o.analysis_window).filter((v) => v != null) ?? []
    const y = byFrame.get(to)?.observations.map((o) => o.analysis_window).filter((v) => v != null) ?? []
    edges.set(key, { from, to, kind, roots: [...new Set([...(previous?.roots ?? []), ...roots])],
      cross_window: Boolean(x.length && y.length && !x.some((v) => y.includes(v))) })
  }
  for (const row of b.catalog.frames) {
    const rootSet = new Set(), groups = new Set()
    for (const o of row.observations) {
      for (const root of o.evidence_root_ids) rootSet.add(root)
      if (o.analysis_window != null) groups.add(o.analysis_window)
      for (const to of o.source_comparison_frames) {
        add(row.frame_id, to, 'comparison_outgoing', o.evidence_root_ids)
        add(to, row.frame_id, 'comparison_incoming', o.evidence_root_ids)
      }
    }
    for (const r of rootSet) rootMembers.set(r, [...(rootMembers.get(r) ?? []), row.frame_id])
    for (const group of groups) byWindow.set(group, [...(byWindow.get(group) ?? []), row])
  }
  // The historical P2-16 topology connects neighbors within each shared-root
  // group and each time window with shared non-timing dimensions.
  for (const [root, ids] of rootMembers) for (let i = 1; i < ids.length; i++) {
    add(ids[i - 1], ids[i], 'shared_root_neighbor', [root]); add(ids[i], ids[i - 1], 'shared_root_neighbor', [root])
  }
  for (const members of byWindow.values()) for (let i = 1; i < members.length; i++) {
    const [first, second] = members.slice(i - 1, i + 1)
    const dims = new Set(first.observations.map((o) => o.dimension_id))
    const shared = second.observations.some((o) => o.dimension_id !== 'D6' && dims.has(o.dimension_id))
    if (shared) { add(first.frame_id, second.frame_id, 'same_window_neighbor', []); add(second.frame_id, first.frame_id, 'same_window_neighbor', []) }
  }
  const catalogue = b.catalog.frames
  for (const receipt of receipts.values()) {
    if (byFrame.has(receipt.frame_id)) continue
    let left = 0, right = catalogue.length
    while (left < right) {
      const mid = left + Math.floor((right - left) / 2)
      if (catalogue[mid].pts < receipt.pts) left = mid + 1
      else right = mid
    }
    for (const neighbor of [catalogue[left - 1], catalogue[left]].filter(Boolean)) {
      add(receipt.frame_id, neighbor.frame_id, 'source_time_neighbor', []); add(neighbor.frame_id, receipt.frame_id, 'source_time_neighbor', [])
    }
  }
  if (!pool.has(anchor)) throw new Error('Anchor must be a coarse frame or an original already delivered in this session.')
  const queue = [[anchor, 0]], seen = new Set([anchor]), result = [], missing = []
  for (let i = 0; i < queue.length; i++) {
    const [from, depth] = queue[i]
    if (depth >= hops) continue
    for (const edge of adjacency.get(from)?.values() ?? []) {
      if (!pool.has(edge.to)) { missing.push(edge); continue }
      const row = pool.get(edge.to)
      result.push({ ...edge, depth: depth + 1, endpoint: { frame_id: edge.to, pts: row.pts, seconds: row.seconds ?? row.t, in_catalog: byFrame.has(edge.to) } })
      if (!seen.has(edge.to)) { seen.add(edge.to); queue.push([edge.to, depth + 1]) }
    }
  }
  return { result, missing }
}

export function installNavigationLinks(tool, config, { backend, page, loadResult }) {
  if (config.linkedRetrieval === false) return
  tool('evidence_routes', 'Find linked catalogue and local figure evidence, original and crop receipts, objects, states, revisions and text by source time. Filter by candidate, revision or root. Compare two times by nomination roles or follow graph neighbors. Results retain their support boundary.',
    { start: number, end: number, anchor_pts: integer, hops: integer, root: text, relation: text, query: text,
      candidate_id: text, revision_id: text, compare_to_pts: integer,
      cursor: integer, page_size: integer, channel: { type: 'string', enum: ['catalogue', 'original', 'derived', 'ocr', 'asr', 'subtitle', 'semantic', 'computed', 'object', 'state', 'observation', 'graph'] } }, [], async (b, a) => {
      const history = (await b.ledger.read()).filter((r) => r.video_sha256 === b.manifest.video_sha256)
      const source = b.manifest.video_sha256
      const { lo, hi } = backend.selection(b, a.start, a.end)
      const rows = []
      const delivered = backend.evidence(history, source)
      const receipts = new Map(delivered.map((r) => [r.pts, { ...r, frame_id: frameRoot(b, r) }]))
      const originals = new Map(delivered.map((r) => [r.evidence_id, r]))
      const revisions = history.filter((r) => r.event === 'observation_revision')
      const latest = new Map(revisions.map((r) => [r.observation_id, r.revision_id]))
      const disputes = new Map()
      for (const r of history) if (r.event === 'observation_dispute') {
        if (!disputes.has(r.revision_id)) disputes.set(r.revision_id, [])
        disputes.get(r.revision_id).push(r)
      }
      const derived = new Map(history.filter((r) => r.event === 'derived_image').map((r) => [r.derived_id, r]))
      const timeFrames = new Map(b.manifest.pool.map((r) => [r.pts, r]))
      for (const r of receipts.values()) timeFrames.set(r.pts, r)
      const add = (r) => {
        if (a.channel && a.channel !== r.channel) return
        if (a.candidate_id && !r.candidate_ids?.includes(a.candidate_id)) return
        if (a.revision_id && !r.revision_ids?.includes(a.revision_id)) return
        if (a.root && !r.roots.includes(a.root)) return
        if (a.relation && !(r.relation ?? '').toLowerCase().includes(a.relation.toLowerCase())) return
        if (a.query && !(r.label ?? '').toLowerCase().includes(a.query.toLowerCase())) return
        if (r.start > hi || r.end < lo) return
        rows.push(r)
      }
      for (const r of b.catalog.frames) for (const o of r.observations) {
        add({ channel: 'catalogue', start: r.t, end: r.t, pts: r.pts, frame_id: r.frame_id, roots: [r.frame_id, ...o.evidence_root_ids],
          label: o.role, relation: o.dimension_id === 'D3' ? o.role : null, analysis_window: o.analysis_window,
          support: 'nomination and image-plane measurement' })
      }
      for (const r of receipts.values()) add({ channel: 'original', start: r.seconds, end: r.seconds, pts: r.pts,
        frame_id: r.frame_id, citation: r.evidence_id, roots: [r.evidence_id, r.frame_id], in_catalogue: r.in_catalog,
        origin: r.via, label: r.via, support: 'exact delivered original sample' })
      for (const r of derived.values()) {
        const parent = originals.get(r.parent_evidence_id)
        if (parent) add({ channel: 'derived', start: parent.seconds, end: parent.seconds, pts: parent.pts,
          label: r.reason, citation: r.derived_id, roots: [frameRoot(b, parent), r.parent_evidence_id, r.derived_id],
          parent_evidence_id: r.parent_evidence_id, native_xywh: r.native_xywh, support: 'derived crop of one delivered original' })
      }
      for (const r of history) {
        if (r.event === 'object_candidate_view') add({ channel: 'object', start: r.view.seconds, end: r.view.seconds, pts: r.view.pts,
          label: r.label, role: r.role, candidate_ids: [r.candidate_id], identity_id: r.observation_id, native_xywh: r.view.native_xywh,
          roots: [r.view.frame_id, ...r.evidence_ids], support: 'recorded candidate view; physical instance unresolved' })
        if (r.event === 'state_chain_entry') add({ channel: 'state', start: r.frame.seconds, end: r.frame.seconds,
          pts: r.frame.pts, label: r.content, relation: r.relationship, roots: r.roots, state_id: r.state_id,
          candidate_ids: [r.candidate_id, r.related_candidate_id].filter(Boolean), revision_ids: r.revision_ids,
          support: 'recorded interpretation; check cited images' })
      }
      const refs = new Map(history.filter((r) => r.event === 'extension_call' && r.result_ref && r.status === 'completed').map((r) => [r.result_ref, r]))
      const ignored = []
      const loaded = new Map()
      const get = (ref) => {
        if (!loaded.has(ref)) loaded.set(ref, loadResult(b, ref))
        return loaded.get(ref)
      }
      if (!a.channel || a.channel === 'observation') for (const r of revisions) {
        const counter = disputes.get(r.revision_id) ?? []
        const link = { channel: 'observation', label: r.content, observation_id: r.observation_id,
          revision_ids: [r.revision_id, ...(r.depends_on_revisions ?? [])], is_current: latest.get(r.observation_id) === r.revision_id,
          dispute_ids: counter.map((d) => d.dispute_id), support: 'recorded observation or counter-evidence; assess citation scope and visual support' }
        for (const id of new Set([...r.evidence_ids, ...counter.flatMap((d) => d.counter_evidence.map((c) => c.id))])) {
          const parent = derived.get(id)?.parent_evidence_id
          const original = originals.get(parent ?? id)
          if (original) add({ ...link, start: original.seconds, end: original.seconds, pts: original.pts,
            roots: [...new Set([frameRoot(b, original), original.evidence_id, id])], citation: id })
          else if (id.startsWith('vo-')) {
            try {
              const { data, channel } = await get(id)
              for (const f of sourceMoments(data)) add({ ...link, start: f.seconds, end: f.seconds, pts: f.pts,
                roots: [frameRoot(b, f), id], citation: id })
              if (['subtitle', 'asr'].includes(channel)) for (const item of data.unfiltered_items ?? data.items ?? []) {
                add({ ...link, start: item.start_seconds, end: item.end_seconds, roots: [id], citation: id })
              }
            } catch (error) { ignored.push({ result_ref: id, reason: error.message }) }
          }
        }
      }
      const figureRoles = new Map()
      const figures = new Map()
      for (const [ref, call] of refs) {
        const channel = { frame_ocr: 'ocr', video_text_search: call.result_channel, semantic_search: 'semantic', visual_compare: 'computed',
          compare_object_views: 'computed', relation_figure2: 'computed', change_recheck: 'computed' }[call.tool]
        if (!channel || (a.channel && a.channel !== channel && !(a.compare_to_pts != null && call.tool === 'relation_figure2'))) continue
        let data
        try { ({ data } = await get(ref)) } catch (error) { ignored.push({ result_ref: ref, reason: error.message }); continue }
        const clue = (start, end, label, extra = {}) => add({ channel, start, end, label, result_ref: ref,
          roots: [ref, ...(extra.roots ?? [])], pts: extra.pts ?? null,
          relation: extra.relation ?? null, index_layer: extra.index_layer ?? null, candidate_ids: extra.candidate_ids,
          support: channel === 'computed' ? 'stored calculation, inspect scope' : 'retrieval or text clue; use originals for visual claims' })
        if (channel === 'ocr' && data.source_frame) for (const item of data.items ?? []) {
          const { seconds, pts } = data.source_frame
          clue(seconds, seconds, item.text, { pts, roots: [frameRoot(b, data.source_frame)] })
        }
        if (channel === 'asr' || channel === 'subtitle') for (const item of data.unfiltered_items ?? data.items ?? []) {
          clue(item.start_seconds, item.end_seconds, item.text)
        }
        if (channel === 'semantic') for (const item of data.all_scores ?? []) clue(item.seconds, item.seconds, data.query, { pts: item.pts, roots: item.evidence_roots })
        if (channel === 'computed') {
          for (const f of sourceMoments(data)) clue(f.seconds, f.seconds, call.tool, { pts: f.pts,
            roots: [frameRoot(b, f), ...(data.root ? [data.root] : [])], candidate_ids: data.candidate_id ? [data.candidate_id] : undefined })
          if (data.requested_range) clue(...data.requested_range, call.tool)
          if (call.tool === 'relation_figure2' && data.figure2) {
            const at = timeFrames.get(data.pts)
            if (!at) continue
            const roots = [at.frame_id, ...(data.evidence_id ? [data.evidence_id] : [])]
            const layer = data.index_layer ?? 'frozen_catalogue'
            if (!figures.has(data.pts)) figures.set(data.pts, [])
            figures.get(data.pts).push({ result_ref: ref, index_layer: layer, region_count: data.figure1?.I,
              measured_projected_pairs: data.figure2.distinct_pairs })
            for (const o of data.nomination_observations ?? []) {
              if (!figureRoles.has(data.pts)) figureRoles.set(data.pts, new Map())
              figureRoles.get(data.pts).set(roleKey(o), new Set([...(figureRoles.get(data.pts).get(roleKey(o)) ?? []), ...o.evidence_root_ids]))
              clue(at.seconds, at.seconds, o.role, { pts: data.pts, roots: [...roots, ...o.evidence_root_ids], index_layer: layer })
            }
            for (const edge of data.figure2?.relations ?? []) for (const type of edge.types) {
              clue(at.seconds, at.seconds, type, { pts: data.pts, roots: [...roots, ...edge.roots], relation: type, index_layer: layer })
            }
          }
        }
      }
      let missing = []
      if (a.anchor_pts !== undefined && a.compare_to_pts == null && (!a.channel || a.channel === 'graph')) {
        const anchor = receipts.get(a.anchor_pts) ?? b.manifest.pool.find((r) => r.pts === a.anchor_pts)
        if (!anchor) throw new Error('Reread the anchor first or choose a coarse PTS.')
        const hops = a.hops ?? 1
        if (!Number.isSafeInteger(hops) || hops < 1) throw new Error('Provide a positive integer graph depth.')
        const found = catalogueFrontier(b, anchor.frame_id, hops, receipts)
        missing = found.missing
        for (const edge of found.result) add({ channel: 'graph', start: edge.endpoint.seconds, end: edge.endpoint.seconds,
          pts: edge.endpoint.pts, label: edge.kind, roots: edge.roots, edge, support: 'navigation only; relation unknown' })
      }
      let nomination_comparison
      if (a.compare_to_pts != null) {
        if (!timeFrames.has(a.anchor_pts) || !timeFrames.has(a.compare_to_pts) || a.anchor_pts === a.compare_to_pts) {
          throw new Error('Provide two different source PTS already indexed or delivered. Use anchor_pts and compare_to_pts.')
        }
        const summarize = (pts) => {
          const nominations = new Map([...(figureRoles.get(pts) ?? [])].map(([k, roots]) => [k, new Set(roots)]))
          for (const o of b.catalog.frames.find((r) => r.pts === pts)?.observations ?? []) {
            nominations.set(roleKey(o), new Set([...(nominations.get(roleKey(o)) ?? []), ...o.evidence_root_ids]))
          }
          return [...nominations].map(([role, roots]) => ({ role, roots: [...roots] }))
        }
        const at_anchor = summarize(a.anchor_pts), at_comparison = summarize(a.compare_to_pts)
        nomination_comparison = { anchor_pts: a.anchor_pts, comparison_pts: a.compare_to_pts, at_anchor, at_comparison,
          recorded_figures: { at_anchor: figures.get(a.anchor_pts) ?? [], at_comparison: figures.get(a.compare_to_pts) ?? [] },
          additional_roles_at_comparison: at_comparison.filter((x) => !at_anchor.some((y) => x.role === y.role)).map((x) => x.role),
          scope: 'Metadata comparison only. Identical roles or roots do not make distinct source times redundant. Unlisted roles are unassessed or unnominated.' }
      }
      rows.sort((x, y) => x.start - y.start || x.channel.localeCompare(y.channel))
      return { ...page(rows, a), source_sha256: source, unavailable_result_refs: ignored, missing_endpoints: missing, nomination_comparison,
        scope: 'Union of existing metadata and receipts. Each unique time remains queryable. A neighbor, text hit or same root makes a route, not independent visual corroboration.' }
    })
}
