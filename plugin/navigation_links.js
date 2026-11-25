const text = { type: 'string' }
const number = { type: 'number' }
const integer = { type: 'integer' }

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
  tool('evidence_routes', 'Link catalogue, original rereads, prior OCR/ASR/subtitles, semantic matches, change evidence, local states and relation roots by source PTS. Optional anchor and hops expand a navigation graph. Entries keep their source and support boundary.',
    { start: number, end: number, anchor_pts: integer, hops: integer, root: text, relation: text, query: text,
      cursor: integer, page_size: integer, channel: { type: 'string', enum: ['catalogue', 'original', 'ocr', 'asr', 'subtitle', 'semantic', 'computed', 'state', 'graph'] } }, [], async (b, a) => {
      const history = (await b.ledger.read()).filter((r) => r.video_sha256 === b.manifest.video_sha256)
      const source = b.manifest.video_sha256
      const { lo, hi } = backend.selection(b, a.start, a.end)
      const rows = []
      const receipts = new Map(backend.evidence(history, source).map((r) => [r.pts, { ...r,
        frame_id: [source, r.source_frame, r.pts, r.time_base ?? b.manifest.time_base].join(':') }]))
      const add = (r) => {
        if (a.channel && a.channel !== r.channel) return
        if (a.root && !r.roots.includes(a.root)) return
        if (a.relation && !(r.relation ?? '').toLowerCase().includes(a.relation.toLowerCase())) return
        if (a.query && !(r.label ?? '').toLowerCase().includes(a.query.toLowerCase())) return
        if (r.start > hi || r.end < lo) return
        rows.push(r)
      }
      for (const r of b.catalog.frames) for (const o of r.observations) {
        add({ channel: 'catalogue', start: r.t, end: r.t, pts: r.pts, frame_id: r.frame_id, roots: o.evidence_root_ids,
          label: o.role, relation: o.dimension_id === 'D3' ? o.role : null, analysis_window: o.analysis_window,
          support: 'nomination and image-plane measurement' })
      }
      for (const r of receipts.values()) add({ channel: 'original', start: r.seconds, end: r.seconds, pts: r.pts,
        frame_id: r.frame_id, citation: r.evidence_id, roots: [r.evidence_id, r.frame_id], in_catalogue: r.in_catalog,
        origin: r.via, label: r.via, support: 'exact delivered original sample' })
      for (const r of history) {
        if (r.event === 'state_chain_entry') add({ channel: 'state', start: r.frame.seconds, end: r.frame.seconds,
          pts: r.frame.pts, label: r.content, relation: r.relationship, roots: r.roots, state_id: r.state_id, support: 'recorded interpretation; check cited images' })
      }
      const refs = new Map(history.filter((r) => r.event === 'extension_call' && r.result_ref && r.status === 'completed').map((r) => [r.result_ref, r]))
      const ignored = []
      for (const [ref, call] of refs) {
        const channel = { frame_ocr: 'ocr', video_text_search: call.result_channel, semantic_search: 'semantic', visual_compare: 'computed',
          compare_object_views: 'computed', change_recheck: 'computed' }[call.tool]
        if (!channel || (a.channel && a.channel !== channel)) continue
        let data
        try { ({ data } = await loadResult(b, ref)) } catch (error) { ignored.push({ result_ref: ref, reason: error.message }); continue }
        const clue = (start, end, label, extra = {}) => add({ channel, start, end, label, result_ref: ref,
          roots: [ref, ...(extra.roots ?? [])], pts: extra.pts ?? null,
          support: channel === 'computed' ? 'stored calculation, inspect scope' : 'retrieval or text clue; use originals for visual claims' })
        if (channel === 'ocr' && data.source_frame) for (const item of data.items ?? []) {
          const { seconds, pts } = data.source_frame
          clue(seconds, seconds, item.text, { pts })
        }
        if (channel === 'asr' || channel === 'subtitle') for (const item of data.unfiltered_items ?? data.items ?? []) {
          clue(item.start_seconds, item.end_seconds, item.text)
        }
        if (channel === 'semantic') for (const item of data.all_scores ?? []) clue(item.seconds, item.seconds, data.query, { pts: item.pts, roots: item.evidence_roots })
        if (channel === 'computed') {
          const frames = data.source_originals ?? data.original ?? []
          for (const f of frames) clue(f.seconds, f.seconds, call.tool, { pts: f.pts, roots: data.root ? [data.root] : [] })
          if (data.requested_range) clue(...data.requested_range, call.tool)
        }
      }
      let missing = []
      if (a.anchor_pts !== undefined && (!a.channel || a.channel === 'graph')) {
        const anchor = receipts.get(a.anchor_pts) ?? b.manifest.pool.find((r) => r.pts === a.anchor_pts)
        if (!anchor) throw new Error('Reread the anchor first or choose a coarse PTS.')
        const hops = a.hops ?? 1
        if (!Number.isSafeInteger(hops) || hops < 1) throw new Error('Provide a positive integer graph depth.')
        const found = catalogueFrontier(b, anchor.frame_id, hops, receipts)
        missing = found.missing
        for (const edge of found.result) add({ channel: 'graph', start: edge.endpoint.seconds, end: edge.endpoint.seconds,
          pts: edge.endpoint.pts, label: edge.kind, roots: edge.roots, edge, support: 'navigation only; relation unknown' })
      }
      rows.sort((x, y) => x.start - y.start || x.channel.localeCompare(y.channel))
      return { ...page(rows, a), source_sha256: source, unavailable_result_refs: ignored, missing_endpoints: missing,
        scope: 'Union of existing metadata and receipts. Each unique time remains queryable. A neighbor, text hit or same root makes a route, not independent visual corroboration.' }
    })
}
