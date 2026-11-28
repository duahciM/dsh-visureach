/** Optional project research adapter joining existing nodes to measured evidence by exact scope. */
export function relationEvidenceAtNode({sourceSha,edges}, node, {sourceSha256, enabled=false}={}) {
  if (sourceSha256!==sourceSha) throw new Error('STALE_OR_DIFFERENT_SOURCE')
  const [lo,hi]=node.bounds_pts
  if (!Number.isSafeInteger(lo)||!Number.isSafeInteger(hi)||lo>hi) throw new Error('Use actual ordered source PTS bounds.')
  const rows=enabled?edges.filter(row=>row.pts.every(t=>lo<=t&&t<=hi)):[]
  return {
    source_sha256:sourceSha,node_id:node.id,parent_id:node.parent_id,bounds_pts:node.bounds_pts,
    relationship_research_enabled:enabled,rows,
    original_stop_states:node.stops,unknown_native_intervals:rows.filter(r=>!r.scope_dependency || Object.values(r.axes).some(x=>x===null)).map(r=>r.pts),
    scope:'Local sampled relation axes. Identity dependencies never grant physical identity or limit original access.',
  }
}
