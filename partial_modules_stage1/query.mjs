/** Optional research-only view. Time windows must come from the same source index. */
export function queryEvidence(arms, scope, {m2=false,m5=false}={}) {
  if (scope.source_sha256 !== arms.source_sha256) throw new Error('STALE_OR_DIFFERENT_SOURCE')
  const [lo,hi]=scope.bounds_pts
  if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || lo>hi) throw new Error('Use two ordered exact source ticks.')
  const within=t=>lo<=t&&t<=hi
  const states=Object.fromEntries(Object.entries(arms.roi_states).map(([name,rows])=>[name,rows.filter(r=>within(r.pts))]).filter(([,rows])=>rows.length))
  const candidates=new Set(Object.keys(states))
  return {
    source_sha256:scope.source_sha256,node_id:scope.node_id,bounds_pts:[lo,hi],
    observed_views:states,
    ...(m2?{candidate_hypotheses:arms.m2.filter(row=>row.episodes.some(e=>candidates.has(e)))}:{}),
    ...(m5?{sampled_relation_edges:arms.m5.map(series=>({episode:series.episode,edges:series.edges.filter(edge=>edge.pts.every(within))})).filter(r=>r.edges.length)}:{}),
    ...(m2&&m5?{cross_episode_hypotheses:arms.bridges.filter(row=>row.measure?.pts.every(within))}:{}),
    scope:'Same-source time and ROI evidence lookup; cross-episode physical identity and unsampled time remain unknown.',
  }
}
