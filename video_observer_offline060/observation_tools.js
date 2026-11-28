import { createHash, randomUUID } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { frameRoot } from './proof.js'

const text = { type: 'string' }
const integer = { type: 'integer' }
const number = { type: 'number' }
const array = (items) => ({ type: 'array', items })
const xywh = { ...array(integer), minItems: 4, maxItems: 4 }
const range = { start: number, end: number, cursor: integer, page_size: integer }
const view = { type: 'object', properties: { pts: integer, native_xywh: xywh }, required: ['pts', 'native_xywh'] }
const identityKeys = ['observation_id', 'appearance_id', 'slot_id', 'context_id', 'instance_candidate_id', 'track_id']
const matchesIdentity = (record, key) => identityKeys.some((field) => record[field] === key)
const idOf = (prefix, ...parts) => `${prefix}-${createHash('sha256').update(JSON.stringify(parts)).digest('hex').slice(0, 24)}`
const sourceRows = (b, rows) => rows.filter((r) => r.video_sha256 === b.manifest.video_sha256)
const statusScope = 'Sampled image-plane evidence. Open time gaps, camera pose, and physical identity remain unknown.'

function stamp(binding, row) {
  return { ...row, frame_id: [binding.manifest.video_sha256, row.source_frame, row.pts, row.time_base].join(':') }
}

function revisionMap(rows) {
  return new Map(rows.filter((r) => r.event === 'observation_revision').map((r) => [r.revision_id, r]))
}

function latestMap(revisions) {
  return new Map([...revisions.values()].map((r) => [r.observation_id, r.revision_id]))
}

export function validateRevisionDependencies(rows, requested = []) {
  if (!Array.isArray(requested)) throw new Error('Pass a list of existing revision IDs.')
  const revisions = revisionMap(rows)
  const distinct = [...new Set(requested)]
  if (distinct.some((id) => !revisions.has(id))) throw new Error('Revision dependency must already exist in this session and source.')
  return distinct
}

function ancestors(id, revisions, found = new Set()) {
  if (found.has(id)) return found
  found.add(id)
  const row = revisions.get(id)
  for (const parent of row?.depends_on_revisions ?? []) ancestors(parent, revisions, found)
  return found
}

function originalRoots(b, ref) {
  return (Number.isSafeInteger(ref.pts) ? [ref] : ref.source_moments ?? [])
    .filter((r) => Number.isSafeInteger(r.source_frame)).map((r) => frameRoot(b, r))
}

export function installObservationTools(tool, config, api) {
  const { backend, checkRefs, exact, compute, page } = api

  async function sourceViews(b, exec, rows) {
    if (!Array.isArray(rows) || !rows.length) throw new Error('Provide one or more exact source views.')
    const f = []
    for (const row of rows) {
      const a = row.native_xywh
      if (!Array.isArray(a) || a.length !== 4 || a.some((x) => !Number.isSafeInteger(x)) || a[0] < 0 || a[1] < 0
        || a[2] <= 0 || a[3] <= 0 || a[0] + a[2] > b.manifest.probe.width || a[1] + a[3] > b.manifest.probe.height) {
        throw new Error('Use an integer native ROI inside the video image.')
      }
      f.push({ ...stamp(b, await exact(b, exec, row.pts)), native_xywh: a })
    }
    return f
  }

  if (config.objectObservation !== false) {
    tool('object_candidate', 'Keep or list local object candidate views at exact PTS and native rectangles. A label may have multiple prototypes and recurrence, replacement or conflict hypotheses. This does not merge physical identities.',
      { action: { type: 'string', enum: ['add', 'list'] }, candidate_id: text, identity_id: text, label: text, view, evidence_ids: array(text),
        role: { type: 'string', enum: ['prototype', 'recurrence_candidate', 'replacement_candidate', 'conflict', 'observed_view'] }, ...range }, ['action'], async (b, a, exec) => {
        const history = sourceRows(b, await b.ledger.read())
        const options = history.filter((r) => r.event === 'object_candidate_view')
        if (a.action === 'list') {
          const { lo, hi } = backend.selection(b, a.start, a.end)
          return page(options.filter((r) => (!a.candidate_id || r.candidate_id === a.candidate_id) && (!a.identity_id || matchesIdentity(r, a.identity_id)) && r.view.seconds >= lo && r.view.seconds <= hi), a)
        }
        if (a.action !== 'add' || !a.label?.trim() || !['prototype', 'recurrence_candidate', 'replacement_candidate', 'conflict', 'observed_view'].includes(a.role)) {
          throw new Error('Pass a candidate label and an explicit observed-view or hypothesis role.')
        }
        if (a.candidate_id && !options.some((r) => r.candidate_id === a.candidate_id)) throw new Error('Unknown same-source candidate group.')
        const [frame] = await sourceViews(b, exec, [a.view ?? {}])
        const citations = await checkRefs(b, history, [...new Set(a.evidence_ids ?? [])])
        if (!citations.some((r) => r.type === 'original' && r.pts === frame.pts)) throw new Error('Record a candidate view with the actual original image receipt at that PTS.')
        const candidate_id = a.candidate_id ?? randomUUID()
        const key = [b.manifest.video_sha256, frame.pts, frame.native_xywh, candidate_id]
        // Mirrors IF14 identifiers, keeping a local chart, context, appearance,
        // episode and physical-instance candidate distinct in every view.
        const ids = Object.fromEntries(Object.entries({ observation_id: 'obs', appearance_id: 'appearance', slot_id: 'slot', context_id: 'context',
          instance_candidate_id: 'instance-candidate', track_id: 'sampled-track' }).map(([k, prefix]) => [k, idOf(prefix, ...key)]))
        const entry = { event: 'object_candidate_view', video_sha256: b.manifest.video_sha256, task_id: b.taskId, candidate_id,
          ...ids, role: a.role, label: a.label.trim(), view: frame, evidence_ids: citations.map((r) => r.id),
          physical_instance: 'unresolved', at: new Date().toISOString(), scope: statusScope }
        await b.ledger.append(entry)
        return entry
      })

    tool('compare_object_views', 'Compare exact native-pixel ROIs. In direct mode use one or more prototypes or saved reference IDs. In A/B mode set mode=ab and supply reference_a and reference_b separately; samples are only test views. Findings are 2D and may abstain.',
      { mode: { type: 'string', enum: ['direct', 'ab'] }, reference_a: { ...view, description: 'Fixed A reference, not a sample.' },
        reference_b: { ...view, description: 'Fixed B reference, not a sample.' }, prototypes: { ...array(view), description: 'Direct references, or legacy ordered A/B with ab_comparison.' },
        candidate_id: text, reference_view_ids: array(text), samples: { ...array(view), description: 'Views to test against the fixed references; never used as references automatically.' },
        ab_comparison: {type: 'boolean', description: 'Legacy mode switch; prefer mode=ab.'} }, ['samples'], async (b, a, exec) => {
        const history = a.candidate_id || a.reference_view_ids ? sourceRows(b, await b.ledger.read()).filter((r) => r.event === 'object_candidate_view') : []
        const selected = (a.reference_view_ids ?? []).map((id) => history.find((r) => matchesIdentity(r, id)))
        const recovery = (code, message, example) => ({ status: 'invalid_parameters', code, message, retry_tool: 'compare_object_views',
          next_action: 'Use exact PTS from catalog_search or source_resample, and x,y,width,height in original pixels.', example, scope: statusScope })
        if (selected.some((r) => !r)) return recovery('UNKNOWN_REFERENCE_VIEW', 'Look up the view in object_candidate action=list, or supply exact prototype views.', { mode: 'direct', prototypes: [], samples: a.samples })
        const named = Boolean(a.reference_a || a.reference_b)
        const ab = a.mode === 'ab' || a.ab_comparison === true || named
        const stored = history.filter((r) => r.candidate_id === a.candidate_id && r.role === 'prototype')
        const chosen = named ? [a.reference_a, a.reference_b] : (a.prototypes ?? (a.reference_view_ids ? selected.map((r) => r.view) : stored.map((r) => r.view)))
        if (ab && (chosen.length !== 2 || chosen.some((r) => !r))) return recovery('MISSING_AB_REFERENCES',
          'A/B requires two fixed references, independent of the test samples. Add reference_a and reference_b, then keep the views to be tested in samples. For one reference, use direct mode.',
          { mode: 'ab', required_fields: ['reference_a', 'reference_b', 'samples'], sample_view_shape: { pts: 'exact source PTS', native_xywh: ['x', 'y', 'width', 'height'] } })
        if (ab && chosen[0].pts === chosen[1].pts && JSON.stringify(chosen[0].native_xywh) === JSON.stringify(chosen[1].native_xywh)) {
          return recovery('DUPLICATE_AB_REFERENCES', 'A and B point to the same PTS and native rectangle. Choose a different reference, or use direct mode with one reference.',
            { mode: 'direct', prototypes: [chosen[0]], samples: a.samples })
        }
        if (!chosen.length || !a.samples?.length) return recovery('MISSING_COMPARISON_VIEWS', 'Provide at least one reference and one separate sample.',
          { mode: 'direct', required_fields: ['prototypes or candidate_id or reference_view_ids', 'samples'] })
        const prototypes = await sourceViews(b, exec, chosen)
        const samples = await sourceViews(b, exec, a.samples)
        const originals = [...prototypes, ...samples]
        const targets = await backend.nativeBatch(b, originals, exec.signal)
        const keys = await Promise.all(targets.map(async (p) => createHash('sha256').update(await readFile(p)).digest('hex')))
        const prepared = originals.map((f, i) => ({ ...f, png_sha256: keys[i], path: targets[i] }))
        const measured = await compute(b, exec, 'local_observation', { prototypes: prepared.slice(0, prototypes.length), samples: prepared.slice(prototypes.length) })
        if (!measured.rows) return measured
        const rows = measured.rows.map((row, i) => {
          const sample = samples[i]
          const comparisons = row.comparisons.map((result, j) => {
            const ref = prototypes[j]
            const root = idOf('source-pair', b.manifest.video_sha256, ...[ref.pts, sample.pts].sort((x, y) => x - y))
            const local = result.local.state === 'heldout_supported'
            const context = result.excluded_context.state === 'heldout_supported'
            const motion = result.residual.target_projection_median_px
            return { ...result, reference_index: j, source_root: root,
              explanations: [local ? 'local_appearance_correspondence' : 'local_identity_evidence_insufficient',
                context ? 'excluded_context_consistent_2d' : 'view_change_edit_or_context_insufficient',
                motion > 3 ? 'target_motion_depth_or_alignment_compete' : 'target_motion_unresolved',
                'occlusion_replacement_and_similar_instance_remain_competing'] }
          })
          return { sample, comparisons }
        })
        const order = [...rows].sort((x, y) => x.sample.seconds - y.sample.seconds)
        const markers = ab ? abMarkers(order, prototypes) : null
        return { status: 'completed', mode: ab ? 'ab' : 'direct', candidate_id: a.candidate_id ?? null, prototypes, source_sha256: b.manifest.video_sha256,
          source_originals: prepared.map(({ path, ...f }) => f), sample_count: rows.length, rows, markers,
          measurement_definitions: {
            target_mad_8bit: 'Arithmetic mean of absolute aligned 8-bit luma residuals in the usable target ROI after background offset correction.',
            context_mad_8bit: 'Arithmetic mean of absolute aligned 8-bit luma residuals in excluded context after background offset correction.',
            median_px: 'Median holdout reprojection error in analysis pixels.',
            target_projection_median_px: 'Median target reprojection residual in analysis pixels for the excluded-context fit.',
          }, scope: statusScope }
      })
  }

  if (config.stateMemory !== false) {
    tool('state_chain', 'Record or query timed candidate states and relationships, with source citations, earlier revisions and unknown gaps. Select by candidate, relationship, time or source root. No automatic object continuity between samples.',
      { action: { type: 'string', enum: ['add', 'list'] }, candidate_id: text, related_candidate_id: text, identity_id: text, relationship: text,
        phase: { type: 'string', enum: ['initial', 'change', 'interaction', 'final', 'observation', 'unknown'] }, content: text,
        pts: integer, root: text, evidence_ids: array(text), revision_ids: array(text), ...range }, ['action'], async (b, a, exec) => {
        const history = sourceRows(b, await b.ledger.read())
        const { lo, hi } = backend.selection(b, a.start, a.end)
        const revisions = revisionMap(history)
        const latest = latestMap(revisions)
        const identity = a.identity_id && history.find((r) => r.event === 'object_candidate_view' && matchesIdentity(r, a.identity_id))
        if (a.identity_id && !identity) throw new Error('Find an existing same-source identity_id with object_candidate action=list.')
        const candidate = a.candidate_id ?? identity?.candidate_id
        if (a.action === 'list') {
          const entries = history.filter((r) => r.event === 'state_chain_entry' && (!candidate || [r.candidate_id, r.related_candidate_id].includes(candidate))
            && (!a.phase || r.phase === a.phase) && (!a.relationship || r.relationship === a.relationship) && (!a.root || r.roots.includes(a.root))
            && lo <= r.frame.seconds && r.frame.seconds <= hi).sort((x, y) => x.frame.seconds - y.frame.seconds)
          const pts = [...new Set(entries.map((r) => r.frame.seconds))]
          return { ...page(entries.map((r) => ({ ...r, review_revisions: r.revision_ids.filter((id) => revisions.has(id) && latest.get(revisions.get(id).observation_id) !== id) })), a),
            unknown_between_observed_states: pts.slice(1).map((end, i) => [pts[i], end]), scope: statusScope }
        }
        const candidates = new Set(history.filter((r) => r.event === 'object_candidate_view').map((r) => r.candidate_id))
        if (a.action !== 'add' || !a.content?.trim() || !['initial', 'change', 'interaction', 'final', 'observation', 'unknown'].includes(a.phase)
          || !candidates.has(candidate) || (a.related_candidate_id && !candidates.has(a.related_candidate_id))) throw new Error('Provide an existing candidate, valid phase and the observed content.')
        const frame = stamp(b, await exact(b, exec, a.pts))
        if (identity && (identity.view.pts !== frame.pts || identity.candidate_id !== candidate)) throw new Error('A view identity_id binds only its recorded candidate and exact PTS. Use candidate_id for a different PTS.')
        const refs = await checkRefs(b, history, [...new Set(a.evidence_ids ?? [])])
        if (a.phase !== 'unknown' && !refs.some((r) => r.type === 'original' && r.pts === frame.pts)) throw new Error('A visible state must cite an original received at its exact PTS.')
        const roots = [...new Set([frame.frame_id, ...refs.flatMap((r) => originalRoots(b, r)), ...refs.map((r) => r.id)])]
        const record = { event: 'state_chain_entry', state_id: randomUUID(), video_sha256: b.manifest.video_sha256, task_id: b.taskId,
          candidate_id: candidate, identity_id: a.identity_id ?? null, related_candidate_id: a.related_candidate_id ?? null, relationship: a.relationship ?? null,
          frame, phase: a.phase, content: a.content.trim(), roots, evidence_ids: refs.map((r) => r.id),
          revision_ids: validateRevisionDependencies(history, a.revision_ids), at: new Date().toISOString() }
        await b.ledger.append(record)
        return record
      })

    tool('observation_review', 'Inspect revision histories, upstream dependencies, downstream affected revisions and conflicts. Optional final claims are checked for citation type, shared original roots, and stale revision links. Semantic contradiction requires review.',
      { observation_id: text, revision_id: text, query: text, evidence_ids: array(text), omitted_observation_ids: array(text), ...range,
        final_claims: array({ type: 'object', properties: { content: text, evidence_ids: array(text), revision_ids: array(text), claim_scope: text,
          kind: { type: 'string', enum: ['direct', 'inference', 'unknown'] }, stated_details: array(text) }, required: ['content'] }) }, [], async (b, a) => {
        const history = sourceRows(b, await b.ledger.read())
        const revisions = revisionMap(history)
        const latest = latestMap(revisions)
        const disputes = history.filter((r) => r.event === 'observation_dispute')
        const stateByRevision = new Map()
        for (const state of history) if (state.event === 'state_chain_entry') for (const revision of state.revision_ids ?? []) {
          if (!stateByRevision.has(revision)) stateByRevision.set(revision, [])
          stateByRevision.get(revision).push({ state_id: state.state_id, candidate_id: state.candidate_id, pts: state.frame.pts })
        }
        const stale = (id) => revisions.has(id) && latest.get(revisions.get(id).observation_id) !== id
        const { lo, hi } = backend.selection(b, a.start, a.end)
        let selected = [...revisions.values()].filter((r) => (!a.observation_id || r.observation_id === a.observation_id) && (!a.revision_id || r.revision_id === a.revision_id)
          && (!a.query || r.content.toLowerCase().includes(a.query.toLowerCase())) && (!a.evidence_ids || a.evidence_ids.some((id) => r.evidence_ids.includes(id))))
        const unresolved_time_refs = []
        if (a.start != null || a.end != null) {
          const timed = new Set()
          for (const id of new Set(selected.flatMap((r) => r.evidence_ids))) {
            try {
              const [ref] = await checkRefs(b, history, [id], true)
              const samples = Number.isFinite(ref.seconds) ? [ref.seconds] : (ref.source_moments ?? []).map((f) => f.seconds)
              if (samples.some((t) => lo <= t && t <= hi) || ref.text_time_spans?.some(([start, end]) => start <= hi && end >= lo)) timed.add(id)
            } catch (error) { unresolved_time_refs.push({ id, reason: error.message }) }
          }
          selected = selected.filter((r) => r.evidence_ids.some((id) => timed.has(id)))
        }
        const items = selected.map((r) => ({ ...r, is_current: !stale(r.revision_id), revision_dependencies: [...ancestors(r.revision_id, revisions)].slice(1),
            needs_review_upstream: [...ancestors(r.revision_id, revisions)].slice(1).filter(stale),
            affected_downstream: [...revisions.keys()].filter((id) => id !== r.revision_id && ancestors(id, revisions).has(r.revision_id)),
            disputes: disputes.filter((v) => v.revision_id === r.revision_id).map((v) => v.dispute_id), linked_states: stateByRevision.get(r.revision_id) ?? [] }))
        const claims = await Promise.all((a.final_claims ?? []).map(async (claim) => {
          const refs = [], bad = []
          for (const id of new Set(claim.evidence_ids ?? [])) {
            try { refs.push(...await checkRefs(b, history, [id])) } catch (error) { bad.push({ id, reason: String(error.message) }) }
          }
          const rootMap = new Map()
          for (const ref of refs) for (const root of originalRoots(b, ref)) {
            rootMap.set(root, [...(rootMap.get(root) ?? []), ref.id])
          }
          const dependencies = [...new Set(claim.revision_ids ?? [])]
          const connected = [...new Set(dependencies.filter((id) => revisions.has(id)).flatMap((id) => [...ancestors(id, revisions)]))]
          const scope = claim.claim_scope ?? 'visual'
          const literalNotInClaim = (claim.stated_details ?? []).filter((detail) => !claim.content.toLowerCase().includes(detail.toLowerCase()))
          return { content: claim.content, kind: claim.kind ?? 'unclassified', declared_details_not_written_verbatim: literalNotInClaim,
            missing_or_invalid_refs: bad, scope_mismatches: refs.filter((r) => !r.allowed_scopes.includes(scope)).map((r) => r.id),
            original_roots: Object.fromEntries(rootMap), same_source_groups: [...rootMap.values()].filter((ids) => ids.length > 1),
            origin_unresolved: refs.filter((r) => ![...rootMap.values()].some((ids) => ids.includes(r.id))).map((r) => r.id),
            citation_state: refs.length ? 'reference_present' : 'no_valid_reference', requested_scope: scope,
            stale_revision_ids: connected.filter(stale), unknown_revision_ids: dependencies.filter((id) => !revisions.has(id)),
            open_dispute_ids: disputes.filter((r) => connected.includes(r.revision_id)).map((r) => r.dispute_id), semantic_support: 'requires_independent_review' }
        }))
        const omitted = [...new Set(a.omitted_observation_ids ?? [])].map((id) => ({ observation_id: id, revision_id: latest.get(id) ?? null, reason: 'agent_marked_omitted' }))
        return { ...page(items, a), final_claim_checks: claims, declared_omissions: omitted, unresolved_time_refs,
          scope: 'Mechanical source, literal-detail and dependency check. Time filters use original and crop timestamps, decoded calculation inputs or transcript intervals. Independent visual review decides correctness or semantic omission.' }
      })
  }
}

function abMarkers(rows, prototypes) {
  const minimum = Math.min(...prototypes.map((f) => f.seconds))
  const eligible = rows.filter(({ sample }) => sample.seconds >= minimum && !prototypes.some((f) => f.pts === sample.pts))
    .filter((r, i, all) => i === 0 || r.sample.pts !== all[i - 1].sample.pts)
  const exclusiveB = ({ comparisons: [a, b] }) => a.local.state !== 'heldout_supported' && b.local.state === 'heldout_supported'
  const after = eligible.findIndex((r, i) => exclusiveB(r) && eligible[i + 1] && exclusiveB(eligible[i + 1]))
  const XD = eligible.filter(({ comparisons: [a] }) => a.local.state === 'heldout_supported' && a.residual.state === 'measured_on_overlap'
    && a.residual.target_mad_8bit > 7 && a.residual.target_mad_8bit > a.residual.context_mad_8bit)
    .map(({ sample, comparisons: [a] }) => ({ sample, root: a.source_root, state: 'A_local_support_with_residual_candidate' }))
  return { XB: after < 0 ? {state: 'insufficient_two_distinct_sampled_B_only_points'}
    : {state: 'two_consecutive_supplied_samples_support_B_only', first_sample: eligible[after].sample, confirmation_sample: eligible[after + 1].sample,
      previous_supplied_sample: eligible[after - 1]?.sample ?? null, earlier_presence: 'unknown'}, XD,
  scope: 'These are image comparison candidates among supplied samples. No interpolation, physical crossing, or proof that A disappeared.' }
}
