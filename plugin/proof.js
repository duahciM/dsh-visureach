import { createHash, randomUUID } from 'node:crypto'
import { link, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

const hash = (data) => createHash('sha256').update(data).digest('hex')
const scopes = {
  original: ['visual', 'visual_region'], derived: ['visual_region'], ocr: ['ocr_text'], asr: ['transcribed_audio'],
  subtitle: ['subtitle_text'], track_metadata: ['stream_metadata'], computed: ['measurement'], external: ['external_statement'],
}

function folder(binding) {
  return join(dirname(binding.ledger.file), '../evidence-results-v3', binding.ledger.file.split(/[\\/]/).at(-1), binding.manifest.video_sha256)
}

export async function saveResult(binding, channel, data) {
  const payload = JSON.stringify({ channel, source_sha256: binding.manifest.video_sha256, data })
  const id = 'vo-' + hash(payload)
  await mkdir(folder(binding), { recursive: true })
  const file = join(folder(binding), id + '.json')
  const temp = `${file}.${randomUUID()}.tmp`
  await writeFile(temp, payload, { flag: 'wx' })
  try {
    await link(temp, file)
  } catch (error) {
    if (error.code !== 'EEXIST' || (await readFile(file, 'utf8')) !== payload) throw error
  } finally { await rm(temp, { force: true }) }
  return id
}

export async function loadResult(binding, id) {
  if (!/^vo-[\da-f]{64}$/.test(id ?? '')) throw new Error('Use the complete result_ref from a tool in this session and source.')
  let raw
  try { raw = await readFile(join(folder(binding), id + '.json'), 'utf8') } catch (error) {
    if (error.code === 'ENOENT') throw new Error('Unknown result_ref in this session and source.')
    throw error
  }
  if (id !== 'vo-' + hash(raw)) throw new Error('Stored result hash changed.')
  return JSON.parse(raw)
}

export function resultPage(data, args) {
  const segments = args.path ?? []
  let value = data
  for (const segment of segments) {
    if (value == null || !Object.hasOwn(value, segment) || (typeof value !== 'object')) throw new Error('No field at that path.')
    value = value[segment]
  }
  const cursor = Math.max(0, args.cursor ?? 0)
  const pageSize = Math.min(60, Math.max(1, args.page_size ?? 10))
  if (typeof value === 'string' && value.length > 400) {
    const width = pageSize * 200
    return { path: segments, text: value.slice(cursor, cursor + width), total: value.length, next_cursor: cursor + width < value.length ? cursor + width : null }
  }
  if (value == null || typeof value !== 'object') return { path: segments, value }
  const items = Object.entries(value)
  return { path: segments, entries: items.slice(cursor, cursor + pageSize).map(([key, v]) => (
    v != null && typeof v === 'object' ? { key, type: Array.isArray(v) ? 'array' : 'object', size: Object.keys(v).length }
      : typeof v === 'string' && v.length > 400 ? { key, type: 'text', length: v.length } : { key, value: v })),
    total: items.length, next_cursor: cursor + pageSize < items.length ? cursor + pageSize : null }
}

export async function reference(binding, records, evidence, id) {
  if (id.startsWith('vo-')) {
    const found = await loadResult(binding, id)
    return { id, type: found.channel, allowed_scopes: scopes[found.channel] ?? [], integrity: 'stored_result_hash_verified',
      support: 'scope only; subject matter must be independently checked' }
  }
  const original = evidence.get(id)
  if (original) return { id, type: 'original', allowed_scopes: scopes.original, pts: original.pts, source_frame: original.source_frame, integrity: 'source_receipt' }
  const row = records.find((r) => r.video_sha256 === binding.manifest.video_sha256 && (r.derived_id === id || r.external_id === id))
  if (row?.derived_id) {
    if (!evidence.has(row.parent_evidence_id)) throw new Error('Derived reference has no available same-session parent original.')
    return { id, type: 'derived', allowed_scopes: scopes.derived, parent_evidence_id: row.parent_evidence_id,
      native_xywh: row.native_xywh, pts: row.pts, integrity: 'derived_receipt_with_parent' }
  }
  if (row?.external_id) return { id, type: 'external', allowed_scopes: scopes.external, integrity: 'agent_declared_not_recomputed' }
  throw new Error('Use an existing same-session, same-source evidence ID.')
}
