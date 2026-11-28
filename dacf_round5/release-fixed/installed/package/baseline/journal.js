import { randomUUID, createHash } from 'node:crypto'
import { mkdir, open, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { withProcessLock } from './process_lock.js'

const queues = new Map()
const fingerprint = (row) => createHash('sha256').update(JSON.stringify(row)).digest('hex')

async function read(path) {
  try { return await readFile(path, 'utf8') } catch (error) {
    if (error.code === 'ENOENT') return ''
    throw error
  }
}

function parse(raw) {
  const rows = [], damaged = []
  for (const [i, line] of raw.split('\n').entries()) {
    if (!line.trim()) continue
    try {
      const row = JSON.parse(line)
      if (!row || typeof row.event !== 'string') throw new Error('missing event')
      rows.push(row)
    } catch { damaged.push(i + 1) }
  }
  return { rows, damaged }
}

function union(first, second) {
  const known = new Map()
  const add = (source) => {
    const sequence = new Map()
    for (const row of source) {
      const hash = row.journal_id ?? fingerprint(row)
      const ordinal = (sequence.get(hash) ?? 0) + 1
      sequence.set(hash, ordinal)
      const key = row.journal_id ?? `${hash}/${ordinal}`
      const previous = known.get(key)
      if (previous && fingerprint(previous) !== fingerprint(row)) throw new Error('Conflicting contents for one ledger journal ID.')
      if (!previous) known.set(key, row)
    }
  }
  add(first); add(second)
  return [...known.values()].sort((a, b) => {
    if (a.journal_seq != null && b.journal_seq != null) return a.journal_seq - b.journal_seq
    if (a.journal_seq == null && b.journal_seq != null) return -1
    if (a.journal_seq != null && b.journal_seq == null) return 1
    return String(a.at ?? '').localeCompare(String(b.at ?? ''))
  })
}

async function durableAppend(path, record) {
  await mkdir(dirname(path), { recursive: true })
  const file = await open(path, 'a')
  try { await file.writeFile(`\n${JSON.stringify(record)}\n`); await file.sync() } finally { await file.close() }
}

async function reset(path, raw, rows) {
  await mkdir(dirname(path), { recursive: true })
  if (parse(raw).damaged.length) await writeFile(`${path}.damaged-${randomUUID()}`, raw, { flag: 'wx' })
  const temporary = `${path}.${randomUUID()}.tmp`
  const file = await open(temporary, 'wx')
  try { await file.writeFile(rows.map((r) => JSON.stringify(r)).join('\n') + '\n'); await file.sync() } finally { await file.close() }
  await rename(temporary, path)
}

export function journal(cache, base, lockProcesses = true) {
  const name = base.file.split(/[\\/]/).at(-1)
  const replica = join(cache, 'active-journal-v3', name)
  function serial(run) {
    const prev = queues.get(base.file) ?? Promise.resolve()
    const next = prev.catch(() => {}).then(() => lockProcesses ? withProcessLock(cache, name, run) : run())
    queues.set(base.file, next)
    return next.finally(() => { if (queues.get(base.file) === next) queues.delete(base.file) })
  }
  async function recover() {
    let [live, mirrored] = await Promise.all([read(base.file), read(replica)])
    if (!live && !mirrored) { await base.read(); live = await read(base.file) }
    const a = parse(live), b = parse(mirrored)
    const rows = union(a.rows, b.rows)
    if (!rows.length && (a.damaged.length || b.damaged.length)) throw new Error('No valid record could be recovered from either active ledger copy.')
    if (a.damaged.length || a.rows.length !== rows.length) await reset(base.file, live, rows)
    if (b.damaged.length || b.rows.length !== rows.length) await reset(replica, mirrored, rows)
    return rows
  }
  return {
    file: base.file, replica,
    async read() { return serial(async () => (await recover()).slice()) },
    async append(record, validate) {
      return serial(async () => {
        const rows = await recover()
        if (validate) validate(rows)
        const state = { ...record, journal_id: randomUUID(), journal_seq: (rows.at(-1)?.journal_seq ?? 0) + 1 }
        await durableAppend(replica, state)
        await durableAppend(base.file, state)
      })
    },
  }
}
