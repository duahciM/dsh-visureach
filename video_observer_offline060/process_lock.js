import { mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { join } from 'node:path'
import { setTimeout } from 'node:timers/promises'

const sqlite = process.getBuiltinModule('node:sqlite')

export const processLocksAvailable = Boolean(sqlite?.DatabaseSync)

// SQLite supplies a process-owned filesystem lock. It releases it on close or
// process death; the JSONL remains the portable record and recovery format.
export async function withProcessLock(cache, key, execute, signal) {
  if (!sqlite?.DatabaseSync) throw new Error('Cross-process protection requires the bundled Node SQLite runtime. Disable processLocks only for single-process use.')
  const folder = join(cache, 'process-locks-v4')
  await mkdir(folder, { recursive: true })
  const name = createHash('sha256').update(key).digest('hex')
  const db = new sqlite.DatabaseSync(join(folder, name + '.db'))
  try {
    for (;;) {
      signal?.throwIfAborted()
      try {
        db.exec('BEGIN IMMEDIATE')
        break
      } catch (error) {
        if (error.errcode !== 5 && error.errcode !== 6) throw error
        await setTimeout(35, undefined, { signal })
      }
    }
    const result = await execute()
    db.exec('COMMIT')
    return result
  } finally {
    db.close()
  }
}
