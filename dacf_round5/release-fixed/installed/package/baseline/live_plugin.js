import { createHash } from 'node:crypto'
import { resolve } from 'node:path'
import { registerGuided } from './guide.js'

export const name = 'video-observer-live'
export const inject = ['tools']
const { DatabaseSync } = process.getBuiltinModule('node:sqlite') ?? {}
const integer = { type: 'integer' }
const scope = 'Committed stream originals only. Unknown gaps and the future are unobserved. Measurements are navigation.'
const fmt = { pts: integer, start_pts: integer, end_pts: integer, cursor: integer, page_size: integer }
const pngBlock = (ref) => ({type: 'image', attachment: ref})
const inspect = 'PTS are integer source ticks, not seconds. Query live_status for time_base and the committed watermark.'

function limit(a) {
  if (a.cursor != null && (!Number.isSafeInteger(a.cursor) || a.cursor < 0)) throw new Error('Use the next_cursor from the preceding page.')
  return Math.max(1, Math.min(24, a.page_size ?? 8))
}

function bounds(a) {
  const lo = a.start_pts ?? Number.MIN_SAFE_INTEGER, hi = a.end_pts ?? Number.MAX_SAFE_INTEGER
  if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || hi < lo) throw new Error(`Use a valid integer start_pts/end_pts interval. ${inspect}`)
  return [lo, hi]
}

export function apply(ctx, config) {
  if (!config?.streamDb || !DatabaseSync) throw new Error('Configure the stream database and Node SQLite before enabling the live plugin.')
  const register = registerGuided(ctx, config.guide !== false, 'The live tools expose committed original frames, known gaps and incremental visual navigation from a separate producer. Choose when to query and which PTS or density to read. The live observer has no source-file tool; future frames cannot be read.')
  const path = resolve(config.streamDb)
  function connection(operation) {
    const db = new DatabaseSync(path, {readOnly: true})
    try {
      db.exec('PRAGMA busy_timeout=3000; BEGIN')
      const key = Object.fromEntries(db.prepare('SELECT key,value FROM metadata').all().map((r) => [r.key,r.value]))
      const tip = db.prepare('SELECT sequence,committed_pts FROM events ORDER BY sequence DESC LIMIT 1').get()
      return operation(db, { stream_id: key.stream_id, source_sha256: key.source_sha256, time_base: key.time_base,
        closed: key.closed === 'true', sequence: tip?.sequence ?? null, committed_through_pts: tip?.committed_pts ?? null, scope })
    } finally { db.close() }
  }

  const jsonOutput = { schema: {type:'object', additionalProperties:true}, render: (_, value) => [{type:'text', text:JSON.stringify(value)}] }
  register({ name: 'live_status', description: `Check the committed live watermark, explicit producer gaps, recovery events and stream time base. ${inspect}`,
    parameters: {type:'object', properties:{after_sequence:integer,page_size:integer}}, output: jsonOutput,
    async execute(a) { return connection((db, state) => {
      const n = limit(a), after = a.after_sequence ?? -1
      if (!Number.isSafeInteger(after) || after < -1) throw new Error('Use an event sequence integer from live_status.')
      const events = db.prepare('SELECT document FROM events WHERE sequence>? ORDER BY sequence LIMIT ?').all(after,n+1).map((r)=>JSON.parse(r.document))
      const page = events.slice(0,n)
      const gaps = db.prepare("SELECT document FROM events WHERE event='gap'").all().map((r) => JSON.parse(r.document).range_pts)
      return { ...state, available_originals: db.prepare('SELECT COUNT(*) n FROM frames').get().n, missing_pts_ranges: gaps,
        events:page, next_after_sequence: events.length>n ? page.at(-1).sequence : null }
    }) }
  })

  for (const wave of [false,true]) register({ name: wave?'live_wave':'live_catalog',
    description: wave ? `Inspect committed edge density, brightness and adjacent received-frame residual. These are navigation hints. ${inspect}`
      : `Find already committed native-frame PTS and source hashes. This live catalogue grows with arrivals. ${inspect}`,
    parameters: {type:'object', properties:fmt}, output: jsonOutput,
    async execute(a) { return connection((db, state) => {
      const n=limit(a), offset=a.cursor??0, [lo,hi]=bounds(a)
      const fields = wave ? 'pts,seconds,information,delta_mad,exposure' : 'pts,seconds,source_frame,width,height,sha AS png_sha256'
      const items = db.prepare(`SELECT ${fields} FROM frames WHERE pts BETWEEN ? AND ? ORDER BY pts LIMIT ? OFFSET ?`).all(lo,hi,n+1,offset)
      return { ...state, items:items.slice(0,n), next_cursor:items.length>n ? offset+n : null,
        measurement:wave?'Image edge density and unregistered luma differences. A missing predecessor/gap leaves delta null.':undefined }
    }) }
  })

  const output = {schema:{type:'object',properties:{envelope:{type:'string'},images:{type:'array',items:{type:'object',additionalProperties:true}}},required:['envelope','images']},
    render: (_, value) => [{type:'text',text:value.envelope},...value.images.map(pngBlock)] }
  register({name:'live_read', description:`Read exact committed original PNGs by pts_list or by a PTS interval at an optional caller-chosen fps. Continue pages as needed. ${inspect}`,
    parameters:{type:'object',properties:{...fmt,pts_list:{type:'array',items:integer},fps:{type:'number'}}}, output,
    async execute(a) {
      const n=limit(a), offset=a.cursor??0, [lo,hi]=bounds(a)
      const query = connection((db,state) => {
        const one=db.prepare('SELECT * FROM frames WHERE pts=?')
        const unknown = []
        let selected
        if (a.pts_list?.length) {
          if (a.fps != null || a.start_pts != null || a.end_pts != null) throw new Error('Use pts_list alone, or a PTS range with optional fps.')
          if (a.pts_list.some((p)=>!Number.isSafeInteger(p))) throw new Error(`Use integer pts_list. ${inspect}`)
          selected = [...new Set(a.pts_list)].map((p) => {
            const row = one.get(p)
            if (!row) unknown.push(p)
            return row
          }).filter(Boolean)
          return {state,unknown,selected:selected.slice(offset,offset+n),next:offset+n<selected.length?offset+n:null}
        }
        if (a.fps != null && (!Number.isFinite(a.fps) || a.fps <= 0)) throw new Error('Use a finite positive caller-selected fps or omit it.')
        if (a.fps == null) {
          const items=db.prepare('SELECT * FROM frames WHERE pts BETWEEN ? AND ? ORDER BY pts LIMIT ? OFFSET ?').all(lo,hi,n+1,offset)
          return {state,unknown,selected:items.slice(0,n),next:items.length>n?offset+n:null}
        }
        const stamps=db.prepare('SELECT pts,seconds FROM frames WHERE pts BETWEEN ? AND ? ORDER BY pts').all(lo,hi)
        let next=-Infinity
        const chosen=stamps.filter((r) => {
          if (r.seconds + 1e-9 < next) return false
          next = r.seconds + 1/a.fps
          return true
        })
        return {state,unknown,selected:chosen.slice(offset,offset+n).map((r)=>one.get(r.pts)),next:offset+n<chosen.length?offset+n:null}
      })
      const images=[], receipts=[]
      for (const row of query.selected) {
        const png=Buffer.from(row.png)
        const hash=createHash('sha256').update(png).digest('hex')
        if (hash!==row.sha) throw new Error(`Stored committed original failed SHA check at PTS ${row.pts}.`)
        const image=await ctx.get('attachments').saveImage({data:png,mediaType:'image/png',name:`live_${row.pts}.png`})
        if (image.attachmentId!==`sha256:${hash}` || image.width!==row.width || image.height!==row.height) throw new Error('Transport changed the committed original.')
        images.push(image)
        receipts.push({pts:row.pts,seconds:row.seconds,source_frame:row.source_frame,source_png_sha256:hash,
          evidence_id:`live-${query.state.stream_id}-${row.pts}-${hash.slice(0,12)}`,width:row.width,height:row.height})
      }
      return {envelope:JSON.stringify({...query.state, originals:receipts, no_original_at_requested_pts:query.unknown,next_cursor:query.next}),images}
    }
  })
}
