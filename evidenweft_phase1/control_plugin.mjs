import { appendFile, realpath } from 'node:fs/promises'
import { readFileSync } from 'node:fs'
import { apply as frozenApply } from './frozen/visureach/index.js'
import { guideReply } from './frozen/visureach/guide.js'

const schema = { type: 'object', additionalProperties: true }
const plain = { schema, render: (_, value) => [{ type: 'text', text: JSON.stringify(value) }] }
const common = new Set(['video_open', 'catalog_search', 'observation_goal', 'delivery_request', 'source_resample', 'evidence_list'])
const catalogArgs = { type: 'object', properties: {
  start: { type: 'number', description: 'Earliest source seconds.' },
  end: { type: 'number', description: 'Latest source seconds.' },
  mode: { type: 'string', enum: ['overview', 'earliest', 'latest'], description: 'Even overview, or paginated time order.' },
  cursor: { type: 'integer' }, page_size: { type: 'integer' },
} }

export function apply(ctx, config) {
  const plan = JSON.parse(readFileSync(config.catalogPlan, 'utf8'))
  const opened = new WeakSet()
  const names = []
  async function log(row) {
    await appendFile(config.controlLog, JSON.stringify({ at: new Date().toISOString(), ...row }) + '\n', 'utf8')
  }
  const register = (tool) => {
    const name = tool.name
    if (!config.full && !common.has(name)) return
    names.push(name)
    const execute = async (a, exec) => {
      if (name === 'video_open') {
        if (await realpath(a.path) !== await realpath(config.authorizedSource)) throw new Error('Only the assigned source can be opened.')
        const v = await tool.execute(a, exec)
        if (v.video_sha256 !== plan.source_sha256) throw new Error('Frozen source hash mismatch.')
        opened.add(exec.agent.session)
        return { video_sha256: v.video_sha256, source_frames: plan.source_frames, width: v.width, height: v.height,
          time_base: v.time_base, duration_seconds: plan.end - plan.start, catalogue_count: plan.frames.length }
      }
      if (name === 'catalog_search') {
        if (!opened.has(exec.agent.session)) throw new Error('Open the assigned video before querying the catalogue.')
        const lo = a.start ?? plan.start
        const hi = a.end ?? plan.end
        const size = Math.min(30, Math.max(1, a.page_size ?? 12))
        const cursor = Math.max(0, a.cursor ?? 0)
        const direction = a.mode ?? 'overview'
        const inRange = plan.frames.filter((r) => lo <= r.seconds && r.seconds <= hi)
        if (direction === 'overview' && inRange.length > size) {
          const subset = Array.from({ length: size }, (_, i) =>
            inRange[Math.round(i * (inRange.length - 1) / Math.max(1, size - 1))])
          return { total: inRange.length, items: subset, guidance: 'See delivery_request for exact native images or source_resample for any source interval.' }
        }
        const ordered = direction === 'latest' ? [...inRange].reverse() : inRange
        const items = ordered.slice(cursor, cursor + size)
        return { total: inRange.length, items, next_cursor: cursor + size < inRange.length ? cursor + size : null }
      }
      const result = await tool.execute(a, exec)
      if (name === 'delivery_request') {
        result.envelope = result.envelope.replace(/ time_base=\S+| png_sha256=\S+| \((?:off-index: sampled but never nominated|direct source PTS, outside the coarse index)\)/g, '')
      }
      return result
    }
    ctx.tools.register({
      ...tool,
      ...(name === 'video_open' ? { description: 'Bind the assigned source. Returns source dimensions, time base and catalogue count.', output: plain } : {}),
      ...(name === 'catalog_search' ? { description: 'Find available catalogue frame IDs, source PTS and seconds. Results are metadata for navigation. Query any valid interval and choose exact originals independently.',
        parameters: catalogArgs, output: plain } : {}),
      async execute(a, exec) {
        const started = new Date().toISOString()
        try {
          const value = await execute(a ?? {}, exec)
          await log({ tool: name, started, args: a, status: value.status ?? 'completed',
            returned_catalogue_pts: name === 'catalog_search' ? value.items.map((r) => r.pts) : undefined,
            result_ref: value.result_ref, images: value.images?.length })
          return value
        } catch (error) {
          await log({ tool: name, started, args: a, status: 'error', reason: error.message })
          throw error
        }
      },
    })
  }
  frozenApply(new Proxy(ctx, {
    get(target, key) {
      if (key === 'tools') return { register }
      const value = Reflect.get(target, key, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  }), { ...config, guide: false, integration: config.full })
  if (config.full) register({ name: 'observer_guide', description: 'Look up available evidence tools, examples, and optional ways to combine them.',
    parameters: { type: 'object', properties: { tool: { type: 'string' }, topic: { type: 'string' } } }, output: plain,
    async execute(args) { return guideReply(args, names) } })
  void log({ event: 'tool_surface', tools: names })
}
