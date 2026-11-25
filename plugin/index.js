/**
 * Video observation tools for DeepSeek Harness.
 *
 * Four tools let an image-capable Agent work over one local video:
 *   video_open        index a video (cached by content hash) and bind a handle
 *   catalog_search    query nominated keyframes and change hints, never images
 *   delivery_request  deliver exact source frames as native-resolution PNGs
 *   source_resample   re-decode the original video over a chosen interval
 *
 * Images travel as attachment references rendered into the tool result, so the
 * model receives the real PNG rather than a text placeholder. The plugin imports
 * no Harness package: it only uses services available on the context.
 */
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { appendFile, mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const INDEXER = join(HERE, 'python', 'index_video.py')

export const name = 'video-observer'
export const inject = ['tools']

/** Per-Session video bindings, keyed by the owning Agent id. */
const bindings = new Map()

/**
 * Append-only delivery ledger. Every catalogue query, frame read and resample
 * is recorded per session so "requested" stays distinguishable from "delivered",
 * and a failure or an interrupted run leaves a recoverable prefix behind.
 */
function ledger(cache, agentId) {
  const folder = join(cache, 'sessions')
  const file = join(folder, `${String(agentId ?? 'unbound').replace(/[^\w.-]/g, '_')}.jsonl`)
  return {
    file,
    async append(record) {
      await mkdir(folder, { recursive: true })
      await appendFile(file, `${JSON.stringify(record)}\n`, 'utf8')
    },
    async read() {
      try {
        const text = await readFile(file, 'utf8')
        return text.split('\n').filter((line) => line.trim().length > 0).map((line) => JSON.parse(line))
      } catch {
        return []
      }
    },
  }
}

/**
 * Snapshot every ledger file that has never been archived before.
 *
 * The ledger is append-only; this archive exists so an accidental delete of the
 * live directory can never destroy an experiment's evidence again. Files are
 * deduplicated by content hash, so repeated passes are cheap and idempotent.
 */
async function archiveLedgers(cache, reason) {
  const sessions = join(cache, 'sessions')
  const archiveRoot = join(cache, 'archive')
  const indexFile = join(archiveRoot, 'index.json')
  let names
  try {
    names = (await readdir(sessions)).filter((name) => name.endsWith('.jsonl'))
  } catch {
    return { archived: 0, skipped: 0, error: 'no sessions directory' }
  }
  if (names.length === 0) return { archived: 0, skipped: 0 }

  let index = { entries: {} }
  try {
    index = JSON.parse(await readFile(indexFile, 'utf8'))
  } catch {
    index = { entries: {} }
  }
  index.entries ??= {}

  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  let archived = 0
  let skipped = 0
  for (const name of names) {
    const source = join(sessions, name)
    let bytes
    try {
      bytes = await readFile(source)
    } catch {
      continue
    }
    const hash = createHash('sha256').update(bytes).digest('hex')
    const prior = index.entries[name]
    if (prior !== undefined) {
      // Same content already archived, or only an append happened. Archive the
      // newer content too rather than overwriting the older snapshot.
      if (prior.hashes?.includes(hash)) {
        skipped += 1
        continue
      }
    }
    const targetDir = join(archiveRoot, stamp)
    await mkdir(targetDir, { recursive: true })
    const target = join(targetDir, name)
    await writeFile(target, bytes)
    const record = prior ?? { name, hashes: [], snapshots: [] }
    record.hashes = [...new Set([...(record.hashes ?? []), hash])]
    record.snapshots = [...(record.snapshots ?? []), { at: stamp, path: target, hash, bytes: bytes.length }]
    index.entries[name] = record
    archived += 1
  }
  await mkdir(archiveRoot, { recursive: true })
  await writeFile(indexFile, JSON.stringify({ ...index, reason, updatedAt: new Date().toISOString() }, null, 1), 'utf8')
  return { archived, skipped, reason }
}

/**
 * Collect the evidence ids this session actually holds for one video.
 *
 * Evidence ids are written only by delivery_request and source_resample, as a
 * side effect of attaching an image. Nothing here can create one, so an id that
 * appears in this list is a frame that was really delivered.
 */
function listEvidence(records, videoSha) {
  const seen = new Map()
  for (const record of records) {
    if (record.event !== 'delivery_request' && record.event !== 'source_resample') continue
    if (record.video_sha256 !== videoSha) continue
    for (const frame of record.frames ?? []) {
      const id = frame.evidence_id
      if (id === undefined) continue
      const prior = seen.get(id)
      if (prior !== undefined) {
        prior.receipts += 1
        continue
      }
      seen.set(id, {
        evidence_id: id,
        seconds: frame.seconds,
        pts: frame.pts,
        source_frame: frame.source_frame,
        png_sha256: frame.png_sha256,
        via: record.event,
        task_id: record.task_id,
        at: record.at,
        reason: record.reason,
        in_catalog: frame.in_catalog ?? true,
        receipts: 1,
      })
    }
  }
  return [...seen.values()].sort((a, b) => a.seconds - b.seconds)
}

function cacheRoot(config) {
  if (config?.cacheDir) return resolve(config.cacheDir)
  const home = process.env.DSH_HOME ?? join(homedir(), '.dsh')
  return join(home, 'video-observer')
}

function resolvePython(config) {
  if (config?.pythonPath) return config.pythonPath
  for (const candidate of [
    'D:/codex/p1-transition/.venv/Scripts/python.exe',
    join(
      process.env.DSH_HOME ?? join(homedir(), '.dsh'),
      'dsh-runtimes',
      'dsh-primary-runtime',
      'dependencies',
      'python',
      'python.exe',
    ),
  ]) {
    if (existsSync(candidate)) return candidate
  }
  return 'python'
}

function run(command, argv, signal, cwd) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, argv, {
      cwd,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      windowsHide: true,
    })
    let stdout = ''
    let stderr = ''
    const onAbort = () => child.kill()
    signal?.addEventListener('abort', onAbort, { once: true })
    child.stdout?.on('data', (chunk) => { stdout += chunk.toString('utf8') })
    child.stderr?.on('data', (chunk) => { stderr += chunk.toString('utf8') })
    child.on('error', (error) => {
      signal?.removeEventListener('abort', onAbort)
      reject(error)
    })
    child.on('close', (code) => {
      signal?.removeEventListener('abort', onAbort)
      if (signal?.aborted) {
        reject(new Error('cancelled'))
        return
      }
      if (code !== 0) {
        reject(new Error(`${command} exited ${code}: ${(stderr || stdout).slice(-1500)}`))
        return
      }
      resolvePromise({ stdout, stderr })
    })
  })
}

const readJson = async (path) => JSON.parse(await readFile(path, 'utf8'))

async function sha256File(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex')
}

/** Resolve one video handle for the calling Agent, or explain why it is absent. */
function bindingFor(exec) {
  const binding = bindings.get(exec.agent?.id)
  if (binding === undefined) {
    throw new Error(
      'No video is open in this session. Call video_open with the path to a local video first.',
    )
  }
  return binding
}

function selection(binding, start, end) {
  const lo = start === undefined || start === null ? binding.manifest.start : Number(start)
  const hi = end === undefined || end === null ? binding.manifest.end : Number(end)
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo > hi) {
    throw new Error('start and end must be finite source seconds with start <= end')
  }
  const clampedLo = Math.max(binding.manifest.start, lo)
  const clampedHi = Math.min(binding.manifest.end, hi)
  return { lo: clampedLo, hi: clampedHi, clamped: clampedLo !== lo || clampedHi !== hi }
}

const seconds = (value) => Number(Number(value).toFixed(3))

function frameItem(binding, row) {
  const index = binding.catalogByFrame.get(row.frame_id)
  return {
    frame_id: row.frame_id,
    source_frame: row.source_frame,
    pts: row.pts,
    seconds: seconds(row.seconds),
    in_catalog: row.in_catalog,
    dimensions: index === undefined ? [] : index.dimensions,
    roles: index === undefined ? [] : index.roles,
  }
}

/** Compact per-frame view: dimensions and roles only, no raw analysis text. */
function catalogIndex(catalog) {
  const byFrame = new Map()
  for (const row of catalog.frames) {
    const dimensions = new Set()
    const roles = new Set()
    for (const observation of row.observations ?? []) {
      dimensions.add(observation.dimension_id)
      roles.add(observation.role)
    }
    byFrame.set(row.frame_id, {
      dimensions: [...dimensions].sort(),
      roles: [...roles].sort().slice(0, 5),
    })
  }
  return byFrame
}

async function indexVideo(config, videoPath, signal) {
  const absolute = resolve(videoPath)
  const info = await stat(absolute).catch(() => undefined)
  if (info === undefined || !info.isFile()) throw new Error(`not a regular file: ${absolute}`)

  const digest = await sha256File(absolute)
  const root = cacheRoot(config)
  // The cache key covers the sampling parameters, not just the bytes: a
  // degraded index must never be silently served as if it were a full scan.
  const fps = Number(config?.sampleFps) > 0 ? Number(config.sampleFps) : 5
  const limit = Number(config?.frameLimit) > 0 ? Number(config.frameLimit) : 0
  const folder = join(root, `${digest.slice(0, 16)}-fps${fps}${limit > 0 ? `-cap${limit}` : ''}`)
  const manifestPath = join(folder, 'manifest.json')

  if (existsSync(manifestPath)) {
    const manifest = await readJson(manifestPath)
    if (manifest.video_sha256 !== digest) {
      throw new Error(
        `cached index at ${folder} belongs to different bytes; delete that folder to rebuild`,
      )
    }
    // Same bytes, possibly a new path (a renamed or moved file). The manifest
    // records where the frames were decoded from, so it must follow the file.
    if (resolve(manifest.video_path) !== absolute) {
      manifest.video_path = absolute
      await writeFile(manifestPath, JSON.stringify(manifest, null, 1), 'utf8')
    }
    return {
      manifest,
      catalog: await readJson(manifest.catalog_path),
      wave: await readJson(manifest.wave_path),
      reused: true,
    }
  }

  await mkdir(folder, { recursive: true })
  const argv = ['-B', '-X', 'utf8', INDEXER, '--video', absolute, '--out', folder]
  if (Number.isFinite(fps) && fps > 0) argv.push('--fps', String(fps))
  if (limit > 0) argv.push('--limit', String(limit))
  await run(resolvePython(config), argv, signal, HERE)
  const manifest = await readJson(manifestPath)
  return {
    manifest,
    catalog: await readJson(manifest.catalog_path),
    wave: await readJson(manifest.wave_path),
    reused: false,
  }
}

/**
 * Before the first keyframe is delivered, ask for the viewing purpose in these
 * exact words. The same sentence leads the refusal, so the agent reads it as the
 * reason delivery is withheld rather than as decoration.
 */
/**
 * A purpose belongs to one observation task: session + video + task. Delivering
 * the first image of a task requires that task's purpose, so a new task asks
 * again even on the same video. Watch history belongs to session + video only,
 * so switching tasks never erases what was already seen.
 */
const PURPOSE_QUESTION = '请回答你观看视频的目的，这会影响后续的工具调用'

async function purposeFor(binding, taskId) {
  if (binding.goal != null
    && binding.taskId === taskId
    && String(binding.goal).trim().length > 0) return binding.goal
  const records = await binding.ledger.read()
  const match = [...records].reverse().find(
    (record) => record.event === 'goal'
      && record.video_sha256 === binding.manifest.video_sha256
      && record.task_id === taskId,
  )
  return match === undefined ? undefined : match.goal
}

async function requirePurpose(binding) {
  const goal = await purposeFor(binding, binding.taskId)
  if (goal !== undefined && String(goal).trim().length > 0) {
    binding.goal = goal
    return
  }
  throw new Error(
    `${PURPOSE_QUESTION}\n`
    + (binding.taskId == null
      ? 'No observation task is open for this video. Call observation_goal with the purpose to '
        + 'open one, then retry.'
      : `Task ${binding.taskId} has no recorded purpose yet. Call observation_goal with the `
        + 'purpose, then retry.')
    + ' The purpose decides which frames later calls should serve. Recording a new one opens a '
    + 'new task; frames already watched stay on record.',
  )
}

/**
 * Stable evidence identifier for one delivered frame.
 *
 * Deterministic in (session, video, exact pts, time base, PNG hash), so the same
 * frame re-cited later resolves to the same id without re-delivering it, while a
 * different session or a different video can never collide with it.
 */
function evidenceId(sessionId, binding, frame) {
  const payload = [
    String(sessionId ?? 'unbound'),
    binding.manifest.video_sha256,
    String(frame.pts),
    binding.manifest.time_base,
    String(frame.png_sha256),
  ].join('|')
  return `ev-${createHash('sha256').update(payload).digest('hex').slice(0, 12)}`
}

/**
 * Persist one PNG as a durable attachment and expose it as an image block.
 * Native pixels are exported on demand from the source video, then cached, so
 * the index never pays for frames nobody reads.
 */
async function attachPng(ctx, buffer, displayName) {
  const attachments = ctx.get('attachments')
  if (attachments === undefined) {
    throw new Error('no attachment service is mounted, so native frames cannot be delivered')
  }
  const ref = await attachments.saveImage({ data: buffer, mediaType: 'image/png', name: displayName })
  return ref
}

/** Every source frame time, so a requested second maps to one exact frame. */
async function sourceTimeline(binding, config, signal) {
  if (binding.timeline !== undefined) return binding.timeline
  // ffprobe sits beside ffmpeg; derive it from the configured ffmpeg path so the
  // plugin exposes one binary setting rather than two.
  const ffmpeg = config?.ffmpegPath || ''
  const ffprobe = ffmpeg.length > 0
    ? join(dirname(ffmpeg), process.platform === 'win32' ? 'ffprobe.exe' : 'ffprobe')
    : 'ffprobe'
  const { stdout } = await run(ffprobe, [
    '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'frame=pts,pts_time', '-of', 'json',
    binding.manifest.video_path,
  ], signal, HERE)
  const parsed = JSON.parse(stdout)
  const entries = (parsed.frames ?? [])
    .map((frame, index) => ({
      index,
      pts: Number(frame.pts),
      seconds: Number(frame.pts_time),
    }))
    .filter((entry) => Number.isFinite(entry.seconds))
  if (entries.length === 0) throw new Error('ffprobe returned no source frame times')
  binding.timeline = entries
  return entries
}

/** Nearest real source frame to a requested second. */
function nearestFrame(timeline, target) {
  let best = timeline[0]
  let bestGap = Math.abs(best.seconds - target)
  for (const entry of timeline) {
    const gap = Math.abs(entry.seconds - target)
    if (gap < bestGap) {
      best = entry
      bestGap = gap
    }
  }
  return best
}

/** Export one exact source frame at native resolution and cache the PNG. */
async function nativeFrame(binding, config, row, signal) {
  const folder = join(dirname(binding.manifest.catalog_path), 'native')
  const target = join(folder, `frame_${row.source_frame}.png`)
  if (existsSync(target)) return target
  await mkdir(folder, { recursive: true })
  const ffmpeg = config?.ffmpegPath || 'ffmpeg'
  // Select by frame index so the pixel comes from exactly the catalogued frame,
  // rather than from a time guess that may land on a neighbour.
  const filter = `select=eq(n\\,${row.source_frame})`
  await run(ffmpeg, [
    '-v', 'error', '-noautorotate',
    '-i', binding.manifest.video_path,
    '-vf', filter,
    '-fps_mode', 'passthrough', '-frames:v', '1', '-y', target,
  ], signal, folder)
  if (!existsSync(target)) {
    throw new Error(`source frame ${row.source_frame} could not be decoded from the original video`)
  }
  return target
}

const imageBlock = (ref) => ({
  type: 'image',
  attachment: {
    attachmentId: ref.attachmentId,
    mediaType: ref.mediaType,
    bytes: ref.bytes,
    width: ref.width,
    height: ref.height,
    ...(ref.name === undefined ? {} : { name: ref.name }),
  },
})

async function assertImageCapableRoute(ctx, exec) {
  const routed = exec.agent?.session?.requestHeader?.()?.config
  const provider = routed?.provider ?? exec.agent?.options?.provider
  const model = routed?.model ?? exec.agent?.options?.model
  const llm = ctx.get('llm')
  if (provider === undefined || model === undefined || llm === undefined) {
    throw new Error('cannot resolve the current model route, so image delivery is refused')
  }
  const active = await llm.resolveModelInfo(provider, model, exec.signal)
  if (!Array.isArray(active?.inputModalities) || !active.inputModalities.includes('image')) {
    throw new Error(
      `model "${model}" does not declare image input; switch to an image-capable model to receive frames`,
    )
  }
}

const imageArraySchema = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      attachmentId: { type: 'string' },
      mediaType: { type: 'string' },
      bytes: { type: 'integer' },
      width: { type: 'integer' },
      height: { type: 'integer' },
      name: { type: 'string' },
    },
    required: ['attachmentId', 'mediaType', 'bytes', 'width', 'height'],
  },
}

export function apply(ctx, config = {}) {
  ctx.effect(() => () => bindings.clear())

  // Archive on every plugin load. Cheap (content-hash deduplicated) and it means
  // an experiment's evidence survives an accidental delete of the live folder.
  ctx.effect(() => {
    let cancelled = false
    archiveLedgers(cacheRoot(config), 'plugin-load').catch(() => undefined)
    return () => { cancelled = true; void cancelled }
  })

  ctx.tools.register({
    name: 'video_open',
    description:
      'Index a local video into a reusable keyframe catalogue and bind it to this session. '
      + 'Re-opening the same file reuses the cached index. Later frame arguments must use the '
      + 'frame_id or pts values that this session catalogue tools returned.',
    parameters: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Absolute path to a local video file that ffmpeg can decode.',
        },
      },
      required: ['path'],
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          video_sha256: { type: 'string' },
          duration_seconds: { type: 'number' },
          source_frames: { type: 'integer' },
          sampled_frames: { type: 'integer' },
          catalog_frames: { type: 'integer' },
          width: { type: 'integer' },
          height: { type: 'integer' },
          time_base: { type: 'string' },
          reused_index: { type: 'boolean' },
          sampling: { type: 'string' },
          nominations_by_dimension: { type: 'string' },
        },
        required: ['video_sha256', 'duration_seconds', 'width', 'height', 'catalog_frames'],
      },
      render: (_args, value) => [{
        type: 'text',
        text: `Video bound: ${value.duration_seconds}s, ${value.width}x${value.height}, `
          + `${value.source_frames} source frames, ${value.catalog_frames} catalogued keyframes `
          + `from ${value.sampled_frames} sampled at ${value.sampling}. `
          + `Nominations ${value.nominations_by_dimension}. `
          + `Times are source PTS seconds (time_base ${value.time_base}). `
          + `${value.reused_index ? 'Reused the cached index.' : 'Built a new index.'}`,
      }],
    },
    async execute(args, exec) {
      if (typeof args?.path !== 'string' || args.path.trim().length === 0) {
        throw new Error('path must be a non-empty string')
      }
      const { manifest, catalog, wave, reused } = await indexVideo(config, args.path, exec.signal)
      const agentId = exec.agent?.id
      const sessionLedger = ledger(cacheRoot(config), agentId)
      // Archive whatever the previous sessions wrote before this one appends.
      await archiveLedgers(cacheRoot(config), `before-video-open:${manifest.video_sha256.slice(0, 8)}`)
        .catch(() => undefined)
      // Reopening a video restores its most recent task, so continuing the same
      // observation does not ask again. A *new* purpose still opens a new task.
      const history = await sessionLedger.read()
      const lastTask = [...history].reverse().find(
        (record) => record.event === 'goal' && record.video_sha256 === manifest.video_sha256,
      )
      await sessionLedger.append({
        at: new Date().toISOString(),
        event: 'video_open',
        video_sha256: manifest.video_sha256,
        path: manifest.video_path,
        seconds: seconds(manifest.end - manifest.start),
        catalog_frames: manifest.catalog_frame_count,
        sampling: manifest.sampling,
        reused_index: reused,
      })
      bindings.set(agentId, {
        manifest,
        catalog,
        wave,
        catalogByFrame: catalogIndex(catalog),
        ledger: sessionLedger,
        goal: lastTask?.goal ?? null,
        taskId: lastTask?.task_id ?? null,
      })
      const counts = Object.entries(manifest.nomination_counts)
        .map(([key, value]) => `${key}:${value}`)
        .join(' ')
      return {
        video_sha256: manifest.video_sha256,
        duration_seconds: seconds(manifest.end - manifest.start),
        source_frames: manifest.source_frame_count,
        sampled_frames: manifest.sampled_frame_count,
        catalog_frames: manifest.catalog_frame_count,
        width: manifest.probe.width,
        height: manifest.probe.height,
        time_base: manifest.time_base,
        reused_index: reused,
        sampling: `${manifest.sampling.fps} fps`
          + (manifest.sampling.downgrade === null || manifest.sampling.downgrade === undefined
            ? ', full density'
            : `, DOWNGRADED: ${manifest.sampling.downgrade}`),
        nominations_by_dimension: counts.length === 0 ? 'none' : counts,
      }
    },
  })

  ctx.tools.register({
    name: 'catalog_search',
    description:
      'Query catalogue metadata for the open video: frame identifiers, source times and '
      + 'nomination dimensions. Returns no images. Use it to decide which frames are worth '
      + 'reading; a nomination is a low-level image measurement, not a verified object or event.',
    parameters: {
      type: 'object',
      properties: {
        start: { type: 'number', description: 'Earliest source second. Defaults to the video start.' },
        end: { type: 'number', description: 'Latest source second. Defaults to the video end.' },
        mode: {
          type: 'string',
          enum: ['overview', 'earliest', 'latest'],
          description:
            'overview samples evenly across the range for a first look; earliest and latest page '
            + 'forward and backward from a cursor.',
        },
        dimension: {
          type: 'string',
          enum: ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'],
          description:
            'Keep only frames nominated by this dimension. D1 local region, D2 local state '
            + 'difference, D3 image relation, D4 viewpoint, D5 unknown or conflict, '
            + 'D6 temporal coverage.',
        },
        cursor: { type: 'integer', description: 'Zero-based page offset. Defaults to 0.' },
        page_size: { type: 'integer', description: 'Frames per page, 1 to 30. Defaults to 12.' },
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          resolved_range: { type: 'string' },
          clamped: { type: 'boolean' },
          matched: { type: 'integer' },
          next_cursor: { type: 'integer' },
          frames: { type: 'string' },
          note: { type: 'string' },
        },
        required: ['resolved_range', 'clamped', 'matched', 'frames', 'note'],
      },
      render: (_args, value) => [{
        type: 'text',
        text: `${value.matched} frame(s) in ${value.resolved_range}`
          + `${value.clamped ? ' (range clipped to the video)' : ''}.\n${value.frames}\n`
          + `${value.next_cursor === undefined ? '' : `next_cursor=${value.next_cursor}\n`}`
          + value.note,
      }],
    },
    async execute(args, exec) {
      const binding = bindingFor(exec)
      const { lo, hi, clamped } = selection(binding, args?.start, args?.end)
      const dimension = args?.dimension
      const mode = args?.mode ?? 'overview'
      const pageSize = Math.max(1, Math.min(30, args?.page_size ?? 12))
      const offset = Math.max(0, args?.cursor ?? 0)

      const inRange = binding.manifest.pool.filter(
        (row) => row.seconds >= lo && row.seconds <= hi,
      )
      const rows = dimension === undefined
        ? inRange
        : inRange.filter((row) => {
          if (!row.in_catalog) return false
          const index = binding.catalogByFrame.get(row.frame_id)
          return index !== undefined && index.dimensions.includes(dimension)
        })

      let picked
      let nextCursor
      if (mode === 'overview' && rows.length > pageSize) {
        picked = Array.from({ length: pageSize }, (_, slot) => (
          rows[Math.round((slot * (rows.length - 1)) / Math.max(1, pageSize - 1))]
        ))
      } else if (mode === 'latest') {
        picked = [...rows].reverse().slice(offset, offset + pageSize)
        nextCursor = offset + pageSize < rows.length ? offset + pageSize : undefined
      } else {
        picked = rows.slice(offset, offset + pageSize)
        nextCursor = offset + pageSize < rows.length ? offset + pageSize : undefined
      }

      const lines = picked.map((row) => {
        const item = frameItem(binding, row)
        const where = `${item.seconds}s pts=${item.pts} source_frame=${item.source_frame}`
        const what = item.in_catalog
          ? `[${item.dimensions.join(',')}] ${item.roles.slice(0, 3).join(' ')}`
          : 'sampled, not nominated'
        return `${item.frame_id} | ${where} | ${what}`
      })
      return {
        resolved_range: `${seconds(lo)}-${seconds(hi)}s`,
        clamped,
        matched: rows.length,
        ...(nextCursor === undefined ? {} : { next_cursor: nextCursor }),
        frames: lines.length === 0 ? '(no frames in this range)' : lines.join('\n'),
        note: 'Metadata only. Nomination dimensions and roles are local image measurements, '
          + 'not object identities, events or causes.',
      }
    },
  })

  ctx.tools.register({
    name: 'wave_query',
    description:
      'Query navigation cues derived from adjacent catalogued frames: where the measured counts '
      + 'rise, fall or plateau, where a relation signature changed, and which intervals were never '
      + 'observed. Returns no images and asserts no event. A cue is a reason to look at the '
      + 'original frames, never a finding.',
    parameters: {
      type: 'object',
      properties: {
        start: { type: 'number', description: 'Earliest source second. Defaults to the video start.' },
        end: { type: 'number', description: 'Latest source second. Defaults to the video end.' },
        signal: {
          type: 'string',
          enum: [
            'all', 'rise', 'fall', 'plateau',
            'equal_height_change', 'peak', 'gap', 'transition',
          ],
          description:
            'all returns ranked cues; rise and fall select count changes on either axis; plateau '
            + 'selects an unchanged relation count; equal_height_change selects a signature change '
            + 'at an unchanged count; peak selects a local maximum; gap selects unobserved intervals; '
            + 'transition is accepted for compatibility and currently yields no cues.',
        },
        cursor: { type: 'integer', description: 'Zero-based page offset. Defaults to 0.' },
        page_size: { type: 'integer', description: 'Cues per page, 1 to 30. Defaults to 12.' },
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          resolved_range: { type: 'string' },
          signal: { type: 'string' },
          matched: { type: 'integer' },
          next_cursor: { type: 'integer' },
          cues: { type: 'string' },
          unknown_intervals: { type: 'string' },
          note: { type: 'string' },
        },
        required: ['resolved_range', 'signal', 'matched', 'cues', 'unknown_intervals', 'note'],
      },
      render: (_args, value) => [{
        type: 'text',
        text: `${value.matched} cue(s) for signal "${value.signal}" in ${value.resolved_range}.\n`
          + `${value.cues}\nUnobserved intervals: ${value.unknown_intervals}\n`
          + `${value.next_cursor === undefined ? '' : `next_cursor=${value.next_cursor}\n`}`
          + value.note,
      }],
    },
    async execute(args, exec) {
      const binding = bindingFor(exec)
      const { lo, hi } = selection(binding, args?.start, args?.end)
      const signal = args?.signal ?? 'all'
      const pageSize = Math.max(1, Math.min(30, args?.page_size ?? 12))
      const offset = Math.max(0, args?.cursor ?? 0)

      const matches = (cue) => {
        const flags = cue.flags ?? []
        switch (signal) {
          case 'all': return true
          case 'rise': return flags.includes('I_rise') || flags.includes('S_rise')
          case 'fall': return flags.includes('I_fall') || flags.includes('S_fall')
          case 'plateau': return flags.includes('observed_S_plateau')
          case 'equal_height_change': return flags.includes('relation_signature_change_at_equal_count')
          case 'peak': return flags.some((flag) => flag.endsWith('_local_peak'))
          case 'gap': return flags.includes('unobserved_gap')
          case 'transition': return false
          default: throw new Error(`unrecognised signal: ${signal}`)
        }
      }

      const cues = (binding.wave.cues ?? [])
        .filter((cue) => cue.window[1] >= lo && cue.window[0] <= hi && matches(cue))
        .sort((a, b) => {
          // gap and peak cues first, then larger absolute deltas, then time order
          const weight = (cue) => (cue.flags.includes('unobserved_gap') ? 2 : 0)
            + (cue.flags.some((flag) => flag.endsWith('_local_peak')) ? 1 : 0)
            + (cue.flags.includes('relation_signature_change') ? 1 : 0)
          const byWeight = weight(b) - weight(a)
          if (byWeight !== 0) return byWeight
          const byDelta = Math.abs(b.delta_I) + Math.abs(b.delta_S)
            - (Math.abs(a.delta_I) + Math.abs(a.delta_S))
          return byDelta !== 0 ? byDelta : a.window[0] - b.window[0]
        })

      const page = cues.slice(offset, offset + pageSize)
      const lines = page.map((cue, index) => {
        const deltas = `dI=${cue.delta_I >= 0 ? '+' : ''}${cue.delta_I} `
          + `dS=${cue.delta_S >= 0 ? '+' : ''}${cue.delta_S}`
        const roots = (cue.root_ids ?? []).length
        return `${offset + index + 1}. ${seconds(cue.window[0])}-${seconds(cue.window[1])}s `
          + `${cue.flags.join(' ')} ${deltas} roots=${roots}`
      })
      const unknown = (binding.wave.unknown_intervals ?? [])
        .filter(([a, b]) => b >= lo && a <= hi)
        .map(([a, b]) => `${seconds(a)}-${seconds(b)}s`)

      await binding.ledger.append({
        at: new Date().toISOString(),
        event: 'wave_query',
        range: [lo, hi],
        signal,
        matched: cues.length,
        returned: page.length,
        images_delivered: 0,
      })

      return {
        resolved_range: `${seconds(lo)}-${seconds(hi)}s`,
        signal,
        matched: cues.length,
        ...(offset + pageSize < cues.length ? { next_cursor: offset + pageSize } : {}),
        cues: lines.length === 0 ? '(no cue for this signal in the range)' : lines.join('\n'),
        unknown_intervals: unknown.length === 0 ? 'none in range' : unknown.join(', '),
        note: 'Navigation only, zero images, and never an authorisation to read a frame: the '
          + 'purpose recorded in observation_goal decides that, not a cue. Counts are distinct '
          + 'region tokens (I) and distinct relation structures keyed by region and evidence '
          + 'family (S) over observations actually made; they are not object counts, '
          + 'contact, interaction or causality. Read the original frames to observe anything.',
      }
    },
  })

  // A0/A2 get no purpose prompt at all, so the tool itself must be absent.
  if (config?.purposeGate !== false) {
    ctx.tools.register({
      name: 'observation_goal',
      description:
        'Record the observation goal for this session, or read the current ledger. A goal states '
        + 'what must be verified about the video; every later frame read and resample is filed '
        + 'against it, so an answer can show which evidence served which question.',
      parameters: {
        type: 'object',
        properties: {
          goal: {
            type: 'string',
            description:
              'The observation goal to record. Omit to read the ledger instead of setting a goal.',
          },
        },
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            goal: { type: 'string' },
            recorded: { type: 'string' },
            note: { type: 'string' },
          },
          required: ['goal', 'recorded', 'note'],
        },
        render: (_args, value) => [{
          type: 'text',
          text: `Goal: ${value.goal || '(none set)'}\n${value.recorded}\n${value.note}`,
        }],
      },
      async execute(args, exec) {
        const binding = bindingFor(exec)
        const goal = typeof args?.goal === 'string' ? args.goal.trim() : ''

        if (goal.length === 0) {
          const records = await binding.ledger.read()
          const tasks = records.filter(
            (record) => record.event === 'goal'
              && record.video_sha256 === binding.manifest.video_sha256,
          )
          const current = await purposeFor(binding, binding.taskId)
          if (current !== undefined) binding.goal = current
          const reads = records.filter(
            (record) => (record.event === 'delivery_request' || record.event === 'source_resample')
              && record.video_sha256 === binding.manifest.video_sha256,
          )
          const images = reads.reduce((total, r) => total + (r.images_delivered ?? 0), 0)
          const lines = reads.slice(-8).map((record) => {
            const what = record.event === 'delivery_request'
              ? `${record.delivered} frame(s)`
              : `resample ${seconds(record.range[0])}-${seconds(record.range[1])}s`
            return `- ${record.at} [${record.task_id ?? 'no-task'}] ${what} — ${record.reason}`
          })
          return {
            goal: binding.goal ?? '',
            recorded: `${tasks.length} observation task(s) on this video; `
              + `current task ${binding.taskId ?? 'none'}. `
              + `${images} image(s) actually delivered for this video.`,
            note: lines.length === 0
              ? 'No evidence has been read for this video yet.'
              : `Most recent evidence reads:\n${lines.join('\n')}`,
          }
        }

        if (goal.length < 8) {
          throw new Error('state a goal specific enough to check against the video')
        }
        // Every recorded purpose opens a new task, so a new question re-arms the
        // delivery gate without touching the watch history of the video.
        const taskId = `task-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
        binding.goal = goal
        binding.taskId = taskId
        await binding.ledger.append({
          at: new Date().toISOString(),
          event: 'goal',
          video_sha256: binding.manifest.video_sha256,
          task_id: taskId,
          goal,
        })
        return {
          goal,
          recorded: `Opened observation task ${taskId}. `
            + 'The next image delivery in this task is now allowed.',
          note: 'A goal constrains what to verify; it does not license any claim the frames do not '
            + 'show. Recording another goal opens a further task; frames already watched stay counted.',
        }
      },
    })
  }

  // A2/A3 expose the unobserved axis; A0/A1 must not, or the factor is not isolated.
  if (config?.unobservedAxis !== false) {
    ctx.tools.register({
      name: 'observation_coverage',
      description:
        'Report which source frames were actually delivered to this agent, from the delivery '
        + 'receipts rather than from intent. Distinguishes three states: the background was scanned, '
        + 'the catalogue nominated it, and this agent actually watched it. Query by time range and '
        + 'by zoom level. It never reports a whole-video unviewed ratio and never demands full '
        + 'coverage.',
      parameters: {
        type: 'object',
        properties: {
          start: { type: 'number', description: 'Earliest source second. Defaults to the video start.' },
          end: { type: 'number', description: 'Latest source second. Defaults to the video end.' },
          zoom: {
            type: 'string',
            enum: ['frames', 'windows', 'summary'],
            description:
              'frames lists each delivered source PTS; windows groups deliveries into contiguous '
              + 'passes; summary reports counts only. Defaults to windows.',
          },
          window_seconds: {
            type: 'number',
            description:
              'Grouping width for the windows zoom, in seconds. Defaults to the sampling interval.',
          },
        },
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            range: { type: 'string' },
            goal: { type: 'string' },
            scanned: { type: 'integer' },
            nominated: { type: 'integer' },
            watched: { type: 'integer' },
            watched_detail: { type: 'string' },
            note: { type: 'string' },
          },
          required: ['range', 'scanned', 'nominated', 'watched', 'watched_detail', 'note'],
        },
        render: (_args, value) => [{
          type: 'text',
          text: `${value.range}: background scanned ${value.scanned}, catalogue nominated `
            + `${value.nominated}, this agent watched ${value.watched}.\n`
            + `${value.goal === undefined ? '' : `Current purpose: ${value.goal}\n`}`
            + `${value.watched_detail}\n${value.note}`,
        }],
      },
      async execute(args, exec) {
        const binding = bindingFor(exec)
        const { lo, hi } = selection(binding, args?.start, args?.end)
        const zoom = args?.zoom ?? 'windows'
        const interval = binding.manifest.sampling?.frame_interval_seconds
          ?? (1 / (binding.manifest.sampling?.fps ?? 5))
        const windowSize = Number.isFinite(Number(args?.window_seconds)) && Number(args?.window_seconds) > 0
          ? Number(args.window_seconds)
          : interval

        const records = await binding.ledger.read()
        // Coverage reports the video's watch history, which spans tasks; the
        // purpose shown alongside it is the current task's.
        const currentPurpose = await purposeFor(binding, binding.taskId)
        if (currentPurpose !== undefined) binding.goal = currentPurpose

        // Only real deliveries of *this* video count as watched: a request that
        // produced no image is not an observation, and frames delivered from a
        // different video in this session must never be attributed to this one.
        // Records written before the video hash was logged cannot be attributed,
        // so they are counted separately instead of being silently trusted.
        const currentVideo = binding.manifest.video_sha256
        const watched = []
        let unattributable = 0
        for (const record of records) {
          const attributable = record.video_sha256 !== undefined
          if (attributable && record.video_sha256 !== currentVideo) continue
          const isDelivery = record.event === 'delivery_request' || record.event === 'source_resample'
          if (!attributable && isDelivery) {
            unattributable += 1
            continue
          }
          if (record.event === 'delivery_request' && Array.isArray(record.frames)) {
            const ratio = record.requested > 0 ? (record.images_delivered ?? 0) / record.requested : 0
            for (const frame of record.frames) {
              if (ratio >= 1 && Number(frame.seconds) >= lo && Number(frame.seconds) <= hi) {
                watched.push({
                  seconds: Number(frame.seconds),
                  pts: frame.pts,
                  source_frame: frame.source_frame,
                  via: 'delivery_request',
                  at: record.at,
                  reason: record.reason,
                })
              }
            }
          }
          if (record.event === 'source_resample' && Array.isArray(record.frames)) {
            for (const frame of record.frames) {
              if (Number(frame.seconds) >= lo && Number(frame.seconds) <= hi) {
                watched.push({
                  seconds: Number(frame.seconds),
                  pts: frame.pts,
                  source_frame: frame.source_frame,
                  via: 'source_resample',
                  at: record.at,
                  reason: record.reason,
                })
              }
            }
          }
        }
        watched.sort((a, b) => a.seconds - b.seconds)

        const scanned = binding.manifest.pool
          .filter((row) => row.seconds >= lo && row.seconds <= hi).length
        const nominated = binding.manifest.pool
          .filter((row) => row.in_catalog && row.seconds >= lo && row.seconds <= hi).length

        let detail
        if (zoom === 'frames') {
          detail = watched.length === 0
            ? 'No frame from this range has been delivered to this agent.'
            : watched.map((frame) => (
              `- ${seconds(frame.seconds)}s pts=${frame.pts} source_frame=${frame.source_frame} `
              + `via ${frame.via}`
            )).join('\n')
        } else if (zoom === 'summary') {
          const viaCount = {}
          for (const frame of watched) viaCount[frame.via] = (viaCount[frame.via] ?? 0) + 1
          detail = watched.length === 0
            ? 'No frame from this range has been delivered to this agent.'
            : `Delivered by ${Object.entries(viaCount).map(([k, v]) => `${k}:${v}`).join(', ')}.`
        } else {
          const groups = []
          for (const frame of watched) {
            const last = groups[groups.length - 1]
            // Grouping follows the sampling interval, never the user's zoom
            // window: two frames one whole second apart were not viewed
            // continuously just because a wide window covers both.
            if (last !== undefined && frame.seconds - last.end <= interval + 1e-9) {
              last.end = frame.seconds
              last.count += 1
              last.via.add(frame.via)
            } else {
              groups.push({ start: frame.seconds, end: frame.seconds, count: 1, via: new Set([frame.via]) })
            }
          }
          detail = groups.length === 0
            ? 'No frame from this range has been delivered to this agent.'
            : groups.map((group) => {
              const gap = group.end - group.start
              const shape = group.count === 1
                ? 'single observation'
                : gap <= interval + 1e-9
                  ? 'contiguous'
                  : `${group.count} discrete observations, unobserved in between`
              return `- ${seconds(group.start)}-${seconds(group.end)}s: ${shape} `
                + `via ${[...group.via].join('+')}`
            }).join('\n')
        }

        await binding.ledger.append({
          at: new Date().toISOString(),
          event: 'observation_coverage',
          range: [lo, hi],
          zoom,
          watched: watched.length,
          images_delivered: 0,
        })

        return {
          range: `${seconds(lo)}-${seconds(hi)}s`,
          ...(binding.goal == null ? {} : { goal: binding.goal }),
          scanned,
          nominated,
          watched: watched.length,
          watched_detail: detail,
          note: 'Counts cover this range only. Background scanning reaches every sampled frame; '
            + 'a catalogue nomination adds an observation but not an image. Watched means an image '
            + 'was actually delivered for this video in this session. Unwatched frames are '
            + 'unobserved, not empty.'
            + (unattributable === 0
              ? ''
              : ` ${unattributable} older delivery record(s) lack a video hash and were not `
                + 'attributed to any video; treat them as unknown rather than as coverage.'),
        }
      },
    })
  }

  ctx.tools.register({
    name: 'evidence_list',
    description:
      'List the evidence ids this session actually holds for the open video, one per delivered '
      + 'frame. Cite these in important visual claims. Re-citing an id costs nothing and does not '
      + 're-deliver the image; only frames already delivered have an id.',
    parameters: {
      type: 'object',
      properties: {
        start: { type: 'number', description: 'Earliest source second. Defaults to the video start.' },
        end: { type: 'number', description: 'Latest source second. Defaults to the video end.' },
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          total: { type: 'integer' },
          listed: { type: 'integer' },
          rows: { type: 'string' },
          note: { type: 'string' },
        },
        required: ['total', 'listed', 'rows', 'note'],
      },
      render: (_args, value) => [{
        type: 'text',
        text: `${value.total} evidence id(s) delivered in this session`
          + `${value.listed === value.total ? '' : `, ${value.listed} in the requested range`}.\n`
          + `${value.rows}\n${value.note}`,
      }],
    },
    async execute(args, exec) {
      const binding = bindingFor(exec)
      const { lo, hi } = selection(binding, args?.start, args?.end)
      const records = await binding.ledger.read()
      const all = listEvidence(records, binding.manifest.video_sha256)
      const inRange = all.filter((row) => row.seconds >= lo && row.seconds <= hi)
      const rows = inRange.length === 0
        ? 'No evidence in this range has been delivered to this agent yet.'
        : inRange.map((row) => (
          `${row.evidence_id} | ${seconds(row.seconds)}s pts=${row.pts} `
          + `source_frame=${row.source_frame} via ${row.via}`
        )).join('\n')
      return {
        total: all.length,
        listed: inRange.length,
        rows,
        note: 'An evidence id proves that this exact frame was delivered to this session. '
          + 'It does not by itself prove that the frame shows what a claim says; that is a '
          + 'separate judgement. Frames never delivered have no id and cannot be cited.',
      }
    },
  })

  ctx.tools.register({
    name: 'external_source',
    description:
      'Record that a conclusion came from a computation outside the image tools (shell, ffmpeg, '
      + 'hashing, diffing), and state what it can and cannot support. Pixel measurement can '
      + 'establish hashes, frame indices and brightness; it cannot establish object action, '
      + 'contact, intent or causality. Claims about those still need a delivered image.',
    parameters: {
      type: 'object',
      properties: {
        tool: { type: 'string', description: 'Tool used, for example ops_bash.' },
        purpose: { type: 'string', description: 'What the computation was for.' },
        supports: {
          type: 'string',
          description: 'What it actually establishes, for example "frames 65-67 differ from frame 0".',
        },
      },
      required: ['tool', 'purpose', 'supports'],
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          recorded: { type: 'boolean' },
          note: { type: 'string' },
        },
        required: ['recorded', 'note'],
      },
      render: (_args, value) => [{ type: 'text', text: value.note }],
    },
    async execute(args, exec) {
      const binding = bindingFor(exec)
      const tool = String(args?.tool ?? '').trim()
      const purpose = String(args?.purpose ?? '').trim()
      const supports = String(args?.supports ?? '').trim()
      if (tool.length === 0 || purpose.length === 0 || supports.length === 0) {
        throw new Error('tool, purpose and supports are all required')
      }
      await binding.ledger.append({
        at: new Date().toISOString(),
        event: 'external_computation',
        video_sha256: binding.manifest.video_sha256,
        task_id: binding.taskId,
        tool,
        purpose,
        supports,
        command_kind: 'declared_by_agent',
        not_evidence_for: 'object action, contact, intent, causality',
      })
      return {
        recorded: true,
        note: `Recorded external computation via ${tool}. It is filed separately from image `
          + 'evidence and cannot be cited as an evidence_id. Declare any claim about action, '
          + 'contact, intent or cause against a delivered frame instead.',
      }
    },
  })

  ctx.tools.register({
    name: 'delivery_request',
    description:
      'Deliver exact original frames from the open video at native resolution. Identify frames by '
      + 'the frame_id or integer pts that catalog_search returned, and state the observable need '
      + 'the comparison serves. Each call returns the real PNG frames.',
    parameters: {
      type: 'object',
      properties: {
        frame_ids: {
          type: 'array',
          items: { type: 'string' },
          description: 'Frame identifiers from catalog_search. At most 8 per call.',
        },
        pts: {
          type: 'array',
          items: { type: 'integer' },
          description: 'Integer source PTS values from catalog_search. At most 8 per call.',
        },
        reason: {
          type: 'string',
          description:
            'What comparison or view this observation needs, for example "compare cup presence '
            + 'before and after the pan". A tool request alone is not an observable need.',
        },
      },
      required: ['reason'],
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          delivered: { type: 'integer' },
          envelope: { type: 'string' },
          images: imageArraySchema,
        },
        required: ['delivered', 'envelope', 'images'],
      },
      render: (_args, value) => {
        const blocks = [{ type: 'text', text: value.envelope }]
        for (const image of value.images) blocks.push(imageBlock(image))
        return blocks
      },
    },
    async execute(args, exec) {
      const binding = bindingFor(exec)
      const sessionId = exec.agent?.id
      const reason = String(args?.reason ?? '').trim()
      if (reason.length < 8) {
        throw new Error('state the observable need for these frames, not just a tool request')
      }
      await assertImageCapableRoute(ctx, exec)
      if (config?.purposeGate !== false) await requirePurpose(binding)
      const ids = Array.isArray(args.frame_ids) ? args.frame_ids : []
      const ptsList = Array.isArray(args.pts) ? args.pts : []
      if (ids.length === 0 && ptsList.length === 0) {
        throw new Error('pass frame_ids or pts values returned by catalog_search')
      }
      if (ids.length + ptsList.length > 8) {
        throw new Error(`at most 8 frames per call, received ${ids.length + ptsList.length}`)
      }

      const byId = new Map(binding.manifest.pool.map((row) => [row.frame_id, row]))
      const byPts = new Map(binding.manifest.pool.map((row) => [row.pts, row]))
      const timeline = await sourceTimeline(binding, config, exec.signal)
      const byTimelinePts = new Map(timeline.map((entry) => [entry.pts, entry]))
      const wanted = []
      const offIndex = []
      for (const id of ids) {
        const row = byId.get(id)
        if (row === undefined) throw new Error(`unknown frame_id: ${id}`)
        wanted.push(row)
      }
      for (const pts of ptsList) {
        const sampled = byPts.get(pts)
        if (sampled !== undefined) {
          wanted.push(sampled)
          continue
        }
        // Any real source PTS is readable even when the coarse index never
        // sampled it, so a flat wave or an unnominated frame cannot block a read.
        const entry = byTimelinePts.get(pts)
        if (entry === undefined) {
          throw new Error(
            `pts ${pts} is not a source frame of this video; pass an integer PTS from `
            + 'catalog_search or source_resample',
          )
        }
        offIndex.push(entry)
      }
      if (new Set(wanted.map((row) => row.frame_id)).size !== wanted.length) {
        throw new Error('each frame may appear once per call')
      }
      if (new Set(offIndex.map((entry) => entry.pts)).size !== offIndex.length) {
        throw new Error('each pts may appear once per call')
      }

      const catalogRows = new Map(binding.catalog.frames.map((row) => [row.frame_id, row]))

      const images = []
      const lines = []
      const delivered = []
      for (const row of wanted) {
        const catalogRow = catalogRows.get(row.frame_id)
        if (catalogRow === undefined) {
          // Sampled but never nominated: the original frame still exists, so
          // read it rather than refusing. Say plainly that it is off-index.
          const file = await nativeFrame(binding, config, row, exec.signal)
          const buffer = await readFile(file)
          const ref = await attachPng(ctx, buffer, `frame_${row.source_frame}.png`)
          const { width, height } = binding.manifest.probe
          if (ref.width !== width || ref.height !== height) {
            throw new Error(
              `frame ${row.frame_id} was normalized to ${ref.width}x${ref.height} instead of the `
              + `source ${width}x${height}; native delivery failed`,
            )
          }
          images.push(ref)
          delivered.push({
            evidence_id: evidenceId(sessionId, binding, {
              pts: row.pts,
              png_sha256: createHash('sha256').update(buffer).digest('hex'),
            }),
            frame_id: row.frame_id,
            seconds: seconds(row.seconds),
            pts: row.pts,
            source_frame: row.source_frame,
            in_catalog: false,
            png_sha256: createHash('sha256').update(buffer).digest('hex'),
          })
          lines.push(
            `${images.length}. ${seconds(row.seconds)}s pts=${row.pts} `
            + `source_frame=${row.source_frame} ${ref.width}x${ref.height} `
            + `evidence_id=${delivered[delivered.length - 1].evidence_id} `
            + '(off-index: sampled but never nominated)',
          )
          continue
        }
        const file = await nativeFrame(binding, config, row, exec.signal)
        const buffer = await readFile(file)
        const ref = await attachPng(ctx, buffer, `frame_${row.source_frame}.png`)
        const { width, height } = binding.manifest.probe
        if (ref.width !== width || ref.height !== height) {
          throw new Error(
            `frame ${row.frame_id} was normalized to ${ref.width}x${ref.height} instead of the `
            + `source ${width}x${height}; native delivery failed`,
          )
        }
        images.push(ref)
        const frameHash = createHash('sha256').update(buffer).digest('hex')
        delivered.push({
          evidence_id: evidenceId(sessionId, binding, { pts: row.pts, png_sha256: frameHash }),
          frame_id: row.frame_id,
          seconds: seconds(row.seconds),
          pts: row.pts,
          source_frame: row.source_frame,
          in_catalog: true,
          png_sha256: frameHash,
        })
        // Provenance the evidence contract requires: source file hash, source
        // frame index, exact PTS and time base, and the delivered PNG hash.
        lines.push(
          `${images.length}. ${seconds(row.seconds)}s pts=${row.pts} `
          + `time_base=${binding.manifest.time_base} source_frame=${row.source_frame} `
          + `${ref.width}x${ref.height} png_sha256=${frameHash.slice(0, 16)} `
          + `evidence_id=${delivered[delivered.length - 1].evidence_id}`,
        )
      }

      // Off-index reads are appended after the catalogued ones and recorded the
      // same way, so coverage counts them as genuinely watched frames.
      for (const entry of offIndex) {
        const pseudo = {
          source_frame: entry.index,
          pts: entry.pts,
          seconds: entry.seconds,
        }
        const file = await nativeFrame(binding, config, pseudo, exec.signal)
        const buffer = await readFile(file)
        const ref = await attachPng(ctx, buffer, `frame_${entry.index}.png`)
        const { width, height } = binding.manifest.probe
        if (ref.width !== width || ref.height !== height) {
          throw new Error(
            `pts ${entry.pts} was normalized to ${ref.width}x${ref.height} instead of the `
            + `source ${width}x${height}; native delivery failed`,
          )
        }
        images.push(ref)
        delivered.push({
          evidence_id: evidenceId(sessionId, binding, {
            pts: entry.pts,
            png_sha256: createHash('sha256').update(buffer).digest('hex'),
          }),
          seconds: seconds(entry.seconds),
          pts: entry.pts,
          source_frame: entry.index,
          in_catalog: false,
          png_sha256: createHash('sha256').update(buffer).digest('hex'),
        })
        lines.push(
          `${images.length}. ${seconds(entry.seconds)}s pts=${entry.pts} `
          + `source_frame=${entry.index} ${ref.width}x${ref.height} `
          + `evidence_id=${delivered[delivered.length - 1].evidence_id} `
          + '(direct source PTS, outside the coarse index)',
        )
      }

      const envelope = `Delivered ${images.length} original frame(s) at native `
        + `${binding.manifest.probe.width}x${binding.manifest.probe.height} `
        + `from source ${binding.manifest.video_sha256.slice(0, 16)}.\n`
        + `${lines.join('\n')}\nObservable need: ${reason}\n`
        + 'These are discrete samples; a gap between two frames is unobserved, not a transition.'
      await binding.ledger.append({
        at: new Date().toISOString(),
        event: 'delivery_request',
        video_sha256: binding.manifest.video_sha256,
        task_id: binding.taskId,
        reason,
        goal: binding.goal,
        requested: wanted.length + offIndex.length,
        delivered: images.length,
        frames: delivered,
        images_delivered: images.length,
        state: 'delivered',
      })
      return {
        delivered: images.length,
        envelope,
        images: images.map((ref) => ({
          attachmentId: ref.attachmentId,
          mediaType: ref.mediaType,
          bytes: ref.bytes,
          width: ref.width,
          height: ref.height,
          ...(ref.name === undefined ? {} : { name: ref.name }),
        })),
      }
    },
  })

  ctx.tools.register({
    name: 'source_resample',
    description:
      'Re-decode the original video over a chosen interval and deliver evenly spaced native frames. '
      + 'Use it where the index is too sparse or has no frame where the observation needs one. '
      + 'Frames come from the source file, never from earlier output.',
    parameters: {
      type: 'object',
      properties: {
        start: { type: 'number', description: 'Interval start in source seconds.' },
        end: { type: 'number', description: 'Interval end in source seconds; must be after start.' },
        count: { type: 'integer', description: 'Frames to return, 1 to 8. Defaults to 4.' },
        reason: { type: 'string', description: 'What this resample is meant to establish.' },
      },
      required: ['start', 'end', 'reason'],
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          delivered: { type: 'integer' },
          envelope: { type: 'string' },
          images: imageArraySchema,
        },
        required: ['delivered', 'envelope', 'images'],
      },
      render: (_args, value) => {
        const blocks = [{ type: 'text', text: value.envelope }]
        for (const image of value.images) blocks.push(imageBlock(image))
        return blocks
      },
    },
    async execute(args, exec) {
      const binding = bindingFor(exec)
      const reason = String(args?.reason ?? '').trim()
      if (reason.length < 8) throw new Error('state what this resample is meant to establish')
      const sessionId = exec.agent?.id
      await assertImageCapableRoute(ctx, exec)
      if (config?.purposeGate !== false) await requirePurpose(binding)

      if (!Number.isFinite(Number(args?.start)) || !Number.isFinite(Number(args?.end))) {
        throw new Error('start and end must be finite numbers of source seconds')
      }
      const { lo, hi } = selection(binding, args.start, args.end)
      if (!(hi > lo)) throw new Error('end must be greater than start inside the video')
      const count = Math.max(1, Math.min(8, args?.count ?? 4))

      const times = Array.from({ length: count }, (_, index) => (
        count === 1 ? (lo + hi) / 2 : lo + ((hi - lo) * index) / (count - 1)
      ))
      // Resolve each requested second to one exact source frame first, so the
      // delivered PTS is recorded rather than a seek guess.
      const timeline = await sourceTimeline(binding, config, exec.signal)
      const chosen = times.map((time) => nearestFrame(timeline, time))

      const folder = join(dirname(binding.manifest.catalog_path), 'resample')
      await mkdir(folder, { recursive: true })
      const ffmpeg = config?.ffmpegPath || 'ffmpeg'

      const images = []
      const lines = []
      const delivered = []
      for (let index = 0; index < chosen.length; index += 1) {
        const frame = chosen[index]
        const target = join(folder, `f_${frame.index}.png`)
        if (!existsSync(target)) {
          await run(ffmpeg, [
            '-v', 'error', '-noautorotate',
            '-i', binding.manifest.video_path,
            '-vf', `select=eq(n\\,${frame.index})`,
            '-fps_mode', 'passthrough', '-frames:v', '1', '-y', target,
          ], exec.signal, folder)
        }
        const buffer = await readFile(target)
        const ref = await attachPng(ctx, buffer, `resample_${frame.seconds.toFixed(3)}.png`)
        images.push(ref)
        const frameHash = createHash('sha256').update(buffer).digest('hex')
        delivered.push({
          evidence_id: evidenceId(sessionId, binding, { pts: frame.pts, png_sha256: frameHash }),
          requested_seconds: seconds(times[index]),
          source_frame: frame.index,
          pts: frame.pts,
          seconds: seconds(frame.seconds),
          png_sha256: frameHash,
        })
        lines.push(
          `${images.length}. ${seconds(frame.seconds)}s pts=${frame.pts} `
          + `source_frame=${frame.index} ${ref.width}x${ref.height} `
          + `evidence_id=${delivered[delivered.length - 1].evidence_id} `
          + `(requested ${seconds(times[index])}s)`,
        )
      }

      const envelope = `Resampled ${seconds(lo)}-${seconds(hi)}s of the original video: `
        + `${images.length} frame(s) at ${binding.manifest.probe.width}x`
        + `${binding.manifest.probe.height}.\n${lines.join('\n')}\nPurpose: ${reason}\n`
        + 'Each line gives the exact source frame delivered and the second that was requested; '
        + 'the interval between returned frames was not observed.'
      await binding.ledger.append({
        at: new Date().toISOString(),
        event: 'source_resample',
        video_sha256: binding.manifest.video_sha256,
        task_id: binding.taskId,
        reason,
        goal: binding.goal,
        range: [lo, hi],
        frames: delivered,
        delivered: images.length,
        images_delivered: images.length,
        state: 'delivered',
      })
      return {
        delivered: images.length,
        envelope,
        images: images.map((ref) => ({
          attachmentId: ref.attachmentId,
          mediaType: ref.mediaType,
          bytes: ref.bytes,
          width: ref.width,
          height: ref.height,
          ...(ref.name === undefined ? {} : { name: ref.name }),
        })),
      }
    },
  })
}
