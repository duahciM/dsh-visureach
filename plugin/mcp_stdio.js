import { createHash, randomUUID } from 'node:crypto'
import { readFile, realpath } from 'node:fs/promises'
import { isAbsolute, relative, resolve, sep } from 'node:path'
import { createInterface } from 'node:readline'
import { apply } from './index.js'

const configPath = process.argv[2]
if (!configPath) throw new Error('Usage: node mcp_stdio.js <configuration.json> <authorized-source-directory>...')
const config = JSON.parse(await readFile(configPath, 'utf8'))
const roots = await Promise.all(process.argv.slice(3).map((path) => realpath(resolve(path))))
if (!roots.length) throw new Error('The transport operator must specify authorized source directories.')
const tools = new Map(), attachments = new Map()
const runtime = {
  effect: (run) => run(), tools: { register: (tool) => tools.set(tool.name, tool) },
  get(name) {
    if (name === 'llm') return { resolveModelInfo: async () => ({ inputModalities: ['image'] }) }
    if (name === 'attachments') return { async saveImage({ data, mediaType, name }) {
      const attachmentId = 'sha256:' + createHash('sha256').update(data).digest('hex')
      attachments.set(attachmentId, data)
      return { attachmentId, mediaType, bytes: data.length, width: data.readUInt32BE(16), height: data.readUInt32BE(20), name }
    } }
  },
}
apply(runtime, config)
const exec = { agent: { id: 'mcp-' + randomUUID(), session: { requestHeader: () => ({ config: { provider: 'mcp', model: 'image-client' } }) } } }
const trustedInputPath = async (path) => {
  const actual = await realpath(resolve(path))
  if (!roots.some((root) => {
    const rel = relative(root, actual)
    return rel !== '..' && !rel.startsWith('..' + sep) && !isAbsolute(rel)
  })) throw new Error('The source is outside the transport operator’s configured directories.')
  return actual
}

async function dispatch(method, params) {
  if (method === 'initialize') return { protocolVersion: params.protocolVersion ?? '2025-11-25',
    capabilities: { tools: { listChanged: false } }, serverInfo: { name: 'video-observer', version: '0.4.0' } }
  if (method === 'ping') return {}
  if (method === 'tools/list') return { tools: [...tools.values()].map((tool) => ({ name: tool.name, description: tool.description, inputSchema: tool.parameters })) }
  if (method !== 'tools/call') return null
  const tool = tools.get(params.name)
  if (!tool) return { isError: true, content: [{ type: 'text', text: 'Unknown observation tool.' }] }
  try {
    const args = { ...params.arguments }
    if (tool.name === 'video_open') args.path = await trustedInputPath(args.path)
    const value = await tool.execute(args, exec)
    const blocks = tool.output?.render?.(args, value) ?? [{ type: 'text', text: JSON.stringify(value) }]
    return { content: blocks.map((block) => {
      if (block.type !== 'image') return block
      const ref = block.attachment
      const png = attachments.get(ref.attachmentId)
      if (!png) throw new Error('This call did not mint the exported image receipt.')
      return { type: 'image', data: png.toString('base64'), mimeType: ref.mediaType }
    }) }
  } catch (error) {
    return { isError: true, content: [{ type: 'text', text: String(error.message) }] }
  } finally { attachments.clear() }
}

for await (const line of createInterface({ input: process.stdin })) {
  let request
  try { request = JSON.parse(line) } catch { process.stdout.write('{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Invalid JSON"}}\n'); continue }
  if (!Object.hasOwn(request ?? {}, 'id')) continue
  const result = await dispatch(request.method, request.params ?? {})
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, ...(result ? { result } : { error: { code: -32601, message: 'Method not found' } }) }) + '\n')
}
