import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {readFile, writeFile} from 'node:fs/promises'
import {join, resolve} from 'node:path'
import {apply} from '../dacf_round6/persistent.mjs'

const out = resolve(process.argv[2])
const probe = JSON.parse(await readFile(join(out, 'request.json'), 'utf8'))
const tools = new Map()
const attachments = new Map()
const sha = data => createHash('sha256').update(data).digest('hex')
const services = {
  llm: {resolveModelInfo: async () => ({inputModalities: ['image']})},
  attachments: {saveImage: async ({data, mediaType}) => {
    const id = sha(data)
    attachments.set(id, data)
    return {attachmentId: 'sha256:' + id, mediaType, width: data.readUInt32BE(16),
      height: data.readUInt32BE(20), bytes: data.length}
  }}
}
apply({effect: f => f(), tools: {register: tool => tools.set(tool.name, tool)}, get: k => services[k]}, {
  arm: 'A2', automaticPolicy: 'S0', authorizedSource: resolve(probe.source),
  planPath: join(out, 'plan.json'), cacheDir: join(out, 'cache'),
  pythonPath: resolve('.venv/Scripts/python.exe'), controlLog: join(out, 'calls.jsonl')
})
const exec = {agent: {id: 'stage8-evidence-probe', session: {
  requestHeader: () => ({config: {provider: 'test', model: 'test'}})
}}}
const call = (tool, args) => tools.get(tool).execute(args, exec)
const start = performance.now()
await call('video_open', {path: resolve(probe.source)})
await call('observation_goal', {goal: 'Check original images inside one independently proposed comparison gap.'})
const result = await call('amr_refine', {
  node_id: probe.node, mode: 'time', pts: [probe.interior.pts], reason: 'Inspect an actual interior observation.'
})
assert.equal(result.images.length, 3)
assert.ok(result.summary.originals[probe.interior.pts])
const child = await call('amr_query', {node_id: result.summary.children[0].id, detail: 'proof'})
assert.ok(child.original_addresses.items.includes(probe.interior.pts))
for (const [id, data] of attachments) await writeFile(join(out, id + '.png'), data, {flag: 'wx'})
await writeFile(join(out, 'host_receipt.json'), JSON.stringify({
  sources: result.summary.originals, child: child.id, proof: child.proof,
  attachments: result.images.map(image => image.attachmentId), host_elapsed_ms: performance.now() - start
}, null, 2), {flag: 'wx'})
