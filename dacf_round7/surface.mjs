import { appendFileSync, readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { apply as research } from '../dacf_round6/discovery_corrected.mjs'

export const name = 'visureach-round7-offline'
export const inject = ['tools', 'systemPrompt', 'deepseekLlmApiExtensions']
const hash = value => createHash('sha256').update(JSON.stringify(value ?? null)).digest('hex')
const record = (path, value) => appendFileSync(path, JSON.stringify(value) + '\n', 'utf8')

function documentParameters(schema, definitions = {}) {
  return {...schema, properties: Object.fromEntries(Object.entries(schema.properties ?? {}).map(([key, value]) =>
    [key, definitions[key] ? {...value, description: definitions[key]} : value]))}
}

export function apply(ctx, config) {
  const docs = config.docsPath ? JSON.parse(readFileSync(config.docsPath, 'utf8')) : null
  const improved = Boolean(docs)
  const newExamples = improved ? docs.tools : null
  const map = new Map()
  const bridge = new Proxy(ctx, {get(target, key) {
    if (key === 'tools') return {register(tool) {
      map.set(tool.name, tool)
      const description = newExamples?.[tool.name]
      if (!description) return target.tools.register(tool)
      const parameters = documentParameters(tool.parameters, description.parameters)
      return target.tools.register({...tool, parameters, description: description.description,
        async execute(args, exec) {
          const result = await tool.execute(args, exec)
          if (tool.name !== 'observer_guide') return result
          if (args.tool && newExamples[args.tool] && result.status === 'unknown_tool') {
            return result
          }
          if (args.tool && newExamples[args.tool] && result.tool === args.tool) {
            const notes = newExamples[args.tool]
            return {...result, why: notes.description, result_example: notes.result_example,
              skip_when: notes.skip_when, ...(notes.example ? {example: notes.example} : {})}
          }
          return result.overview ? {...result, overview: docs.overview} : result
        }})
    }}
    const value = Reflect.get(target, key, target)
    return typeof value === 'function' ? value.bind(target) : value
  }})
  research(bridge, {...config, arm: 'A3'})
  if (docs && config.overviewInSystem) {
    ctx.systemPrompt.register('r7_observer_overview', {order: 5050, render: () => docs.overview})
  }
  let index = 0
  ctx.deepseekLlmApiExtensions.register('r7_provider_request_audit', {prepare(request) {
    if (request.purpose) return
    const {system, tools = [], messages = []} = request.body
    const results = messages.flatMap((m, message_index) => Array.isArray(m.content)
      ? m.content.filter(b => b.type === 'tool_result').map(({content, tool_use_id, is_error}) => ({message_index, tool_use_id, is_error,
          content: Array.isArray(content) ? content.map(c => c.type === 'text' ? {type: 'text', text: c.text} :
              {type: c.type, media_type: c.source?.media_type, encoded_length: c.source?.data?.length}) : content})) : [])
    record(config.requestAudit + '.effective', {index: ++index, at_ms: Date.now(), system_hash: hash(system), tool_hash: hash(tools),
      ...(index === 1 ? {system, tools} : {}), message_count: messages.length, tool_results: results})
  }})
}
