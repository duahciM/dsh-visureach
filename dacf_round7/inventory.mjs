import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { apply } from './surface.mjs'
import { GUIDE, OVERVIEW } from '../dacf_round5/release-fixed/installed/package/baseline/guide.js'

const dest = process.argv[2]
const dir = resolve('dacf_round6/independent')
const tools = new Map()
const ctx = {tools: {register(tool) {tools.set(tool.name, tool)}}, effect: fn => fn(), get: () => null,
  deepseekLlmApiExtensions: {register() {}}, systemPrompt: {register() {}}}
apply(ctx, {planPath: resolve(dir, 'plan.json'), authorizedSource: resolve(dir, 'source.mp4'),
  docsPath: process.argv[3], controlLog: resolve('dacf_round7/inventory-unused.jsonl')})
writeFileSync(dest, JSON.stringify({overview: OVERVIEW, tools: Object.fromEntries([...tools].map(([name, t]) => [name,
  {description: t.description, input: t.parameters, output: t.output?.schema, guide: GUIDE[name]}]))}, null, 2), {flag: 'wx'})
