import {strict as assert} from 'node:assert'
import {readFileSync,writeFileSync} from 'node:fs'
import {spawnSync} from 'node:child_process'
import {createHash} from 'node:crypto'
import {resolve} from 'node:path'
import {treeJournal} from './journal.mjs'

const db=resolve('dacf_round6/engineering/cache/amr-v1.sqlite')
const planPath=resolve('dacf_round5/inputs/fixture/plan.json')
const planBytes=readFileSync(planPath), plan=JSON.parse(planBytes)
const hash=createHash('sha256').update(planBytes).digest('hex')
const base=()=>new Map(plan.nodes.map(n=>[n.id,structuredClone(n)]))
const editor=treeJournal(db)
const restored=base(), scope='r6-durable-session'
assert.equal(editor.restore(scope,plan.sha256,hash,restored),2)
const old=JSON.parse(readFileSync(resolve('dacf_round6/engineering/result.json'),'utf8'))
assert.equal(restored.get(old.recovered.id).parent_id,old.recovered.parent_id)
assert.equal(editor.restore(scope,plan.sha256,'different_plan',base()),0)
assert.equal(editor.restore(scope,'different_source',hash,base()),0)
const child=spawnSync(process.execPath,['--input-type=module','-e',
  'const D=process.getBuiltinModule("node:sqlite").DatabaseSync;const d=new D(process.argv[1]);'
  +'d.exec("BEGIN IMMEDIATE");d.prepare("INSERT INTO records VALUES(?,?,?,?,?)").run(process.argv[2],process.argv[3],process.argv[4],3,"{}");process.exit(19)',
  db,scope,plan.sha256,hash],{encoding:'utf8',windowsHide:true})
assert.equal(child.status,19)
assert.equal(editor.restore(scope,plan.sha256,hash,base()),2)
writeFileSync(resolve('dacf_round6/engineering/recovery.json'),JSON.stringify({cold_process_version:2,cross_plan_denied:true,
 cross_source_denied:true,uncommitted_transaction_exited_mid_write:true,after_abrupt_exit_version:2},null,2),{flag:'wx'})
