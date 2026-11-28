import { mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
const {DatabaseSync}=process.getBuiltinModule('node:sqlite')??{}

export function treeJournal(file){
  if(!DatabaseSync) throw new Error('AMR restoration requires Node SQLite.')
  mkdirSync(dirname(file),{recursive:true})
  const use=fn=>{
    const db=new DatabaseSync(file)
    try{
      db.exec('PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS records(scope TEXT, source TEXT, plan TEXT, version INTEGER, record TEXT, PRIMARY KEY(scope,source,plan,version));')
      return fn(db)
    }finally{db.close()}
  }
  function replay(db,scope,source,plan,map){
    let latest=0
    for(const r of db.prepare('SELECT version,record FROM records WHERE scope=? AND source=? AND plan=? ORDER BY version').all(scope,source,plan)){
      if(r.version!==latest+1) throw new Error('AMR journal has a revision gap.')
      const step=JSON.parse(r.record)
      const parent=map.get(step.parent)
      if(!parent||step.children.some(n=>n.parent_id!==parent.id||map.has(n.id))) throw new Error('AMR journal has invalid lineage.')
      for(const child of step.children){
        map.set(child.id,child)
        parent.children.push(child.id)
      }
      parent.calculation_refs=[...new Set([...(parent.calculation_refs??[]),...step.refs])]
      latest=r.version
    }
    return latest
  }
  return {
    restore(scope,source,plan,base){ return use(db=>replay(db,scope,source,plan,base)) },
    add(scope,source,plan,record){
      return use(db=>{
        db.exec('BEGIN IMMEDIATE')
        try{
          const row=db.prepare('SELECT coalesce(max(version),0) AS latest FROM records WHERE scope=? AND source=? AND plan=?').get(scope,source,plan)
          const version=row.latest+1
          db.prepare('INSERT INTO records VALUES(?,?,?,?,?)').run(scope,source,plan,version,JSON.stringify(record))
          db.exec('COMMIT')
          return version
        }catch(e){db.exec('ROLLBACK');throw e}
      })
    }
  }
}
