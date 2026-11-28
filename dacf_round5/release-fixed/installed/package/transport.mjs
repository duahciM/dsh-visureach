import { apply as study } from './entry.mjs'

export const name='visureach-amr-research'
export const inject=['tools']

export function apply(ctx,config) {
  const transport=new Proxy(ctx,{get(target,key){
    if(key==='tools') return {register(tool){
      target.tools.register({...tool,async execute(args,exec){
        const value=await tool.execute(args,exec)
        return JSON.parse(JSON.stringify(value))
      }})
    }}
    const value=Reflect.get(target,key,target)
    return typeof value==='function'?value.bind(target):value
  }})
  study(transport,config)
}
