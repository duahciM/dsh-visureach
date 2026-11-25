import { createServer } from 'node:http'
import { createHash, randomUUID } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'

const [manifestFile, ledgerFile] = process.argv.slice(2).map((p) => resolve(p))
if (!manifestFile || !ledgerFile) throw new Error('Usage: node viewer.js <source manifest.json> <session ledger.jsonl>')
const manifest = JSON.parse(await readFile(manifestFile, 'utf8'))
const origin = manifest.video_sha256
const folder = dirname(manifestFile)
const key = randomUUID()
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex')

async function data() {
  const records = (await readFile(ledgerFile, 'utf8')).split('\n').flatMap((line) => {
    try { const row = JSON.parse(line); return row.video_sha256 === origin ? [row] : [] } catch { return [] }
  })
  const originals = new Map(), derived = []
  for (const row of records) {
    if (['delivery_request','source_resample'].includes(row.event)) for (const f of row.frames ?? []) {
      originals.set(f.evidence_id, {id:f.evidence_id,type:'Original',pts:f.pts,seconds:f.seconds,frame:f.source_frame,
        hash:f.png_sha256,in_catalog:f.in_catalog,origin:row.event})
    }
    if (row.event==='derived_image') derived.push({id:row.derived_id,type:'Crop',pts:row.pts,seconds:row.seconds,
      parent:row.parent_evidence_id, region:row.native_xywh, scale:row.scale, hash:row.attachment_sha256})
  }
  const items = [...originals.values(), ...derived].sort((a,b)=>a.seconds-b.seconds)
  return {source:basename(manifest.video_path),source_sha256:origin,time_base:manifest.time_base,start:manifest.start,end:manifest.end,
    catalogue:manifest.pool.filter((r)=>r.in_catalog).map(({pts,seconds})=>({pts,seconds})),items,originals}
}

const html = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video evidence</title><style>
body{background:#101820;color:#e8eef3;font:15px system-ui;margin:auto;padding:24px;max-width:1160px}h1{font-size:25px;margin:0 0 4px}p,small{color:#aebccb}button,input{font:inherit;color:inherit;background:#1b2937;border:1px solid #4a6279;border-radius:5px;padding:7px 10px}button{cursor:pointer}button:focus-visible{outline:2px solid #85d6da}#track{width:100%;margin:12px 0;accent-color:#75c4c6}#ruler{position:relative;background:#203344;height:25px;border-radius:4px;width:100%}.mark{position:absolute;width:2px;height:8px;background:#7da0bb;top:10px}.read{height:18px;top:3px;background:#77d1c9}#layout{display:grid;gap:16px;grid-template-columns:minmax(260px,2fr) minmax(270px,3fr);margin-top:18px}#entries{display:flex;flex-direction:column;align-items:stretch;gap:5px;max-height:60vh;overflow:auto}.item{text-align:left}.item[aria-pressed=true]{border-color:#77d1c9}img{max-width:100%;height:auto}code{overflow-wrap:anywhere}#detail{min-height:5em}#canvas{position:relative;display:inline-block;max-width:100%}#box{position:absolute;box-shadow:0 0 0 2px #f9ae60;pointer-events:none;display:none}@media(max-width:720px){#layout{grid-template-columns:1fr}} </style>
<h1>Video evidence</h1><p id="source"></p><label for="track">Source time</label><input id="track" type="range" step="0.001"><div id="ruler" aria-hidden="true"></div><p id="time"></p><div id="layout"><section><input id="find" placeholder="Filter by evidence ID or type" aria-label="Filter evidence"><div id="entries"></div></section><section><h2 id="heading">Select an evidence receipt</h2><div id="canvas"><img id="original" alt="Exact original source frame" hidden><span id="box"></span></div><p id="detail"></p><button id="parent" hidden>Open parent original</button></section></div>
<script>
const $=id=>document.getElementById(id);const track=$('track'),entries=$('entries'),image=$('original'),parent=$('parent'),box=$('box');let current;
const f=(n)=>Number(n).toFixed(3)+' s';
async function main(){const d=await (await fetch('data')).json();$('source').textContent=d.source+' • PTS time base '+d.time_base+' • '+d.source_sha256;track.min=d.start;track.max=d.end;track.value=d.start;const map=new Map(d.items.map(r=>[r.id,r]));
function position(at){return 100*(at-d.start)/Math.max(.001,d.end-d.start)+'%'}
for(const [points,klass] of [[d.catalogue,'mark'],[d.items.filter(r=>r.type==='Original'),'mark read']])for(const r of points){const mark=document.createElement('span');mark.className=klass;mark.style.left=position(r.seconds);$('ruler').append(mark)}
function select(id){const r=map.get(id);if(!r)return;current=id;track.value=r.seconds;const original=r.parent?map.get(r.parent):r;image.hidden=!original;box.style.display='none';if(original){image.src='image/'+encodeURIComponent(original.id);image.onload=()=>{if(current!==id||!r.region)return;const [x,y,w,h]=r.region;Object.assign(box.style,{display:'block',left:100*x/image.naturalWidth+'%',top:100*y/image.naturalHeight+'%',width:100*w/image.naturalWidth+'%',height:100*h/image.naturalHeight+'%'})}}
$('heading').textContent=r.parent?'Crop parent and source area':'Original source frame'; $('detail').textContent=r.id+' • '+f(r.seconds)+' • PTS '+r.pts+' • SHA-256 '+r.hash+(r.parent?' • Parent '+r.parent+' • ROI '+r.region.join(', ')+' • '+r.scale+'× derived receipt':'');parent.hidden=!r.parent;parent.onclick=()=>select(r.parent);refresh();}
function refresh(){const q=$('find').value.trim().toLowerCase();const nearest=Number(track.value);$('time').textContent=f(nearest)+' • Blue marks show catalogue points. Teal marks show delivered originals.';const rows=d.items.filter(r=>(r.id+' '+r.type).toLowerCase().includes(q));rows.sort((a,b)=>Math.abs(a.seconds-nearest)-Math.abs(b.seconds-nearest));entries.replaceChildren(...rows.map(r=>{const b=document.createElement('button');b.className='item';b.textContent=r.type+' · '+f(r.seconds)+' · PTS '+r.pts+' · '+r.id;b.setAttribute('aria-pressed',String(r.id===current));b.onclick=()=>select(r.id);return b}));}
track.oninput=refresh;$('find').oninput=refresh;refresh();}main().catch(e=>{$('source').textContent='Unable to load evidence: '+e.message});
</script></html>`

const server = createServer(async (req,res)=>{
  const path = new URL(req.url,'http://127.0.0.1').pathname
  const reply = (status,body,mime='text/plain') => {res.writeHead(status,{'Content-Type':mime,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(body)}
  if (req.method!=='GET' || !path.startsWith(`/${key}/`)) return reply(404,'Not found')
  const relative = path.slice(key.length+2)
  try {
    if (!relative) return reply(200,html,'text/html; charset=utf-8')
    const resolved = await data()
    if (relative==='data') { const {originals,...visible}=resolved;return reply(200,JSON.stringify(visible),'application/json') }
    const id = /^image\/(ev-[a-f0-9]+)$/.exec(relative)?.[1]
    const original = resolved.originals.get(id)
    if (!original || !Number.isSafeInteger(original.frame)) return reply(404,'No received original at this ID')
    const png = await readFile(join(folder,'native',`frame_${original.frame}.png`))
    if (sha(png)!==original.hash) return reply(409,'Cached original no longer matches its receipt')
    return reply(200,png,'image/png')
  } catch (error) {return reply(500,error.code==='ENOENT'?'Source receipt file is unavailable':'Evidence view failed')}
})
server.listen(0,'127.0.0.1',()=>console.log(`http://127.0.0.1:${server.address().port}/${key}/`))
