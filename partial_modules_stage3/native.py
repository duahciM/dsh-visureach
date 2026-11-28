"""Independent max-effort DSH acceptance with the actual 0.6.0 offline tool surface."""
import json
from time import monotonic

from dacf_round7.run import PROTOCOL,TrackedHarness,parse_events,strip_private,native_artifacts,answer_of,SYSTEM_WIDE_TOOLS
from dacf_research.prepare_inputs import ROOT,digest,read,write
from partial_modules_stage3.material import H

if __name__=='__main__':
    out=H/'native'; out.mkdir()
    work=out/'workspace';work.mkdir()
    p=out/'profile.yml'
    mode={'insert':[{'id':'visureach-0.6.0-scoped-check','name':(ROOT/'video_observer_offline060/index.js').as_uri(),
       'config':{'cacheDir':str(out/'cache'),'pythonPath':str(ROOT/'.venv/Scripts/python.exe'),'integration':True}}]}
    p.write_text(SYSTEM_WIDE_TOOLS+'\n- '+json.dumps(mode)+'\n',encoding='utf8')
    check=read(H/'engineering/result.json')
    source=H/'inputs/roles/source.mkv'; a,b=check['pts']; ref=check['role_roi'][1]['box']
    obj=next(r['box'] for r in read(H/'references/sealed.json')['targets']['roles'] if r['pts']==b and r['episode']=='target_before')
    prompt=('Run a short real offline evidence check, then explain it. Open only '+source.as_posix()+'. Record your intended question if needed. '
      f'Read actual originals at native integer source ticks {a} and {b}, and compare these image regions at the second sample using a suitable '
      f'projected relation tool: {obj} and {ref} in native XYWH. Cite actual originals. Measure a projected relation if supported, keep hidden intervals and physical contact unclear. '
      'In your own words distinguish environment reference, independently useful observation object, local identity and optional cross-range identity. '
      'Give one novel application, one likely misuse, one reasonable no-use case, and explain why an identity failure does not invalidate all single-time geometry. '
      'Return short JSON with summary, facts, and answers.')
    write(out/'test_lock.json',{'source_sha256':digest(source),'pts':[a,b],'uses_published_tools_only':True,
           'scope':'Specified original tool workflow for verification, not automatic selection or AMR test','route':PROTOCOL['route']})
    started=monotonic();h=None;events=[]
    try:
        h=TrackedHarness(out,p)
        h.result(h.send('initialize',PROTOCOL['route']|{'cwd':str(work)}),50)
        stop=h.turn(prompt,240,events)
    except (OSError,RuntimeError,ValueError,TimeoutError) as e:
        stop={'kind':'technical_failure','message':str(e)}
    finally:
        if h:
            h.close()
            (out/'raw_runtime_private.jsonl').unlink(missing_ok=True)
    raw,answer=answer_of(events); counts=parse_events(events,[])
    receipt=native_artifacts(events)
    write(out/'events.json',strip_private(events))
    write(out/'result.json',receipt|{'stop':stop,'source_sha256':digest(source),'prompt':prompt,'answer':answer,'answer_text':raw,
         'usage':counts['provider_usage_tokens'],'seconds':round(monotonic()-started,3),'route':PROTOCOL['route']})
    print(stop,len(receipt['images']),len(receipt['calls']),flush=True)
