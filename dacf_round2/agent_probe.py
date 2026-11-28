"""Paired first-answer native DSH smoke. The source, prompt and basic tool rights match."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from random import Random
import shutil
from time import monotonic, sleep

from dacf_research.prepare_inputs import ROOT, digest, read, write
from evidenweft_phase1.run_native import TrackedHarness, prime_index
from plugin_integration_20261124.dsh_compare import SYSTEM_WIDE_TOOLS, native_artifacts
from p224_research.agent import parse_events, strip_private

HERE=Path(__file__).resolve().parent


def logrows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def trial(spec, source, cohort_source, public, frozen, setup, output_root=None):
    enable=spec=='region_metadata'
    out=(output_root or HERE/'agent_probe')/spec; out.mkdir()
    (out/'workspace').mkdir()
    cache=prime_index(out,'doorway',cohort_source['sha256'],source,frozen,False)
    plugin={'insert':[{'id':'observer-investigation-pilot','name':(HERE/'probe_entry.mjs').as_uri(),'config':{
      'authorizedSource':str(source),'catalogPlan':str(public/'catalogue.json'),'cacheDir':str(cache),
      'pythonPath':str(ROOT/'.venv/Scripts/python.exe'),'sampleFps':5,'full':False,'purposeGate':True,
      'controlLog':str(out/'calls.jsonl'),**({'regionPlan':str(public/'regions.json')} if enable else {})
    }}]}
    patch=out/'profile.yml'
    patch.write_text(SYSTEM_WIDE_TOOLS+'\n- '+json.dumps(plugin)+'\n',encoding='utf-8')
    status={'condition':spec,'source_sha256':cohort_source['sha256'],'model_route_requested':setup['route'],
      'prompt':setup['prompt']}
    h=None; events=[]; t=monotonic()
    try:
        h=TrackedHarness(out,patch)
        status['sdk_session']=h.session
        h.result(h.send('initialize',setup['route']|{'cwd':str(out/'workspace')}),40)
        deadline=monotonic()+12
        while monotonic()<deadline:
            try:
                registered=logrows(out/'calls.jsonl')
                common=next((r['tools'] for r in registered if r.get('event')=='tool_surface'),[])
                if set(common)==set(frozen['common_tools']) and (not enable or any(r.get('event')=='investigation_tool_registered' for r in registered)):
                    status['registered']=common+(['investigation_regions'] if enable else [])
                    break
            except (FileNotFoundError,ValueError):
                pass
            sleep(.2)
        if not status.get('registered'):
            raise RuntimeError('Expected native tool-registration receipts missing')
        status['stop']=h.turn(setup['prompt'], setup['watchdog_seconds'], events)
    except (OSError,RuntimeError,TimeoutError,ValueError) as e:
        status['stop']={'kind':'technical_failure','message':str(e)}
    finally:
        if h:
            h.close()
            raw=out/'raw_runtime_private.jsonl'
            if raw.is_file():
                raw.unlink()
        result=parse_events(events,[])
        status.update(native_artifacts(events))
        status.update(provider=result['provider_context'],reasoning=result['declared_reasoning_effort_in_headers'],
             usage=result['provider_usage_tokens'],answer=result['final_answer'],answer_text=result['final_answer_text'],
             wall_seconds=round(monotonic()-t,3))
        write(out/'events.json',strip_private(events))
        write(out/'result.json',status)
    print(spec,status['stop'],len(status['images']),status['wall_seconds'],flush=True)


def main():
    protocol=read(ROOT/'evidenweft_phase1/public_protocol.json')
    case=protocol['sources']['doorway']
    folder=HERE/'agent_probe';folder.mkdir()
    public=folder/'public';public.mkdir()
    source=public/'video.mp4'
    shutil.copyfile(ROOT/'dacf_research/inputs/doorway.mp4',source)
    assert digest(source)==case['sha256']
    shutil.copyfile(ROOT/'evidenweft_phase1/prediction_inputs/doorway/multi.json',public/'catalogue.json')
    result=read(HERE/'replay/doorway/regions.json')
    m=read(ROOT/'evidenweft_phase1/inventory/doorway/index/manifest.json')
    tb={r['pts']:r['seconds'] for r in m['pool']}
    regions=[]
    for row in result['regions']:
        if row.get('children'):continue
        lo,hi=row['bounds_pts']
        regions.append({'id':row['id'],'start':tb[lo],'end':tb[hi],
            'scope':row['kind'],'support_channels':row.get('distinction',{}).get('supported_by',[])})
    write(public/'regions.json',regions)
    plans=['basic','region_metadata']
    Random(2026112602).shuffle(plans)
    order={'seed':2026112602,'conditions':plans,'route':protocol['route'],'watchdog_seconds':360,
           'limitation':'equal source and tool rights with one independent run per condition; outcome is descriptive only',
           'prompt':f'Observe this local video: `{source.as_posix()}`. {protocol["task"]}\n{protocol["answer"]}',
           'sha256':{p.name:digest(p) for p in public.iterdir()}}
    write(folder/'lock.json',order)
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(lambda cond:trial(cond,source,case,public,protocol,order),plans))

if __name__=='__main__':
    main()
