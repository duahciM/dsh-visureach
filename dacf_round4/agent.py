"""Run the real native DSH probe after a model-free tool-chain acceptance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from random import Random
import shutil
from time import monotonic, sleep

from dacf_research.prepare_inputs import ROOT, digest, read, write
from evidenweft_phase1.run_native import TrackedHarness
from plugin_integration_20261124.dsh_compare import native_artifacts, SYSTEM_WIDE_TOOLS
from p224_research.agent import parse_events, strip_private
from dacf_round4.offline import HERE, R3, verify

CASE='rolling'


def configuration(folder, arm, public):
    return {'arm':arm,'case':CASE,'project':str(ROOT),'authorizedSource':str(public/'source.mp4'),
            'catalogPlan':str(public/'catalogue.json'),'regionPlan':str(public/'regions.json'),
            'pythonPath':str(ROOT/'.venv/Scripts/python.exe'),'controlLog':str(folder/'calls.jsonl'),
            'refinementDir':str(folder/'refinements'),'cacheDir':str(folder/'cache'),
            'purposeGate':True,'full':False,'sampleFps':5}


def prepare():
    lock=read(HERE/'lock.json')
    verify(lock)
    c=lock['cases'][CASE]
    folder=HERE/'agent'
    public=folder/'public'
    public.mkdir(parents=True)
    source=public/'source.mp4'
    shutil.copyfile(ROOT/c['source'],source)
    raw=read(ROOT/c['index']/'manifest.json')
    assert digest(source)==raw['video_sha256']
    frame_list=[{'frame_id':r['frame_id'],'pts':r['pts'],'seconds':r['seconds'],'source_frame':r['source_frame']} for r in raw['pool'] if r['in_catalog']]
    write(public/'catalogue.json',{'source_sha256':raw['video_sha256'],'source_frames':raw['source_frame_count'],
                                  'start':raw['start'],'end':raw['end'],'frames':frame_list})
    history=read(R3/'run'/CASE/'tree-v2.json')
    stops=read(HERE/'offline'/CASE/'initial_stops.json')
    edges=read(R3/'run'/CASE/'region_graph.json')['edges']
    nodes=[]
    for key,node in history['nodes'].items():
        if key!='source' and '/' in key and not key.startswith(('E5-gray/','E5-edge/')):
            continue
        root=key.split('/')[1] if '/' in key else key
        policy={}
        for ch,item in stops.items():
            options=[o for o in item['candidates'] if o['root']==root]
            policy[ch]={'S0_initial_source_stop':item['s0_initial_stop'],
                        'S1_state':'local_gap_remaining' if any(o['local']['eligible'] for o in options) else 'no_eligible_local_gap',
                        'eligible_local_count':sum(o['local']['eligible'] for o in options)}
        item=node|{'stop':policy,'can_request_refinement':True}
        nodes.append(item)
    ids={n['id'] for n in nodes}
    for node in nodes:
        node['children']=[i for i in node['children'] if i in ids]
        for revision in node.get('refinements',[]):
            trace=revision['trace']
            p=revision['actual_pts']; a,b=trace['bounds']
            old=next((e['value'] for e in trace.get('original_edges',[]) if e['pts']==[a,b] and e['applicable']),None)
            new=[e['value'] for e in trace.get('revised_edges',[]) if e['pts'] in ([a,p],[p,b]) and e['applicable']]
            revision['path_increment']=sum(new)-old if old is not None and len(new)==2 else None
    write(public/'regions.json',{'source_sha256':raw['video_sha256'],'nodes':nodes,'edges':edges})
    guide=read(HERE/'protocol.json')
    order=['A0','A1','A2']
    Random(2026112704).shuffle(order)
    prompt=(f'Observe the assigned video at {source.as_posix()}. Describe the most important visible sequence, including useful references '
            'for where things are in the picture. Check directly from original images when useful. Keep gaps and unverified explanations distinct. '
            'Choose your own tools and stop when you have enough evidence for a concise response. '
            'Return a JSON object with facts (list of objects with claim, kind direct or inference, evidence_ids from actual original image receipts), '
            'open_questions (list of strings), and summary. Prefer a few well-supported facts.')
    write(folder/'protocol_lock.json',{'order':order,'prompt':prompt,'watchdog':guide['agent_watchdog'],
            'route':guide['agent_route'],'case':CASE,'public_hashes':{p.name:digest(p) for p in public.iterdir()},
            'tool_and_driver_hashes':{p.name:digest(p) for p in (HERE/'probe.mjs',HERE/'bridge.py',HERE/'agent.py')}})
    engineering=folder/'engineering'
    engineering.mkdir()
    (engineering/'refinements').mkdir()
    write(engineering/'config.json',configuration(engineering,'A2',public))


def trial(arm, settings, folder):
    public=folder/'public'
    out=folder/arm
    out.mkdir()
    for name in ('workspace','refinements'):
        (out/name).mkdir()
    config=configuration(out,arm,public)
    patch=out/'profile.yml'
    patch.write_text(SYSTEM_WIDE_TOOLS+'\n- '+json.dumps({'insert':[{'id':'amr-research','name':(HERE/'probe.mjs').as_uri(),'config':config}]})+'\n',encoding='utf-8')
    h=None
    events=[]
    start=monotonic()
    status={'arm':arm,'route':settings['route']}
    try:
        h=TrackedHarness(out,patch)
        status['session_id']=h.session
        h.result(h.send('initialize',settings['route']|{'cwd':str(out/'workspace')}),40)
        deadline=monotonic()+18
        while monotonic()<deadline:
            try:
                rows=[json.loads(l) for l in (out/'calls.jsonl').read_text(encoding='utf-8').splitlines() if l]
                commons=next((r['tools'] for r in rows if r.get('event')=='tool_surface'),[])
                extras=next((r['tools'] for r in rows if r.get('event')=='research_registered'),[])
                if len(commons)==6 and len(extras)=={'A0':0,'A1':1,'A2':2}[arm]:
                    status['registered']=commons+extras
                    break
            except (FileNotFoundError,ValueError):
                pass
            sleep(.15)
        if 'registered' not in status:
            raise RuntimeError('Native DSH tool registration not verified')
        status['stop']=h.turn(settings['prompt'],settings['watchdog'],events)
    except (RuntimeError,OSError,ValueError,TimeoutError) as error:
        status['stop']={'kind':'technical_failure','message':str(error)}
    finally:
        if h:
            h.close()
            private=out/'raw_runtime_private.jsonl'
            if private.exists(): private.unlink()
        result=parse_events(events,[])
        status.update(native_artifacts(events))
        status.update(provider=result['provider_context'],reasoning=result['declared_reasoning_effort_in_headers'],
                      usage=result['provider_usage_tokens'],answer=result['final_answer'],answer_text=result['final_answer_text'],
                      wall_seconds=round(monotonic()-start,4))
        write(out/'events.json',strip_private(events))
        write(out/'result.json',status)
    print(arm,status['stop'],len(status['images']),status['wall_seconds'],flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare:
        prepare()
        return
    folder=HERE/'agent'
    settings=read(folder/'protocol_lock.json')
    assert (folder/'engineering'/'acceptance.json').is_file()
    assert all(digest(folder/'public'/f)==h for f,h in settings['public_hashes'].items())
    assert all(digest(HERE/f)==h for f,h in settings['tool_and_driver_hashes'].items())
    with ThreadPoolExecutor(max_workers=3) as workers:
        list(workers.map(lambda set_:trial(set_,settings,folder), settings['order']))


if __name__=='__main__':
    main()
