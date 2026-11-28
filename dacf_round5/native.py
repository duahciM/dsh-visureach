"""Native DSH repetitions of a complete, shared offline tool surface."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from random import Random
from time import monotonic, sleep

from dacf_research.prepare_inputs import ROOT,digest,read,write
from evidenweft_phase1.run_native import TrackedHarness
from plugin_integration_20261124.dsh_compare import SYSTEM_WIDE_TOOLS,native_artifacts
from p224_research.agent import parse_events,strip_private
from dacf_round5.setup import HERE,verify


def trial(item,config,locked,root,entry):
    folder=root/f"{item['source']}-{item['repeat']}-{item['arm']}"
    folder.mkdir()
    workspace=folder/'workspace'; workspace.mkdir()
    source=HERE/'inputs'/item['source']/'source.mp4'
    plan=source.with_name('plan.json')
    assert digest(source)==locked['source_hashes'][item['source']]
    parameters={'arm':item['arm'],'authorizedSource':str(source),'planPath':str(plan),'automaticPolicy':'S0',
                'cacheDir':str(folder/'cache'),'controlLog':str(folder/'calls.jsonl'),
                'pythonPath':str(ROOT/'.venv/Scripts/python.exe'),'sampleFps':5,'purposeGate':True,
                'semanticAssets':str(ROOT/'p215_discovery_20261121'),'semanticPythonPaths':[str(ROOT/'if13_research_20261120/runtime_onnx')]}
    patch=folder/'profile.yml'
    patch.write_text(SYSTEM_WIDE_TOOLS+'\n- '+json.dumps({'insert':[{'id':'visureach-amr-research','name':entry.as_uri(),'config':parameters}]})+'\n',encoding='utf-8')
    prompt=locked['prompts'][item['source']]
    route=config['provider']
    result={'item':item,'prompt':prompt,'route':route,'source_sha256':digest(source)}
    run=None; events=[]; start=monotonic()
    try:
        run=TrackedHarness(folder,patch)
        result['session']=run.session
        run.result(run.send('initialize',route|{'cwd':str(workspace)}),45)
        until=monotonic()+20
        while monotonic()<until:
            try:
                rows=[json.loads(s) for s in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
                tool=next(row for row in rows if row['event']=='tool_surface')
                assert set(tool['tools'])==set(locked['common_tools'])
                result['tool_surface']=tool
                break
            except (FileNotFoundError,StopIteration,ValueError):
                sleep(.2)
        if 'tool_surface' not in result:
            raise RuntimeError('Tool registration missing in native DSH')
        result['stop']=run.turn(prompt,config['watchdog_seconds'],events)
    except (RuntimeError,OSError,ValueError,TimeoutError,AssertionError) as error:
        result['stop']={'kind':'technical_failure','reason':str(error)}
    finally:
        if run:
            run.close()
            private=folder/'raw_runtime_private.jsonl'
            if private.exists(): private.unlink()
        parsed=parse_events(events,[])
        result.update(native_artifacts(events))
        result.update(provider=parsed['provider_context'],reasoning=parsed['declared_reasoning_effort_in_headers'],usage=parsed['provider_usage_tokens'],
                      answer=parsed['final_answer'],answer_text=parsed['final_answer_text'],wall_seconds=round(monotonic()-start,6))
        write(folder/'events.json',strip_private(events)); write(folder/'result.json',result)
    print(item,result['stop'],len(result['images']),result['wall_seconds'],flush=True)


def main():
    lock=read(HERE/'lock.json');verify(lock)
    config=read(HERE/'protocol.json')
    acceptance=read(HERE/'engineering/result.json'); assert acceptance['status']=='passed'
    entry=HERE/'release/installed/package/entry.mjs'
    assert digest(entry)==digest(HERE/'entry.mjs')
    folder=HERE/'native';folder.mkdir()
    jobs=[]; prompts={}; hashes={}
    rng=Random(config['seed'])
    for case,plan in config['sources'].items():
        source=HERE/'inputs'/case/'source.mp4'
        hashes[case]=digest(source)
        prompts[case]=(f'Observe the assigned video at {source.as_posix()}. Explain what happens and which visual relationships change. '
          'Look for brief or local changes as relevant and ground important claims in actual source images you received. '
          'Keep unobserved intervals, identity, contact, and causal explanations appropriately qualified. Use any available tools on your own terms. '
          'Return one concise JSON object with facts (objects with claim, kind direct or inference, evidence_ids from actual image receipts), '
          'open_questions (strings), and summary.')
        for index in range(plan['repeats']):
            arms=list(config['arms']);rng.shuffle(arms)
            jobs.extend({'source':case,'repeat':index+1,'arm':arm} for arm in arms)
    locked={'jobs':jobs,'prompts':prompts,'source_hashes':hashes,'entry_sha256':digest(entry),'common_tools':acceptance['common_tools'],
            'predictor_hashes':{p.name:digest(p) for p in (HERE/'native.py',HERE/'entry.mjs',HERE/'protocol.json')}}
    write(folder/'lock.json',locked)
    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(lambda item:trial(item,config,locked,folder,entry),jobs))


if __name__=='__main__':main()
