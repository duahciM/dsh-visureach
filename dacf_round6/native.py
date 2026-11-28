from concurrent.futures import ThreadPoolExecutor
import argparse,json
from pathlib import Path
from random import Random
from time import monotonic,sleep

from dacf_research.prepare_inputs import ROOT,digest,read,write
from evidenweft_phase1.run_native import TrackedHarness
from plugin_integration_20261124.dsh_compare import SYSTEM_WIDE_TOOLS,native_artifacts
from p224_research.agent import parse_events,strip_private

H=Path(__file__).resolve().parent


def trial(out,case,condition,arm,prompt,entry,extras=None):
    out.mkdir(); workspace=out/'workspace'; workspace.mkdir()
    source=H/'independent/source.mp4' if case=='independent' else ROOT/'dacf_round5/inputs/fixture/source.mp4'
    cfg={'arm':arm,'automaticPolicy':'S0','authorizedSource':str(source),'planPath':str(source.with_name('plan.json')),
         'cacheDir':str(out/'cache'),'controlLog':str(out/'calls.jsonl'),'pythonPath':str(ROOT/'.venv/Scripts/python.exe'),
         'condition':condition,'guidePath':str(H/'guide.json'),'requestAudit':str(out/'request-audit.jsonl')} | (extras or {})
    (out/'profile.yml').write_text(SYSTEM_WIDE_TOOLS+'\n- '+json.dumps({'insert':[{'id':'round6-study','name':entry.as_uri(),'config':cfg}]})+'\n',encoding='utf-8')
    route={'provider':'deepseek-account','model':'deepseek-flash','reasoningEffort':'max','maxTokens':32768}
    h=None; events=[]; start=monotonic()
    status={'case':case,'condition':condition,'arm':arm,'source_sha256':digest(source),'prompt':prompt}
    try:
        h=TrackedHarness(out,out/'profile.yml')
        status['session']=h.session
        h.result(h.send('initialize',route|{'cwd':str(workspace)}),50)
        limit=monotonic()+20
        while not (out/'calls.jsonl').exists() and monotonic()<limit: sleep(.25)
        if not (out/'calls.jsonl').exists(): raise RuntimeError('Tool registration missing')
        status['stop']=h.turn(prompt,420,events)
    except (OSError,RuntimeError,ValueError,TimeoutError) as e:
        status['stop']={'kind':'technical_failure','message':str(e)}
    finally:
        if h:
            h.close()
            private=out/'raw_runtime_private.jsonl'
            if private.exists():private.unlink()
        stats=parse_events(events,[])
        status.update(native_artifacts(events))
        status.update(usage=stats['provider_usage_tokens'],answer=stats['final_answer'],answer_text=stats['final_answer_text'],
                      provider=stats['provider_context'],reasoning=stats['declared_reasoning_effort_in_headers'],seconds=round(monotonic()-start,3))
        write(out/'events.json',strip_private(events));write(out/'result.json',status)
    print(out.name,status['stop'],len(status['images']),status['seconds'],flush=True)


def task(case):
    source=H/'independent/source.mp4' if case=='independent' else ROOT/'dacf_round5/inputs/fixture/source.mp4'
    request=('Describe the important visible changes. Distinguish repeated or separate changes, and describe a bracket for their timing using actual source images. '
             'Keep visual gaps and uncertain explanations distinct. Choose how to investigate and stop when useful evidence is sufficient.'
             if case=='independent' else 'Describe the opening image layout and cite an original you actually received. A brief answer suffices.')
    return f'Observe the assigned video at {source.as_posix()}. '+request+' Return a concise JSON object with facts (claim, kind, evidence_ids), open_questions and summary.'


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prompt','discovery'],default='prompt');p.add_argument('--retry',action='store_true');a=p.parse_args()
    lock=read(H/('lock-retry.json' if a.retry or a.phase=='discovery' else 'lock.json')); assert all(digest(ROOT/path)==sha for path,sha in lock['files'].items())
    folder=H/(a.phase+'-retry' if a.retry else a.phase); folder.mkdir(); jobs=[]
    if a.phase=='prompt':
        entry=H/'measure.mjs'
        for case in ['independent','fixture']:
            for k in range(4):jobs.append((case,f'P{k}','A2'))
    else:
        assert (H/'prompt-retry/summary.json').is_file(),'Finish prompt study first'
        entry=H/'discovery.mjs'
        for case in ['independent','fixture']:
            for k in range(4):jobs.append((case,'P0',f'A{k}'))
    Random(62117).shuffle(jobs)
    write(folder/'design.json',{'jobs':jobs,'entry_sha256':digest(entry),'prompts':{case:task(case) for case in ('independent','fixture')},
                             'scope':'Independent sessions with difficult unfamiliar task and simple historical control, not within-task replication.'})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda j:trial(folder/('-'.join(j)),*j,task(j[0]),entry),jobs))


if __name__=='__main__':main()
