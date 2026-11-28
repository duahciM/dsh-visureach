"""Transport-only retry. Preserve original lock, raw attempts, and research design."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from threading import Lock
from time import monotonic, sleep, time

from evidenweft_phase1 import run_native as runner
from evidenweft_phase1.prepare import HERE, PUBLIC, digest, read, verify, write
from plugin_integration_20261124.dsh_compare import SYSTEM_WIDE_TOOLS

ENTRY = (HERE/'control_plugin_retry.mjs').as_uri()
COMMON = {'video_open','catalog_search','observation_goal','delivery_request','source_resample','evidence_list'}


class InjectedHarness(runner.TrackedHarness):
    def __init__(self, folder, patch):
        self.folder = folder
        self.full = '"full": true' in patch.read_text(encoding='utf-8')
        runtime = folder/'activated-profile.yml'
        runtime.write_text(patch.read_text(encoding='utf-8').replace((HERE/'control_plugin.mjs').as_uri(), ENTRY), encoding='utf-8')
        super().__init__(folder,runtime)

    def result(self, message, timeout):
        response = super().result(message, timeout)
        deadline = monotonic() + 12
        path = self.folder/'calls.jsonl'
        while monotonic() < deadline:
            if path.exists():
                try:
                    data = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                except ValueError:
                    sleep(0.15)
                    continue
                tools = next((set(x['tools']) for x in data if x.get('event') == 'tool_surface'),set())
                if tools and (COMMON < tools if self.full else tools == COMMON):
                    return response
            sleep(0.15)
        raise RuntimeError('Native registration receipt missing or wrong; observation was not started.')


def preflight(protocol):
    root = HERE/'transport_preflight'
    root.mkdir(exist_ok=False)
    source = PUBLIC/'bench'/protocol['sources']['bench']['public_name']
    for full in (False,True):
        out = root/('complete' if full else 'common')
        out.mkdir()
        patch = out/'profile.yml'
        config = {'cacheDir':str(out/'cache'), 'full':full, 'catalogPlan':str(PUBLIC/'bench/multi.json'),
                  'authorizedSource':str(source),'controlLog':str(out/'calls.jsonl')}
        plugin = {'insert':[{'id':'observer-registry-probe','name':(HERE/'control_plugin.mjs').as_uri(),'config':config}]}
        patch.write_text(SYSTEM_WIDE_TOOLS + '\n- ' + json.dumps(plugin) + '\n',encoding='utf-8')
        harness = None
        result = {}
        try:
            harness = InjectedHarness(out,patch)
            receipt = harness.result(harness.send('initialize',protocol['route'] | {'cwd':str(out)}),50)
            tools = readlines(out/'calls.jsonl')[0]['tools']
            result = {'passed':True,'full':full,'tool_count':len(tools),'tools':tools,'initialized':bool(receipt)}
        except (RuntimeError,TimeoutError,OSError) as error:
            result = {'passed':False,'error':str(error)}
        finally:
            if harness:
                harness.close()
                raw = out/'raw_runtime_private.jsonl'
                if raw.exists():
                    raw.unlink()
            write(out/'result.json',result)
        assert result['passed'], result
        print('native registration verified',full,result['tool_count'],flush=True)


def readlines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    lock = verify(code_only=True)
    protocol = read(HERE/'public_protocol.json')
    for rel, sig in read(HERE/'prediction_inputs_lock.json').items():
        assert digest(HERE / rel) == sig, rel
    if args.preflight:
        preflight(protocol)
        return
    assert all(read(HERE/'transport_preflight'/label/'result.json')['passed'] for label in ('common','complete'))
    schedule = []
    first_attempts = []
    for r in lock['schedule']:
        folder = HERE/'sessions'/r['blind_id']/'attempt-1'
        result = read(folder/'result.json')
        text = (folder/'runtime.stderr.log').read_text(encoding='utf-8')
        if 'cannot get property "tools" without inject' in text:
            entry = r | {'recorded_status':result['status'],'effective_status':'technical_failure','reason':'plugin entry failed during tool dependency injection'}
            schedule.append(r)
        else:
            entry = r | {'recorded_status':result['status'],'effective_status':result['status']}
            if result['status'] == 'technical_failure':
                schedule.append(r)
        first_attempts.append(entry)
    write(HERE/'transport_amendment.json',{'sealed_at':datetime.now().astimezone().isoformat(),
         'initial_protocol_sha256':digest(HERE/'protocol_lock.json'),
         'changed':{'control_plugin_retry.mjs':digest(HERE/'control_plugin_retry.mjs'),'retry_native.py':digest(HERE/'retry_native.py')},
         'rationale':'The trial wrapper failed Cordis tool service injection. Add only the entry export, native registry preflight, and startup receipt guard. Same model, tasks, input bytes, catalogue plans, schedule, and scoring.',
         'first_attempts':first_attempts,'retry_schedule':schedule})
    runner.TrackedHarness = InjectedHarness
    mutex = Lock()
    def dispatch(job):
        with mutex, (HERE/'dispatch-2.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(job | {'start_epoch':time()})+'\n')
    with ThreadPoolExecutor(max_workers=protocol['workers']) as ex:
        rr = list(ex.map(lambda job:runner.trial(job,protocol,2,dispatch),schedule))
    print({k:sum(r['status'] == k for r in rr) for k in ('completed','partial','technical_failure')},flush=True)


if __name__ == '__main__':
    main()
