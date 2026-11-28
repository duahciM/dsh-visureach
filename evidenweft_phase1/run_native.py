"""Launch balanced native DSH runs. No score asset is read by this process."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
from time import monotonic, time

from p224_research.agent import NativeHarness, parse_events, strip_private
from plugin_integration_20261124.dsh_compare import native_artifacts, SYSTEM_WIDE_TOOLS
from evidenweft_phase1.prepare import HERE, PUBLIC, ROOT, digest, read, verify, write


class TrackedHarness(NativeHarness):
    def turn(self, prompt, watchdog, events):
        token = self.send('session/prompt', {'sessionId': self.session, 'contentBlocks': [{'type':'text','text':prompt}]})
        stop = None
        deadline = monotonic() + watchdog
        while True:
            m = self.receive(deadline)
            if m.get('id') == token and m.get('error'):
                raise RuntimeError(str(m['error']))
            p = m.get('params', {})
            if p.get('sessionId') != self.session:
                continue
            if m.get('method') == 'session.event':
                e = p['event']
                events.append(e)
                if e['type'] == 'turn/end':
                    stop = e['data'].get('reason')
            if m.get('method') == 'session.status' and p.get('status') == 'idle' and stop:
                return stop


def prime_index(out, case, sha, source, protocol, full):
    index = HERE / 'inventory' / case / 'index'
    cache = out / 'cache'
    # Same frozen index cache for every arm. Only public catalogue projection
    # and enabled tools vary. Exact reads use the same index decoder and source.
    suffix = '-analysis' if full else ''
    target = cache / 'observer-index-v5' / f'{sha}-fps{protocol["coarse_fps"]}{suffix}'
    target.mkdir(parents=True)
    manifest = read(index / 'manifest.json')
    assert manifest['video_sha256'] == sha
    for field in ('catalog_path', 'wave_path', 'index_path', 'analysis_path'):
        name = Path(manifest[field]).name
        manifest[field] = str(target / name)
        if field == 'analysis_path':
            shutil.copytree(index / name, target / name)
        else:
            shutil.copyfile(index / name, target / name)
    manifest['video_path'] = str(source)
    write(target / 'manifest.json', manifest)
    # The production cache's analysis suffix follows the integration setting.
    return cache


def trial(job, protocol, attempt, dispatch=None):
    blind = job['blind_id']
    case, arm = job['case'], job['arm']
    out = HERE / 'sessions' / blind / f'attempt-{attempt}'
    if (out / 'result.json').exists():
        return read(out / 'result.json')
    out.mkdir(parents=True)
    workspace = out / 'workspace'
    workspace.mkdir()
    item = protocol['sources'][case]
    public = PUBLIC / case
    source = public / item['public_name']
    assert digest(source) == item['sha256']
    cache = prime_index(out, case, item['sha256'], source, protocol, arm == 'A2')
    if dispatch:
        dispatch(job)
    config = {'cacheDir': str(cache), 'pythonPath': str(ROOT / '.venv/Scripts/python.exe'),
              'sampleFps': protocol['coarse_fps'], 'purposeGate': True, 'unobservedAxis': True,
              'full': arm == 'A2', 'catalogPlan': str(public / ('uniform.json' if arm == 'A0' else 'multi.json')),
              'semanticAssets': str(ROOT / 'p215_discovery_20261121'),
              'semanticPythonPaths': [str(ROOT / 'if13_research_20261120/runtime_onnx')],
              'whisperModel': str(ROOT / 'plugin_integration_20261124/dependencies/whisper.bin'),
              'authorizedSource': str(source), 'controlLog': str(out / 'calls.jsonl')}
    plug = {'insert': [{'id': 'evidence-observer', 'name': (HERE / 'control_plugin.mjs').as_uri(), 'config': config}]}
    patch = out / 'profile.yml'
    patch.write_text(SYSTEM_WIDE_TOOLS + '\n- ' + json.dumps(plug) + '\n', encoding='utf-8')
    first = f'Observe this local video: `{source.as_posix()}`. {protocol["task"]}\n{protocol["answer"]}'
    follow = protocol['follow'] + '\n' + protocol['answer']
    receipt = {**job, 'attempt': attempt, 'source_sha256': item['sha256'], 'source': str(source), 'prompts': [first, follow],
               'started_epoch': time(), 'started': datetime.now().astimezone().isoformat(), 'rounds': []}
    h = None
    start = monotonic()
    try:
        h = TrackedHarness(out, patch)
        receipt['session_id'] = h.session
        h.result(h.send('initialize', protocol['route'] | {'cwd': str(workspace)}), 50)
        for index, prompt in enumerate((first, follow), 1):
            events = []
            beginning = monotonic()
            failure = None
            try:
                stop = h.turn(prompt, protocol['watchdog_seconds_per_turn'], events)
            except (TimeoutError, RuntimeError, BrokenPipeError) as e:
                failure = str(e)
                stop = {'kind': 'technical_error', 'message': failure}
            parsed = parse_events(events, [])
            block = native_artifacts(events)
            block.update(stop=stop, model=parsed['provider_context'], reasoning=parsed['declared_reasoning_effort_in_headers'],
                         usage=parsed['provider_usage_tokens'], answer=parsed['final_answer'],
                         answer_text=parsed['final_answer_text'], seconds=round(monotonic()-beginning, 3))
            receipt['rounds'].append(block)
            write(out / f'events-{index}.json', strip_private(events))
            write(out / f'checkpoint-{index}.json', receipt)
            if failure or stop.get('kind') != 'completed':
                break
        applicable = [r for turn in receipt['rounds'] for r in turn['model']]
        efforts = {e for turn in receipt['rounds'] for e in turn['reasoning']}
        receipt['model_verified'] = bool(applicable and efforts == {'max'}
            and all(r['provider'] == protocol['route']['provider'] and r['model'] == protocol['route']['model'] for r in applicable))
        complete = receipt['model_verified'] and len(receipt['rounds']) == 2 and all(
            isinstance(r['answer'], dict) and r['stop'].get('kind') == 'completed' for r in receipt['rounds'])
        receipt['status'] = 'completed' if complete else 'partial' if any(
            r['images'] or r['answer'] for r in receipt['rounds']) else 'technical_failure'
    except (TimeoutError, RuntimeError, ValueError, OSError) as e:
        receipt.update(status='technical_failure', error=str(e))
    finally:
        if h:
            h.close()
            raw = out / 'raw_runtime_private.jsonl'
            if raw.exists():
                with raw.open(encoding='utf-8') as f, (out/'transport-events.jsonl').open('w', encoding='utf-8') as d:
                    for line in f:
                        try:
                            d.write(json.dumps(strip_private(json.loads(line)), ensure_ascii=False) + '\n')
                        except ValueError:
                            continue
                raw.unlink()
        receipt['wall_seconds'] = round(monotonic() - start, 3)
        receipt['tool_counts'] = dict(Counter(c['name'] for r in receipt['rounds'] for c in r['calls']))
        write(out / 'result.json', receipt)
    print(blind, case, arm, attempt, receipt['status'], receipt['wall_seconds'], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    protocol = read(HERE / 'public_protocol.json')
    lock = verify(code_only=True)
    for rel, sig in read(HERE / 'prediction_inputs_lock.json').items():
        assert digest(HERE / rel) == sig, rel
    schedule = lock['schedule']
    if args.attempt == 2:
        schedule = [j for j in schedule if read(HERE / 'sessions' / j['blind_id'] / 'attempt-1/result.json')['status'] == 'technical_failure']
    # jobs enter the predetermined queue in order. Record actual native starts
    # after isolated cache preparation, since parallel preparation may reorder.
    from threading import Lock
    serial = Lock()
    def dispatch(job):
        with serial, (HERE / f'dispatch-{args.attempt}.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(job | {'start_epoch': time()}) + '\n')
    print('locked launch queue', schedule, flush=True)
    with ThreadPoolExecutor(max_workers=protocol['workers']) as worker:
        reports = list(worker.map(lambda job: trial(job, protocol, args.attempt, dispatch), schedule))
    print(Counter(r['status'] for r in reports), flush=True)


if __name__ == '__main__':
    main()
