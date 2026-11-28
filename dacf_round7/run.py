"""Run sealed cognition/transfer tasks separately from scoring and source inspection."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import argparse
import json
from pathlib import Path
from random import Random
from time import monotonic, time

from dacf_research.prepare_inputs import ROOT, digest, read, write
from evidenweft_phase1.run_native import TrackedHarness
from p224_research.agent import parse_events, strip_private
from plugin_integration_20261124.dsh_compare import SYSTEM_WIDE_TOOLS, native_artifacts

H = Path(__file__).resolve().parent
PROTOCOL = read(H / 'protocol.json')


def prompts(form):
    intro = ('You are new to this video observation environment. Rely on the tool descriptions and schemas exposed in this session. '
             'You may read the available help with metadata-only guide calls. Do not open a video or use image, analysis, shell, or web calls; '
             'this is a conceptual assessment. Unknowns are acceptable. Answer independently in English with compact JSON; do not copy descriptions.\n')
    for i, group in enumerate(PROTOCOL['groups']):
        yield f'tools-{i+1}', intro + f'Cover exactly these tools: {", ".join(group)}.\n' + (
            'Return {"summary":"short", "tools":[{"tool":"exact name", "purpose":"observer difficulty and why", '
            '"contrast":"name a similar tool and a real difference", "supports":"what the result supports", "limit":"what remains unknown", '
            '"scenarios":[{"phenomenon":"what happens", "known":"what observer already has", "question":"unresolved", '
            '"fit":"reason and handoff/input", "result":"expected output", "supports":"justified claim", '
            '"unknown":"what cannot be concluded"}], "misuse":{"phenomenon":"...","known":"...","question":"...","fit":"why wrong",'
            '"result":"...","supports":"...","unknown":"..."}, "skip":{"phenomenon":"...","known":"...","question":"...",'
            '"fit":"other solution or evidence sufficient","result":"...","supports":"...","unknown":"..."}}]}. '
            'For each tool write exactly three distinct valid scenario records, a misuse and a skip. Change the type of observation problem, '
            'evidence available, or tool use from any shown examples. Do not rename only the objects or colors. Keep each value short but concrete.'
        )
    yield 'architecture', intro + ('Answer each numbered question with a short, concrete explanation. Return {"summary":"short",'
        '"answers":[{"number":1,"text":"..."}],"combinations":[{"question":"...","known":"...",'
        '"steps":[{"tool":"name","arguments":{},"handoff":"next input"}], "optional":"...", "recover":"...", "stop":"..."}]}.\n') + '\n'.join(
            f'{i}. {question}' for i, question in enumerate(PROTOCOL['architecture_questions'], 1))
    for i in range(2):
        bank = PROTOCOL['forms'][form][5*i:5*i+5]
        yield f'match-{i+1}', intro + ('Propose a solution for each item. Several methods may be sound. If no call is needed say so. '
            'Give legal tool and argument forms when useful. State which placeholders must come from earlier results. '
            'Return {"summary":"short","answers":[{"id":"item id","choice":"why and alternative",'
            '"calls":[{"tool":"name","arguments":{}}],"boundary":"scope or unknown","stop":"..."}]}.\n') + '\n'.join(
              f'{qid}. {question}' for qid, question, _ in bank)


def answer_of(events):
    text = ''
    for e in events:
        if e['type'] == 'assistant/message':
            items = e['data'].get('message', {}).get('content', [])
            stanza = '\n'.join(b['text'] for b in items if b.get('type') == 'text')
            if stanza:
                text = stanza
    for idx, c in enumerate(text):
        if c == '{':
            try:
                possible = json.JSONDecoder().raw_decode(text[idx:])[0]
            except json.JSONDecodeError:
                continue
            if isinstance(possible, dict) and {'summary','tools','answers','facts','scores'}.intersection(possible):
                return text, possible
    return text, None


def launch(folder, prompt, docs=None, source=None, plain=False, overview=True, watchdog=None):
    folder.mkdir(parents=True)
    work = folder / 'workspace'
    work.mkdir()
    out = folder/'profile.yml'
    if plain:
        out.write_text(SYSTEM_WIDE_TOOLS, encoding='utf8')
    else:
        source = source or ROOT / PROTOCOL['source']
        cfg = {'arm':'A3','automaticPolicy':'S0','authorizedSource':str(source),'planPath':str(source.with_name('plan.json')),
               'cacheDir':str(folder/'cache'),'controlLog':str(folder/'calls.jsonl'),'pythonPath':str(ROOT/'.venv/Scripts/python.exe'),
               'requestAudit':str(folder/'provider.jsonl'),'overviewInSystem':overview}
        if docs:
            cfg['docsPath'] = str(docs)
        stanza = {'insert':[{'id':'visureach-round7','name':(H/'surface.mjs').as_uri(),'config':cfg}]}
        out.write_text(SYSTEM_WIDE_TOOLS + '\n- ' + json.dumps(stanza)+'\n', encoding='utf8')
    h = None
    events = []
    start = monotonic()
    receipt = {'prompt':prompt,'started_epoch':time(),'requested_model':PROTOCOL['route']}
    try:
        h = TrackedHarness(folder, out)
        receipt['session'] = h.session
        h.result(h.send('initialize', PROTOCOL['route'] | {'cwd':str(work)}), 50)
        if not plain and not (folder/'provider.jsonl').parent.is_dir():
            raise RuntimeError('Missing request audit directory')
        receipt['stop'] = h.turn(prompt, watchdog or PROTOCOL['watchdog_seconds'], events)
    except (OSError, RuntimeError, ValueError, TimeoutError) as e:
        receipt['stop'] = {'kind':'technical_failure','message':str(e)}
    finally:
        if h:
            h.close()
            (folder/'raw_runtime_private.jsonl').unlink(missing_ok=True)
    raw, answer = answer_of(events)
    stats = parse_events(events, [])
    receipt.update(native_artifacts(events))
    receipt.update(answer=answer, answer_text=raw, usage=stats['provider_usage_tokens'],
                   seconds=round(monotonic()-start, 3), provider=stats['provider_context'],
                   reasoning=stats['declared_reasoning_effort_in_headers'])
    write(folder/'events.json', strip_private(events))
    write(folder/'result.json', receipt)
    print(folder.name, receipt['stop'], 'parsed='+str(answer is not None),
          'calls='+str(len(receipt['calls'])), 'sec='+str(receipt['seconds']), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['baseline','transfer','audit'])
    parser.add_argument('--docs', type=Path)
    parser.add_argument('--retry', metavar='NAME')
    parser.add_argument('--parallel', type=int, default=6)
    args = parser.parse_args()
    if args.phase == 'transfer' and (not args.docs or not (H/'MISUNDERSTANDING_REPORT.md').is_file()):
        raise ValueError('Finish and record original understanding before transfer.')
    if args.phase == 'audit':
        source = (ROOT/PROTOCOL['source']).as_posix()
        jobs = [('audit', f'Perform a read-only interface request audit on {source}. '
            'First call observer_guide with no args and again with tool=compare_object_views. '
            'Then open the given video and use catalog_search with page_size 14, wave_query with page_size 14, '
            'and amr_query with detail guide. Ask amr_query for invalid node ID missing-r7. '
            'Read a legal node afterward and acknowledge the tool errors and recoveries in brief JSON. This is not an observation trial.')]
    else:
        jobs = list(prompts('late' if args.phase == 'transfer' else 'early'))
    Random(PROTOCOL['seed']).shuffle(jobs)
    folder = H/(args.retry or args.phase)
    if not folder.is_relative_to(H) or '..' in folder.parts:
        raise ValueError('Run output must be in the study directory.')
    folder.mkdir()
    assignment = {digest(H/'protocol.json')[:5]+f'-{i:02}': {'task':k,'prompt':p} for i,(k,p) in enumerate(jobs,1)}
    write(folder/'assignment.json', assignment)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(lambda job: launch(folder/job[0], job[1]['prompt'], args.docs and args.docs.resolve()), assignment.items()))


if __name__ == '__main__':
    main()
