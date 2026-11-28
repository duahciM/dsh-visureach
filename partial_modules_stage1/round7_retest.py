"""Same-task Round 7 close-out on the frozen research plugin."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from random import Random

from dacf_research.prepare_inputs import digest, write
from dacf_round7.run import H as R7, PROTOCOL, ROOT, launch

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / PROTOCOL['source']

QUESTIONS = '''You are new to this video tool environment. Answer four unrelated cases in compact JSON. Use actual schemas, independent alternatives are allowed. Do not open a video. You may read interface help.
1. A catalogue frame has saved comparison neighbours. A colleague wants instead to reconcile a saved crop, state claim and text clue at a later source tick not in the catalogue. Which handles and tools are suitable for each, and why?
2. An observation has a seconds interval of 2.0 to 2.4 and a separate exact native tick 25600. What calls can obtain useful originals, and how are the units distinct? Show one argument object of each type.
3. A crop and its parent are cited as two independent originals. How would you recover source and scope, and what can a fresh integrity check still fail to establish?
4. A patch aligns under a camera pan but nearby background has a different fit. What output and applicability would you compare? What conclusions are still unavailable?
Return {"summary":"brief", "answers":[{"item":1,"choice":"...","args":{},"boundary":"..."}]} with all four items. Do not copy a tool description.'''

PRACTICE = f'''Open the assigned source {SOURCE.as_posix()}. This is a short tool operation check. Find one catalogue entry from the interval 2.0 to 2.4 source seconds. Use its returned frame ID to retrieve its comparison neighbours. Obtain the exact source tick 25600 and a different valid sample in that seconds interval using suitable original-image calls. Derive a small in-bounds crop using the first original, recover its parent and scope without counting it twice, and run one paired geometric comparison. Use the joint evidence index to recover at least the original or crop there. Return JSON with summary, facts (claim, kind, evidence_ids), and recovered_errors. Keep point/plane fits, physical identity and contact distinct. Do not explore unrelated events.'''

RUBRIC = {
    '1': 'Catalogue frame ID and its comparison/shared-root neighbours versus range/anchor/candidate cross-layer routes. Crop lineage uses exact citation and source root.',
    '2': 'Range uses source seconds plus fps/count. Exact delivery uses integer source ticks in pts array or full frame ID. No interpolation or conversion without time_base and actual sample.',
    '3': 'Recover the crop parent/type via evidence_reference. Shared root means dependent observations. claim_verify re-decodes eligible original or crop parent, does not prove crop pixels or semantics by itself.',
    '4': 'Local/excluded-context fit or global/local role plus applicability, overlap, holdout, and residual. Fits are scoped 2D evidence with competing appearance/view explanations, no identity/3D/contact.'
}

if __name__ == '__main__':
    out = HERE/'round7_retest'; out.mkdir()
    docs = R7/'revised_surface.json'
    jobs = [(phase, task) for phase in ('original','candidate') for task in ('blind','operation')]
    Random(1128).shuffle(jobs)
    mapping = {sha256(f'partial-v1-{i}'.encode()).hexdigest()[:7]: {'version':v,'task':k}
               for i,(v,k) in enumerate(jobs)}
    write(out/'lock.json', {'source':digest(SOURCE), 'guide':digest(docs),'live_tool_surface':digest(R7/'surface.mjs'),
         'route':PROTOCOL['route'],'rubric':RUBRIC,'blind_prompts':[QUESTIONS,PRACTICE],'assignments':mapping})
    for task in ('blind','operation'):
        batch=[(job,item) for job,item in mapping.items() if item['task']==task]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda x: launch(out/x[0], QUESTIONS if task=='blind' else PRACTICE,
                docs if x[1]['version']=='candidate' else None,overview=False,watchdog=260), batch))
