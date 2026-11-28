"""A small final observation transfer probe after conceptual and operation checks."""
from concurrent.futures import ThreadPoolExecutor
from random import Random

from dacf_research.prepare_inputs import ROOT,digest,read,write
from dacf_round7.run import H,PROTOCOL,launch

if __name__=='__main__':
    assert read(H/'engineering-retry/result.json')['pass']
    assert all(read(H/'operations'/tag/'result.json')['stop']['kind']=='completed' for tag in ('interior-evidence','saved-local-result'))
    out=H/'autonomous';out.mkdir()
    path=(ROOT/PROTOCOL['source']).as_posix()
    task=(f'Observe the assigned source video {path}. Investigate a visibly recurring or similar change in a small part of the image. '
      'Locate and compare separate occurrences if supported, including what stays unclear. Decide where and how to inquire and when you have enough evidence. '
      'Return compact JSON with summary, facts (claim, kind, evidence_ids), and open_questions. Use citations to original evidence actually received. '
      'A scoped partial answer is fine if the observations do not establish recurrence.')
    conditions=['original','revised'];Random(PROTOCOL['seed']+19).shuffle(conditions)
    docs=H/'revised_surface.json'
    write(out/'assignment.json',{'task':task,'conditions':conditions,'docs_hash':digest(docs),
                                 'scope':'One genuine independent session per condition with identical task, no mandatory tool and same underlying A3.'})
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda c:launch(out/c,task,docs if c=='revised' else None,overview=False,watchdog=300),conditions))
