"""Replay baseline-generated observations on original source, separately from conceptual trials."""
from concurrent.futures import ThreadPoolExecutor

from dacf_research.prepare_inputs import ROOT, read, write
from dacf_round7.run import H,PROTOCOL,launch

def jobs():
    path=(ROOT/PROTOCOL['source']).as_posix()
    start=f'Use only the assigned source {path}. Open it and choose a suitable observation goal if needed. '
    result=(' Return JSON with summary, facts (claim, kind, evidence_ids), open_questions, recovered_errors and tool_handoffs. '
            'Report actual received or computed evidence, units and remaining ambiguity. Make no claim about hidden object identity or physical continuity.')
    yield 'saved-local-result', start+(
      'Replay the observer proposal to recover omitted comparison detail without rerunning a comparison. Compare the known exact samples at source ticks 0 and 5120. '
      'Get their native originals so you can check what the comparison means. A colleague mistakenly supplied the original receipt instead of the calculation reference for retrieving detailed fields. '
      'Test that error once and recover using the real returned comparison reference. Find one exact saved field and its scope. '
      'If local ambiguity is relevant, you may derive a crop of an already received image. Use any legal tools and briefly explain each handoff.')+result
    yield 'interior-evidence', start+(
      'Replay the observer proposal for a branch with a useful interior evidence gap after automatic stopping. The source-tree root is available, but a copied child ID `unknown-r7-node` may be stale. '
      'Check the stale ID once with an ordinary node detail request rather than a guide request, then recover. Inspect a real saved interval around roughly three seconds and confirm its status. '
      'If you need an original inside a useful bracket, investigate it. For this engineering verification explicitly exercise the time-subdivision operation once when a legal interior exists, and check what came back. '
      'Choose any later verification method and distinguish original attachments, optional derived evidence, saved calculations and unknown gaps.')+result

if __name__=='__main__':
    area=H/'operations';area.mkdir()
    mapping=dict(jobs())
    write(area/'assignment.json',mapping)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda x: launch(area/x[0],x[1],H/'revised_surface.json',overview=False,watchdog=300),mapping.items()))
