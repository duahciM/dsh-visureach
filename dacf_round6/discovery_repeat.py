from concurrent.futures import ThreadPoolExecutor
from random import Random
from dacf_round6.native import H,task,trial
from dacf_research.prepare_inputs import write,digest

if __name__=='__main__':
    h=H/'discovery-corrected';h.mkdir()
    assert (H/'prompt-retry/summary.json').is_file() and (H/'discovery-engineering/result.json').is_file()
    plan=[('independent','P0','A3'),('fixture','P0','A3'),('independent','P1','A2'),('independent','P1','A3')]
    # P1 is a filename label for replicate 2 in this arm only. No guide is injected by discovery.
    Random(618).shuffle(plan)
    entry=H/'discovery_corrected.mjs'
    write(h/'design.json',{'jobs':plan,'entry_sha256':digest(entry),'title':'A3 renderer repair and additional difficult A2/A3 repetition',
       'scope':'Identical task, shared schema and source permissions. P1 here is an independent repetition identifier only.'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda j:trial(h/('-'.join(j)),*j,task(j[0]),entry),plan))
