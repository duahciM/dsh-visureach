"""Fresh transfer sessions. The overview remains available on demand via observer_guide."""
from concurrent.futures import ThreadPoolExecutor
from random import Random

from dacf_research.prepare_inputs import digest,write
from dacf_round7.run import H,PROTOCOL,launch,prompts

if __name__=='__main__':
    assert (H/'MISUNDERSTANDING_REPORT.md').is_file()
    docs=H/'revised_surface.json'
    area=H/'transfer'
    area.mkdir()
    jobs=list(prompts('late'))
    Random(PROTOCOL['seed']+1).shuffle(jobs)
    assignment={f'{digest(docs)[:5]}-{i:02}':{'task':k,'prompt':p} for i,(k,p) in enumerate(jobs,1)}
    write(area/'assignment.json',assignment)
    write(area/'guide_lock.json',{'hash':digest(docs),'overview':'on_demand_existing_observer_guide','entry':digest(H/'surface.mjs')})
    with ThreadPoolExecutor(max_workers=7) as pool:
        list(pool.map(lambda job: launch(area/job[0],job[1]['prompt'], docs, overview=False),assignment.items()))
