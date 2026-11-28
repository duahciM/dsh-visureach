"""Fresh compact concept test after the first model merged two identity levels."""
from dacf_round7.run import launch,PROTOCOL
from dacf_research.prepare_inputs import write
from partial_modules_stage3.material import H

if __name__=='__main__':
    out=H/'native_scope_retry';out.mkdir()
    prompt=('Short independent observer tool check. The local observer can store evidence-scoped links across the two '
            'actual endpoints when local support is sufficient. No link implies physical sameness. Consider a colored tab '
            'above a fixture at ticks 20 and 24, with a valid local visual link; after an unseen gap a similar tab appears at tick 80. '
            'What does the one-frame region ID refer to? What can the local identity support across 20 and 24? '
            'What would the 80 association require? Which single-time geometric evidence remains if the link is withdrawn? '
            'Reply in JSON with summary and answers. Do not copy a guide.')
    write(out/'contract.json',{'scope':'Independent comprehension follow-up; not a second automated observation run. The first answer confused one-frame identity with evidence-scoped linkage.','route':PROTOCOL['route']})
    launch(out/'independent',prompt,plain=True,watchdog=165)
