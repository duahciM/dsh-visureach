"""Repeat the evidence-description comparison inside the frozen live browsing loop."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from if18_research import observe
from if17_research.catalog import read,ref
from p1.io import dump

from .fixed import ADD
from .inputs import PROTOCOL

original_condition=observe.condition
ARMS={"U0":"B0","U1":"B1"}


def condition(case,arm):
    base=original_condition(case,"B1").replace(
        'facts [{claim,kind:"direct",pts:[integer,...]}]',
        'facts [{claim,kind:"direct",optional pts:[integer,...]}]')
    if arm=="U1":
        return base+"\n"+ADD
    return base


def trial(root,c,a):
    attempts=[]
    for rep in (1,"1_retry"):
        r=observe.run_session(root,c,a,rep)
        p=root/"sessions"/f"{c}_{a}_{rep}"/"result.json"
        attempts.append(ref(p)|{"status":r["status"]})
        if r["status"]=="completed":
            break
    return {"case":c,"condition":ARMS[a],"transport_id":a,"status":r["status"],"all_attempts":attempts,
           "accepted":attempts[-1] if r["status"]=="completed" else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--workers",type=int,default=2)
    x=p.parse_args()
    o=x.out.resolve()
    assert read(o/"A"/"manifest.json")["each_case_has_comparable_same_image_arms"]
    assert observe.ROUTE==PROTOCOL["route"]
    root=o/"B"
    root.mkdir(exist_ok=False)
    observe.condition=condition
    cases=PROTOCOL["stage_B"]["cases"]
    jobs=[(c,a) for c in cases for a in ARMS]
    with ThreadPoolExecutor(max_workers=x.workers) as pool:
        tasks=[pool.submit(trial,root,c,a) for c,a in jobs]
        outcomes=[f.result() for f in tasks]
    dump(root/"manifest.json",{"groups":outcomes,"both_arms_have_same_multidimensional_index":True,
         "base_user_goals":{c:observe.QUESTIONS[c] for c in cases},"caps":observe.CAP,
         "different_realized_reads_allowed":True,"no_extra_final_review_instruction":True,
         "caller_source":"Exact frozen P3-09 host and full P2-13 catalog"})


if __name__=="__main__":
    main()
