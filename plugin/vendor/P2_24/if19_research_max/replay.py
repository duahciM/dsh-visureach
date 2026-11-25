"""Replay P3-10 with the account Flash route at max effort and unchanged protocol inputs."""
import argparse
import json
from pathlib import Path
from shutil import copyfile
import sys

from if11_research.agent import Harness
from if10_research import freeze
from if17_research import freeze as history_verify
from if17_research.catalog import read,ref,sha
from if18_research import observe
from if19_research import blind_review,fixed,inputs,tool_loop,verify
from p1.io import dump

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OFF=ROOT/"if19_research_20261121"
BASE_RULES=read(ROOT/"if19_research"/"protocol.json")
ROUTE=BASE_RULES["route"]|{"reasoningEffort":"max"}


def prepare(out):
    prefix=out/"fixed_input"
    prefix.mkdir(exist_ok=False,parents=True)
    original=read(OFF/"fixed_input"/"manifest.json")
    copies={}
    for case, item in original["input_by_case"].items():
        path=Path(item["path"])
        assert sha(path)==item["sha256"]
        c=prefix/case
        c.mkdir()
        dst=c/"input.json"
        copyfile(path,dst)
        assert sha(dst)==item["sha256"]
        p=read(dst)
        assert sha(p["image"]["path"])==p["image"]["sha256"] and p["native_audit"]["pass"]
        copies[case]=ref(dst)
    old_reference=OFF/"evaluation_only"
    evaluate=out/"evaluation_only"
    evaluate.mkdir()
    for p in old_reference.glob("*_reference.json"):
        copyfile(p,evaluate/p.name)
    dump(prefix/"manifest.json",{"input_by_case":copies,"one_common_input_per_case":True,
         "original":ref(OFF/"fixed_input"/"manifest.json"),"replay_effective_route":ROUTE})
    dump(out/"protocol_lock.json",{"base":ref(ROOT/"if19_research"/"protocol.json"),"effective_observer_route":ROUTE,
         "observer_transport_timeout_seconds":600 if ROUTE["maxTokens"]>=16384 else 150,"reviewer_route":BASE_RULES["route"],"same_questions_and_observer_caps":True,
         "frozen_off_run":ref(OFF/"run_manifest.json"),"same_fixed_input":copies,
         "stage_B_catalog_and_budget":ref(OFF/"B"/"manifest.json")})


class LongTurnHarness(Harness):
    def turn(self,text,images,ordinal):
        self.long_turn=True
        try:
            return super().turn(text,images,ordinal)
        finally:
            self.long_turn=False

    def receive(self,deadline):
        return super().receive(deadline+(450 if getattr(self,"long_turn",False) else 0))


def use_max():
    for module in (fixed,tool_loop,verify,inputs):
        module.PROTOCOL={**module.PROTOCOL,"route":ROUTE}
    fixed.ROUTE=ROUTE
    observe.ROUTE=ROUTE
    if ROUTE["maxTokens"]>=16384:
        fixed.Harness=LongTurnHarness
        observe.Harness=LongTurnHarness


def existing_trial(out,case,arm,replicate):
    attempts=[]
    for number in (1,2):
        path=out/"A"/case/f"{arm}_{replicate}"/f"attempt_{number}"/"result.json"
        if path.exists():
            r=read(path)
            p=ref(path)
            assert r["route"]==ROUTE
            assert r["input"]["sha256"]==sha(out/"fixed_input"/case/"input.json")
        else:
            r,p=fixed.run_one(out,case,arm,replicate,number)
        attempts.append(p|{"status":r["status"]})
        if r["status"]=="completed":
            break
    return {"case":case,"arm":arm,"replicate":replicate,"attempts":attempts,"status":r["status"],
            "accepted":p if r["status"]=="completed" else None}

def run_history(out,checking):
    freeze.HERE=HERE
    freeze.CARD=freeze.MEDIA.parent/"video-keyframes-research-CP-IF18-visual-evidence-accountability-next.md"
    if checking:
        history_verify.verify(out)
    else:
        sys.argv=["history","--out",str(out)]
        freeze.main()


def main():
    global ROUTE
    a=argparse.ArgumentParser()
    a.add_argument("--out",required=True,type=Path)
    a.add_argument("--phase",choices=("prepare","freeze","history","A","B","review","audit","probe"),required=True)
    a.add_argument("--workers",type=int)
    a.add_argument("--max-tokens",type=int,choices=(1800,8192,16384,32768),default=1800)
    p=a.parse_args()
    o=p.out.resolve()
    ROUTE=BASE_RULES["route"]|{"reasoningEffort":"max","maxTokens":p.max_tokens}
    if p.phase=="prepare":
        prepare(o)
        return
    if p.phase in ("freeze","history"):
        run_history(o,p.phase=="history")
        return
    # The observer intervention is reasoning only. Reviewer settings are kept at the prior setting.
    if p.phase!="review":
        use_max()
    if p.phase=="A":
        fixed.job=existing_trial
    if p.phase=="probe":
        fixed.run_one(o,"g3","A0",1,1)
        return
    sys.argv=[p.phase,"--out",str(o)]
    if p.workers:
        sys.argv.extend(["--workers",str(p.workers)])
    {"A":fixed.main,"B":tool_loop.main,"review":blind_review.main,"audit":verify.main}[p.phase]()


if __name__=="__main__":
    main()



