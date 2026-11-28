"""Freeze P2-17 inputs, former results and an observer-input protocol."""

import argparse
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p1.io import dump
from p217_research.prepare import ROOT, EXCLUDED_SHA256


HERE=Path(__file__).resolve().parent
OLD=ROOT/"p217_triggered_20261121"
CFG=read(HERE/"protocol.json")
TOKENIZER=ROOT/"p215_discovery_20261121/assets/tokenizer.json"
VENDOR_TOKENS=ROOT/"p215_discovery_20261121/assets/token_runtime"
PRIMARY={"A0":"T3","A1":"T5"}


def freeze(out,card):
    out.mkdir(parents=True,exist_ok=True)
    if (out/"protocol_lock.json").exists():
        raise ValueError("Protocol lock already exists")
    prior=read(OLD/"inputs_lock.json")
    inputs={}
    def check(path,digest=None):
        p=Path(path).resolve()
        actual=sha(p)
        if digest and actual!=digest:
            raise ValueError(f"Frozen historical input changed: {p}")
        inputs[str(p)]=actual
    for path,digest in prior["files"].items():
        check(path,digest)
    trial_manifest=read(OLD/"prediction_manifest.json")
    for trial in trial_manifest["trials"].values():
        check(trial["path"],trial["sha256"])
    for p in [OLD/"inputs_lock.json",OLD/"public_goals.json",OLD/"library/manifest.json",OLD/"P2_17_REPORT.md",
              OLD/"prediction_manifest.json",OLD/"source_audit.json",TOKENIZER,ROOT/"if20_research/direct.py",
              ROOT/"v07/audit.py",ROOT/"v05/engine.py",ROOT/"if15_research/audit.py",
              *[HERE/name for name in ("freeze.py","protocol.json","observer.py","baseline.py","primary.py")]]:
        check(p)
    inventory=read(OLD/"library/manifest.json")
    assert all(v["source"]["sha256"]!=EXCLUDED_SHA256 for v in inventory.values())
    dump(out/"public_goals.json",read(OLD/"public_goals.json"))
    dump(out/"public_library.json",inventory)
    for p in (out/"public_goals.json",out/"public_library.json"):
        check(p)
    dump(out/"protocol_lock.json",{"inputs":inputs,"task_card":ref(card),"rules":CFG,"historical_runs_frozen":len(trial_manifest["trials"]),
          "explicit_known_source_exclusion":EXCLUDED_SHA256,"prediction_has_no_reference_answers":True,
          "p217_solver":"unchanged T3/T5","a3":"must have separately versioned later lock"})
    print("Frozen",len(inputs),"files",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--card",required=True,type=Path)
    opts=parser.parse_args()
    freeze(opts.out.resolve(),opts.card.resolve())
