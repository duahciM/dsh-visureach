"""Freeze P2-19 archives and new positive-evidence experiment inputs."""

import argparse
import csv
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p215_research.prepare import ROOT, EXCLUDED_SHA256, supplemental
from p215_research.semantic import Clip, indexed_case
from p216_research.navigate import checked_json


HERE=Path(__file__).resolve().parent
OLD=ROOT/"p219_goal_appropriate_20261121"
MODEL=ROOT/"p215_discovery_20261121"
CFG=read(HERE/"protocol.json")
EXTRA=[{"id":"bottle_single","case":"bottle","intent":"single","text":"Show one direct image of a visible bottle and the immediate surface around it."},
       {"id":"bottle_scene","case":"bottle","intent":"scene","text":"Compare an opening view of the working surface with a later visible object view. Keep viewpoint explicit."},
       {"id":"people_recur","case":"people","intent":"recurrence","text":"Show visually checkable appearances of people at different parts of the source. Their physical identity across time is unknown."},
       {"id":"people_scene","case":"people","intent":"scene","text":"Show the visible scene geometry and a later reference with people in view. Keep camera direction separate from movement."}]


def lock(out,card):
    out.mkdir(exist_ok=True,parents=True)
    prev=read(OLD/"protocol_lock.json")
    files=dict(prev["files"])
    for path in [OLD/"protocol_lock.json",OLD/"source_and_goal_manifest.json",OLD/"per_goal_dimension_results.csv",OLD/"P2_19_REPORT.md",OLD/"routes_frozen.json",
                 OLD/"paired_uniform_manifest.json",OLD/"run_manifest.json",*[HERE/k for k in ("prepare.py","protocol.json","visual.py","run.py","uniform.py")]]:
        files[str(path)]=sha(path)
    for key,row in read(OLD/"routes_frozen.json")["routes"].items():
        if key.startswith("S2/"):
            for ptr in row.values():
                assert sha(ptr["path"])==ptr["sha256"]
                files[ptr["path"]]=ptr["sha256"]
    checked_json(out/"protocol_lock.json",{"files":files,"task_card":ref(card),"rules":CFG,"holdout_frozen_before_visual_review":True,
                 "physically_deleted_and_excluded_source":EXCLUDED_SHA256})


def input_manifest(out):
    frozen=read(out/"protocol_lock.json")
    assert all(sha(HERE/name)==frozen["files"][str(HERE/name)] for name in ("prepare.py","visual.py","run.py","uniform.py"))
    prior=read(OLD/"source_and_goal_manifest.json")
    library=prior["library"]
    source_hashes={x["source"]["sha256"] for x in library.values()}
    with (MODEL/"assets/openimages.csv").open(encoding="utf-8-sig",newline="") as f:
        lexicon=sorted({r[1].casefold() for r in csv.reader(f) if len(r)==2})
    encoder=Clip(MODEL)
    text=encoder.text([f"a photo of a {name}." for name in lexicon])
    for case,url in CFG["holdouts"].items():
        media=out/"assets"/url.split("/")[-1]
        d=sha(media)
        assert d!=EXCLUDED_SHA256 and d not in source_hashes
        source_hashes.add(d)
        item=supplemental(out/"holdouts"/case,"fresh",media)
        folder=out/"holdouts"/case/"semantic"
        folder.mkdir(exist_ok=True)
        generated,_,errors=indexed_case(read(item["catalog"]["path"]),lexicon,text,encoder)
        checked_json(folder/"candidates.json",{"rows":generated,"errors":errors,"scope":"unchanged local CLIP model and words"})
        library[case]=item|{"semantic":str(folder),"original_provenance":url,"license":"CC BY 4.0",
                            "semantic_sha256":sha(folder/"candidates.json")}
    goals=prior["goals"]+[v|{"cohort":"new_independent"} for v in EXTRA]
    checked_json(out/"inputs_and_goals.json",{"library":library,"goals":goals,"public_scope_rule":"late hint: last 1/3 of declared source; otherwise complete source",
                  "reference_not_available_to_predictor":True})
    checked_json(out/"input_receipt.json",{"public":ref(out/"inputs_and_goals.json"),"protocol":ref(out/"protocol_lock.json")})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--card",type=Path)
    a=p.parse_args()
    if a.card:
        lock(a.out.resolve(),a.card.resolve())
    else:
        input_manifest(a.out.resolve())
