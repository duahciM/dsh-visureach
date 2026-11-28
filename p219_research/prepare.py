"""Freeze the public task surface and the exact historical stream provenance."""

import argparse
import csv
from pathlib import Path

import numpy as np

from if17_research.catalog import read, ref, sha
from p216_research.navigate import commit
from p215_research.prepare import supplemental, EXCLUDED_SHA256, ROOT
from p215_research.semantic import Clip, indexed_case


HERE=Path(__file__).resolve().parent
OLD=ROOT/"p218_equal_visible_20261121"
P217=ROOT/"p217_triggered_20261121"
MODEL=ROOT/"p215_discovery_20261121"
CONF=read(HERE/"protocol.json")
INTENTS={"short":"relation","g3":"relation","flower":"open","flower-explicit":"boundary","f01":"scene","long":"recurrence",
         "g1":"open","g4":"open","ppt":"open","dissolve":"open","local_object":"open","rabbit":"open","still":"open",
         "return":"open","bear_holdout":"open"}
CROSSED=[
    {"id":"bear_late_single","case":"bear","intent":"single","scope_hint":"late","text":"Show one direct view of the dark animal near the left frame edge at a late time. Keep the wire grid as context."},
    {"id":"short_single","case":"short","intent":"single","scope_hint":"late","text":"Show one late direct view of the red ball and blue tray. Keep the nearby context if visible."},
    {"id":"f01_scene","case":"f01","intent":"scene","text":"Compare the door-side view with an interior view. Preserve the image geometry needed for the comparison."},
    {"id":"ppt_boundary","case":"ppt","intent":"boundary","text":"Show a comparable blank prestate and the first visible small mark. Locate the observable onset interval."},
    {"id":"street_open_semantics","case":"fresh","intent":"open","text":"What local changes can be checked directly? Start with the observed region and compare the available views."},
    {"id":"bolt_single","case":"bolt","intent":"single","scope_hint":"any","text":"Show one view where one fastener and the surface immediately around it can be inspected."},
    {"id":"bolt_scene","case":"bolt","intent":"scene","text":"Show a view of the working surface and a later reference if the view changes. Keep viewpoint and motion separate."},
    {"id":"bolt_recur","case":"bolt","intent":"recurrence","text":"Compare visibly checkable stages at different parts of the source. Do not assume the same physical fastener between views."},
    {"id":"bolt_open","case":"bolt","intent":"open","text":"What changes should be investigated? Show local evidence and its uncertainty."}
]


def freeze_logic(out,card):
    out.mkdir(parents=True,exist_ok=True)
    old=read(OLD/"protocol_lock.json")
    paths=dict(old["inputs"])
    for p in (OLD/"P2_18_REPORT.md",OLD/"paired_uniform_manifest.json",OLD/"paired_role_scores.csv",OLD/"candidate_protocol_lock.json",
              OLD/"primary_score_freeze.json",OLD/"run_manifest.json",OLD/"candidate_algorithm_freeze.json",OLD/"holdout_algorithm_freeze.json",
              ROOT/"p218_research/candidate.py",ROOT/"p218_research/observer.py",ROOT/"p218_research/baseline.py",
              *[HERE/name for name in ("prepare.py","protocol.json","choose.py","run.py","controls.py")]):
        paths[str(p)]=sha(p)
    for name in ("candidate","holdout"):
        for row in read(OLD/f"{name}_algorithm_freeze.json")["routes"].values():
            for ref_item in row.values():
                assert sha(ref_item["path"])==ref_item["sha256"]
                paths[ref_item["path"]]=ref_item["sha256"]
    commit(out/"protocol_lock.json",{"card":ref(card),"rules":CONF,"files":paths,"freeze_scope":"Prediction code and immutable historical evidence before holdout visual review"})


def prepare(out):
    lock=read(out/"protocol_lock.json")
    assert all(sha(HERE/name)==lock["files"][str(HERE/name)] for name in ("prepare.py","choose.py","run.py","controls.py"))
    lib=read(OLD/"public_library.json")|read(OLD/"holdout_public_library.json")
    file=out/"assets"/CONF["new_source"]["name"]
    assert sha(file) not in {EXCLUDED_SHA256,*(v["source"]["sha256"] for v in lib.values())}
    # The existing 2 fps independent M7 intake is reused for new content.
    data=supplemental(out/"independent", "fresh",file)
    dest=out/"independent/semantics"
    dest.mkdir(parents=True)
    with (MODEL/"assets/openimages.csv").open(encoding="utf-8-sig",newline="") as f:
        lexicon=sorted({r[1].casefold() for r in csv.reader(f) if len(r)==2})
    model=Clip(MODEL)
    words=model.text([f"a photo of a {x}." for x in lexicon])
    semantic,vectors,errors=indexed_case(read(data["catalog"]["path"]),lexicon,words,model)
    commit(dest/"candidates.json",{"rows":semantic,"errors":errors,"scope":"frozen previously used model and taxonomy"})
    np.save(dest/"vectors.npy",vectors,allow_pickle=False)
    lib["bolt"]=data|{"semantic":str(dest),"provenance":CONF["new_source"]}
    tasks=read(OLD/"public_goals.json")+read(OLD/"holdout_public_goals.json")
    authored=[r|{"intent":INTENTS[r["id"]],"cohort":"retained"} for r in tasks]+[r|{"cohort":"holdout" if r["case"]=="bolt" else "crossed"} for r in CROSSED]
    assert all(v["source"]["sha256"]!=EXCLUDED_SHA256 for v in lib.values())
    manifest={"library":lib,"goals":authored,"holdout":ref(file)|CONF["new_source"],"excluded_sha256":EXCLUDED_SHA256,
              "goal_permissions":"Public prompts and observation intent only. Goals expose no reviewer timestamp, region or answer."}
    commit(out/"source_and_goal_manifest.json",manifest)
    seal={"public":ref(out/"source_and_goal_manifest.json"),"protocol":ref(out/"protocol_lock.json"),"holdout_new_code_frozen_before_review":True}
    for k in ("source","catalog","index","coverage"):
        seal[k]=data[k]
    seal["semantic"]=ref(dest/"candidates.json")
    commit(out/"public_input_lock.json",seal)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--card",type=Path)
    a=p.parse_args()
    if a.card:
        freeze_logic(a.out.resolve(),a.card.resolve())
    else:
        prepare(a.out.resolve())
