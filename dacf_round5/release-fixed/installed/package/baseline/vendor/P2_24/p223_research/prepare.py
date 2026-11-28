"""Freeze existing research and intake new, video-grouped source material."""
import argparse
import csv
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p215_research.prepare import EXCLUDED_SHA256, ROOT, supplemental
from p215_research.semantic import Clip, indexed_case
from p216_research.navigate import checked_json


HERE=Path(__file__).resolve().parent
OLD=ROOT/"p222_extremes_20261121"
EARLIER=ROOT/"p221_recovery_20261121"
PROTO=read(HERE/"protocol.json")
SOURCES={
    "pin_gripper": {"file":"pin-gripper.mp4", "origin":"https://github.com/siggraph-pin-pression-gripper/pin-pression-gripper-video#real-world-demo", "group":"Pin-pression grippers real-world passive demo"},
    "articulation": {"file":"articulation.mp4", "origin":"https://github.com/SungHyunC/video2urdf", "group":"Independent Video2URDF phone recording"},
}
GOALS=(
    ("pin_scene","pin_gripper","scene","Show opening work surface and a later view of gripper and object in the same visible arrangement."),
    ("pin_relation","pin_gripper","relation","Compare a wide gap and later closer image-space positions of tool and object. Physical contact can remain unknown."),
    ("articulation_scene","articulation","scene","Compare opening and later reference views of the object and nearby visible structure."),
    ("articulation_open","articulation","open","Show directly checkable changes in the image. Distinguish changes of the projected object from camera direction."),
)


def freeze(out,card):
    history=read(OLD/"protocol_lock.json")["files"].copy()
    for name in ("algorithm_lock.json","source_and_goal_manifest.json","protocol_lock.json","E0_reference_lock.json","run_manifest.json","video_segment_and_transition_audit.json"):
        file=OLD/name
        history[str(file)]=sha(file)
    manifest=read(OLD/"run_manifest.json")
    history.update({p["path"]:p["sha256"] for p in manifest["artifacts"].values()})
    for key,pointers in read(OLD/"algorithm_lock.json")["routes"].items():
        if key.startswith(("W0_C4/","L-only/")):
            history.update({p["path"]:p["sha256"] for p in pointers.values()})
    for p in (HERE/"protocol.json",*[HERE/n for n in ("prepare.py","features.py","timeline.py","delivery.py","run.py","uniform.py","fixture.py")]):
        history[str(p)]=sha(p)
    media={key:ref(out/"assets"/m["file"]) for key,m in SOURCES.items()}
    old=read(OLD/"source_and_goal_manifest.json")
    user_path=Path("C:/Users/Administrator/Desktop/视觉插件/素材")
    inventory=[ref(p) for p in sorted(user_path.iterdir()) if p.is_file() and p.suffix.lower() in (".mp4",".mkv",".mov",".webm",".avi")]
    used_hashes={v["source"]["sha256"] for v in old["library"].values()}
    checked_json(out/"protocol_lock.json",{"files":history,"task_card":ref(card),"rules":PROTO,"new_media":media,
                 "before_new_source_image_review":True,"excluded_source_sha256":EXCLUDED_SHA256,
                 "user_materials_inventory":[r|{"used_in_historical_input_batch":r["sha256"] in used_hashes} for r in inventory]})


def public(out):
    locked=read(out/"protocol_lock.json")
    assert all(sha(p)==h for p,h in locked["files"].items() if Path(p).parent==HERE)
    prior=read(OLD/"source_and_goal_manifest.json")
    library=prior["library"]
    used={v["source"]["sha256"] for v in library.values()}|{EXCLUDED_SHA256}
    model=ROOT/"p215_discovery_20261121"
    with (model/"assets/openimages.csv").open(encoding="utf-8-sig",newline="") as handle:
        words=sorted({r[1].casefold() for r in csv.reader(handle) if len(r)==2})
    clip=Clip(model)
    encoded=clip.text([f"a photo of a {v}." for v in words])
    for name,group in SOURCES.items():
        ptr=locked["new_media"][name]
        assert sha(ptr["path"])==ptr["sha256"] and ptr["sha256"] not in used
        used.add(ptr["sha256"])
        active=out/"holdouts"/name
        entry=supplemental(active,"fresh",Path(ptr["path"]))
        sem=active/"semantic"
        sem.mkdir()
        rows,_,errors=indexed_case(read(entry["catalog"]["path"]),words,encoded,clip)
        checked_json(sem/"candidates.json",{"rows":rows,"errors":errors,"scope":"legacy intake compatibility; I/R does not use these labels"})
        library[name]=entry|{"semantic":str(sem),"semantic_sha256":sha(sem/"candidates.json"),"public_provenance":group}
    goals=prior["goals"]+[{"id":g,"case":c,"intent":i,"text":t,"cohort":"P2_23_new_source"} for g,c,i,t in GOALS]
    file=out/"source_and_goal_manifest.json"
    checked_json(file,{"library":library,"goals":goals,"E0_private":True,"no_training":True,"late":"last third from public scope"})
    checked_json(out/"input_receipt.json",{"public":ref(file),"protocol":ref(out/"protocol_lock.json")})
    checked_json(out/"learned_weight_audit.json",PROTO["learning"]|{"fitted_coefficients":None,"training_videos":[],"validation_videos":[],
                 "new_source_groups":[v["group"] for v in SOURCES.values()],"manual_I_plus_R":False,"U_ranking_target_leakage":False})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--card",type=Path)
    a=p.parse_args()
    freeze(a.out.resolve(),a.card.resolve()) if a.card else public(a.out.resolve())
