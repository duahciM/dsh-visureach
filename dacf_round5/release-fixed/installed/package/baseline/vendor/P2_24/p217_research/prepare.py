"""Freeze the inherited artifact set and build fresh independent candidate catalogs."""

import argparse
import csv
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from if17_research.catalog import read, ref, sha
from p1.io import dump
from p215_research.prepare import EXCLUDED_SHA256, ROOT, supplemental
from p215_research.semantic import Clip, indexed_case


HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/"p216_navigation_20261121"
MODEL=ROOT/"p215_discovery_20261121"
CFG=read(HERE/"protocol.json")
MDN_URL="https://raw.githubusercontent.com/mdn/learning-area/main/accessibility/multimedia/rabbit320.mp4"
NEUTRAL="What changes are directly checkable in this video? Show evidence, and separate uncertainty from observation."
CASES=("short","g3","flower","f01","long","g1","g4","ppt","dissolve","local_object","rabbit","still","return")


def fixtures(out):
    folder=out/"assets"
    w,h=640,360
    yy,xx=np.indices((h,w))
    base=np.empty((h,w,3),dtype=np.uint8)
    base[:,:,0]=120+(xx//19%2)*25
    base[:,:,1]=91+(yy//17%2)*28
    base[:,:,2]=89+((xx+yy)//21%2)*22
    view=Image.fromarray(base)
    draw=ImageDraw.Draw(view)
    draw.rounded_rectangle((147,76,492,279),radius=21,fill="#24445a",outline="#d3d7b7",width=5)
    for i in range(8):
        x=162+i*41
        draw.line((x,100,480-x//2,250),fill="#697c91",width=2)
    draw.ellipse((270,130,360,222),fill="#0b5f86",outline="#b6d7e2",width=4)
    unchanged=np.asarray(view).tobytes()
    changed=view.copy()
    ImageDraw.Draw(changed).ellipse((287,159,343,210),fill="#ff358a",outline="#ffe8ce",width=4)
    flash=np.asarray(changed).tobytes()
    for case in ("still","return"):
        video=folder/f"{case}.mp4"
        if video.exists():
            continue
        cmd=["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r","30","-i","pipe:0",
             "-an","-c:v","libx264","-qp","0","-pix_fmt","yuv444p",str(video)]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
            for i in range(150):
                proc.stdin.write(flash if case=="return" and 66<=i<=68 else unchanged)
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("Control video encoding failed")


def fresh_catalog(out,case,video):
    scratch=out/"new_library"/case
    entry=supplemental(scratch,"fresh",video)
    target=out/"semantic"/case
    target.mkdir(parents=True)
    started=perf_counter()
    book=read(entry["catalog"]["path"])
    if case=="rabbit":
        with (MODEL/"assets/openimages.csv").open(encoding="utf-8-sig",newline="") as source:
            vocabulary=sorted({row[1].casefold() for row in csv.reader(source) if len(row)==2})
        encoder=Clip(MODEL)
        vectors=encoder.text([f"a photo of a {term}." for term in vocabulary])
        rows,_,errors=indexed_case(book,vocabulary,vectors,encoder)
    else:
        rows=[{"frame_id":r["frame_id"],"status":"synthetic_control","nodes":[],"relations":[]} for r in book["frames"]]
        errors=[]
    assert [r["frame_id"] for r in rows]==[r["frame_id"] for r in book["frames"]]
    dump(target/"candidates.json",{"rows":rows,"errors":errors,"model_cache":str(MODEL),
         "explicit_control_policy":"Synthetic controls use no semantic labels." if case!="rabbit" else None,
         "elapsed_seconds":perf_counter()-started})
    return entry|{"semantic":str(target),"basis":"new P2-17 source, inherited 2 fps M7 intake",
                  "provenance":MDN_URL if case=="rabbit" else "deterministic P2-17 synthetic control"}


def prepare(out,card):
    if (out/"inputs_lock.json").exists():
        raise ValueError("Prediction inputs already sealed")
    fixtures(out)
    items=read(PREVIOUS/"library/manifest.json")
    sha_list={v["source"]["sha256"] for v in items.values()}
    new_paths={case:out/"assets"/("rabbit320.mp4" if case=="rabbit" else f"{case}.mp4") for case in ("rabbit","still","return")}
    for case,video in new_paths.items():
        if sha(video) in sha_list|{EXCLUDED_SHA256}:
            raise ValueError("Duplicate or user-excluded new source")
        items[case]=fresh_catalog(out,case,video)
    choices={"short":"Compare visible ball, gripper and tray states without assuming their physical interaction.",
             "g3":"Compare observations of the small object relative to the other visible structures.",
             "long":"Compare separable views of elevated ball and gripper at different parts of the source. Identity and contact can remain uncertain.",
             "f01":"Which image structures and viewpoints can be compared? Leave physical room and object identity open."}
    goals=[{"id":case,"case":case,"text":choices.get(case,NEUTRAL)} for case in CASES]
    prior=next(g for g in read(PREVIOUS/"public_goals.json") if g["id"]=="flower-detail")
    goals.append(prior|{"id":"flower-explicit"})
    (out/"library").mkdir(exist_ok=True)
    dump(out/"library/manifest.json",items)
    dump(out/"public_goals.json",goals)
    inputs={}
    def lock(path,digest=None):
        path=Path(path).resolve()
        actual=sha(path)
        if digest is not None and digest!=actual:
            raise ValueError(f"Frozen input changed: {path}")
        inputs[str(path)]=actual
    for entry in items.values():
        assert entry["source"]["sha256"]!=EXCLUDED_SHA256
        for name in ("catalog","index","coverage","source"):
            lock(entry[name]["path"],entry[name]["sha256"])
        for frame in read(entry["catalog"]["path"])["frames"]:
            native=frame["source_image"]
            lock(native["path"],native["sha256"])
        lock(Path(entry["semantic"])/"candidates.json")
        vectors=Path(entry["semantic"])/"vectors.npy"
        if vectors.exists():
            lock(vectors)
    prior_manifest=read(PREVIOUS/"navigation_manifest.json")
    for receipt in prior_manifest["trials"].values():
        lock(receipt["path"],receipt["sha256"])
    frozen=[HERE/n for n in ("prepare.py","triggers.py","runner.py","protocol.json")]
    frozen.extend(ROOT/name for name in ("p1/core.py","if16_research/nominate.py","if17_research/browse.py","p215_research/retrieval.py","p216_research/navigate.py","v05/engine.py"))
    frozen.extend(PREVIOUS/n for n in ("public_goals.json","library/manifest.json","inputs_lock.json","P2_16_REPORT.md","evaluation.json","navigation_manifest.json"))
    for p in [*frozen,out/"library/manifest.json",out/"public_goals.json"]:
        lock(p)
    dump(out/"inputs_lock.json",{"files":inputs,"task_card":ref(card),"excluded_by_user_sha256":EXCLUDED_SHA256,
         "prior_result_count":len(prior_manifest["trials"]),"architecture":"shadow derived triggers, no M7 membership mutation",
         "predictor_scoring_files_accessed":False})
    print("Frozen",len(inputs),"files;",len(goals),"goals;",len(items),"catalogs",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--card",required=True,type=Path)
    args=parser.parse_args()
    prepare(args.out.resolve(),args.card.resolve())
