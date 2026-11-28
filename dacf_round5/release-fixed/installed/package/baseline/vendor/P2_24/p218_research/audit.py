"""Independently re-decode the formal visible frames and audit continuation."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory

from if15_research.audit import pts_ledger
from if17_research.catalog import read, ref, sha
from p216_research.navigate import commit
from v05.engine import select_expression

from .freeze import EXCLUDED_SHA256
from .observer import observe


def verify_history(o):
    lock=read(o/"protocol_lock.json")
    failed=[p for p,h in lock["inputs"].items() if not Path(p).is_file() or sha(p)!=h]
    if sha(lock["task_card"]["path"])!=lock["task_card"]["sha256"]:
        failed.append(lock["task_card"]["path"])
    return failed,len(lock["inputs"])


def audit_source(src, by_frame):
    tb,ledger=pts_ledger(src)
    w,h=src["probe"]["width"],src["probe"]["height"]
    frames=sorted(by_frame)
    conditions=[]
    if not frames:
        return {"pass":True,"source_rgb_frames":0,"png_file_variants":0,"rows":[]}
    with TemporaryDirectory(prefix="p218_rgb_") as temp:
        f=Path(temp)/"filter.txt"
        f.write_text(f"select='{select_expression(frames)}',format=rgb24",encoding="ascii")
        command=["ffmpeg","-v","error","-noautorotate","-i",src["path"],"-/filter:v",str(f),"-fps_mode","passthrough",
                 "-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as pipe:
            for frame in frames:
                data=pipe.stdout.read(w*h*3)
                variants=by_frame[frame]
                wanted=ledger[frame]
                source_pixels=sha256(data).hexdigest()
                equal=bool(len(data)==w*h*3) and all(
                    row["pts"]==wanted["pts"] and row["time_base"]==tb and abs(row["seconds"]-float(wanted["pts_time"]))<=1.1e-6
                    and row["attachment"]["rgb_sha256"]==source_pixels and row["attachment"]["width"]==w and row["attachment"]["height"]==h
                    for row in variants.values())
                conditions.append({"frame":frame,"pts":wanted["pts"],"time_base":tb,"rgb_sha256":source_pixels,
                                  "exact_native_pts_rgb":equal,"attachments":[{"path":r["path"],"sha256":r["attachment"]["sha256"]} for r in variants.values()]})
            trailing=pipe.stdout.read(1)
            failure=pipe.stderr.read().decode(errors="replace")
            success=pipe.wait()==0 and not trailing and not failure
    okay=success and all(row["exact_native_pts_rgb"] for row in conditions)
    return {"pass":okay,"source_rgb_frames":len(conditions),"png_file_variants":sum(len(v) for v in by_frame.values()),"rows":conditions}


def test_resume(out,source,catalog,goal):
    folder=out/"observer_continuation_test"
    folder.mkdir(exist_ok=True)
    report=folder/"check.json"
    if report.exists():
        return read(report)
    example=catalog[0]
    src=example["source_image"]
    spot=folder/"recovered.png"
    first={"path":src["path"],"sha256":src["sha256"],"seconds":example["t"],"pts":example["pts"],
           "source_frame":example["source_frame"],"time_base":example["time_base"],"kind":"catalog","in_catalog":True,"in_coarse_pool":True}
    second=first|{"path":str(spot)}
    initial=observe(folder,goal,"repeat_resume_demo",source,[first,second],"demo")
    if not initial["technical_abort"] or initial["N_visible"]!=1:
        raise ValueError("Missing-image continuation did not record the real visible prefix")
    copyfile(src["path"],spot)
    final=observe(folder,goal,"repeat_resume_demo",source,[first,second],"demo")
    good=not final["technical_abort"] and final["N_visible"]==2 and final["N_unique_pts"]==1 and final["N_repeated"]==1 and initial["items"][0]==final["items"][0]
    audit={"pass":good,"initial_visible":initial["N_visible"],"recovered_visible":final["N_visible"],"unique_after_recovery":final["N_unique_pts"],
           "real_repeat_count":final["N_repeated"],"interrupted_page":ref(folder/"observer_pages/interrupted_00001.json"),"final_page":ref(folder/"observer_pages/00001.json"),
           "scope":"Separation test only, never in official counts."}
    commit(report,audit)
    return audit


def run(out):
    changed,count=verify_history(out)
    phase=read(out/"paired_uniform_manifest.json")
    library=read(out/"public_library.json")|read(out/"holdout_public_library.json")
    books={case:read(entry["catalog"]["path"]) for case,entry in library.items()}
    sources={case:row["source"] for case,row in books.items()}
    assert all(src["sha256"]!=EXCLUDED_SHA256 and sha(src["path"])==src["sha256"] for src in sources.values())
    before=read(out/"primary_score_freeze.json")
    candidate=read(out/"candidate_protocol_lock.json")
    required=(before["primary_scores"],before["primary_counts"],before["primary_routes"],before["primary_controls"],
              before["score_program"],candidate["primary_score_freeze"],candidate["candidate_code"],candidate["support_code"])
    assert all(sha(x["path"])==x["sha256"] for x in required)
    frames=defaultdict(lambda:defaultdict(dict))
    file_hashes={}
    streams=0
    total=0
    for item in phase["paired_stream_receipts"].values():
        assert sha(item["path"])==item["sha256"]
        r=read(item["path"])
        streams+=1
        case=r["goal"]["case"]
        unpacked=[]
        for receipt in r["page_receipts"]:
            assert sha(receipt["path"])==receipt["sha256"]
            unpacked.extend(read(receipt["path"])["items"])
        assert unpacked==r["items"] and r["N_visible"]==len(unpacked)
        assert sum(im["repeated_visible_pts"] for im in unpacked)==r["N_repeated"]
        total+=len(unpacked)
        for item in unpacked:
            assert item["source_sha256"]==sources[case]["sha256"]
            assert item["delivered_to_interface"] and not item["sdk_acknowledged_by_model"] and item["format"]=="full_native_png"
            if item["path"] not in file_hashes:
                file_hashes[item["path"]]=sha(item["path"])
            assert file_hashes[item["path"]]==item["attachment"]["sha256"]
            frames[case][item["source_frame"]][item["path"]]=item

    def verify(case):
        row=audit_source(sources[case],frames[case])
        print(case,row["source_rgb_frames"],row["pass"],flush=True)
        return case,row
    with ThreadPoolExecutor(max_workers=3) as pool:
        pixels=dict(pool.map(verify,frames))
    goal=next(g for g in read(out/"public_goals.json") if g["id"]=="flower")
    restart=test_resume(out,sources["flower"],books["flower"]["frames"],goal)
    passed=not changed and restart["pass"] and all(x["pass"] for x in pixels.values())
    result={"pass":passed,"frozen_files":count,"historical_differences":changed,"native_interface_streams":streams,
            "native_interface_items":total,"validated_file_paths":len(file_hashes),"distinct_source_pts":sum(v["source_rgb_frames"] for v in pixels.values()),
            "rgb_checks":pixels,"continued_native_context":restart,"sdk_acknowledgements":0,"excluded_source_sha256":EXCLUDED_SHA256}
    commit(out/"source_and_history_audit.json",result)
    print("Audited",streams,"streams",total,"delivered items",passed,flush=True)
    return passed


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    raise SystemExit(not run(p.parse_args().out.resolve()))
