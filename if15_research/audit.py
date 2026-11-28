"""Independent original-image and source PTS verification."""
import argparse
from pathlib import Path
from traceback import format_exc

from identity_catalog_matching.common import read,sha
from p1.io import command,dump
from v07.audit import audit_images
from .intake import HERE
from . import freeze

import json


def audit_history(o):
    lock=read(o/"history_lock.json")
    actual=freeze.freeze.history(o)
    old=lock["history"]
    changed=sorted(k for k in old if k in actual and old[k]!=actual[k])
    missing=sorted(old.keys()-actual.keys())
    extra=sorted(actual.keys()-old.keys())
    src={r["path"]:r["sha256"] for r in lock["media"]+list(lock["historical_sources"].values())}
    src[lock["card"]["path"]]=lock["card"]["sha256"]
    mismatch=[p for p,h in src.items() if not Path(p).is_file() or sha(p)!=h]
    result={"files":len(old),"outside_sources_and_card":len(src),"changed":changed,"missing":missing,"added":extra,
            "source_mismatch":mismatch,"pass":not any((changed,missing,extra,mismatch))}
    dump(o/"history_audit.json",result)
    print("history",result["files"],result["outside_sources_and_card"],result["pass"],flush=True)


def pts_ledger(source):
    p=json.loads(command(["ffprobe","-v","error","-select_streams","v:0","-show_frames",
                          "-show_entries","frame=pts,pts_time:stream=time_base","-of","json", source["path"]]))
    return p["streams"][0]["time_base"],p["frames"]


def verify_case(o,d):
    c=d.name
    manifest=read(d/"frames.json")
    source=manifest["source"]
    selected=read(d/"selection.json")
    rows=[r for arm in selected["arms"].values() for r in arm]
    p2_count=len(rows)
    resampled=[]
    for p in d.glob("query_*/keyframes.json"):
        resampled.extend(read(p)["frames"])
    for p in o.glob(f"P3*/{c}/*/*/keyframes.json"):
        resampled.extend(read(p)["frames"])
    geos=list((o/"IF15").glob(f"{c}/summary.json"))
    geometry=[r for p in geos for r in read(p)["native"]]
    extra=[o/"source_review"/"cycle_examples"/c/"summary.json"]
    if c=="new_6a8c74a6":
        extra.append(o/"source_review"/"G3"/"summary.json")
    geometry.extend(r for p in extra if p.is_file() for r in read(p)["native"])
    return checks_for(source,rows,resampled,geometry,manifest["frames"],p2_count)


def checks_for(source,rows,resampled,geometry,pool,p2_count):
    exact,entries=pts_ledger(source)
    all_uses=rows+resampled+geometry+pool
    pts_failures=[]
    for r in all_uses:
        z=entries[r["source_frame"]]
        if not (r["pts"]==z["pts"] and r["time_base"]==exact and abs(float(z["pts_time"])-r["t"])<=1e-6):
            pts_failures.append({k:r[k] for k in ("source_frame","pts","time_base","t")})
    checks={}
    for key,chunk in (("p2",rows),("resample",resampled),("geometry",geometry)):
        found={}
        for r in chunk:
            found.setdefault(r["native_image"],r)
        values=list(found.values())
        checks[key]=[it for i in range(0,len(values),15) for it in audit_images(source,values[i:i+15])]
    image_count=sum(map(len,checks.values()))
    same=sha(source["path"])==source["sha256"]
    ok=same and not pts_failures and all(k["exact_source_rgb_equal"] and k["file_hash_equal"]
                                       for a in checks.values() for k in a)
    return {"source_sha256_equal":same,"pts_uses_checked":len(all_uses),"recorded_selection_slots":p2_count,
            "pts_mismatch":pts_failures,"native_images_checked":image_count,"checks":checks,"pass":ok}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--history",action="store_true")
    x=a.parse_args()
    o=x.out.resolve()
    if x.history:
        audit_history(o)
        return
    rows={}
    for d in sorted((o/"P2").iterdir()):
        if not (d/"selection.json").is_file():
            continue
        try:
            rows[d.name]=verify_case(o,d)
        except Exception:
            rows[d.name]={"pass":False,"error":format_exc()}
        print(d.name,rows[d.name]["pass"],rows[d.name].get("native_images_checked"),flush=True)
    target=o/"IF15"/"F01"/"summary.json"
    if not target.is_file():
        target=o/"IF15"/"F01_recovered"/"summary.json"
    if target.is_file():
        f=read(target)
        refs=[o/"source_review"/k/"summary.json" for k in ("F01_native_adjacent","F01_native_heldout")]
        reviewed=[r for p in refs if p.is_file() for r in read(p)["native"]]
        rows["F01"]=checks_for(f["source"],[],[],f["native"]+reviewed,[],0)
        print("F01",rows["F01"]["pass"],rows["F01"]["native_images_checked"],flush=True)
    dump(o/"source_audit.json",{"cases":rows,"total_image_checks":sum(r.get("native_images_checked",0) for r in rows.values()),
         "total_pts_uses":sum(r.get("pts_uses_checked",0) for r in rows.values()),
         "pass":bool(rows) and all(r["pass"] for r in rows.values())})


if __name__=="__main__":
    main()


