"""Two fair, count-matched baselines without selected or scored PTS inputs."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from if15_research.audit import pts_ledger
from if17_research.catalog import read, ref, sha
from p216_research.navigate import checked_json
from p218_research.baseline import materialize, uniform
from p218_research.observer import observe
from p219_research.run import for_observer

from .prepare import OLD, EXCLUDED_SHA256


def scope_for(public_range,goal):
    start,end=public_range
    if goal.get("scope_hint")=="late":
        return start+2*(end-start)/3,end,"pre-locked last third from public late-only wording"
    return start,end,"no objective public narrowing; same range as U-global"


def run(out):
    count=read(out/"algorithm_lock.json")["count_input"]
    assert sha(count["path"])==count["sha256"]
    groups=defaultdict(list)
    for row in read(count["path"])["pairs"]:
        groups[row["goal"]["case"]].append(row)
    library=read(out/"inputs_and_goals.json")["library"]
    modes=("U-global","U-public-scope")

    def run_case(pair):
        case,entries=pair
        row=library[case]
        book=read(row["catalog"]["path"])
        source=book["source"]
        assert sha(source["path"])==source["sha256"] and source["sha256"]!=EXCLUDED_SHA256
        tb,ledger=pts_ledger(source)
        catalog={r["source_frame"] for r in book["frames"]}
        coarse={r["source_frame"] for r in read(row["coverage"]["path"])["candidate_frames"]}
        planned={}
        for request in entries:
            goal=request["goal"]
            assert request["source_sha256"]==source["sha256"] and request["public_range"]==[source["start"],source["end"]]
            for mode in modes:
                lo,hi,reason=scope_for(request["public_range"],goal) if mode=="U-public-scope" else (*request["public_range"],"full public source")
                options=[(i,float(r["pts_time"])) for i,r in enumerate(ledger) if lo<=float(r["pts_time"])<hi]
                indices=[options[j][0] for j in uniform([t for i,t in options],request["n"])]
                planned[(mode,request["arm"]+"/"+goal["id"])]=(request,indices,len(options),[lo,hi],reason)
        needed={i for _,indices,_,_,_ in planned.values() for i in indices}
        folder=out/"controls_native"/case
        folder.mkdir(exist_ok=True,parents=True)
        # Selection above never consults this independent cache.
        for i in needed:
            p=OLD/"controls_native"/case/f"{i:07d}.json"
            copy=folder/p.name
            if p.exists() and not copy.exists():
                previous=read(p)
                assert previous["pts"]==ledger[i]["pts"]
                checked_json(copy,previous)
        decoded,cache=materialize(out,case,source,tb,ledger,needed)
        controls={mode:{} for mode in modes}
        for (mode,key),(request,indices,capacity,span,note) in planned.items():
            result=[]
            for i in indices:
                r=decoded[i]
                result.append({"path":r["native_image"],"sha256":r["native_image_sha256"],"source_frame":i,"pts":r["pts"],
                    "time_base":tb,"seconds":r["t"],"in_catalog":i in catalog,"in_coarse_pool":i in coarse,"kind":"uniform",
                    "source_cache_reuse":i in cache})
            label=mode+"_for_"+request["arm"]
            destination=out/"streams"/label/request["goal"]["id"]
            got=observe(destination,request["goal"],label,source,result,"uniform_completed")
            assert got["N_visible"]==min(request["n"],capacity)
            controls[mode][key]={"observer":ref(destination/"receipt.json"),"requested_n":request["n"],"capacity":capacity,
                "resolved_scope_seconds":span,"basis":note,"equal_actual_n":got["N_visible"]==request["n"],
                "source_frame_indices":indices,"no_ref_times_or_algorithm_pts":True}
        return controls

    with ThreadPoolExecutor(max_workers=3) as pool:
        all_results=list(pool.map(run_case,groups.items()))
    for mode in modes:
        output="global" if mode=="U-global" else "public_scope"
        checked_json(out/f"paired_uniform_{output}.json",{"pairs":{k:v for part in all_results for k,v in part[mode].items()},"count_only_export":count})
    old=read(OLD/"paired_uniform_manifest.json")["separate_natural_D6"]
    controls={}
    goals={g["id"]:g for group in groups.values() for g in [x["goal"] for x in group]}
    for case,ptr in old.items():
        assert sha(ptr["path"])==ptr["sha256"]
        source=read(library[case]["catalog"]["path"])["source"]
        dest=out/"streams"/"D6_natural"/case
        observe(dest,goals[case],"D6_natural",source,[for_observer(r) for r in read(ptr["path"])["items"]],"frozen_natural_coverage")
        controls[case]=ref(dest/"receipt.json")
    checked_json(out/"natural_D6.json",controls)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out.resolve())
