"""Equal-count controls isolated from the algorithmic PTS and answer manifests."""

import argparse
from bisect import bisect_left, insort
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from if15_research.audit import pts_ledger
from if17_research.catalog import read, ref, sha
from p216_research.navigate import checked_json, commit
from v05.engine import export_native

from .freeze import EXCLUDED_SHA256
from .observer import observe


def nearest(sequence,target,lo,hi):
    stop=bisect_left(sequence,target,lo,hi+1)
    return min({min(hi,stop),max(lo,stop-1)},key=lambda i:(abs(sequence[i]-target),i))


def uniform(times,n):
    n=min(n,len(times))
    if not n:
        return []
    if n==1:
        return [nearest(times,(times[0]+times[-1])/2,0,len(times)-1)]
    begin,end=times[0],times[-1]
    result=[]
    lo=0
    for i in range(n):
        target=begin+(end-begin)*i/(n-1)
        pick=nearest(times,target,lo,len(times)-n+i)
        result.append(pick)
        lo=pick+1
    return result


def coverage(times,anchors,n):
    n=min(n,len(times))
    if not n:
        return []
    registered=sorted(set([times[0],times[-1],*anchors]))
    midpoints=sorted(zip(registered,registered[1:]),key=lambda z:(z[0]-z[1],z[0]))
    targets=[times[0],times[-1]]+[(a+c)/2 for a,c in midpoints]+registered
    order=[]
    visited=set()
    def add(index):
        if index not in visited:
            order.append(index)
            visited.add(index)
    for target in targets:
        if len(order)>=n:
            return order
        add(nearest(times,target,0,len(times)-1))
    sorted_indices=sorted(visited)
    while len(order)<n:
        a,c=max(((a,c) for a,c in zip(sorted_indices,sorted_indices[1:]) if c-a>1),key=lambda z:(times[z[1]]-times[z[0]],-z[0]))
        k=nearest(times,(times[a]+times[c])/2,a+1,c-1)
        add(k)
        insort(sorted_indices,k)
    return order


def materialize(out,case,source,tb,ledger,indices):
    """Independent point store consulted only after the control selection is fixed."""
    folder=out/"controls_native"/case
    folder.mkdir(exist_ok=True,parents=True)
    available={i:read(folder/f"{i:07d}.json") for i in indices if (folder/f"{i:07d}.json").is_file()}
    missing=sorted(set(indices)-available.keys())
    if missing:
        dest=folder/f"decode_{uuid4().hex}"
        rows=[{"source_frame":i,"pts":ledger[i]["pts"],"t":float(ledger[i]["pts_time"]),"time_base":tb,"sample":j,
               "image":source["path"]} for j,i in enumerate(missing)]
        raw=export_native(source,rows,list(range(len(rows))),dest)
        for r in raw:
            i=r["source_frame"]
            available[i]=r
            commit(folder/f"{i:07d}.json",r)
    return available,set(indices)-set(missing)


def run(out,counts_path,library_path,tag):
    pairs=read(counts_path)["pairs"]
    inventory=read(library_path)
    grouped=defaultdict(list)
    for p in pairs:
        grouped[p["goal"]["case"]].append(p)

    def run_case(case,entries):
        item=inventory[case]
        book=read(item["catalog"]["path"])
        source=book["source"]
        assert source["sha256"]!=EXCLUDED_SHA256 and sha(source["path"])==source["sha256"]
        tb,ledger=pts_ledger(source)
        eligible=[i for i,frame in enumerate(ledger) if source["start"]<=float(frame["pts_time"])<source["end"]]
        points=[float(ledger[i]["pts_time"]) for i in eligible]
        by_catalog={r["source_frame"]:r for r in book["frames"]}
        coarse={r["source_frame"] for r in read(item["coverage"]["path"])["candidate_frames"]}
        cat=list(by_catalog)
        cat_times=[by_catalog[i]["t"] for i in cat]
        anchors=[r["t"] for r in book["frames"] if "D6" in r["image_dimensions"]]
        selected={}
        for p in entries:
            assert p["public_range"]==[source["start"],source["end"]] and p["source_sha256"]==source["sha256"]
            key=p["arm"]+"/"+p["goal"]["id"]
            selected[key]={"algorithm_arm":p["arm"],"goal":p["goal"],"n":p["n"],"n_catalog":p["n_catalog"],
                           "technical_abort":p["technical_abort"],"source_capacity":len(eligible),"catalog_capacity":len(cat),
                           "uniform_source_indices":[eligible[i] for i in uniform(points,p["n"])],
                           "uniform_catalog_indices":[cat[i] for i in uniform(cat_times,p["n_catalog"])],
                           "coverage_source_indices":[eligible[i] for i in coverage(points,anchors,p["n"])]}
        locks=out/"control_selection"/tag
        locks.mkdir(parents=True,exist_ok=True)
        checked_json(locks/f"{case}.json",selected)
        needed={i for p in selected.values() for k in ("uniform_source_indices","coverage_source_indices") for i in p[k]}
        exact,prior_cache=materialize(out,case,source,tb,ledger,sorted(needed))
        outcomes={}
        for name,p in selected.items():
            records={}
            for suffix,key in (("A2","uniform_source_indices"),("A2_catalog_only","uniform_catalog_indices"),("A4","coverage_source_indices")):
                sequence=[]
                for i in p[key]:
                    if suffix=="A2_catalog_only":
                        raw=by_catalog[i]
                        image=raw["source_image"]
                        path,digest,seconds=image["path"],image["sha256"],raw["t"]
                    else:
                        raw=exact[i]
                        path,digest,seconds=raw["native_image"],raw["native_image_sha256"],raw["t"]
                    sequence.append({"path":path,"sha256":digest,"seconds":seconds,"pts":raw["pts"],"source_frame":i,"time_base":tb,
                                     "kind":"full_source_control" if suffix!="A2_catalog_only" else "catalog_control",
                                     "in_catalog":i in by_catalog,"in_coarse_pool":i in coarse,"source_cache_reuse":i in prior_cache})
                label=suffix+"_for_"+p["algorithm_arm"]
                folder=out/"streams"/label/p["goal"]["id"]
                receipt=observe(folder,p["goal"],label,source,sequence,"control_selection_complete")
                count=p["n_catalog"] if suffix=="A2_catalog_only" else p["n"]
                assert receipt["N_visible"]==min(count,p["catalog_capacity"] if suffix=="A2_catalog_only" else p["source_capacity"])
                records[suffix]=ref(folder/"receipt.json")
            outcomes[name]={"equal_input":p["n"]<=p["source_capacity"],"saturation":p["n"]>p["source_capacity"],
                            "counts_only":{k:p[k] for k in ("n","n_catalog","source_capacity","catalog_capacity","technical_abort")},
                            "arms":records,"selected":ref(locks/f"{case}.json")}
            print(name,"matched",p["n"],flush=True)
        return outcomes

    started=perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        controls={k:v for table in pool.map(lambda args:run_case(*args),grouped.items()) for k,v in table.items()}
    checked_json(out/f"{tag}_paired_uniform_manifest.json",{"controls":controls,"count_export":ref(counts_path),
                 "received_algorithm_pts":False,"received_reference_labels":False})
    commit(out/f"{tag}_control_runtime.json",{"seconds":perf_counter()-started})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--counts",required=True,type=Path)
    parser.add_argument("--library",required=True,type=Path)
    parser.add_argument("--tag",required=True,choices=("primary","candidate","holdout"))
    a=parser.parse_args()
    run(a.out.resolve(),a.counts.resolve(),a.library.resolve(),a.tag)
