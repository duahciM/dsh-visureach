"""Read-only evaluation of the emitted eight-frame arms against sealed visible roles."""
import argparse
import csv
from pathlib import Path

from identity_catalog_matching.common import read,sha
from p1.io import dump


def taken(frames, windows):
    return [r for r in frames if any(lo<=r["t"]<=hi for lo,hi in windows)]


def measure(frames, roles, case):
    out={}
    for role, windows in roles.items():
        rows=taken(frames,windows)
        out[role]={"eligible_intervals":windows,"seen":bool(rows),
                   "frames":[{k:f[k] for k in ("t","source_frame","pts","time_base","native_image")} for f in rows]}
    if case=="new_6a8c74a6":
        bracket=all(out[k]["seen"] for k in ("onset","process","outcome"))
        for k in ("process","competing"):
            out[k]["seen"]=bracket
            out[k]["needs_both_sides"]="Occluder-only image has no standalone process credit"
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    x=ap.parse_args()
    o=x.out.resolve()
    path=o/"intake"/"observed_roles_sealed.json"
    annotations=read(path)
    assert sha(path)==read(o/"intake"/"selection_addendum.json")["roles_sha256"]
    summary=[]
    for c,roles in annotations["roles"].items():
        d=o/"P2"/c
        q=read(d/"selection.json")
        rows={}
        for arm,frames in q["arms"].items():
            r=measure(frames,roles,c)
            rows[arm]={"budget":len(frames),"roles":r,"seen":sorted(k for k,v in r.items() if v["seen"]),
                       "missed":sorted(k for k,v in r.items() if not v["seen"]),
                       "unassessed":[k for k in ("initial","onset","process","outcome","competing") if k not in r],
                       "coverage":q["diagnostics"][arm]}
            summary.append({"case":c,"arm":arm,"covered":len(rows[arm]["seen"]),"eligible":len(roles),
                "roles":" ".join(rows[arm]["seen"]),"missed":" ".join(rows[arm]["missed"]),"unassessed":" ".join(rows[arm]["unassessed"]),
                "maximum_gap_s":q["diagnostics"][arm]["maximum_gap_s"],"candidate_count":q["candidate_count"]})
        comparisons={}
        for base in ("V","uniform","S2","K_roles","K_IF11_shadow","K_motion_only"):
            if base in rows:
                a,b=set(rows["relation"]["seen"]),set(rows[base]["seen"])
                comparisons[base]={"gained_role_proxies":sorted(a-b),"lost_role_proxies":sorted(b-a),
                    "relation_new_pts":[r["t"] for r in q["arms"]["relation"] if r["pts"] not in {z["pts"] for z in q["arms"][base]}],
                    "base_missing_pts":[r["t"] for r in q["arms"][base] if r["pts"] not in {z["pts"] for z in q["arms"]["relation"]}]}
        dump(d/"role_evaluation.json",{"case":c,"assessment":"visible time-neighborhood proxy, source images are available for all hits; no causal gold",
               "annotation_hash":sha(path),"arm_results":rows,"relation_comparisons":comparisons,
               "observed_context":annotations["notes"].get(c)})
    with (o/"P2"/"role_summary.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        w=csv.DictWriter(stream,fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    dump(o/"P2"/"role_summary.json",summary)
    for row in summary:
        if row["arm"] in ("V","uniform","relation"):
            print(row["case"],row["arm"],f'{row["covered"]}/{row["eligible"]}',row["missed"],flush=True)


if __name__=="__main__":
    main()
