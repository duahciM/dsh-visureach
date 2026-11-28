"""Evaluate candidate recall and delivered roles separately; never imported by prediction."""
import argparse
from collections import Counter
import csv
from pathlib import Path

from if15_research.evaluate import taken
from p1.io import dump

from .common import IF15, RULES, get_meta, read, sha


G3="new_6a8c74a6"
LONG="new_4740e05f"
MAIN="room_corridor_dev"
PRIORITY=[G3,LONG,MAIN,"new_a31953d0","local_object","ppt","dissolve","new_7e08d03a","new_a32f9e1f"]


def references():
    prior=IF15/"intake"/"observed_roles_sealed.json"
    add=IF15/"P2"/"source_spot_check_addendum.json"
    base=read(prior)["roles"]
    for clip,details in read(add)["observations"].items():
        for r in details["supplement"]:
            base[clip][r["role"]].append([r["time"]-1e-6,r["time"]+1e-6])
    # CP-IF15 source-confirmed right-bearing images include 4.4 and 4.5 s.
    # The 4.6 s image was separately visually inspected from the same frozen pool.
    base[G3]={"initial":[[0,4]],"onset":[[4.399999,4.700001]],"process":[[4.8,5.3]],
              "outcome":[[5.399999,5.500001]],"competing":[[4.8,5.3]]}
    return base,{"IF15_original":sha(prior),"IF15_reviewed_points":sha(add),
        "CP_IF15_G3":"Actual first appearance right 4.4/4.5; coarse 4.6 visually reviewed; left at 5.4/5.5. Middle sampled absence only."}


def measure(frames,roles,case):
    r={role:[{k:f[k] for k in ("source_frame","t","pts","time_base")} for f in taken(frames,spans)]
       for role,spans in roles.items()}
    raw=sorted(k for k in r if r[k])
    covered=list(raw)
    if case==G3 and not {"onset","process","outcome"}<=set(raw):
        covered=[k for k in covered if k not in {"process","competing"}]
    return {"covered":covered,"present_before_bracket_rule":raw,"missed":sorted(set(roles)-set(covered)),
            "counts":{k:len(v) for k,v in r.items()},"witnesses":r,"role_count":len(roles)}


def target_regions(library):
    details=[]
    expected={
        2640:("right",274,318),
        2760:("right",214,260),
        3240:("left",39,79),
        3300:("left",21,61)
    }
    for f in library:
        t=f["t"]
        visible,lo,hi=expected.get(f["pts"],(None,None,None))
        for n in f["nominations"]:
            box=n["applicable_region"]
            if not box:
                continue
            x,y,w,h=box
            cx,cy=x+w/2,y+h/2
            support=visible is not None and 22<=cy<=85 and lo<=cx<=hi
            name="target_approximate_region" if support else "off_target_region"
            details.append({"pts":f["pts"],"t":t,"dimension":n["dimension_id"],"visible_target":visible,
                            "status":name,"region":box,"scope":"G3 human reviewed coarse side/height only"})
    return details


def diagnose(role,m_pool,m_library,m_final):
    if not m_pool["counts"][role]:
        return "M0_no_eligible_frame_in_coarse_pool"
    if not m_library["counts"][role]:
        return "M1_M5_no_nominated_eligible_frame"
    if role not in m_final["covered"]:
        return "M7_not_delivered_or_bracket_incomplete"
    return "delivered"


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    x=a.parse_args()
    o=x.out.resolve()
    annotations,provenance=references()
    dump(o/"evaluation_reference.json",{"roles":annotations,"provenance":provenance,
         "timing_proxy_only":True,"never_read_by_nominations":True})
    reduced=[]
    outputs={}
    eligibility=[]
    long_data={}
    for case in PRIORITY:
        d=o/"P2"/case
        selection=read(d/"N2_selection.json")
        lib=read(d/"N1_library.json")
        meta=get_meta(case)
        op=read(d/"N3"/"selection.json")
        roles=annotations[case]
        per={k:measure(v,roles,case) for k,v in selection["N0"].items()}
        pset={"M0":meta["frames"],"N1":lib,"N2":selection["N2"],**op["arms"]}
        for k,v in pset.items():
            per[k]=measure(v,roles,case)
        selected=set(per["N2"]["covered"])
        sets=[set(per[k]["covered"]) for k in selection["N0"]]
        no_old_coverage=sorted(selected-set.union(*sets))
        losses={k:sorted(set(v["covered"])-selected) for k,v in per.items() if k in selection["N0"]}
        assert all(len(v)==RULES["budget"] for v in [selection["N2"],*op["arms"].values()])
        result={"arms":per,"diagnosis":{role:diagnose(role,per["M0"],per["N1"],per["N2"]) for role in roles},
                "gained_vs_every_N0_eligible_for_visual_gate":no_old_coverage,"lost_vs_N0":losses,
                "n2_seconds":selection["seconds"],"n1_count":len(lib),
                "distinct_roots":dict(Counter(k for f in lib for k in f["image_dimensions"])),
                "multi_dimension_frame_count":sum(len(f["image_dimensions"])>1 for f in lib),
                "same_frame_nominations_removed_as_duplicate_presentations":sum(f["nomination_count"]-1 for f in lib),
                "multi_independent_group_frame_count":sum(len(f["independent_evidence_groups"])>1 for f in lib),
                "n3_read_cost":op["read_cost"]}
        for arm,r in per.items():
            frames=pset.get(arm,selection["N0"].get(arm,[]))
            obs=[meta["source"]["start"],*[f["t"] for f in frames],meta["source"]["end"]]
            reduced.append({"case":case,"arm":arm,"covered":len(r["covered"]),"denominator":r["role_count"],
                         "role_ids":" ".join(r["covered"]),"frames":len(frames),
                         "max_unobserved_spacing":max(b-a for a,b in zip(obs,obs[1:]))})
        if no_old_coverage:
            eligibility.append({"case":case,"roles":no_old_coverage,
                                "witnesses":{k:per["N2"]["witnesses"][k] for k in no_old_coverage}})
        if case==G3:
            invalid=target_regions(lib)
            result["region_review"]=invalid
            result["target_screen_by_dimension"]={k:dict(Counter(q["status"] for q in invalid if q["dimension"]==k))
                                                   for k in RULES["modules"]}
        if case==LONG:
            for method in ("fixed","overlap"):
                x=o/"P2_12"/method
                s=read(x/"summary.json")
                long_data[method]={k:measure(v,roles,case) for k,v in s["arms"].items()}
        dump(d/"evaluation.json",result)
        outputs[case]=result
        detail=" ".join(f'{k}:{len(per[k]["covered"])}/{per[k]["role_count"]}' for k in ("M0","N1","N2","N3","uniform","same_read_uniform"))
        print(case,detail,no_old_coverage,flush=True)
    dump(o/"P2_summary.json",{"cases":outputs,"candidate_gates_not_yet_manually_verified":eligibility,
                            "compared_arms":reduced,"long_window_results":long_data})
    with (o/"role_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(reduced[0]))
        w.writeheader()
        w.writerows(reduced)


if __name__=="__main__":
    main()

