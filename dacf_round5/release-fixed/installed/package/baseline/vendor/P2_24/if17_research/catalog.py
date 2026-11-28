"""Build time-sorted full catalogs over original-source frames without a display allocator."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from traceback import format_exc

from identity_catalog_matching.common import read,sha
from if16_research.common import frame_key,get_meta
from p1.io import dump
from v05.engine import export_native

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
IF16=ROOT/"if16_research_20261120"
CFG=read(HERE/"protocol.json")


def ref(p):
    return {"path":str(p.resolve()),"sha256":sha(p)}


def native_references(fs,src,library,d,existing=None):
    have=existing or {}
    needed={r["frame_id"] for r in library}
    indices=[i for i,f in enumerate(fs) if frame_key(src,f) in needed-have.keys()]
    assert len(indices)+len(needed&have.keys())==len(needed)
    add=export_native(src,fs,indices,d) if indices else []
    index={k:have[k] for k in needed&have.keys()}
    for f in add:
        index[frame_key(src,f)]={"path":f["native_image"],"sha256":f["native_image_sha256"],
          "source_frame":f["source_frame"],"pts":f["pts"],"time_base":f["time_base"],
          "width":src["probe"]["width"],"height":src["probe"]["height"],"native_orientation":True}
    return index


def observation_status(n):
    if n["dimension_id"]=="D5":
        return "unknown_or_competing_explanations"
    if n["dimension_id"]=="D2":
        return "observed_image_difference"
    if n["dimension_id"]=="D6":
        return "timing_reference"
    return "image_space_observation"


def region_id(row,n):
    if not n["applicable_region"]:
        return None
    token=json.dumps([row["frame_id"],n["applicable_region"]],separators=(",",":"))
    return "region-"+sha256(token.encode()).hexdigest()[:20]


def build_catalog(rows,src,native,d):
    index={g:defaultdict(list) for g in ("dimension","role","root","region_candidate","relationship","uncertainty","window","time_group")}
    pts,frames={},{}
    chronological=sorted(rows,key=lambda f:f["pts"])
    assert all(a["pts"]<b["pts"] for a,b in zip(chronological,chronological[1:]))
    anchors=[]
    previous=None
    linked=[]
    for r in chronological:
        assert frame_key(src,r)==r["frame_id"]
        k=r["frame_id"]
        when=r["t"]
        if not anchors or when-anchors[-1]>CFG["P2_13"]["soft_neighbor_seconds"]:
            anchors.append(when)
        group=f"near-{len(anchors):04d}"
        stamps={"source_image":native[k],"time_seconds":when,"chronological_ordinal":len(linked)+1,
                "soft_nearby_group":group,"open_gap_since_previous_catalog_image":
                [previous,when] if previous is not None else None}
        previous=when
        index["time_group"][group].append(k)
        obs=[]
        for n in r["nominations"]:
            tag=observation_status(n)
            box=n["applicable_region"]
            region=region_id(r,n)
            channels={"dimension":[n["dimension_id"]],"role":[n["role"]],
                      "root":n["evidence_root_ids"],"uncertainty":[tag]}
            if region:
                channels["region_candidate"]=[region]
            if n["dimension_id"]=="D3":
                channels["relationship"]=[n["role"]]
            if "analysis_window" in n:
                channels["window"]=[str(n["analysis_window"])]
            for namespace,keys in channels.items():
                for key in keys:
                    ledger=index[namespace][key]
                    if not ledger or ledger[-1]!=k:
                        ledger.append(k)
            obs.append({"dimension_id":n["dimension_id"],"role":n["role"],"status":tag,
                "region_xywh_analysis":box,"region_candidate":region,"evidence_root_ids":n["evidence_root_ids"],
                "source_comparison_frames":n["comparison_frame_ids"],"confidence_scope":n["confidence_scope"],
                "uncertainty":n.get("warning") or n.get("alternative_explanations"),
                "analysis_window":n.get("analysis_window"),"related_evidence_family":n["evidence_family"]})
        # Preserve each original record as well as the normalized/source-indexed view.
        fields={k:v for k,v in r.items() if k not in ("delivered_in_first_batch","deferred_reason","display_reason")}
        linked.append(fields|stamps|{"observations":obs})
        pts[str(r["pts"])]=k
        frames[str(r["source_frame"])]=k
    receipt={"source":src,"frame_count":len(linked),"frames":linked,
        "coordinate_contract":{"analysis_width":320,"analysis_height":max(8,round(src["probe"]["height"]*320/src["probe"]["width"])),
           "pixel_center_mapping":"native pixel = (analysis pixel + .5) * native extent / analysis extent - .5"},
        "catalog_range":[linked[0]["t"],linked[-1]["t"]] if linked else None,
        "delivery_records":"written to query/session receipts separate from this catalog",
        "membership":"at least one nomination; complete union by original source frame; same root retained without independent vote",
        "region_scope":"local observed region candidates; repeated geometry alone leaves object instance unknown"}
    dump(d/"catalog.json",receipt)
    dump(d/"index.json",{"source_sha256":src["sha256"],"pts":pts,"source_frame":frames,"filters":index,
          "timeline":[r["frame_id"] for r in linked]})
    return linked


def verify_lossless(prev,rows):
    one={r["frame_id"]:r for r in prev}
    two={r["frame_id"]:r for r in rows}
    assert len(one)==len(prev)==len(two)==len(rows) and one.keys()==two.keys()
    for key,r in one.items():
        assert all(r[col]==two[key][col] for col in ("nominations","independent_evidence_groups",
            "source_frame","pts","time_base","full_resolution_reference","unique_role_ids"))
    return {"exact_membership":True,"original_nomination_records_equal":True,"evidence_root_grouping_equal":True}


def build_case(case,d):
    start=perf_counter()
    meta=get_meta(case)
    fs,src=meta["frames"],meta["source"]
    assert sha(src["path"])==src["sha256"]
    prior=IF16/"P2"/case
    old=read(prior/"N1_library.json")
    oldsel=read(prior/"N2_selection.json")
    old_measured=read(prior/"local_measurements.json")
    previous_arms={f"IF16_N0_{k}":v for k,v in oldsel["N0"].items()}
    previous_arms["IF16_N2"]=oldsel["N2"]
    previous_arms["IF16_N3"]=read(prior/"N3"/"selection.json")["arms"]["N3"]
    allowed={frame_key(src,f) for f in fs}
    assert len(allowed)==len(fs) and all(r["frame_id"] in allowed for r in old)
    native=native_references(fs,src,old,d/"native")
    library=build_catalog(old,src,native,d)
    audit=verify_lossless(old,library)
    held={frame_key(src,f) for f in oldsel["N2"]}
    measurable={x["i"]:x["state"] for x in old_measured["foreground"]}
    indexed={r["frame_id"] for r in library}
    coverage=[{"frame_id":frame_key(src,f),"sample":i,"pts":f["pts"],"t":f["t"],
                  "source_frame":f["source_frame"],"in_catalog":frame_key(src,f) in indexed,
                  "D1_assessment":measurable.get(i,"unassessed")} for i,f in enumerate(fs)]
    transition=[{"frame_id":r["frame_id"],"t":r["t"],"pts":r["pts"],
               "source_frame":r["source_frame"],"historical_N1_nominated":True,
               "old_N2_displayed":r["frame_id"] in held,"available_in_catalog":True,
               "old_allocator_reason":old[i].get("deferred_reason"),
               "native_source":r["source_image"]} for i,r in enumerate(library)]
    dump(d/"coverage.json",{"source_start":src["start"],"source_end":src["end"],
        "native_source_frame_count":src["source_frame_count"],"candidate_frames":coverage,
        "source_frames_outside_frozen_pool":src["source_frame_count"]-len(fs),
        "portal_assessment_scope":"Fewer images than decoded source frames. Source outside the frozen observation pool remains unassessed.",
        "source_gaps_between_candidates":[[a["t"],b["t"]] for a,b in zip(fs,fs[1:])]})
    dump(d/"transfer_from_old_allocator.json",transition)
    dump(d/"historical_arms.json",{"arms":previous_arms,
      "N1_source":ref(prior/"N1_library.json"),"N2_source":ref(prior/"N2_selection.json"),"N3_source":ref(prior/"N3"/"selection.json")})
    roles=Counter(n["dimension_id"] for r in library for n in r["nominations"])
    tally={"case":case,"catalog_frames":len(library),"nominations":sum(roles.values()),"dimension_counts":roles,
           "multiple_dimensions":sum(len(r["image_dimensions"])>1 for r in library),
           "multiple_root_groups":sum(len(r["independent_evidence_groups"])>1 for r in library),
           "presentation_reference_reuse":sum(r["nomination_count"]-1 for r in library),"frozen_pool_frames":len(fs),
           "pool_frames_without_nominations":len(fs)-len(library),"native_source_frames_outside_pool":src["source_frame_count"]-len(fs),
           "old_unshown_now_retrievable":sum(not r["old_N2_displayed"] for r in transition),
           "disk_bytes":sum(p.stat().st_size for p in d.rglob("*") if p.is_file()),
           "seconds":perf_counter()-start,"invariants":audit}
    dump(d/"metrics.json",tally)
    print(case,tally["catalog_frames"],tally["nominations"],round(tally["seconds"],3),flush=True)
    return tally


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    x=p.parse_args()
    o=x.out.resolve()
    root=o/"P2_13"
    root.mkdir(exist_ok=False,parents=True)
    results={}
    failures={}
    for case in read(IF16/"P2"/"manifest.json")["cases"]:
        d=root/case
        d.mkdir()
        try:
            results[case]=build_case(case,d)
        except Exception:
            failures[case]=format_exc()
            print(case,failures[case],flush=True)
    dump(root/"run_manifest.json",{"cases":results,"failures":failures,"protocol":ref(HERE/"protocol.json"),
                   "new_selector_fixed_budget":None,"dimensions_display_slots":None,
                   "prior_manifest":ref(IF16/"run_manifest.json")})


if __name__=="__main__":
    main()
