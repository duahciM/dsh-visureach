"""E0 and observer-image scores; all private truth stays in the evaluator."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from if15_research.audit import pts_ledger
from if17_research.catalog import read, ref, sha
from p216_research.navigate import commit


def matches(t,ranges):
    return any(a-1e-6<=t<=b+1e-6 for a,b in ranges)


def checked(pointer):
    if sha(pointer["path"])!=pointer["sha256"]:
        raise ValueError("Input receipt changed")
    return read(pointer["path"])


def grade(receipt,reference,reviewed=None):
    hit={}
    observations=[]
    counts=Counter()
    for image in receipt["items"]:
        roles=[k for k,v in reference["roles"].items() if matches(image["seconds"],v["windows"])]
        new=[k for k in roles if k not in hit]
        for k in new:
            hit[k]={"ordinal":image["image"],"pts":image["pts"],"seconds":image["seconds"],"path":image["path"],
                    "in_catalog":image["in_catalog"],"kind":image["backend_origin"]}
        if new:
            verdict="verified_new_role"
        elif image["repeated_visible_pts"]:
            verdict="verified_repeated_pts_no_new_role"
        elif roles:
            verdict="verified_role_already_seen"
        elif image["seconds"]<reference.get("outside_early_scope",-1):
            verdict="confirmed_outside_requested_time"
        else:
            verdict=(reviewed or {}).get(image["pts"],"unjudged")
        counts[verdict]+=1
        observations.append({"goal":receipt["goal"]["id"],"case":receipt["goal"]["case"],"intent":receipt["goal"]["intent"],
            "arm":receipt["arm"],"ordinal":image["image"],"pts":image["pts"],"seconds":image["seconds"],"source_frame":image["source_frame"],
            "time_base":image["time_base"],"png_path":image["path"],"in_catalog":image["in_catalog"],"role_ids":"|".join(roles),
            "new_role_ids":"|".join(new),"verdict":verdict,"cumulative_roles":len(hit)})
    full=len(hit)==len(reference["roles"]) if reference["roles"] else None
    return {"roles_found":len(hit),"required_roles":len(reference["roles"]),"full":full,"first_complete_k":max(r["ordinal"] for r in hit.values()) if full else None,
            "hits":hit,"unverified":sorted(reference["roles"].keys()-hit.keys()),"image_verdicts":dict(counts),
            "complementary_observation":bool(full and len(hit)>1 and len({r["pts"] for r in hit.values()})>1)},observations


def e0(out,public,truth):
    books={}
    ledger={}
    for goal in public["goals"]:
        case=goal["case"]
        if case in books:
            continue
        row=public["library"][case]
        book=read(row["catalog"]["path"])
        books[case]={"archive":book["frames"],"source":book["source"],"coarse":read(row["coverage"]["path"])["candidate_frames"]}
        ledger[case]=pts_ledger(book["source"])[1]
    by={}
    for goal in public["goals"]:
        case,gid=goal["case"],goal["id"]
        ref=truth[gid]
        book=books[case]
        def pts(seq,stamp):
            return [r["pts"] for r in seq if matches(stamp(r),role["windows"])]
        roles={}
        for name,role in ref["roles"].items():
            archived=[r for r in book["archive"] if matches(r["t"],role["windows"])]
            roles[name]={"source_pts_in_reviewed_scope":pts(ledger[case],lambda r:float(r["pts_time"])),
                         "coarse_pts":pts(book["coarse"],lambda r:r["t"]),"archive_pts":[r["pts"] for r in archived],
                         "archive_dims_by_pts":{str(r["pts"]):r["image_dimensions"] for r in archived},"review_scope":role}
        by[gid]={"case":case,"source_capacity":len(ledger[case]),"archive_count":len(book["archive"]),"roles":roles}
    return by


def failure(role,scored,upper,decision,entries):
    ref=upper["roles"][role]
    catalogue=ref["archive_pts"]
    pts=set(ref["source_pts_in_reviewed_scope"])
    shots={r["frame"]["pts"]:r for r in entries.values()}
    candidate=[r for r in decision.get("candidates",[]) if r["pts"] in catalogue]
    when=ref["review_scope"]["windows"]
    windows=[q for q in decision.get("source_queries",[]) if any(q["trigger"]["interval"][0]<b and a<q["trigger"]["interval"][1] for a,b in when)]
    if role in scored["hits"]:
        stage="observed_interface_reference_match"
    elif not pts:
        stage="source_role_not_verified_in_sealed_scope"
    elif any(pts.intersection(q["internally_opened_pts"]) for q in windows):
        stage="source_pixels_read_internally_but_no_role_image_offered"
    elif any(r["decision"]=="offered_by_role" for r in candidate):
        stage="offered_catalog_member_not_confirmed_at_interface"
    elif catalogue and any(shots[p]["scope_eligible"] for p in catalogue):
        stage="available_catalog_candidate_not_selected_for_goal" if candidate else "legacy_route_has_no_relevant_delivery_proposal"
    elif catalogue:
        stage="within_frozen_archive_but_excluded_by_public_scope"
    elif windows:
        stage="raw_window_sampling_missed_or_time_scope_unresolved"
    elif ref["coarse_pts"]:
        stage="coarse_evidence_not_nominated_and_no_revisit"
    else:
        stage="reference_outside_coarse_pool_and_no_revisit"
    return {"phase":stage,"catalogue_pts":catalogue,"catalogue_evidence_dims":ref["archive_dims_by_pts"],"scope_window_proposals":len(windows),
            "eligible_in_index":sum(shots[p]["scope_eligible"] for p in catalogue),"catalogue_decisions":candidate}


def csv_write(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(out):
    truth=checked(read(out/"evaluation_only/lock.json")["reference"])["goals"]
    review_file=out/"evaluation_only/blind_visual_adjudication.json"
    reviewed=read(review_file)["confirmed_noncontributors"] if review_file.is_file() else []
    overrides={}
    for row in reviewed:
        overrides.setdefault(row["goal"],{})[row["pts"]]="confirmed_noncontributing_sample"
    public=read(out/"source_and_goal_manifest.json")
    evaluation=e0(out,public,truth)
    commit(out/"E0_catalog_and_source_upper.json",{"isolated_from_prediction":True,"goals":evaluation})
    algorithms=read(out/"routes_frozen.json")["routes"]
    manifest=read(out/"paired_uniform_manifest.json")
    assert algorithms.keys()==manifest["pairs"].keys()
    cases={}
    attr={}
    imagery=[]
    counts=[]
    coverage=[]
    audits=[]
    sem=[]
    streams={}
    for key,data in algorithms.items():
        a=checked(data["image_receipt"])
        b=checked(manifest["pairs"][key]["image_receipt"])
        d=checked(data["candidate_decisions"])
        metadata=checked(data["index"])
        gid=a["goal"]["id"]
        goal=a["goal"]
        assert a["source_sha256"]==b["source_sha256"] and a["public_range"]==b["public_range"] and goal==b["goal"]
        sa,rows0=grade(a,truth[gid],overrides.get(gid))
        sb,rows1=grade(b,truth[gid],overrides.get(gid))
        imagery.extend(rows0+rows1)
        streams[key]=data["image_receipt"]
        streams["U/"+key]=manifest["pairs"][key]["image_receipt"]
        ar=len({x["pts"] for x in a["items"] if x["in_catalog"]})
        archive=evaluation[gid]["archive_count"]
        n=a["N_visible"]
        valid=not(a["technical_abort"] or b["technical_abort"])
        entry={"case":goal["case"],"goal":gid,"cohort":goal["cohort"],"intent":goal["intent"],"arm":a["arm"],"N_visible":n,"U_N_visible":b["N_visible"],
               "N_unique_pts":a["N_unique_pts"],"N_repeated":a["N_repeated"],"N_first_batch":a["N_first_batch"],"N_source_revisit":a["N_source_revisit"],
               "roles":sa["roles_found"],"uniform_roles":sb["roles_found"],"required_roles":sa["required_roles"],
               "delta_roles":sa["roles_found"]-sb["roles_found"] if valid else None,"complete":sa["full"],"uniform_complete":sb["full"],
               "first_complete_k":sa["first_complete_k"],"uniform_first_complete_k":sb["first_complete_k"],"archive_frames_visible":ar,"archive_capacity":archive,
               "fraction_archive_seen":round(ar/max(1,archive),5),"fraction_original_seen":round(n/max(1,evaluation[gid]["source_capacity"]),5),
               "algorithm_proxy_text_tokens":a["metadata_text_tokens_clip_proxy"],"uniform_proxy_text_tokens":b["metadata_text_tokens_clip_proxy"],
               "technical_abort":not valid,"sdk_model_acks":0,"total_observable_approximate":False}
        counts.append(entry)
        byver=sa["image_verdicts"]
        coverage.append({**entry,"direct_role_supports":byver.get("verified_new_role",0)+byver.get("verified_role_already_seen",0),
                         "distinct_role_contributing_images":byver.get("verified_new_role",0),"verified_noncontributing_time":byver.get("confirmed_outside_requested_time",0),
                         "verified_noncontributing_sample":byver.get("confirmed_noncontributing_sample",0),
                         "repeated_without_role_increment":byver.get("verified_repeated_pts_no_new_role",0),"unjudged":byver.get("unjudged",0),
                         "all_catalog_delivered":ar==archive,"emission":d["mode"],"all_catalog_metadata_indexed":len(metadata["entries"])==archive,
                         "raw_windows_with_actual_samples":sum(bool(x["internally_opened_pts"]) for x in d.get("source_queries",[]))})
        cases[key]={"summary":entry,"algorithm":sa,"uniform":sb,"unbounded_archive_index_retained":True,"paired_by":manifest["pairs"][key]}
        attr[key]={role:failure(role,sa,evaluation[gid],d,metadata["entries"]) for role in truth[gid]["roles"]}
        audits.extend({"trial":key,**row} for row in d.get("candidates",[]))
        audits.extend({"trial":key,"kind":"source_window","query":r} for r in d.get("source_queries",[]))
        if d["mode"]!="goal_conditioned":
            audits.append({"trial":key,"mode":d["mode"],"frozen_legacy_backend":d.get("backend")})
        sem.extend({"trial":key,**r} for r in d.get("semantic_weak_candidates",[]))
    controls={}
    for gid,pointer in manifest["separate_natural_D6"].items():
        control,rows=grade(checked(pointer),truth[gid])
        imagery.extend(rows)
        streams["D6_natural/"+gid]=pointer
        controls[gid]=control
    commit(out/"paired_scored.json",{"trials":cases,"natural_D6":controls,"references":"independently sealed originals; unsupported positions remain unjudged"})
    commit(out/"failure_attribution.json",{"trials":attr,"hierarchy":"M7 membership, admissible candidate/offer, M8 expansion or source window, native interface"})
    csv_write(out/"role_by_visible_image.csv",imagery)
    csv_write(out/"per_goal_dimension_results.csv",counts)
    csv_write(out/"scan_vs_targeted_report.csv",coverage)
    for name,rows in (("candidate_suitability_audit.jsonl",audits),("weak_semantic_hypotheses.jsonl",sem),
                      ("index_entry_audit.jsonl",({"goal":gid,"record":row} for gid in truth
                        for row in read(out/"routes"/gid/"index.json")["entries"].values())),
                      ("visible_stream_receipts.jsonl",({"stream":key,**r} for key,p in streams.items() for r in checked(p)["items"]))):
        with (out/name).open("w",encoding="utf-8",newline="\n") as f:
            f.writelines(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows)
    commit(out/"final_receipt_index.json",streams)
    print(len(cases),"pairs and",len(imagery),"image judgments including unjudged",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    score(p.parse_args().out.resolve())
