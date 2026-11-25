"""Execute all evidence routes, receive real PNGs and seal count-only pairs."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from types import FunctionType

from if17_research.catalog import read, ref, sha
from p216_research.navigate import checked_json, commit, delivery
from p215_research.retrieval import handoff
from p217_research.runner import trial
from p217_research.triggers import Case, make_candidates
from p218_research import candidate as inherited
from p218_research.observer import observe
from p218_research.primary import ReuseSource, deliveries

from .choose import choose, index
from .prepare import CONF, EXCLUDED_SHA256, OLD


class Source(ReuseSource):
    def __init__(self,book):
        super().__init__(book)
        for p in (OLD/"decoded"/book.case).glob("*.json"):
            i=int(p.stem)
            if i in self.points:
                continue
            row=read(p)
            assert self.ledger[i]["pts"]==row["pts"] and row["source_sha256"]==book.source["sha256"]
            checked_json(self.folder/p.name,row)
            self.points[i]=row


def for_observer(row):
    return {"path":row["path"],"sha256":row["attachment"]["sha256"],"seconds":row["seconds"],"pts":row["pts"],
            "time_base":row["time_base"],"source_frame":row["source_frame"],"kind":row["backend_origin"],"in_catalog":row["in_catalog"],
            "in_coarse_pool":row["in_coarse_pool"],"source_cache_reuse":row.get("source_cache_reuse")}


def visual_packet(book,source,q,pool,archive):
    points=source.request(q["interval"])
    source.ensure(points)
    raw=[]
    prior=None
    for i in points:
        rec=source.points[i]
        item=handoff(book,rec["native_image"],rec["native_image_sha256"],rec,prior)
        prior=item["seconds"]
        raw.append(item|{"kind":"source_window","in_catalog":rec["pts"] in archive,"in_coarse_pool":rec["pts"] in pool,
                         "rgb_proof":rec["rgb_verification"],"source_cache_reuse":True})
    measured=inherited.feedback(book,q,[r for r in raw if not r["in_catalog"]],{})
    by={r["pts"]:r for r in raw}
    classes=measured.get("typed_source_evidence",{})
    selected=[]
    for name,shortlist in classes.items():
        if not shortlist:
            continue
        if name=="comparable_earlier_state_confirmed":
            selected.append((max(shortlist),name))
        if name=="comparable_later_state_confirmed":
            selected.append((min(shortlist),name))
        if name=="new_distinguishable_crop_state":
            sig={r["pts"]:min(r["distance_to_anchor_crops"]) for r in measured["observations"] if "distance_to_anchor_crops" in r}
            selected.append((max(shortlist,key=lambda pts:sig.get(pts,0)),name))
    if not selected and raw:
        t=sum(q["interval"])/2
        best=min(raw,key=lambda r:(r["in_catalog"],abs(t-r["seconds"]),r["pts"]))
        selected.append((best["pts"],"intervening_view_needs_pixel_review"))
    return {"trigger":q,"requested_source_frame_indices":points,"internally_opened_pts":[r["pts"] for r in raw],
            "selected_roles":[{"pts":p,"role":role} for p,role in selected],"competing_interpretations":q["comparison"].get("competing_explanations",[]),
            "feedback_measurement":measured,"source_scope":[q["interval"]],"statement":"Observed image candidate; unknown outside these discrete sampled PTS"},[(by[p],k) for p,k in selected]


def run(out):
    main_lock=read(out/"public_input_lock.json")
    assert sha(main_lock["public"]["path"])==main_lock["public"]["sha256"]
    bundle=read(main_lock["public"]["path"])
    snapshot=read(OLD/"candidate_algorithm_freeze.json")["routes"]|read(OLD/"holdout_algorithm_freeze.json")["routes"]
    proc=FunctionType(trial.__code__,trial.__globals__|{"route":inherited.route,"feedback":inherited.feedback},"frozen_A3_trial")
    jobs=defaultdict(list)
    for task in bundle["goals"]:
        jobs[task["case"]].append(task)
    start=perf_counter()

    def each(pair):
        case,goals=pair
        entry=bundle["library"][case]
        assert entry["source"]["sha256"]!=EXCLUDED_SHA256
        book=Case(out,case,entry)
        source=Source(book)
        facts,missing=make_candidates(book)
        pool={r["pts"] for r in book.coverage}
        archive={r["pts"] for r in book.rows}
        routes={}
        counts=[]
        for goal in goals:
            gid=goal["id"]
            path=out/"routes"/gid
            path.mkdir(parents=True,exist_ok=True)
            scope=index(book,goal,facts,missing)
            checked_json(path/"index.json",{k:scope[k] for k in ("goal","entries","partitions","unread_eligible_count","comparison_endpoints_outside_library")})
            for arm in CONF["arms"]:
                started=perf_counter()
                decision_path=path/f"{arm}.json"
                folder=out/"streams"/arm/gid
                delivery_file=folder/"receipt.json"
                if decision_path.exists() and delivery_file.exists() and not read(delivery_file)["technical_abort"]:
                    decisions=read(decision_path)
                    observed=read(delivery_file)
                elif arm=="S0":
                    if goal["cohort"]=="retained":
                        frozen=snapshot[f"A3/{gid}"]["observer"]
                        assert sha(frozen["path"])==frozen["sha256"]
                        old=read(frozen["path"])
                        images=[for_observer(x) for x in old["items"]]
                        batches=[images[i:i+16] for i in range(0,len(images),16)]
                        observed=observe(folder,goal,arm,book.source,[],old["source_stop"],batches)
                        assert [x["pts"] for x in observed["items"]]==[x["pts"] for x in old["items"]]
                        decisions={"mode":"unchanged_A3_visible_order","input":frozen,"backend":snapshot[f"A3/{gid}"]["backend"]}
                    else:
                        where=out/"new_A3_backend"
                        rr=proc(book,source,goal,"A3",facts,missing,where,False,None)
                        observed=observe(folder,goal,arm,book.source,[],rr["run_state"],deliveries(rr))
                        decisions={"mode":"unchanged_A3_candidate_engine","backend":ref(where/case/"A3"/gid/"result.json")}
                else:
                    proposal,defer,sem=choose(book,scope,arm)
                    accepted=[]
                    raw_reasons={}
                    queries=[]
                    seen=set()
                    prior=None
                    for key in proposal:
                        item=delivery(book,key,prior)
                        prior=item["seconds"]
                        seen.add(item["pts"])
                        accepted.append(item|{"in_catalog":True,"in_coarse_pool":True})
                    if arm in ("S2","S3"):
                        for family in scope["selected_comparison_groups"]:
                            q=family["selected"]
                            item,kept=visual_packet(book,source,q,pool,archive)
                            queries.append(item|{"goalspecific_role":family["channel"],"independent_time_region":family["region"]})
                            for r,role in kept:
                                raw_reasons.setdefault(str(r["pts"]),[]).append({"purpose":role,"trigger_id":q["id"],"roots":q["comparison"]["basis_root_ids"]})
                                if r["pts"] not in seen:
                                    accepted.append(r)
                                    seen.add(r["pts"])
                    observed=observe(folder,goal,arm,book.source,accepted,"scoped_candidates_exhausted" if accepted else "no_supported_offer")
                    entries=[]
                    for key,row in scope["entries"].items():
                        accepted_reason=proposal.get(key,[])
                        entries.append({"frame_id":key,"pts":row["frame"]["pts"],"via":row["dimension_entries"],"time_region":row["time_region"],
                                        "decision":"offered_by_role" if accepted_reason else "indexed_for_later_or_other_goal",
                                        "reasons":accepted_reason or defer.get(key) or ["outside current public time scope" if not row["scope_eligible"] else "no independent evidence contribution selected for this goal"]})
                    decisions={"mode":"goal_conditioned","intent":goal["intent"],"candidates":entries,"source_queries":queries,
                               "scanned_all_archive_metadata":True,"raw_offer_roles":raw_reasons,"semantic_weak_candidates":sem,
                               "untouched_unopened_archive_count":len(archive-seen),"no_identity_contact_cause_or_absence_claim":True,
                               "elapsed_internal_seconds":perf_counter()-started}
                if not decision_path.exists():
                    commit(decision_path,decisions)
                name=arm+"/"+gid
                routes[name]={"image_receipt":ref(delivery_file),"candidate_decisions":ref(decision_path),"index":ref(path/"index.json")}
                counts.append({"arm":arm,"goal":goal,"source_sha256":observed["source_sha256"],"public_range":observed["public_range"],"n":observed["N_visible"],
                               "technical_abort":observed["technical_abort"]})
                print(name,observed["N_visible"],flush=True)
        return routes,counts
    with ThreadPoolExecutor(max_workers=3) as pool:
        result=list(pool.map(each,jobs.items()))
    counts=[v for _,rows in result for v in rows]
    checked_json(out/"count_only.json",{"pairs":counts,"no_algorithm_pts_or_reference_labels":True})
    checked_json(out/"routes_frozen.json",{"routes":{k:v for jobs,_ in result for k,v in jobs.items()},"count_only":ref(out/"count_only.json")})
    commit(out/"algorithm_run_time.json",{"seconds":perf_counter()-start,"scope":"backend and local adapter, not visual score"})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    run(p.parse_args().out.resolve())
