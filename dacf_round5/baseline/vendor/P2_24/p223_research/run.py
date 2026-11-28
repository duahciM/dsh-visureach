"""Lock a common pre-delivery batch, then run separate F, T and receiver stages."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p216_research.navigate import checked_json, delivery
from p217_research.triggers import Case, identifier, make_candidates
from p218_research.observer import observe
from p219_research.run import for_observer
from p219_research.score import checked
from p220_research.run import Exact
from p220_research.score import jsonl
from p221_research.recover import RawGrid, public_scope
from p222_research.extremes import transition
from p222_research.run import fresh

from .delivery import route
from .features import evidence_roots, figure1, figure2, node_bundle
from .fixture import run as fixtures
from .prepare import EXCLUDED_SHA256, OLD, PROTO
from .timeline import time_of, timeline


def freeze_batch(book, goal, w0, original_reasons, core, new_source_queries, legacy_revisits, catalog_pts):
    registered = {r["pts"]: r for r in book.rows}
    pool = {r["pts"] for r in book.coverage}
    scope = public_scope(book, goal)
    items = {}
    def put(row, roles, old_rank):
        stamp=row["pts"]
        if stamp in items:
            return
        old=registered.get(stamp)
        record={"candidate_id":identifier(book.source["sha256"],row["source_frame"],stamp,"pre-delivery"),
                "source_sha256":book.source["sha256"], "source_frame":row["source_frame"], "pts":stamp,
                "time_base":row["time_base"],"seconds":row["seconds"],"image":row,"in_catalog":old is not None,
                "coarse_member":stamp in pool,"available_source_original": True,"D0_order":old_rank,
                "historical_core":stamp in core,"dimensions":old["image_dimensions"] if old else [],
                "legacy_reasons":roles,"evidence_roots":sorted({rt for r in roles for rt in evidence_roots(r)}),
                "actual_observer_acknowledgement_this_step": False}
        assert abs(float(time_of(record))-record["seconds"])<=1.1e-6
        items[stamp]=record
    for i,row in enumerate(w0,1):
        put(row,original_reasons.get(str(row["pts"]),[]),i)
    for pts,reasons in catalog_pts.items():
        old=registered.get(pts)
        if old is None or not (scope[0]<=old["t"]<scope[1]):
            continue
        roles=[{"observation_role":r["role"],"root_ids":r["evidence_root_ids"],"origin":"frozen M7 local observation"} for r in old["observations"]]
        put(delivery(book,old["frame_id"],None)|{"in_catalog":True,"in_coarse_pool":True},roles+reasons,None)
    permissions={"source_sha256":book.source["sha256"],"public_read_range":list(scope),"same_for_all_arms":True,
                 "source_reader":"p217_research.runner.ExactSource / p220_research.run.Exact",
                 "authorized_legacy_queries":new_source_queries,"authorized_legacy_coarse_or_source_reads":legacy_revisits,
                 "raw_frame_pts_in_batch":[k for k,r in items.items() if not r["in_catalog"]],
                 "new_wave_reads": "only exact candidate originals; other legacy permissions retained for audit/replay, no E0-derived window"}
    return list(items.values()), permissions


def run(out):
    public=checked(read(out/"input_receipt.json")["public"])
    locked=checked(read(out/"input_receipt.json")["protocol"])
    assert all(sha(p)==h for p,h in locked["files"].items() if Path(p).parent==Path(__file__).resolve().parent)
    fixtures(out)
    old=read(OLD/"algorithm_lock.json")["routes"]
    segments=read(OLD/"video_segment_and_transition_audit.json")
    groups=defaultdict(list)
    for goal in public["goals"]:
        groups[goal["case"]].append(goal)

    def propose(pair):
        case,goals=pair
        book=Case(out,case,public["library"][case])
        assert sha(book.source["path"])==book.source["sha256"]!=EXCLUDED_SHA256
        cuts=segments.get(case) or transition(book)
        private={}
        if "W0_C4/"+goals[0]["id"] not in old:
            exact=Exact(book)
            facts,missing=make_candidates(book)
            grid=RawGrid(book,exact)
        for goal in goals:
            gid=goal["id"]
            old_trace=old.get("W0_C4/"+gid)
            mapped=defaultdict(list)
            if old_trace:
                receipt=checked(old_trace["observer"])
                t=checked(old_trace["trace"])
                observations=[for_observer(r) for r in receipt["items"]]
                assert [p["pts"] for p in observations]==t["actual_pts"]
                core={r["pts"] for r in observations[:t["prefix_images"]]}
                working={"proposals":checked(old["L-only/"+gid]["trace"])["candidate_priority"]}
                for mark in working["proposals"]:
                    mapped[mark["pts"]].append({"observation_role":mark["role"],"root_ids":mark["root_ids"],"origin":"P2-22 C1 proposal before representative selection"})
                reasons,queries,rereads=t["evidence_by_pts"],t["inherited_source_queries"],t["actual_revisit_receipts"]
            else:
                base,_,queries,decisions,_,(observations,reasons),rereads=fresh(book,goal,out,exact,facts,missing,grid)
                core={r["pts"] for r in base}
                for mark in decisions:
                    if "pts" in mark:
                        mapped[mark["pts"]].append({"observation_role":mark.get("role","catalog review"),"root_ids":mark.get("parents",[]),"origin":"fresh unchanged C1 proposal before representative arbitration"})
            batch,permissions=freeze_batch(book,goal,observations,reasons,core,queries,rereads,mapped)
            folder=out/"traces"/gid
            folder.mkdir(exist_ok=True,parents=True)
            file=folder/"pre_delivery.json"
            checked_json(file,{"goal":goal,"batch":batch,"source_read_permissions":permissions,"transition_status_source":cuts,
                               "proposal_uses_historical_receipt_only_for_exact_D0_handoff_replay":bool(old_trace),
                               "already_sent_by_current_experiment_count":0})
            private[gid]=ref(file)
        return private

    with ThreadPoolExecutor(max_workers=3) as pool:
        sealed={g:p for part in pool.map(propose,groups.items()) for g,p in part.items()}
    checked_json(out/"candidate_batch_lock.json",{"predictions_from_goal_only":True,"proposals":sealed,"four_score_or_deliver_layers_share_batches":True})
    jsonl(out/"candidate_proposals_before_delivery.jsonl",({"goal":g,**r} for g,p in sealed.items() for r in checked(p)["batch"]))

    def compute(pair):
        case,goals=pair
        book=Case(out,case,public["library"][case])
        archive={r["pts"]:r for r in book.rows}
        cache={}
        routes={};counts=[];spatials=[];figures=[];waves=[];navigation=[]
        for goal in goals:
            gid=goal["id"]
            pref=sealed[gid]
            record=checked(pref)
            batch=record["batch"]
            source_rights=record["source_read_permissions"]
            f0,f1=[],[]
            for item in batch:
                cid=item["candidate_id"]
                if item["in_catalog"] and cid in cache:
                    f2,a,b=cache[cid]
                else:
                    node=node_bundle(book,item,archive.get(item["pts"]))
                    f2=figure2(node)
                    a,b=figure1(node),figure1(node,f2)
                    assert a["I"]==b["I"] and a["information_roots"]==b["information_roots"]
                    assert not ({"seconds","pts","time_base"} & f2.keys())
                    if item["in_catalog"]:
                        cache[cid]=f2,a,b
                f0.append(a);f1.append(b)
                spatials.append({"goal":gid,**f2})
                figures.append({"goal":gid,"candidate":cid,"F0":a,"F1-abl":a,"F1":b,"no_time_inputs_to_F":True})
            trace=out/"traces"/gid
            snapshot=trace/"spatial_snapshot.json"
            checked_json(snapshot,{"F0":f0,"F1":f1,"proposal":pref,"stage":"EVALUATED_BEFORE_TIMELINE"})
            restored=checked(ref(snapshot))
            t0, cues0=timeline(batch,restored["F0"],record["transition_status_source"]["window_candidates"],book.source)
            t1, cues1=timeline(batch,restored["F1"],record["transition_status_source"]["window_candidates"],book.source)
            waves.append({"goal":gid,"F0_wave":t0,"F1_wave":t1,"spatial_snapshot":ref(snapshot)})
            navigation.extend({"goal":gid,"regime":f,**q} for f,seq in (("F0",cues0),("F1",cues1)) for q in seq)
            for arm in PROTO["arms"]:
                if arm=="D0":
                    selected=sorted((x for x in batch if x["D0_order"] is not None),key=lambda r:r["D0_order"])
                    ids={r["candidate_id"] for r in selected}
                    d=[{"candidate":x["candidate_id"],"pts":x["pts"],"source_frame":x["source_frame"],"decision":"send" if x["candidate_id"] in ids else "hold",
                        "reason":"frozen complete C4 handoff path", "support":x["D0_order"]} for x in batch]
                else:
                    evaluation=f0 if arm.endswith("_F0") else f1
                    cues=(cues0 if arm.endswith("_F0") else cues1) if arm.startswith("D2") else []
                    selected,d=route(goal,batch,evaluation,cues,book.source)
                target=out/"streams"/arm/gid
                r=observe(target,goal,arm,book.source,[x["image"] for x in selected],"route_finished" if selected else "no_grounded_offer")
                actual=[x["pts"] for x in r["items"]]
                assert r["technical_abort"] or actual==[x["pts"] for x in selected]
                path=trace/f"{arm}.json"
                checked_json(path,{"goal":goal,"arm":arm,"proposal":pref,"spatial_snapshot":ref(snapshot),"decision":d,
                    "actual_pts":actual,"native_source_reads_at_interface":[x for x in actual if x in source_rights["raw_frame_pts_in_batch"]],
                    "inputs_include_figure2_directly":False,"all_stages_frozen_before_score":True})
                routes[arm+"/"+gid]={"observer":ref(target/"receipt.json"),"trace":ref(path)}
                counts.append({"goal":goal,"arm":arm,"n":r["N_visible"],"public_range":r["public_range"],"source_sha256":r["source_sha256"],"technical_abort":r["technical_abort"]})
                print(arm,gid,len(batch),r["N_visible"],flush=True)
        return routes,counts,spatials,figures,waves,navigation

    with ThreadPoolExecutor(max_workers=3) as pool:
        packed=list(pool.map(compute,groups.items()))
    for name,idx in (("figure2_relation_features.jsonl",2),("figure1_information_relation.jsonl",3),
                     ("information_relation_timeline.jsonl",4),("wave_navigation_trace.jsonl",5)):
        jsonl(out/name,(row for job in packed for row in job[idx]))
    total={k:v for job in packed for k,v in job[0].items()}
    checked_json(out/"count_only.json",{"pairs":[row for job in packed for row in job[1]],"no_truth_or_conditioned_pts":True})
    checked_json(out/"algorithm_lock.json",{"routes":total,"count_input":ref(out/"count_only.json"),"pre_delivery_batches":ref(out/"candidate_batch_lock.json")})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out.resolve())
