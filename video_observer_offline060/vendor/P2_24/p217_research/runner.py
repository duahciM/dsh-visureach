"""Execute shadow routes with resumable native PTS delivery and scoped feedback."""

import argparse
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from if15_research.audit import pts_ledger
from if17_research.catalog import read, ref, sha
from p215_research.retrieval import handoff
from p216_research.navigate import checked_json, commit, delivery
from v05.engine import export_native
from v07.audit import audit_images

from .prepare import CFG, EXCLUDED_SHA256
from .triggers import Case, feedback, make_candidates, route


class ExactSource:
    def __init__(self,book):
        self.book=book
        self.folder=book.out/"decoded"/book.case
        self.folder.mkdir(parents=True,exist_ok=True)
        if sha(book.source["path"])!=book.source["sha256"]:
            raise ValueError("Original source changed")
        self.time_base,self.ledger=pts_ledger(book.source)
        self.times=[float(r["pts_time"]) for r in self.ledger]
        self.points={int(p.stem):read(p) for p in self.folder.glob("*.json")}

    def request(self,interval):
        lo,hi=interval
        start=self.book.source["start"]
        fps=CFG["source_fps"]
        if not start<=lo<hi<=self.book.source["end"]:
            raise ValueError("Proposal outside the declared original source")
        requested=[]
        for q in range(ceil((lo-start)*fps-1e-8),ceil((hi-start)*fps-1e-8)):
            t=start+q/fps
            i=bisect_left(self.times,t-1e-7)
            if i<len(self.times) and lo<self.times[i]<hi and (not requested or requested[-1]!=i):
                requested.append(i)
        return requested

    def ensure(self,indices):
        absent=sorted(set(indices)-self.points.keys())
        token=None
        batch=CFG["source_decode_batch"]
        for pos in range(0,len(absent),batch):
            work=absent[pos:pos+batch]
            rows=[{"source_frame":i,"pts":self.ledger[i]["pts"],"time_base":self.time_base,"t":self.times[i],
                   "sample":j,"image":self.book.source["path"]} for j,i in enumerate(work)]
            token=self.folder/f"batch_{uuid4().hex}"
            images=export_native(self.book.source,rows,list(range(len(rows))),token)
            checked=audit_images(self.book.source,images)
            if len(checked)!=len(work) or not all(r["file_hash_equal"] and r["exact_source_rgb_equal"] for r in checked):
                raise ValueError("Independent source native RGB mismatch")
            ref_audit={r["source_frame"]:r for r in checked}
            for row in images:
                i=row["source_frame"]
                item=row|{"source_sha256":self.book.source["sha256"],"rgb_verification":ref_audit[i],"decode_batch":str(token)}
                commit(self.folder/f"{i:07d}.json",item)
                self.points[i]=item
        return len(absent)


def trial(book,source,goal,arm,facts,missing,base,pause,timeout):
    target=base/book.case/arm/goal["id"]
    target.mkdir(parents=True,exist_ok=True)
    plan=route(book,goal,arm,facts,missing)
    part=CFG["page"]
    keys=plan["anchor_ids"]
    plan["coverage_rate"]=CFG["source_fps"]
    for q in plan["proposals"]:
        q["requested_source_frame_indices"]=source.request(q["interval"])
        q["proposed_sampling"]={"fps":CFG["source_fps"],"clock":"relative to source start; strict interval interior"}
    targets=sorted({i for q in plan["proposals"] for i in q["requested_source_frame_indices"]})
    operations=[{"phase":"archive_page","ids":keys[pos:pos+part]} for pos in range(0,len(keys),part)]
    operations.extend({"phase":"source_page","source_frame_indices":targets[pos:pos+part]} for pos in range(0,len(targets),part))
    plan["operations"]=operations
    plan_path=target/"plan.json"
    checked_json(plan_path,plan)
    result_path=target/"result.json"
    if result_path.exists():
        last=read(result_path)
        if last["run_state"]!="technical_abort":
            return last
    events_dir=target/"events"
    events_dir.mkdir(exist_ok=True)
    archive_pts={r["pts"] for r in book.rows}
    pool_pts={r["pts"] for r in book.coverage}
    received=[]
    opened={}
    receipts=[]
    raw_pts={}
    acquired=0
    fault=None
    started=perf_counter()
    previous=None
    for position,work in enumerate(operations):
        page=events_dir/f"page_{position:06d}.json"
        if page.exists():
            event=read(page)
            if event["op"]!=work:
                raise ValueError("Checkpoint does not match locked operation")
        else:
            expired=timeout is not None and perf_counter()-started>=timeout
            if expired or (pause and acquired==1):
                fault={"phase":"watchdog" if expired else "intentional_checkpoint_pause","position":position}
                break
            event={"op":work,"operation_number":position,"images":[],"physical_new_source_frames":0,"duplicate_pts":[],
                   "stages":["proposed","dispatched"]}
            timer=perf_counter()
            prior=previous
            try:
                if work["phase"]=="archive_page":
                    for key in work["ids"]:
                        native=delivery(book,key,prior)
                        prior=native["seconds"]
                        event["images"].append(native|{"in_catalog":True,"in_coarse_pool":True})
                else:
                    event["physical_new_source_frames"]=source.ensure(work["source_frame_indices"])
                    event["stages"].append("exact_source_read_with_rgb_verification_or_cache")
                    for index in work["source_frame_indices"]:
                        raw=source.points[index]
                        if raw["pts"] in opened:
                            event["duplicate_pts"].append(raw["pts"])
                            continue
                        item=handoff(book,raw["native_image"],raw["native_image_sha256"],raw,prior)
                        prior=item["seconds"]
                        event["images"].append(item|{"kind":"source_window","in_catalog":raw["pts"] in archive_pts,
                            "in_coarse_pool":raw["pts"] in pool_pts,"decode_rgb_receipt":str(source.folder/f"{index:07d}.json")})
                event["stages"].append("sent_and_actually_opened_original_png_items")
                event["seconds"]=perf_counter()-timer
                commit(page,event)
                acquired+=1
            except (RuntimeError,OSError,ValueError,AssertionError,KeyboardInterrupt) as exc:
                fault={"phase":"native_decode_or_delivery","position":position,"type":type(exc).__name__,"error":str(exc),"partial_page":event}
                break
        for row in event["images"]:
            if row["previous_sent_seconds"]!=previous or row["pts"] in opened:
                raise ValueError("Repeated or broken source reported by checkpoint")
            previous=row["seconds"]
            received.append(row)
            opened[row["pts"]]=row
        for index in work.get("source_frame_indices",[]):
            raw_pts[index]=source.ledger[index]["pts"]
        receipts.append(ref(page))
    pending=operations[len(receipts):]

    results=[]
    samples={}
    for q in plan["proposals"]:
        requested=q["requested_source_frame_indices"]
        attempted=[index for index in requested if index in raw_pts]
        delivered=[opened[raw_pts[index]] for index in attempted if raw_pts[index] in opened]
        known_intermediate=[row for row in delivered if row["in_catalog"]]
        new=[row for row in delivered if not row["in_catalog"]]
        review=feedback(book,q,new,samples)
        state="technical_abort" if fault and len(requested)!=len(attempted) else "trigger_checked_gap_unresolved"
        if not attempted and not fault:
            state="trigger_no_sample_in_finite_window"
        pts=[row["pts"] for row in delivered]
        span=q["interval"]
        unsettled=sorted({*span,*[d["seconds"] for d in delivered]})
        revised=("Check which competing geometry or visibility explanations remain after the distinct source sample."
                 if review["result"]=="supports_difference" else
                 "Keep local change undecided until a comparable viewpoint is available."
                 if review["result"]=="clarifies_viewpoint" else q["investigation_gap"]["question"])
        results.append({"proposal_id":q["id"],"trigger_kind":q["trigger_kind"],"window":span,"status":state,
            "stages":{"proposal":{"fps":CFG["source_fps"],"requested_frame_indices":requested},
                      "dispatched":attempted,"decoded_native":[{"source_frame":i,"pts":raw_pts[i],"proof":str(source.folder/f"{i:07d}.json")} for i in attempted],
                      "delivered_original_pts":pts},"archive_overlap_pts":[r["pts"] for r in known_intermediate],
            "novel_library_external_pts":[r["pts"] for r in new],"coarse_pool_external_pts":[r["pts"] for r in new if not r["in_coarse_pool"]],
            "feedback":review|{"remaining_scope_unknown":[[a,c] for a,c in zip(unsettled,unsettled[1:])],"remaining_question":revised,
                     "status":"provisional","can_claim_physical_causation":False}})
    state="technical_abort" if fault else "finite_proposals_executed" if results else "no_supported_change_trigger"
    outcome={"run_state":state,"case":book.case,"goal":goal,"arm":arm,"pending":pending,"failure":fault,"opened":received,
             "plan":ref(plan_path),"event_receipts":receipts,"query_results":results,
             "remaining_goal_state":"unresolved_pending_independent_pixel_review","unchecked_scope":"outside opened PTS and all recorded unsettled intervals",
             "unbounded_options":"Direct original point/range may be requested independently of the automatic proposals.",
             "causal_claims":[],"identity_claims":[],"absence_claims":[],"elapsed_invocation_seconds":perf_counter()-started}
    if fault:
        commit(target/f"interrupted_{uuid4().hex}.json",outcome)
    commit(result_path,outcome)
    return outcome


def run(out,only,only_arm,alternate,pause,timeout):
    locked=read(out/"inputs_lock.json")["files"]
    for p in (out/"library/manifest.json",out/"public_goals.json",Path(__file__),Path(__file__).with_name("triggers.py"),Path(__file__).with_name("protocol.json")):
        if sha(p)!=locked[str(p.resolve())]:
            raise ValueError(f"Frozen predictor input differs: {p}")
    inventory=read(out/"library/manifest.json")
    goals=defaultdict(list)
    for goal in read(out/"public_goals.json"):
        if only is None or only==goal["id"]:
            goals[goal["case"]].append(goal)
    if not goals:
        raise ValueError("Unknown public goal")
    inventories=out/"comparisons"
    inventories.mkdir(exist_ok=True)
    start=perf_counter()
    def case_task(case):
        entry=inventory[case]
        if entry["source"]["sha256"]==EXCLUDED_SHA256:
            raise ValueError("User-excluded source")
        book=Case(out,case,entry)
        source=ExactSource(book)
        facts,missing=make_candidates(book)
        checked_json(inventories/f"{case}.json",{"observed_pairs":facts,"missing_library_endpoints":missing})
        files={}
        for goal in goals[case]:
            for arm in [only_arm] if only_arm else CFG["arms"]:
                result=trial(book,source,goal,arm,facts,missing,alternate,pause,timeout)
                key=f'{case}/{arm}/{goal["id"]}'
                files[key]=ref(alternate/key/"result.json")
                print(key,result["run_state"],len(result["opened"]),len(result["query_results"]),flush=True)
        return files
    with ThreadPoolExecutor(max_workers=3) as pool:
        trials={k:v for block in pool.map(case_task,goals) for k,v in block.items()}
    if not only and not only_arm:
        commit(out/"prediction_manifest.json",{"trials":trials,"expected":sum(map(len,goals.values()))*len(CFG["arms"]),
              "wall_seconds":perf_counter()-start,"read_policy":"Native PNG point deduplication, uncapped eligible ranges and pages",
              "execution_scope":"local source receiver; no VLM/Agent calls"})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--goal")
    parser.add_argument("--arm",choices=CFG["arms"])
    parser.add_argument("--pause-after-first-page",action="store_true")
    parser.add_argument("--watchdog-seconds",type=float)
    parser.add_argument("--trial-root",type=Path)
    arguments=parser.parse_args()
    out=arguments.out.resolve()
    if (arguments.trial_root or arguments.pause_after_first_page) and not (arguments.goal and arguments.arm):
        parser.error("Checkpoint demonstrations require one goal and route")
    run(out,arguments.goal,arguments.arm,arguments.trial_root.resolve() if arguments.trial_root else out/"trials",
        arguments.pause_after_first_page,arguments.watchdog_seconds)
