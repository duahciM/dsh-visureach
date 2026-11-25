"""Run and resume the unbounded navigation study using real native image handoffs."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from if17_research.browse import read_original_range
from if17_research.catalog import read, ref, sha
from p1.io import dump
from p215_research.retrieval import goal_spec, handoff
from p215_research.semantic import Clip

from .graph import Book, R_ROUTES, agenda, build, entries, window_plan
from .prepare import CFG, EXCLUDED_SHA256, OLD


def commit(path, data):
    staging=path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    dump(staging,data)
    staging.replace(path)


def checked_json(path, data):
    if path.exists():
        if read(path)!=data:
            raise ValueError(f"Replay inputs or derived plan differ at {path}")
    else:
        commit(path,data)


def source_range(book,window):
    key=(book.source["sha256"],window["start"],window["end"],window["fps"])
    slug=sha256(repr(key).encode()).hexdigest()[:20]
    parent=book.out/"raw"/book.case/slug
    parent.mkdir(parents=True,exist_ok=True)
    pointer=parent/"receipt.json"
    cached=pointer.exists()
    if not cached:
        decoded=parent/f"decode_{uuid4().hex}"
        read_original_range(book.source_book,decoded,*key[1:])
        commit(pointer,ref(decoded/"read_receipt.json"))
    evidence=read(pointer)
    if sha(evidence["path"])!=evidence["sha256"]:
        raise ValueError("Raw receipt checksum mismatch")
    record=read(evidence["path"])
    if record["source_sha256"]!=key[0] or record["requested_time"]!=list(key[1:3]) or record["requested_fps"]!=key[3]:
        raise ValueError("Raw receipt source or time request mismatch")
    return record,evidence,cached


def delivery(book,key,prior):
    row=book.by[key]
    native=row["source_image"]
    return handoff(book,native["path"],native["sha256"],row,prior)|{"kind":"catalog","frame_id":key,
        "dimensions":row["image_dimensions"],
        "root_ids":sorted({r for obs in row["observations"] for r in obs["evidence_root_ids"]})}


def gaps(lo,hi,received):
    times=sorted({lo,hi,*[r["seconds"] for r in received]})
    return [[a,b] for a,b in zip(times,times[1:]) if b>a]


def make_plan(book,graph,goal,arm):
    channels,spec,words,slots=entries(book,goal,arm)
    nav=agenda(channels,{} if arm=="U0" else graph["edges"])
    # Early guidance, graph reach and whole-archive coverage are independently observable.
    alternate=["direct source time point or range query"]
    if arm!="U0":
        alternate.append("full chronological catalog route (available as U0)")
    raw=window_plan(book,goal,arm,graph,nav["generated_order"])
    size=CFG["transport_page_size"]
    order=nav["generated_order"]
    work=[{"kind":"catalog_page","ids":order[pos:pos+size],"cursor":pos,
           "next_cursor":pos+size if pos+size<len(order) else None} for pos in range(0,len(order),size)]
    work.extend({"kind":"source_window","window_index":pos} for pos,_ in enumerate(raw["windows"]))
    return {"goal":goal,"case":book.case,"arm":arm,"cold_overview":book.overview(R_ROUTES[arm]),
            "initial_queries":list(channels),"discovered_words":words,"query_family":spec["family"],
            "roles":{slot:{"hypothesis_members":ids,"state":"awaiting_original_image_review"} for slot,ids in slots.items()},
            "optional_route_frontier":alternate,"graph_enabled":arm!="U0","navigation":nav,"source":raw,"tasks":work}


def execute(book,graph,goal,arm,base,safety_seconds=None,pause_once=False):
    trial=base/book.case/arm/goal["id"]
    trial.mkdir(parents=True,exist_ok=True)
    plan=make_plan(book,graph,goal,arm)
    plan_path=trial/"plan.json"
    checked_json(plan_path,plan)
    tasks=plan["tasks"]
    received=[]
    opened=set()
    completed=[]
    failures=[]
    duplicate_pts=0
    attempted=[]
    events=trial/"events"
    events.mkdir(exist_ok=True)
    start=perf_counter()
    newly_completed=0

    def add(items):
        for item in items:
            expected=received[-1]["seconds"] if received else None
            if item["pts"] in opened or item["previous_sent_seconds"]!=expected:
                raise ValueError("Page ledger has repeated PTS or a broken previous-image link")
            opened.add(item["pts"])
            received.append(item)

    for pos,task in enumerate(tasks):
        event_path=events/f"page_{pos:06d}.json"
        if event_path.exists():
            record=read(event_path)
            if record["position"]!=pos or record["task"]!=task:
                raise ValueError("Checkpoint does not match the locked task")
        else:
            expired=safety_seconds is not None and perf_counter()-start>=safety_seconds
            if expired or (pause_once and newly_completed==1):
                failures.append({"stage":"technical_abort","position":pos,"cause":"watchdog" if expired else "requested_checkpoint_pause"})
                break
            record={"position":pos,"task":task,"delivered":[],"duplicate_pts":[],"operation_stages":[]}
            operation_start=perf_counter()
            try:
                if task["kind"]=="catalog_page":
                    record["operation_stages"].append("proposed_then_dispatched_catalog_page")
                    for key in task["ids"]:
                        if book.by[key]["pts"] in opened:
                            record["duplicate_pts"].append(book.by[key]["pts"])
                            continue
                        prior=(record["delivered"] or received)
                        record["delivered"].append(delivery(book,key,prior[-1]["seconds"] if prior else None))
                else:
                    window=plan["source"]["windows"][task["window_index"]]
                    record["operation_stages"].append("source_window_proposed_and_dispatched")
                    source,evidence,cached=source_range(book,window)
                    record.update(window=window,source_decode=evidence,cache_hit=cached,
                                  exact_sampled_pts=source["covered_exact_original_pts"])
                    record["operation_stages"].append("verified_decode_cache_opened" if cached else "source_decoded_and_native_pixels_verified")
                    for row in source["returned"]:
                        if row["pts"] in opened:
                            record["duplicate_pts"].append(row["pts"])
                            continue
                        prior=(record["delivered"] or received)
                        item=handoff(book,row["native_image"],row["native_image_sha256"],row,
                                     prior[-1]["seconds"] if prior else None)
                        record["delivered"].append(item|{"kind":"direct_source_range","window_index":task["window_index"]})
                record["operation_stages"].append("independent_native_attachments_opened_by_local_reader")
                record["elapsed_seconds"]=perf_counter()-operation_start
                commit(event_path,record)
                newly_completed+=1
            except (OSError,ValueError,RuntimeError,AssertionError,KeyboardInterrupt) as exc:
                failure={"stage":"native_or_source_operation","position":pos,"error":str(exc),"type":type(exc).__name__,
                         "partial_receipt":record}
                failures.append(failure)
                break
        add(record["delivered"])
        completed.append(ref(event_path))
        duplicate_pts+=len(record["duplicate_pts"])
        attempted.extend(task.get("ids",[]))

    pending=tasks[len(completed):]
    state={"goal":goal,"arm":arm,"case":book.case,"plan":ref(plan_path),"events":completed,"delivered":received,
           "failures":failures,"catalog_frames_attempted":attempted,"pending_frontiers":pending,
           "visited_query_keys":[f'{arm}:{p["entry"]}:{p["cursor"]}' for p in plan["navigation"]["metadata_pages"]],
           "source_operations_completed":sum(task["kind"]=="source_window" for task in tasks[:len(completed)]),
           "duplicate_pts_skipped":duplicate_pts,"resume_seconds":perf_counter()-start,
           "query_state":"incomplete_technical_abort" if failures else "goal_unresolved",
           "coverage_claim":{
               "catalog_state":"catalog_paths_exhausted" if len(attempted)==len(plan["navigation"]["generated_order"]) else "catalog_paths_unfinished",
               "scope":"catalog_complete" if len({row["frame_id"] for row in received if row["kind"]=="catalog"})==len(book.by) else "opened_frames",
               "unopened_catalog_ids":[key for key,row in book.by.items() if row["pts"] not in opened],
               "unchecked_source_ranges":gaps(book.source["start"],book.source["end"],received),
               "source_range_reviewed":"discrete original PTS only; window span and return rate in each event",
               "stop_reason":"technical abort with retained pending frontier" if failures else "configured navigation and source proposals exhausted; images await independent judgement",
               "false_identity_links_claimed":0,"false_absence_claims":0,
               "unmet_observation_roles":[{"role":role,"state":"unassessable"} for role in plan["roles"]]}}
    if failures:
        commit(trial/f"interruption_{uuid4().hex}.json",state)
    commit(trial/"result.json",state)
    return state


def run(out,target,one_arm,safety_seconds,pause_once,base):
    lock=read(out/"inputs_lock.json")
    for name in ("public_goals.json","library/manifest.json"):
        path=out/name
        if sha(path)!=lock["files"][str(path)]:
            raise ValueError(f"Frozen input changed: {path}")
    for path in (Path(__file__),Path(__file__).with_name("graph.py"),Path(__file__).with_name("protocol.json")):
        if sha(path)!=lock["files"][str(path)]:
            raise ValueError(f"Frozen navigation code changed: {path}")
    manifest=read(out/"library/manifest.json")
    goals=[g for g in read(out/"public_goals.json") if target is None or target==g["id"]]
    if not goals:
        raise ValueError("Unknown goal")
    grouped=defaultdict(list)
    for goal in goals:
        grouped[goal["case"]].append(goal)
    terms=sorted({term for goal in goals for term in goal_spec(goal["text"])["terms"]})
    vectors=dict(zip(terms,Clip(OLD,vision=False).text([f"a photo of a {word}." for word in terms]),strict=True)) if terms else {}
    graph_base=out/"graphs"
    graph_base.mkdir(exist_ok=True)
    def run_case(group):
        case,goals=group
        if manifest[case]["source"]["sha256"]==EXCLUDED_SHA256:
            raise ValueError("Cannot use user-excluded source")
        book=Book(out,case,manifest[case],vectors)
        graph=build(book)
        checked_json(graph_base/f"{case}.json",graph)
        results={}
        for goal in goals:
            for arm in [one_arm] if one_arm else CFG["arms"]:
                result=execute(book,graph,goal,arm,base,safety_seconds,pause_once)
                key=f'{case}/{arm}/{goal["id"]}'
                results[key]=ref(base/key/"result.json")
                print(key,result["query_state"],len(result["delivered"]),result["source_operations_completed"],flush=True)
        return results
    start=perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        trials={key:value for partial in pool.map(run_case,grouped.items()) for key,value in partial.items()}
    if not target and not one_arm:
        commit(out/"navigation_manifest.json",{"trials":trials,"expected":len(goals)*len(CFG["arms"]),
             "elapsed_seconds":perf_counter()-start,"query_vectors":len(vectors)})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--target")
    parser.add_argument("--arm",choices=CFG["arms"])
    parser.add_argument("--safety-seconds",type=float)
    parser.add_argument("--pause-after-first-page",action="store_true")
    parser.add_argument("--trial-root",type=Path)
    args=parser.parse_args()
    out=args.out.resolve()
    if (args.pause_after_first_page or args.trial_root) and not (args.target and args.arm):
        parser.error("A checkpoint experiment requires a single goal and arm")
    run(out,args.target,args.arm,args.safety_seconds,args.pause_after_first_page,
        args.trial_root.resolve() if args.trial_root else out/"trials")
