"""Audit tool transcripts and the evidence opportunities actually delivered."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import json

from if11_research.agent import walk
from if15_research.audit import pts_ledger
from if17_research.catalog import read,ref,sha
from p1.io import dump

from .video_tools import CASES,CAP,PRIOR

ANCHORS={
 "g3":{
    "right":[[4.399999,4.700001]],"sampled_gap":[[4.799999,5.300001]],"left":[[5.399999,5.500001]]},
 "short":{
    "table":[[.123999,2.124001]],"approach":[[2.624,4.124001]],"ball_claw":[[4.624,6.324]],
    "tray":[[7.124,10.124]]},
 "long":{
    "table_initial":[[0,2.5]],"early_lift":[[4.9,6.5]],"ball_in_tray":[[12,105]],
    "late_lift":[[112.3,115]],"table_final":[[119,119.959]]},
}


def refs(path):
    records=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return {n["text"] for d in records for n in walk(d)
            if n.get("type")=="text" and isinstance(n.get("text"),str)}


def anchor_opportunity(choices,policy):
    return {k:[r["pts"] for r in choices if any(a<=r["seconds"]<=b for a,b in w)] for k,w in policy.items()}


def qualified_frames(r):
    # Only acknowledged image-sheet receipts contribute to visual exposure.
    return {a["pts"]:a for c in r["calls"] if c["returned_to_observer"] for a in c["frame_receipts"]}


def classify_case(path,ledgers):
    r=read(path/"result.json")
    src=ledgers[r["case"]]["source"]
    tb,entries=ledgers[r["case"]]["ledger"]
    assert r["input_lock"]["source_sha256"]==src["sha256"]
    assert len(r["calls"])<=CAP["calls"] and len(r["turns"])<=CAP["turns"]
    if not r["turns"]:
        return r,{"calls_confirmed":0,"reason":"no observer turn"}
    outbound=refs(path/"runtime_events.jsonl")
    assert all(t["prompt"] in outbound for t in r["turns"])
    no_human_hints=True
    before={}
    real_pts={f["pts"] for f in entries}
    frozen_scope=ledgers[r["case"]]["frozen_scope"]
    missing_requests=[]
    index_ops=[]
    error_ops=[]
    raw_reads=[]
    actionable_confirmed=[]
    timing_checks=[]
    turns={t["turn"]:t for t in r["turns"]}
    for c in r["calls"]:
        t=turns[c["turn_requested"]]
        msg=t["response"]["result"]["reply"]
        proposed=msg.get("actions",[])
        assert any(a.get("why")==c["why"] and a.get("args",{})==c["args"] for a in proposed if isinstance(a,dict))
        if c["tool"]=="search_frames":
            used=set(c["args"])&{"dimension","root","region_candidate","region","relationship","uncertainty"}
            if r["arm"]=="B0":
                assert not used
            index_ops.append({"call":c["call"],"used_filters":sorted(used),"source_matches":c["reply"].get("matches"),
                              "page_cursor":c["args"].get("cursor",0),"delivered":c["returned_to_observer"]})
        if c["status"]=="error":
            error_ops.append({"call":c["call"],"status":c["reply"]["error"],"returned":c["returned_to_observer"]})
        for pts in c["reply"].get("not_returned_pts",[]):
            layer=("invalid_source_pts" if pts not in real_pts else
                   "not_in_frozen_pool" if pts not in frozen_scope else
                   "sampled_not_nominated" if not frozen_scope[pts] else "indexed_but_not_returned")
            missing_requests.append({"call":c["call"],"pts":pts,"layer":layer})
        if not c["returned_to_observer"]:
            continue
        received=turns[c["returned_on_turn"]]
        assert c["call"] in received["tool_replies_received"]
        assert all(sha(p["path"])==p["sha256"] and p in received["response"]["attachments"] for p in c["queued_images"])
        # Prompts expose actual tool content only. File names and annotation fields never from the wrapped tool payload.
        reply_json=json.dumps(c["reply"],ensure_ascii=False)
        assert Path(src["path"]).name not in reply_json
        assert "replacement_ground_truth" not in reply_json and "historical_N1_nominated" not in reply_json
        actionable_confirmed.append(c["call"])
        if c["frame_receipts"]:
            assert [x["pts"] for x in c["frame_receipts"]]==sorted(x["pts"] for x in c["frame_receipts"])
        for frame in c["frame_receipts"]:
            ident=entries[frame["source_frame"]]
            assert frame["source_sha256"]==src["sha256"] and frame["time_base"]==tb and frame["pts"]==ident["pts"]
            assert abs(float(Fraction(tb)*frame["pts"])-frame["seconds"])<=5.1e-7
            before[frame["pts"]]=frame
            timing_checks.append(frame["pts"])
        if c["tool"]=="read_source_range" and c["status"]=="returned":
            exact=read(path/f'call_{c["call"]:02d}'/"original"/"read_receipt.json")
            assert exact["source_sha256"]==src["sha256"]
            assert all(f["file_hash_equal"] and f["exact_source_rgb_equal"] for f in exact["native_audit"])
            raw_reads.append({"call":c["call"],"why":c["why"],"requested":c["args"],"full_decode_pts":exact["covered_exact_original_pts"],
                              "verified_returned_pts":[f["pts"] for f in c["frame_receipts"]],
                              "catalog_outside_decoded":exact["returned_outside_catalog"],"receipt":ref(
                                 path/f'call_{c["call"]:02d}'/"original"/"read_receipt.json")})
    confirmed=qualified_frames(r)
    assert set(before)==set(confirmed)==set(r["query_stats"]["delivered_pts"])
    snapshots=[]
    for t in r["turns"]:
        res=t["response"]["result"]["reply"] or {}
        cuts=[c for c in r["calls"] if c["returned_to_observer"] and c["returned_on_turn"]<=t["turn"]]
        delivered_by_t={f["pts"] for c in cuts for f in c["frame_receipts"]}
        claims=t.get("fact_source_checks",[])
        outside=[a for a in claims if a["kind"]=="direct" and not a["all_refs_previously_delivered"]]
        snapshots.append({"turn":t["turn"],"distinct_visual_frames_seen":len(delivered_by_t),
            "direct_with_bad_or_missing_source_pointers":outside,"claim_count":len(claims),
            "unresolved":res.get("unresolved",[]),"update":res.get("update"),
            "stop":res.get("stop"),"final_text":res.get("final_narration")})
    return r,{"calls_confirmed":len(actionable_confirmed),"pts_checks":len(timing_checks),
             "raw_receipts":raw_reads,"tool_errors":error_ops,"search_reads":index_ops,"unreturned_pts_attribution":missing_requests,
             "source_scope_counts":dict(Counter(r["frozen_scope"] for r in confirmed.values())),
             "role_opportunities_delivered":anchor_opportunity(confirmed.values(),ANCHORS[r["case"]]),
             "observation_snapshots":snapshots,"all_sent_text_verified":True,
             "no_annotation_or_source_filename_in_tool_content":no_human_hints}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    x=ap.parse_args()
    o=x.out.resolve()
    ledgers={}
    for tag,k in CASES.items():
        source=read(PRIOR/"P2_13"/k/"catalog.json")["source"]
        coverage=read(PRIOR/"P2_13"/k/"coverage.json")["candidate_frames"]
        ledgers[tag]={"source":source,"ledger":pts_ledger(source),"frozen_scope":{f["pts"]:f["in_catalog"] for f in coverage}}
        assert sha(source["path"])==source["sha256"]
    used={}
    evaluations={}
    for d in sorted((o/"sessions").iterdir()):
        if not (d/"result.json").is_file():
            continue
        run,a=classify_case(d,ledgers)
        evaluations[d.name]=a
        used[d.name]={"case":run["case"],"arm":run["arm"],"status":run["status"],
                     "query_stats":run["query_stats"],"model_usage":run["usage"],"stop_reason":run["stop_reason"],
                     "ended_on_safety_limit":run["ended_on_safety_limit"],"seconds":run["seconds"],
                     "result":ref(d/"result.json"),"self_goal_history":run["self_goal_history"]}
        print(d.name,run["status"],a.get("pts_checks"),flush=True)
    dump(o/"transport_and_source_audit.json",{"sessions":evaluations,
             "anchor_scope":"evaluation-only conservative nearby time windows from prior inspected sources; exposure is not understanding",
             "observed_role_clock":ANCHORS,"actual_source_hashes_valid":True})
    dump(o/"session_metrics.json",used)


if __name__=="__main__":
    main()

