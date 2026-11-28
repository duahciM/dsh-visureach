"""Audit sent P3-07 inputs and run one controlled same-image timestamp treatment."""
import argparse
from collections import Counter
import json
from pathlib import Path
from traceback import format_exc

from PIL import Image, ImageChops

from if10_research.p3_live import contact,query_requests
from if11_research.agent import CLI,Harness,walk
from if15_research.audit import checks_for
from if16_research.common import get_meta
from if16_research.evaluate import G3
from p1.io import dump

from identity_catalog_matching.common import read,sha

HERE=Path(__file__).resolve().parent
HISTORY=HERE.parent/"if16_research_20261120"
CONFIG=read(HERE/"protocol.json")["P3_08"]
ROUTE={k:CONFIG[k] for k in ("provider","model","reasoningEffort","maxTokens")}


def actual_text(path):
    result={}
    for line in path.read_text(encoding="utf-8").splitlines():
        for part in walk(json.loads(line)):
            t=part.get("text")
            if not isinstance(t,str) or part.get("type")!="text":
                continue
            if t.startswith("Describe this short silent video"):
                result.setdefault("first",set()).add(t)
            if t.startswith("The added attachments"):
                result.setdefault("followup",set()).add(t)
    assert set(result)=={"first","followup"} and all(len(v)==1 for v in result.values())
    return {k:next(iter(v)) for k,v in result.items()}


def tile_list(rows,label):
    return [{"ordinal":f"{label}{i+1:02d}","row":i//4+1,"column":i%4+1,"seconds":f["t"],
             "original_pts":f["pts"],"time_base":f["time_base"],"source_frame":f["source_frame"],
             "from_previous_tile_seconds":None if not i else round(f["t"]-rows[i-1]["t"],6)}
             for i,f in enumerate(rows)]


def verify_panel(rows,receipt,rebuild,title,scale,label):
    image=Path(receipt["path"])
    assert sha(image)==receipt["sha256"]
    assert all(a["pts"]<b["pts"] for a,b in zip(rows,rows[1:]))
    contact(rows,rebuild,title,scale=scale)
    with Image.open(image) as actual,Image.open(rebuild) as rebuilt:
        match=ImageChops.difference(actual.convert("RGB"),rebuilt.convert("RGB")).getbbox() is None
        detail={"dimensions":list(actual.size),"pixel_identical_rebuild":match}
    assert match
    return detail|{"receipt":receipt,"tiles":tile_list(rows,label),
            "labels":{"original_pts":True,"seconds_rounded_to_4_places":True,"tile_ordinal":False,
                      "explicit_intertile_gap":False,"time_base_in_image":False},
            "layout":"4 columns, row-major, chronological within sheet",
            "orientation":"Original native RGB rebuilt without horizontal mirror or display rotation"}


def audit(o):
    target=o/"P3_08"
    target.mkdir(parents=True,exist_ok=False)
    src=get_meta(G3)["source"]
    records={}
    for a in ("N2","uniform"):
        old=HISTORY/"P3"/a
        r=read(old/"result.json")
        content=actual_text(old/"runtime_events.jsonl")
        first=verify_panel(r["initial"],r["first"]["attachments"][0],target/f"{a}_rebuilt_first.png",
                         "Initial video frames",380,"I")
        after=[]
        reads=[w for w in r["reads"] if w["state"]=="read_original_video"]
        assert len(reads)==len(r["followup"]["attachments"])
        for i,(query,receipt) in enumerate(zip(reads,r["followup"]["attachments"]),1):
            p=Path(query["trace"]).parent/"keyframes.json"
            fs=read(p)["frames"]
            y=query["requested"]
            after.append(verify_panel(fs,receipt,target/f"{a}_rebuilt_query_{i}.png",
                 f'Query {i} | {float(y["start"]):.3f} to {float(y["end"]):.3f} | actual original source frames',350,f"R{i}."))
        native=checks_for(src,r["initial"],[f for w in reads
             for f in read(Path(w["trace"]).parent/"keyframes.json")["frames"]],[],[],len(r["initial"]))
        assert native["pass"]
        records[a]={"runtime_event_source":str(old/"runtime_events.jsonl"),"wire_text":content,"first":first,
                    "followup":after,"source_checks":native,
                    "time_labels_in_text":False,"explicit_time_order_in_text":False,
                    "gap_as_general_instruction_in_followup":True}
    assert records["N2"]["wire_text"]==records["uniform"]["wire_text"]
    dump(target/"old_input_audit.json",{"cases":records,"findings":{
         "missing_timestamps_hypothesis":"refuted for the actual rendered attachments",
         "missing_paired_text_table":"confirmed",
         "misordering_or_left_right_flip":"not present in delivered attachment pixels",
         "older_than_first_batch_followup":"N2 returned 3.3 to 4.0 appears in a later message, despite earlier time than initial 4.4 to 5.5."}})
    print("P3-07 actual wire text and both arms' initial/returned image pixels audited",flush=True)
    return records


def treatment(sheets):
    all_rows=[]
    lines=["Timing index only. Read tiles left to right, then top to bottom, within each attachment.",
           "Labels identify samples. Open intervals between samples have no delivered intermediate image.",
           "A return sheet can depict an earlier source time than a prior message."]
    for label,frames in sheets:
        for r in tile_list(frames,label):
            all_rows.append(r)
            lines.append(f'{r["ordinal"]} row {r["row"]} col {r["column"]} | {r["seconds"]:.6f} s | '
                         f'original PTS {r["original_pts"]} [{r["time_base"]}] | source frame {r["source_frame"]}')
    by_pts={}
    for row in all_rows:
        by_pts.setdefault(row["original_pts"],[]).append(row)
    previous=None
    lines.append("Global source order (duplicate samples share a line).")
    for pts,items in sorted(by_pts.items()):
        t=items[0]["seconds"]
        labels=", ".join(r["ordinal"] for r in items)
        gap="" if previous is None else f" | open unsampled gap ({previous:.6f}, {t:.6f})"
        lines.append(f"{t:.6f} s: {labels}{gap}")
        previous=t
    return "\n".join(lines)


def do_turn(live,j,report,stage,prompt,images):
    report[stage+"_input"]={"text":prompt,"attachments":[{"path":str(p),"sha256":sha(p)} for p in images]}
    ans=live.turn(prompt,images,j)
    assert [x["sha256"] for x in ans["attachments"]]==[x["sha256"] for x in report[stage+"_input"]["attachments"]]
    report[stage]=ans
    z=ans["result"]["reply"]
    if not z or ans["result"]["tool_calls"] or (j==2 and not z.get("final_narration")):
        raise RuntimeError(f"Invalid answer on round {j}")
    return z


def allocate(reports,src):
    candidates=[]
    selected=[]
    lo,hi=CONFIG["width_s"]
    # Round-robin by request ordinal, then earliest source time, without arm-specific priority.
    for arm,r in reports.items():
        req=r.get("requested",[])
        for i,q in enumerate(req if isinstance(req,list) else []):
            try:
                a,b=float(q["start"]),float(q["end"])
                rate=float(q["fps"])
                eligible=src["start"]-1e-8<=a<b<=src["end"]+1e-8 and lo-1e-7<=b-a<=hi+1e-7 and 1<=rate<=10
            except (KeyError,ValueError,TypeError):
                eligible=False
                a=b=None
            candidates.append({"arm":arm,"ordinal":i,"range":[a,b],"request":q,
                               "state":"eligible" if eligible else "invalid"})
    for c in sorted([c for c in candidates if c["state"]=="eligible"],key=lambda c:(c["ordinal"],c["range"],c["arm"])):
        if c["range"] in [s["range"] for s in selected]:
            c["state"]="shares_identical_selected_window"
            continue
        if len(selected)>=CONFIG["windows_shared_between_arms"]:
            c["state"]="shared_window_budget_exhausted"
            continue
        c["state"]="allocated"
        selected.append(c)
    q=[{"start":r["range"][0],"end":r["range"][1],"fps":CONFIG["reader_fps"]} for r in
       sorted(selected,key=lambda r:r["range"])]
    return q,candidates


def experiment(o,records):
    folder=o/"P3_08"
    m=get_meta(G3)
    src=m["source"]
    recipes=records["N2"]["wire_text"]
    historical=read(HISTORY/"P3"/"N2"/"result.json")
    initial=historical["initial"]
    initial_panel=Path(records["N2"]["first"]["receipt"]["path"])
    both={}
    observers={}
    shared=folder/"common_reads"
    shared.mkdir()
    first_docs=[("I",initial)]
    try:
        for name in ("original","ordinal_time"):
            d=folder/name
            d.mkdir()
            r=both.setdefault(name,{"condition":name,"route":ROUTE,"session":None})
            try:
                live=observers[name]=Harness(d,CLI)
                r.update(sdk=live.result(live.send("initialize",ROUTE|{"cwd":str(d)})),session=live.session)
                prompt=recipes["first"]
                if name=="ordinal_time":
                    prompt+="\n\n"+treatment(first_docs)
                z=do_turn(live,1,r,"first",prompt,[initial_panel])
                r["requested"]=z.get("retrieval",[])
            except Exception:
                r.update(status="failed",error=format_exc())
        requests,decisions=allocate(both,src)
        reads,images=query_requests({"retrieval":requests},shared,src,m)
        all_docs=first_docs.copy()
        for i,r in enumerate(reads,1):
            if r["state"]=="read_original_video":
                all_docs.append((f"R{i}.",read(Path(r["trace"]).parent/"keyframes.json")["frames"]))
        receipt=checks_for(src,initial,[f for _,v in all_docs[1:] for f in v],[],[],len(initial))
        assert receipt["pass"]
        dump(shared/"delivery.json",{"agent_proposals":decisions,"actual_reads":reads,
             "attachment_hashes":[{"path":str(p),"sha256":sha(p)} for p in images],
             "timed_sheets":{k:tile_list(v,k) for k,v in all_docs},"source_audit":receipt,
             "both_arms_receive_identical_post_query_images":True})
        follow=("The added attachments, if any, are original-source frames from the shared original-video reread. "
                "Each observer receives the same returned images. A request alone is not evidence. "
               +recipes["followup"].split(". ",1)[1])
        for name,r in both.items():
            if r.get("status")=="failed":
                continue
            try:
                prompt=follow if name=="original" else follow+"\n\n"+treatment(all_docs)
                do_turn(observers[name],2,r,"followup",prompt,images)
                r["status"]="completed"
            except Exception:
                r.update(status="failed",error=format_exc())
    finally:
        for name,live in observers.items():
            live.close()
        for name,r in both.items():
            dump(folder/name/"result.json",r)
            print(name,r.get("status","failed"),flush=True)
    complete=len(both)==2 and all(r.get("status")=="completed" for r in both.values())
    diffs={}
    if complete:
        a,b=[both[k] for k in ("original","ordinal_time")]
        for stage in ("first","followup"):
            x,y=(r[stage+"_input"] for r in (a,b))
            assert x["attachments"]==y["attachments"] and y["text"].startswith(x["text"]+"\n\nTiming index only.")
            diffs[stage]={"same_images_in_order":True,"only_text_timing_ledger_appended":True}
    dump(folder/"experiment_status.json",{"complete":complete,"matched_input_checks":diffs,
             "old_timestamps_exist":True,"model_conditions":ROUTE,"case":G3,
             "layout_condition":"Shared source reread jointly scheduled for identical follow-up pixels. Proposed ranges remain independently recorded.",
             "scope":"One observed pair; narrative adjudication independent of output eligibility."})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--audit-only",action="store_true")
    x=ap.parse_args()
    o=x.out.resolve()
    path=o/"P3_08"/"old_input_audit.json"
    records=read(path)["cases"] if path.is_file() else audit(o)
    if not x.audit_only:
        experiment(o,records)


if __name__=="__main__":
    main()
