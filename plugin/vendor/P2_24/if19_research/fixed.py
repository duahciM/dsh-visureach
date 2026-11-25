"""Fixed-image A0/A1 test with the shared DeepSeek route."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from traceback import format_exc

from if11_research.agent import CLI,Harness
from if17_research.catalog import read,ref,sha
from p1.io import dump

from .inputs import PROTOCOL,QUESTIONS

ROUTE=PROTOCOL["route"]
SHARED="""Only observe the attached original video frames and their source-time table. Some source-time intervals have no sample in this input.
Answer the user question for a reader who has not seen the images. Distinguish observations from inferences and unresolved intervals. Do not issue functions or ask for more frames.
Return text JSON with facts (up to six concise objects with claim, kind and optional pts as original integer source PTS), hypotheses (list), unresolved (list) and final_narration (short prose, used in the final reply). Keep the output under 350 words."""
ADD="""For each important claim, include a stable claim_id, the actual source frame number, source PTS and seconds, a visible_description of concrete pixels and relative positions, a visual_support statement naming which detail supports the claim and any limit or counterevidence, status (supported, contradicted, or insufficient), and open_questions. These fields describe externally checkable visual evidence. Include the same claim and pts fields. Keep claims and sources concise."""
FINAL="""Using the frames and your prior observations from this session, answer the same question in final_narration and include the facts and unresolved lists. There are no additional images in this round. Use the same answer format."""


def tokens(turns):
    keys=("inputTokens","outputTokens","cacheReadTokens","cacheWriteTokens")
    values=Counter()
    for x in turns:
        previous=None
        for u in x["receipt"]["result"]["usage"]:
            if u!=previous:
                values.update({k:int(u.get(k,0)) for k in keys if u.get(k) is not None})
            previous=u
    return dict(values)


def run_one(out,case,arm,trial,attempt):
    d=out/"A"/case/f"{arm}_{trial}"/f"attempt_{attempt}"
    d.mkdir(exist_ok=False,parents=True)
    packet=read(out/"fixed_input"/case/"input.json")
    image=Path(packet["image"]["path"])
    assert sha(image)==packet["image"]["sha256"]
    table=[{k:v[k] for k in ("ordinal","row","column","source_frame","pts","seconds","time_base")} for v in packet["positions"]]
    base=QUESTIONS[case]+"\n"+SHARED+"\nSource-time table, row-major by sheet tile:\n"+json.dumps(table)
    follow=FINAL+"\nUser question: "+QUESTIONS[case]
    prompts=[base,follow]
    if arm=="A1":
        prompts=[p+"\n"+ADD for p in prompts]
    started=perf_counter()
    r={"case":case,"arm":arm,"trial":trial,"attempt":attempt,"route":ROUTE,
        "input":ref(out/"fixed_input"/case/"input.json"),"transcript":[]}
    dump(d/"lock.json",r|{"source_input":packet["image"],"prompts":prompts,"images_per_turn":[[packet["image"]],[]]})
    live=None
    try:
        live=Harness(d,CLI)
        r["session_id"]=live.session
        r["sdk"]=live.result(live.send("initialize",ROUTE|{"cwd":str(d)}))
        for turn,p in enumerate(prompts,1):
            images=[image] if turn==1 else []
            z=live.turn(p,images,turn)
            assert [a["sha256"] for a in z["attachments"]]==([packet["image"]["sha256"]] if turn==1 else [])
            r["transcript"].append({"turn":turn,"sent":p,"receipt":z})
            answer=z["result"]["reply"]
            if not answer or z["result"]["tool_calls"] or (turn==2 and not answer.get("final_narration")):
                raise RuntimeError("Malformed, missing, or tool-contaminated comparison response")
        r["status"]="completed"
    except Exception:
        r.update(status="failed",error=format_exc())
    finally:
        if live:
            live.close()
        r.update(usage=tokens(r["transcript"]),seconds=perf_counter()-started)
        dump(d/"result.json",r)
    print(case,arm,trial,attempt,r["status"],flush=True)
    return r,ref(d/"result.json")


def job(out,c,a,n):
    trials=[]
    for attempt in (1,2):
        r,ref_out=run_one(out,c,a,n,attempt)
        trials.append(ref_out|{"status":r["status"]})
        if r["status"]=="completed":
            break
    return {"case":c,"arm":a,"replicate":n,"attempts":trials,"status":r["status"],"accepted":ref_out if r["status"]=="completed" else None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workers",type=int,default=3)
    args=ap.parse_args()
    o=args.out.resolve()
    lock=read(o/"fixed_input"/"manifest.json")
    assert lock["one_common_input_per_case"]
    keys=[(c,a,n) for c in PROTOCOL["stage_A"]["cases"] for n in range(1,1+PROTOCOL["stage_A"]["replicates_per_arm"]) for a in ("A0","A1")]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        found=[pool.submit(job,o,*k) for k in keys]
        rows=[fut.result() for fut in found]
    paired={}
    for case in PROTOCOL["stage_A"]["cases"]:
        valid=[r for r in rows if r["case"]==case and r["status"]=="completed"]
        assert len({read(r["accepted"]["path"])["input"]["sha256"] for r in valid})<=1
        complete={a:sum(r["arm"]==a for r in valid) for a in ("A0","A1")}
        paired[case]=complete
    dump(o/"A"/"manifest.json",{"groups":rows,"complete_by_case":paired,
         "each_case_has_comparable_same_image_arms":all(min(x.values())>0 for x in paired.values()),
         "treatment_difference":"Only requested visible detail, explicit source references, limits and claim support. Same two prompts otherwise."})


if __name__=="__main__":
    main()
