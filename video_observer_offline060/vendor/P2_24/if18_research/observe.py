"""Real DeepSeek observations with host-executed model-selected tools."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from traceback import format_exc

from if11_research.agent import CLI,Harness
from p1.io import dump

from .video_tools import CAP,PROTOCOL,VideoTools,read,sha


ROUTE=PROTOCOL["route"]
QUESTIONS={
  "g3":"Determine the small object's directly visible positions at different source times. Cite the original frames and explain which parts of its behavior remain unobserved.",
  "short":"Describe the directly observable states and relative relations of the ball, the robotic arm and the tray as they change. Cite key original-frame times.",
  "long":"Outline the main phases observable in this long clip with original-frame timestamps. Investigate uncertain or visibly changing intervals first."
}
BASE="""You are a video observer. The host has bound one video, and you have not seen any image yet.
Use your observations and the task to decide what to inspect. Stop when adequately answered, evidence is insufficient, or further reads add no useful evidence. Avoid enumerating the entire catalog. There is no required image count.
For each delivered tile verify its original integer PTS, image side, and position relative to a visible reference. A metadata match is not a viewed frame. Do not assume physical identity or a continuous path across open gaps. Separate hypotheses and unresolved questions.
IMPORTANT: This SDK message has zero callable functions. Produce only a normal text JSON answer. The host reads numbered request records from that text, performs the requested actions itself, and returns results in the next message. Never issue an SDK function call.
Request codes and their args:
1: video span, directory count, and window metadata. args {}
2: search the metadata directory. args {start?,end? (source seconds),cursor?,limit?}. Reuse the filters and next_cursor for the next page.
3: view original frame images for selected integer pts. args {pts:[integer,...]}. Or view the next batch of an earlier source reread with args {range_token,cursor?,limit?}.
4: reread the original video. args {start,end,fps}. Use to resolve a concrete gap, width at most 4 seconds, fps 1–15.
5: optionally log an observation. args {pts:[integer,...],claim}. This checks pointers and does not validate meaning.
Write text JSON with facts [{claim,kind:"direct",pts:[integer,...]}], hypotheses [string], unresolved [string], update {change,reason}, goals [{id,question,motivation,pts:[integer,...],revision_reason?}], actions [{code:integer,why,args}], and optional stop {reason,status}, final_narration string. Always include facts, even empty. Ask up to two numbered actions with reasons. On the final reply leave actions empty and set stop status to answered, partial, insufficient, or no_new_evidence. Aim for under 260 words."""
MULTI="""Code 2 additionally accepts dimension (D1–D6), root, region_candidate, relationship, uncertainty, and region [x,y,width,height] in analysis pixels. Lists are permitted for named index fields. Low-level labels are search clues. Dimensions are local residual, pairwise difference, region-to-edge distance, image displacement, detection ambiguity and timing respectively."""
TIME="""In this session code 2 has the chronological directory only, with the same original images and source reread actions."""
SELF="""Choose your own investigation. Ask for a neutral overview first. You may inspect a small neutral preview before writing 1–3 open, falsifiable goals with motivation and observed PTS where available. Register goals before detailed original-source rereads. Reuse a goal ID and give a revision reason when revising it. Give a stop result for each goal."""
FREE="""Explore freely and provide a brief factual account of any noteworthy visually grounded observations. No particular event, object, or relation has been specified."""
SMOKE="""This is a workflow check. Try code 2 with a relevant non-time filter and limit 3. If it offers a continuation, ask for the next page. Use code 3 for the frames you choose. If your first valid image request returns a transient failure, retry. Use code 4 on a gap identified by you. Failed reads do not count as observations."""
ACTION_NAMES={1:"catalog_overview",2:"search_frames",3:"read_frames",4:"read_source_range",5:"record_observation"}


def condition(case,arm):
    start=BASE+"\n"+(TIME if arm=="B0" else MULTI)
    if arm=="C":
        return start+"\n"+SELF
    if arm=="B2":
        return start+"\n"+FREE
    return start+"\nUser question: "+QUESTIONS[case]+("\n"+SMOKE if arm=="A" else "")


def usage_totals(turns):
    wanted=("inputTokens","outputTokens","cacheReadTokens","cacheWriteTokens")
    sum_tokens=Counter()
    for a in turns:
        # The legacy SDK exposes each final usage twice through nested notifications.
        rows=a["response"]["result"]["usage"]
        if rows:
            row=rows[-1]
            sum_tokens.update({k:int(row.get(k,0)) for k in wanted if row.get(k) is not None})
    return dict(sum_tokens)


def update_goals(tracked,reply,turn,tools):
    for item in reply.get("goals",[]) or []:
        if not isinstance(item,dict) or not item.get("id") or not item.get("question"):
            continue
        key=str(item["id"])
        revisions=tracked.setdefault(key,[])
        if revisions and revisions[-1]["goal"]==item:
            continue
        refs=item.get("pts",[])
        cited=[x for x in refs if isinstance(x,int) and x in tools.delivered] if isinstance(refs,list) else []
        revisions.append({"turn":turn,"goal":item,"verified_delivered_refs":cited,
                          "delivered_before_goal":sorted(tools.delivered),
                          "explicit_revision_reason":item.get("revision_reason")})


def diagnostic_state(r,tools):
    facts=r.get("facts",[]) or []
    result=[]
    for f in facts:
        if not isinstance(f,dict):
            continue
        pts=f.get("pts",[])
        refs=pts if isinstance(pts,list) else []
        result.append({"claim":f.get("claim"),"pts":refs,"kind":f.get("kind"),
             "all_refs_previously_delivered":bool(refs) and all(isinstance(x,int) and x in tools.delivered for x in refs)})
    return result


def remember_pending(d,response,prompt,images,n,waiting,reports):
    a={"turn":n,"prompt_sha256":__import__("hashlib").sha256(prompt.encode()).hexdigest(),"prompt":prompt,
       "attachments_requested":[{"path":str(x),"sha256":sha(x)} for x in images],
       "tool_replies_received":[r["call"] for r in waiting],"response":response}
    reports.append(a)
    return a


def run_session(o,case,arm,rep):
    name=f"{case}_{arm}_{rep}"
    folder=o/"sessions"/name
    folder.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    tools=VideoTools(case,folder,arm!="B0",inject_read_failure=arm=="A")
    p=condition(case,arm)
    s={"case":case,"arm":arm,"replicate":rep,"route":ROUTE,"caps":CAP,"input_lock":tools.lock,
       "goal_condition":None if arm in ("C","B2") else QUESTIONS[case],
       "turns":[],"self_goal_history":{},"status":"started","source_filename_visible":False}
    dump(folder/"lock.json",{k:v for k,v in s.items() if k!="turns"}|{"first_message":p})
    live=None
    waiting=[]
    images=[]
    last=None
    safe_stop=False
    try:
        live=Harness(folder,CLI)
        s["session_id"]=live.session
        s["sdk"]=live.result(live.send("initialize",ROUTE|{"cwd":str(folder)}))
        for i in range(1,CAP["turns"]+1):
            receipt=live.turn(p,images,i)
            state=remember_pending(folder,receipt,p,images,i,waiting,s["turns"])
            tools.confirm(i,waiting,receipt["attachments"])
            answer=receipt["result"]["reply"]
            if not answer or receipt["result"]["tool_calls"]:
                raise RuntimeError("Unavailable structured DeepSeek reply or unwanted native tool call")
            update_goals(s["self_goal_history"],answer,i,tools)
            state["fact_source_checks"]=diagnostic_state(answer,tools)
            state["tool_enforcement_notes"]=[]
            last=answer
            dump(folder/f"turn_{i:02d}.json",state)
            todo=answer.get("actions",[]) or []
            has_final=isinstance(answer.get("final_narration"),str) and bool(answer["final_narration"].strip())
            if has_final and not todo:
                s.update(status="completed",stop_reason=answer.get("stop") or {"reason":"final_narration"})
                break
            if i>=CAP["turns"]:
                safe_stop=True
                s.update(status="bounded_without_final",stop_reason={"reason":"safety model-turn cap"})
                break
            waiting=[]
            images=[]
            feedback=[]
            if not isinstance(todo,list):
                todo=[]
            for a in todo[:CAP["actions_per_turn"]]:
                if len(tools.calls)>=CAP["calls"]:
                    safe_stop=True
                    feedback.append({"scope":"tool call safety cap reached","images_returned":0})
                    break
                if not isinstance(a,dict):
                    feedback.append({"error":"actions must contain objects"})
                    continue
                code=a.get("code")
                name_for_code=ACTION_NAMES.get(code) if isinstance(code,int) and not isinstance(code,bool) else None
                if arm=="C" and name_for_code=="read_source_range" and not s["self_goal_history"]:
                    feedback.append({"error":"Record the initial investigation goals before original-source reread.",
                                     "images_returned":0})
                    state["tool_enforcement_notes"].append("raw range held until goals registered")
                    continue
                r,pics=tools.dispatch(a|{"tool":name_for_code},i)
                waiting.append(r)
                images.extend(pics)
                feedback.append({"call":r["call"],"action_code":code,"status":r["status"],"result":r["reply"]})
            if len(todo)>CAP["actions_per_turn"]:
                feedback.append({"error":"Additional actions beyond the per-turn transport cap were not executed."})
            if has_final and todo:
                feedback.append({"note":"This response contained actions. Treat narration as provisional and update from new evidence."})
            reminder=""
            if len(tools.last_new)>CAP["no_new_evidence_reminder_after"] and all(x==0 for x in tools.last_new[-CAP["no_new_evidence_reminder_after"]:]):
                reminder=" If metadata or errors add no useful observations, either read a justified image batch or stop explicitly."
            closing=i>=CAP["turns"]-1 or safe_stop
            if closing:
                reminder+=" This is your last scheduled reply. Use the delivered evidence and give final_narration and stop without actions."
            p="Host result records (image sheets are attached in the same request order):\n"+json.dumps(feedback,ensure_ascii=False)+"\nUpdate observations and next action or stop."+reminder
            dump(folder/f"turn_{i:02d}.json",state)
    except Exception:
        s.update(status="failed",error=format_exc(),stop_reason={"reason":"provider/transport or schema failure"})
    finally:
        if live:
            live.close()
        s.update(final_reply=last,query_stats=tools.summary(),observations_recorded=tools.events,
                 calls=tools.calls,usage=usage_totals(s["turns"]),seconds=perf_counter()-started,
                 ended_on_safety_limit=safe_stop or (s.get("status")=="completed" and len(s["turns"])>=CAP["turns"]))
        dump(folder/"result.json",s)
    print(name,s["status"],s["query_stats"]["distinct_read_frames"],s["query_stats"]["tool_counts"],flush=True)
    return s


def smoke_checks(a):
    returned=[c for c in a["calls"] if c["returned_to_observer"]]
    images=[c for c in returned if c["frame_receipts"]]
    source=[c for c in images if c["tool"]=="read_source_range" or "range_token" in c["args"]]
    filters={"dimension","region_candidate","root","region","relationship","uncertainty"}
    error=[c for c in returned if c["status"]=="error" and "Injected transient" in c.get("internal_error","")]
    recovered=any(c["tool"]=="read_frames" and c["frame_receipts"] and c["call"]>e["call"] for c in images for e in error)
    return {"multidimensional_search":any(c["tool"]=="search_frames" and c["status"]=="returned" and
                    filters&c["args"].keys() and c["reply"]["frames"] for c in returned),
       "actual_catalog_images":any(c["tool"]=="read_frames" and "range_token" not in c["args"] for c in images),
       "actual_original_reread":bool(source),"tool_error_observed":bool(error),"recovered_after_error":recovered,
       "paginated":any(c["tool"]=="search_frames" and c["args"].get("cursor",0) not in (0,"0") for c in returned),
       "completed":a["status"]=="completed"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--phase",choices=("smoke","comparisons","retry"),required=True)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--smoke-replicate",type=int,default=1)
    ap.add_argument("--retry-session")
    x=ap.parse_args()
    o=x.out.resolve()
    (o/"sessions").mkdir(exist_ok=True)
    if x.phase=="smoke":
        s=run_session(o,"g3","A",x.smoke_replicate)
        check=smoke_checks(s)
        accepted=all(check[k] for k in ("multidimensional_search","actual_catalog_images","actual_original_reread",
                                          "recovered_after_error","paginated"))
        dump(o/"smoke.json",{"accepted":accepted,"checks":check,"actual_output":str(o/"sessions"/f"g3_A_{x.smoke_replicate}"/"result.json")})
        print(check,flush=True)
        return
    if not read(o/"smoke.json")["accepted"]:
        raise SystemExit("G3 smoke gate incomplete. Inspect failure before comparison.")
    if x.phase=="retry":
        if not x.retry_session:
            raise SystemExit("--retry-session is required")
        original=read(o/"sessions"/x.retry_session/"result.json")
        if original["status"]!="failed":
            raise SystemExit("Only failed observer sessions can use this one-time transport retry")
        run_session(o,original["case"],original["arm"],str(original["replicate"])+"_retry")
        return
    jobs=[(case,arm,j) for case,arm,n in PROTOCOL["stage_B"]+PROTOCOL["stage_C"] for j in range(1,n+1)]
    with ThreadPoolExecutor(max_workers=x.workers) as pool:
        pending=[pool.submit(run_session,o,c,a,r) for c,a,r in jobs]
        completed=[(c,a,r,f.result()["status"]) for (c,a,r),f in zip(jobs,pending)]
    dump(o/"sessions_manifest.json",{"statuses":[{"case":c,"arm":a,"replicate":r,"status":s} for c,a,r,s in completed]})


if __name__=="__main__":
    main()


