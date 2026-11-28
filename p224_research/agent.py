"""Real DSH account-Flash observation via native MCP calls and PNG admissions."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
from queue import Queue
import random
import subprocess
import sys
from threading import Thread
from time import monotonic
from uuid import uuid4

from if11_research.agent import CLI,Harness
from if17_research.catalog import read,ref
from if20_research.direct import inspect_png
from PIL import Image
from p216_research.navigate import checked_json,commit
from p219_research.score import checked

from .prepare import HERE,PROTO,ROOT,jsonl
from .access import Video
from .mcp_server import append


ROUTE=PROTO["route"]
WATCHDOG=420
BASE="""You are an independent visual observer in a read-only, single-video session. You have not seen any images. Your task is below. Use the available tools to choose your own original pictures. catalog_search returns times, measured low-level features and candidate IDs, no image. delivery_request returns native frames for exact IDs or PTS and a specific observable reason. Prefer useful comparisons and avoid enumerating all candidates. There is no required image count. A metadata or tool request alone is no visual evidence. Take as many tool calls as needed to reach adequate evidence or an explicit partial stop. Keep physical identity, actions between frames, contact and causality uncertain unless directly established. Do not use unrelated tools. In your final answer return JSON only: {"facts":[{"claim":"specific directly visible detail","pts":[123],"kind":"direct"}],"unknown":["..."],"summary":"..."}. Cite only the integer PTS of images you actually received. Distinguish the starting view, changed appearance, or late comparison according to the task rather than assuming all clips need the same evidence."""
WAVE="""The additional wave_query tool returns only navigation clues for changes in regional or projected relation measurements. You may use its suggested time windows to direct an investigation. An index guess or equal-height marker change is never a visually observed action. For any helpful candidate decide separately whether its original image is needed."""


def json_answer(content):
    raw="\n".join(x.get("text","") for x in content if x.get("type")=="text")
    decoder=json.JSONDecoder()
    options=[]
    pos=0
    while (a:=raw.find("{",pos))>=0:
        try:
            j,taken=decoder.raw_decode(raw[a:])
            pos=a+taken
            if isinstance(j,dict) and ("facts" in j or "actions" in j):options.append(j)
        except ValueError:
            pos=a+1
    actual=next((j for j in options if j.get("actions") or j.get("facts") or j.get("summary")),None)
    return raw,actual or (options[0] if options else None)


def sdk_png(bits,original):
    """Losslessly remove container metadata before DSH can transcode."""
    with Image.open(BytesIO(bits)) as pic:
        rgb=pic.convert("RGB")
        out=BytesIO()
        Image.frombytes("RGB",rgb.size,rgb.tobytes()).save(out,format="PNG")
    wire=out.getvalue()
    metadata=inspect_png(wire)
    assert all(metadata[k]==original[k] for k in ("width","height","rgb_sha256"))
    return wire,metadata


class NativeHarness(Harness):
    def __init__(self,folder,patch):
        self.inbox=Queue();self.index=0;self.session=str(uuid4())
        self.console=(folder/"runtime.stderr.log").open("w",encoding="utf-8")
        self.wire=(folder/"raw_runtime_private.jsonl").open("w",encoding="utf-8")
        self.proc=subprocess.Popen(["node",str(CLI),"--profile","sdk","--patch",str(patch)],cwd=folder,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.console,encoding="utf-8",errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,env=os.environ|{"DSH_HOME":str(Path.home()/".dsh")})
        Thread(target=self.pump,daemon=True).start()

    def visual_run(self,prompt,images=()):
        blocks=[{"type":"text","text":prompt}]
        blocks.extend({"type":"image","mimeType":"image/png","data":base64.b64encode(bits).decode("ascii")} for bits in images)
        token=self.send("session/prompt",{"sessionId":self.session,"contentBlocks":blocks})
        receipt=None;events=[];stop=None;deadline=monotonic()+WATCHDOG
        while True:
            m=self.receive(deadline)
            if m.get("id")==token:
                if m.get("error"):
                    raise RuntimeError(str(m["error"]))
                receipt=m.get("result")
            p=m.get("params",{})
            if p.get("sessionId")!=self.session:
                continue
            if m.get("method")=="session.event":
                e=p["event"];events.append(e)
                if e["type"]=="turn/end":
                    stop=e["data"].get("reason")
            if m.get("method")=="session.status" and p.get("status")=="idle" and stop is not None:
                break
        return receipt,events,stop


def strip_private(value):
    if isinstance(value,list):
        return [strip_private(x) for x in value]
    if isinstance(value,dict):
        v={k:strip_private(x) for k,x in value.items() if k!="stream"}
        if str(v.get("type", "")).startswith("reasoning"):
            v.pop("text",None)
        if v.get("type")=="image" and isinstance(v.get("data"),str):
            v.pop("data")
        return v
    return value


def parse_events(events,logs):
    offers={d["call_ref"]:d for d in logs}
    token_counts=Counter();calls=[];results=[];admitted=[];contexts=[];reasons=0;headers=[]
    last_text="";answer=None
    after=[i for i,e in enumerate(events) if e["type"]=="assistant/message"]
    for i,e in enumerate(events):
        d=e.get("data",{}); typ=e["type"]
        if typ=="request/context":
            contexts.append({k:d.get(k) for k in ("provider","model","contextWindow")})
        if typ=="request/header":
            h=d.get("header",{})
            demand=[]
            def scan(j):
                if isinstance(j,dict):
                    demand.extend(v for k,v in j.items() if "reasoningeffort"==k.lower())
                    for v in j.values():scan(v)
                elif isinstance(j,list):
                    for v in j:scan(v)
            scan(h)
            headers.extend(demand)
        elif typ=="tool/call":
            calls.append({k:d.get(k) for k in ("name","arguments","callId","step")})
        elif typ=="tool/result":
            msg=d.get("message",{});blocks=msg.get("content",[])
            callref=None
            for t in blocks:
                if t.get("type")=="text":
                    try:
                        callref=json.loads(t.get("text","" )).get("call_ref") or callref
                    except (ValueError,AttributeError):pass
            offer=offers.get(callref,{});wanted=offer.get("image_offers",[])
            atoms=[t.get("attachment",{}) for t in blocks if t.get("type")=="image"]
            proof=["sha256:"+r["native"]["sha256"] for r in wanted]
            matched=[k.get("attachmentId") for k in atoms]==proof and all(
                a.get("width")==b["native"]["width"] and a.get("height")==b["native"]["height"] and a.get("bytes")==b["native"]["bytes"]
                for a,b in zip(atoms,wanted))
            observed=bool(matched and any(n>i for n in after))
            if observed:
                for atom,row in zip(atoms,wanted):
                    admitted.append({**row,"sdk_attachment":atom,"call_ref":callref,"ordinal":len(admitted)+1,"later_assistant_message":True})
            results.append({"call_ref":callref,"callId":msg.get("callId"),"matched_native_images":matched,"later_assistant_message":observed,
                            "mcp_produced_images":len(wanted),"sdk_image_attachments":len(atoms),"tool":offer.get("tool"),"error":msg.get("isError",False)})
        elif typ=="assistant/message":
            msg=d.get("message",{});content=msg.get("content",[])
            last_text,answer=json_answer(content)
            reasons+=sum(x.get("type", "").startswith("reasoning") and bool(x.get("text")) for x in content)
            token_counts.update({k:int(v) for k,v in d.get("usage",{}).items() if isinstance(v,int) and not isinstance(v,bool)})
    return {"provider_context":contexts,"declared_reasoning_effort_in_headers":headers,"reasoning_nonempty_blocks":reasons,
            "native_calls":calls,"tool_results":results,"received_originals":admitted,"visible_count":len(admitted),
            "provider_usage_tokens":dict(token_counts),"tool_return_text_bytes":sum(x["returned_text_bytes"] for x in logs),
            "final_answer_text":last_text,"final_answer":answer}


def session(root,goal,arm,pointer,attempt=1,prompt_override=None):
    folder=root/goal["id"]/arm/f"attempt_{attempt}"
    result_path=folder/"result.json"
    if result_path.is_file():
        return read(result_path)
    folder.mkdir(exist_ok=True,parents=True)
    packet=checked(pointer); assert packet["goal"]==goal
    log=folder/"mcp_calls.jsonl"
    wave=arm=="A1"
    arguments=["-B","-X","utf8","-m","p224_research.mcp_server","--packet",pointer["path"],"--log",str(log)]
    if wave:arguments.append("--wave")
    patch=[{"id":k,"disabled":True} for k in ("tool-pwsh","tool-bash","tool-fs","tool-fs-search","tool-web","tool-jobs","tool-skill","tool-workflow","tool-subagent","tool-subagent-fork","tool-subagent-control","tool-subagent-list-agents","tool-goal","tool-ralph","session-telemetry-otel","tool-todo")]
    patch.append({"id":"navigation-read-only-mcp","name":"@deepseek-ai/dsh-mcp-client","config":{
        "transport":"stdio","serverName":"video","command":sys.executable,"args":arguments,"cwd":str(ROOT),"failOnStartupError":True,
        "reconnect":{"enabled":False}}})
    patch_file=folder/"native_mcp_patch.yml"
    checked_json(patch_file,patch)
    prompt=prompt_override or BASE+("\n"+WAVE if wave else "")+"\nPublic task: "+goal["text"]
    record={"goal":goal["id"],"arm":arm,"packet":pointer,"attempt":attempt,"observe_route":ROUTE,"prompt":prompt,
            "transport":"native DSH MCP image attachments","prompt_text_bytes":len(prompt.encode("utf-8")),"status":"not_started"}
    host=None;start=monotonic();events=[]
    try:
        host=NativeHarness(folder,patch_file)
        record["session_id"]=host.session
        record["initialize"]=host.result(host.send("initialize",ROUTE|{"cwd":str(folder)}),50)
        receipt,events,stop=host.visual_run(prompt)
        record|={"prompt_receipt":receipt,"termination":stop,"status":"completed" if stop.get("kind")=="completed" else "transport_stop"}
    except (RuntimeError,TimeoutError,OSError) as e:
        record|={"status":"transport_error","error":str(e)}
    finally:
        if host:
            host.close()
        raw=folder/"raw_runtime_private.jsonl"
        if not events and raw.exists():
            events=[m["params"]["event"] for m in jsonl(raw) if m.get("method")=="session.event" and m["params"].get("sessionId")==record.get("session_id")]
        logs=list(jsonl(log)) if log.exists() else []
        final=parse_events(events,logs)
        # Audit on the exact session events; retain only safe observation and
        # provider metadata in the shareable event ledger.
        with (folder/"sdk_session_events.jsonl").open("w",encoding="utf-8") as fp:
            for event in events:
                fp.write(json.dumps(strip_private(event),ensure_ascii=False)+"\n")
        if raw.exists():
            raw.unlink()
        provider_ok=bool(final["provider_context"]) and all(c["provider"]==ROUTE["provider"] and c["model"]==ROUTE["model"] for c in final["provider_context"])
        record|={"wall_seconds":round(monotonic()-start,3),"provider_route_verified":provider_ok,"audit":final,
                 "tool_log":ref(log) if log.exists() else None,"model_events":ref(folder/"sdk_session_events.jsonl")}
        if record["status"]=="completed" and (not provider_ok or final["visible_count"]==0 or final["final_answer"] is None):
            record["status"]="completed_without_verified_visual_answer"
        commit(result_path,record)
    print(goal["id"],arm,record["status"],record["audit"]["visible_count"],record["wall_seconds"],flush=True)
    return record


HOST="""The SDK does not expose native tools in this profile. You still control the explicitly named read-only tools below by writing text JSON. The host executes only your actions and returns results in a subsequent message, with original PNG attachments if requested. Repeated/supplementary reads are allowed when justified. Use this shape on every reply: {"actions":[{"tool":"catalog.search","args":{"start":0,"end":10,"mode":"overview","page_size":12}}],"facts":[],"unknown":[]}. You may batch actions. catalog.search accepts optional start/end (source seconds), mode (overview, earliest, latest), cursor, dimension, page_size. delivery.request accepts {"frame_ids":["..."],"reason":"observable question about these frames"} or integer pts and reason. Catalog metadata is not visual observation. To finish, leave actions empty and return facts with precise image-only claims and integer PTS and a summary. Do not request a fake native function. Do not repeat metadata queries that already answered the same question."""
HOST_WAVE="""You additionally have wave.query with optional start/end, signal (all, rise, fall, plateau, equal_height_change, peak, gap, transition), cursor, page_size. Its output is metadata only and may suggest where to investigate. Decide any delivery.request separately with your observable purpose."""


def host_session(root,goal,arm,pointer,attempt=1):
    folder=root/goal["id"]/arm/f"attempt_{attempt}"
    finalpath=folder/"result.json"
    if finalpath.exists():return read(finalpath)
    folder.mkdir(exist_ok=True,parents=True)
    packet=checked(pointer); backend=Video(packet); log=folder/"host_tool_calls.jsonl"
    patch_file=ROOT/"if11_research/dsh_patch.yml"
    wave=arm=="A1"
    first=BASE+"\n"+HOST+("\n"+HOST_WAVE if wave else "")+f"\nSource span visible to the tools: {backend.scope[0]:.6f} to {backend.scope[1]:.6f} seconds.\nPublic task: "+goal["text"]
    record={"goal":goal["id"],"arm":arm,"packet":pointer,"attempt":attempt,"transport":"DSH model-selected host-dispatched tools, original PNG via SDK user attachments",
            "observe_route":ROUTE,"prompt":first,"prompt_text_bytes":len(first.encode("utf-8")),"status":"not_started"}
    live=None;received=[];turns=[];events=[];all_logs=[];pending=[];images=[];prompt=first;previous=[];start=monotonic();usage=Counter();routes=[];efforts=[]
    try:
        live=NativeHarness(folder,patch_file)
        record["session_id"]=live.session
        record["initialize"]=live.result(live.send("initialize",ROUTE|{"cwd":str(folder)}),50)
        while True:
            if monotonic()-start>WATCHDOG*2:
                record["status"]="wall_watchdog_partial";break
            receipt,stage,stop=live.visual_run(prompt,images)
            events.extend(stage)
            simple=parse_events(stage,[])
            usage.update(simple["provider_usage_tokens"])
            routes.extend(simple["provider_context"]);efforts.extend(simple["declared_reasoning_effort_in_headers"])
            user=[e["data"]["content"] for e in stage if e["type"]=="user/message" and any(x.get("text")==prompt for x in e["data"]["content"] if x.get("type")=="text")]
            atoms=[[x["attachment"] for x in xs if x.get("type")=="image"] for xs in user]
            native_expected=[{"attachmentId":"sha256:"+r["sdk_input"]["sha256"],"mediaType":"image/png",
                              **{k:r["sdk_input"][k] for k in ("width","height","bytes")}} for r in pending]
            assert atoms and all(a==native_expected for a in atoms), "SDK native attachment mismatch"
            observed=bool(simple["provider_context"] and any(e["type"]=="assistant/message" for e in stage)) or bool(any(e["type"]=="assistant/message" for e in stage) and routes)
            if observed:
                for row,attach in zip(pending,native_expected):
                    received.append(row|{"sdk_attachment":attach,"ordinal":len(received)+1,"same_turn_provider_replied":True,"actual_dsh_session_id":live.session})
            a=simple["final_answer"]
            turns.append({"turn":len(turns)+1,"prompt_message_id":receipt,"termination":stop,"prompt_text_bytes":len(prompt.encode("utf-8")),
                          "provider_route_context":simple["provider_context"],"declared_effort":simple["declared_reasoning_effort_in_headers"],
                          "sdk_user_attachment_admissions":native_expected,"sdk_reply_after_admissions":observed,
                          "model_answer":a,"visible_answer":simple["final_answer_text"],"actual_consumed_so_far":len(received)})
            commit(folder/"turns.json",turns)
            if stop.get("kind")!="completed":
                record["status"]="provider_stop_partial";break
            if not isinstance(a,dict):
                record["status"]="nonstructured_reply";break
            actions=a.get("actions",[])
            if not actions:
                record["status"]="completed";break
            canonical=json.dumps(actions,sort_keys=True,ensure_ascii=False)
            if len(previous)>=2 and previous[-2:]==[canonical,canonical]:
                record["status"]="repeated_identical_requests_partial";break
            previous.append(canonical)
            pending=[];images=[];results=[]
            for action in actions:
                stamp=uuid4().hex; tool=action.get("tool"); args=action.get("args",{}); produced=[]
                doc={"turn_requested":len(turns),"call_ref":stamp,"tool":tool,"arguments":args,"model_selected":True}
                try:
                    if tool=="catalog.search":reply=backend.catalog_search(**args)
                    elif tool=="wave.query" and wave:reply=backend.wave_query(**args)
                    elif tool=="delivery.request":
                        shown=backend.delivery_request(args.get("frame_ids"),args.get("pts"),args.get("reason",""))
                        for row,bits in shown:
                            wire,sent=sdk_png(bits,row["native"])
                            native=row|{"call_ref":stamp,"sdk_input":sent,"original_source_rgb_preserved":True,"sdk_input_bytes_same_as_original":sent["sha256"]==row["native"]["sha256"]}
                            pending.append(native);images.append(wire);produced.append(native)
                        reply={"frames":[{k:v for k,v in r.items() if k not in ("native","path","source_sha256")} for r in produced],
                               "note":"Original PNG attachments follow in this exact order"}
                    else:raise ValueError("Unknown tool for this arm; use your available actions")
                    doc["error"]=False
                except (ValueError,TypeError,KeyError,OSError) as ex:
                    reply={"error":str(ex),"images":0};doc["error"]=True
                response={"call_ref":stamp,"tool":tool,"result":reply}
                doc|={"returned_metadata":reply,"returned_text_bytes":len(json.dumps(response,ensure_ascii=False).encode("utf-8")),"image_offers":produced}
                append(log,doc);all_logs.append(doc);results.append(response)
            prompt="Host action results (original PNGs, if any, follow the frame records in order):\n"+json.dumps(results,ensure_ascii=False)+"\nUpdate your review. Return the same JSON schema; if you have enough evidence or reach an unresolved gap, finish with actions empty."
    except (RuntimeError,TimeoutError,OSError,AssertionError,AttributeError,ValueError,TypeError) as ex:
        record|={"status":"transport_error_partial","error":str(ex)}
    finally:
        if live:live.close()
        raw=folder/"raw_runtime_private.jsonl"
        if raw.exists():
            events=[m["params"]["event"] for m in jsonl(raw) if m.get("method")=="session.event" and m["params"].get("sessionId")==record.get("session_id")]
        with (folder/"sdk_session_events.jsonl").open("w",encoding="utf-8") as fp:
            for event in events:fp.write(json.dumps(strip_private(event),ensure_ascii=False)+"\n")
        if raw.exists():raw.unlink()
        verified=bool(routes) and all(c["provider"]==ROUTE["provider"] and c["model"]==ROUTE["model"] for c in routes) and "max" in efforts
        record|={"audit":{"provider_context":routes,"declared_reasoning_effort_in_headers":efforts,"received_originals":received,
            "visible_count":len(received),"final_answer":turns[-1]["model_answer"] if turns else None,"final_answer_text":turns[-1]["visible_answer"] if turns else None,
            "provider_usage_tokens":dict(usage),"tool_return_text_bytes":sum(x["returned_text_bytes"] for x in all_logs),
            "host_prompt_text_bytes":sum(x["prompt_text_bytes"] for x in turns),"model_requested_tool_calls":len(all_logs),"tool_role_native":False},
            "provider_route_verified":verified,"wall_seconds":round(monotonic()-start,3),"turn_count":len(turns),"tool_log":ref(log) if log.exists() else None,
            "model_events":ref(folder/"sdk_session_events.jsonl"),"turn_receipts":ref(folder/"turns.json") if turns else None}
        if record["status"]=="completed" and (not verified or not received):record["status"]="completed_without_verified_visual_answer"
        commit(finalpath,record)
    print(goal["id"],arm,record["status"],len(received),record["wall_seconds"],flush=True)
    return record


def run(out,cohort,workers,retry=False):
    bank=read(out/("historical_packets.json" if cohort=="historical" else "new_packets.json"))
    chosen=set(PROTO["agent_historical"] if cohort=="historical" else PROTO["agent_new"])
    goals=[g for g in bank["goals"] if g["id"] in chosen]
    random.Random(2424).shuffle(goals)
    jobs=[(g,a) for i,g in enumerate(goals) for a in (("A0","A1") if i%2==0 else ("A1","A0"))]
    root=out/"DSH_AGENT_SESSIONS"
    checked_json(root/("schedule_"+cohort+".json"),{"fixed_seed":2424,"launch_order":[[g["id"],a] for g,a in jobs],"watchdog_seconds":WATCHDOG})
    def one(pair):
        g,a=pair
        first=host_session(root,g,a,bank["packets"][g["id"]])
        if retry and first["status"]!="completed":
            return host_session(root,g,a,bank["packets"][g["id"]],attempt=2)
        return first
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result=list(pool.map(one,jobs))
    target=root/("manifest_"+cohort+("_after_recovery" if retry else "")+".json")
    checked_json(target,{"sessions":{r["goal"]+"/"+r["arm"]:ref(root/r["goal"]/r["arm"]/f"attempt_{r['attempt']}/result.json") for r in result},"statuses":dict(Counter(r["status"] for r in result))})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--cohort",choices=("historical","fresh"),required=True);p.add_argument("--workers",type=int,default=4)
    p.add_argument("--retry",action="store_true")
    a=p.parse_args();run(a.out.resolve(),a.cohort,a.workers,a.retry)
