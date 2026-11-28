"""Keep delivery coverage and real DSH answer grounding in separate tables."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import re

from if17_research.catalog import read,ref
from p216_research.navigate import commit
from p219_research.score import checked,csv_write,e0,grade,matches
from p220_research.score import jsonl,metrics,stream_rows

from .prepare import OLD,PROTO,jsonl as read_jsonl


# Descriptive features of the reviewed role, independent of provider language.
# Each tuple is one required concept group; alternative names are pipe-separated.
CONCEPTS={
    "early_elevated": ("ball|sphere|red object","suspend|raised|lift|above|held|elevated|in the air|grip"),
    "late_elevated": ("ball|sphere|red object","suspend|raised|lift|above|held|elevated|in the air|grip"),
    "opening":("road|path|intersection|sidewalk|plaza|pavement|pedestrian|grass|walkway",),
    "central_activity":("pedestrian|people|person|walker|figures|walk|group",),
    "later_activity":("pedestrian|people|person|walker|figures|walk|group|traffic",),
    "early_working_surface":("surface|table|background|cloth|mat|work",),
    "later_seam_reference":("seam|crease|scratch|line|streak|diagonal|fold","surface|cloth|mat|table|bolt|screw|fastener"),
    "early_brown_door":("door|handle",),
    "middle_cabinet_window":("cabinet|doorway|shelf|window|opening|cupboard",),
    "late_brown_door":("door|handle",),
    "first_visible":("dot|mark|spot|circle|speck|point",),
    "last_blank_sample_before_visible_dot":("blank|no dot|without a dot|no mark|no spot|not visible|no black|unmarked|no circle|clear",),
    "frontal_view_right_of_fixed_fence_marks":("animal|bear|dark","fence|grid|right|front"),
    "sideward_profile_near_center":("animal|bear|dark","side|center|centre|middle|profile|fence|grid"),
    "body_partly_out_of_view_at_left":("animal|bear|dark","edge|left|partly|partial"),
    "late_left_boundary_view":("animal|bear|dark","edge|left|grid|mesh|wire|fence"),
    "opening_road_geometry":("road|lane|street|car|vehicle|traffic|highway|tunnel",),
    "later_road_occupancy":("road|lane|street|car|vehicle|traffic|highway|tunnel",),
    "notebook_initial_state":("notebook|book|cover|page|paper|laptop|folder",),
    "notebook_later_state":("notebook|book|cover|page|paper|laptop|folder",),
}


def score_o(out,pub,truth,broad):
    ceiling=e0(out,pub,truth)
    routes=read(out/"algorithm_lock.json")["routes"]
    global_u=read(out/"paired_uniform_global.json")["pairs"]
    public_u=read(out/"paired_uniform_public_scope.json")["pairs"]
    old_u_global=read(OLD/"paired_uniform_global.json")["pairs"]
    old_u_scope=read(OLD/"paired_uniform_public_scope.json")["pairs"]
    results={};records=[];fail={};all_streams={};visible=[];distance=[];broad_rows=[];rollup=defaultdict(Counter)
    for key,item in routes.items():
        arm,gid=key.split("/")
        base=checked(item["observer"]); source_goal=base["goal"];packet=checked(item["batch"])
        candidates={r["pts"]:r for r in packet["proposal"]["batch"]}
        cut=checked(item["decision"]) if "decision" in item else None
        d={r.get("pts"):r for r in cut["decision"] if r.get("pts") is not None} if cut else {}
        paths={"algorithm":item["observer"]}
        if key in global_u:
            paths.update({"global":global_u[key]["observer"],"public_scope":public_u[key]["observer"]})
        elif item["frozen_history"]:
            tag={"O0":"D0","O1":"D1","O4":"D2"}[arm]+"/"+gid
            paths.update({"global":old_u_global[tag]["observer"],"public_scope":old_u_scope[tag]["observer"]})
        grades={}
        for label,pointer in paths.items():
            stream=key if label=="algorithm" else label+"/"+key
            all_streams[stream]=pointer
            actual=checked(pointer)
            assert actual["goal"]==base["goal"] and actual["source_sha256"]==base["source_sha256"] and actual["N_visible"]==base["N_visible"]
            v,pts=grade(actual,truth[gid]);grades[label]=v
            distance.extend({"trial":stream,**r} for r in pts if r["new_role_ids"])
            visible.extend(stream_rows(stream,actual,{str(x["pts"]):[d.get(x["pts"],{})] for x in actual["items"]} if label=="algorithm" else {}))
            if gid in broad:
                wider,_=grade(actual,broad[gid]);broad_rows.append({"trial":key,"route":label,"strict":v["roles_found"],"broad":wider["roles_found"],"images":actual["N_visible"]})
        g=grades["algorithm"]
        case_roles={}
        for role,ev in ceiling[gid]["roles"].items():
            pts=set(ev["source_pts_in_reviewed_scope"]);eligible=sorted(pts&set(candidates))
            hit=g["hits"].get(role)
            layer="M7" if ev["archive_pts"] else "coarse" if ev["coarse_pts"] else "out_of_coarse"
            relevant=[d[p] for p in eligible if p in d]
            failure="delivered_positive" if hit else "proposal_absent" if not eligible else "investigate_not_released" if any(v.get("state")=="investigate" for v in relevant) else "frozen_or_current_delivery_selection"
            case_roles[role]={"phase":failure,"layer":layer,"candidate_pts":eligible,"decisions":relevant,"hit":hit}
        row={"arm":arm,"goal":gid,"case":source_goal["case"],"cohort":source_goal["cohort"],"intent":source_goal["intent"],
             "batch_n":len(candidates),"catalogue_total":ceiling[gid]["archive_count"],"catalogue_delivered":sum(x["in_catalog"] for x in base["items"]),
             "N_visible":base["N_visible"],"positive_roles":g["roles_found"],"required_roles":g["required_roles"],"complete":g["full"],
             "first_positive_k":metrics(g)["first_positive"],"first_complete_k":g["first_complete_k"],
             "global_roles":grades.get("global",{}).get("roles_found"),"public_scope_roles":grades.get("public_scope",{}).get("roles_found"),
             "uniform_status":"verified_equal_actual_count" if len(paths)==3 else "not_run_for_fresh_historical_reference",
             "freeze_reuse":item["frozen_history"]}
        records.append(row);results[key]={"summary":row,"scores":grades};fail[key]=case_roles
        for group in (arm+"/all",arm+"/"+("fresh" if packet["new_source"] else "historical"),arm+"/intent="+row["intent"]):
            counter=rollup[group];counter.update(goals=1,images=row["N_visible"],roles=row["positive_roles"],required=row["required_roles"],complete=int(bool(row["complete"])))
            if row["uniform_status"]=="verified_equal_actual_count":
                counter.update(global_roles=row["global_roles"],public_roles=row["public_scope_roles"],paired_goals=1)
    csv_write(out/"positive_role_scores.csv",records)
    if broad_rows:csv_write(out/"strict_vs_broad_scores.csv",broad_rows)
    jsonl(out/"visible_stream_receipts.jsonl",visible)
    jsonl(out/"offline_role_acquisition.jsonl",distance)
    commit(out/"offline_result.json",{"scores":results,"aggregate":rollup})
    commit(out/"failure_attribution.json",fail)
    commit(out/"final_offline_stream_index.json",all_streams)


def semantic(role,claim):
    groups=CONCEPTS.get(role)
    if not groups:
        return False
    value=claim.casefold()
    if role=="first_visible" and re.search(r"\b(no |without |not visible|unmarked|blank)",value) and not re.search(r"appears?|shows?|visible at|newly|now|added",value):
        return False
    return all(any(w in value for w in group.split("|")) for group in groups)


def score_a(out,pub,truth):
    root=out/"DSH_AGENT_SESSIONS"
    use={};attempts={}
    for cohort in ("historical","fresh"):
        p=root/("manifest_"+cohort+"_after_recovery.json")
        if not p.exists():p=root/("manifest_"+cohort+".json")
        if p.exists():use.update(read(p)["sessions"])
    extra=root/"extra_provider_recovery.json"
    if extra.exists():use.update(read(extra)["sessions"])
    for p in root.glob("*/A*/attempt_*/result.json"):
        r=read(p)
        attempts.setdefault(r["goal"]+"/"+r["arm"],[]).append(ref(p)|{"status":r["status"],"session_id":r.get("session_id")})
    if not use:
        commit(out/"model_use_audit.json",{"status":"DSH_NOT_EXECUTED","reason":"No formal DSH session manifest completed","attempts":attempts})
        return
    rows=[];details={};nav=[];delivered=[];tool_decisions=[];aggregate=defaultdict(Counter)
    for key,ptr in use.items():
        r=checked(ptr);gid=r["goal"];a=r["audit"];got=a.get("received_originals",[]);by_pts={q["pts"]:q for q in got}
        answer=a.get("final_answer") or {}; facts=answer.get("facts",[])
        facts=[f for f in facts if isinstance(f,dict)] if isinstance(facts,list) else []
        goalref=truth[gid]["roles"];covered={};grounded={};review=[]
        for row in got:
            for label,rule in goalref.items():
                if label not in covered and matches(row["seconds"],rule["windows"]):covered[label]=row
        for f in facts:
            claim=f.get("claim",f.get("observation","")); pts=f.get("pts",[])
            if isinstance(pts,int):pts=[pts]
            if not isinstance(pts,list):pts=[]
            supported_here=sorted({label for value in pts if isinstance(value,int) and not isinstance(value,bool) and value in by_pts
                                   for label,v in goalref.items() if matches(by_pts[value]["seconds"],v["windows"]) and semantic(label,str(claim))})
            for label in supported_here:grounded.setdefault(label,{"claim":claim,"cited_pts":pts})
            review.append({"claim":claim,"cited_pts":pts,"all_citations_seen":bool(pts) and all(isinstance(p,int) and p in by_pts for p in pts),
                           "matched_positive_roles":supported_here})
        logs=list(read_jsonl(Path(r["tool_log"]["path"]))) if r.get("tool_log") else []
        for x in logs:
            if x["tool"]=="wave.query":nav.append({"session":r["session_id"],"trial":key,**x})
            if x["tool"]=="delivery.request":tool_decisions.append({"session":r["session_id"],"trial":key,**x})
        admitted={x["call_ref"] for x in got}
        wave_calls=[x for x in logs if x["tool"]=="wave.query" and not x["error"]]
        guided=[]
        for i,call in enumerate(logs):
            if call["tool"]!="delivery.request" or call["call_ref"] not in admitted:
                continue
            prior=[x for x in logs[:i] if x["tool"]=="wave.query" and not x["error"]]
            pointed={q["pts"] for v in prior for link in v["returned_metadata"].get("hints",[]) for q in link["candidate_endpoints"]}
            used=sorted({x["pts"] for x in got if x["call_ref"]==call["call_ref"]}&pointed)
            if used:guided.append({"call":call["call_ref"],"earlier_wave_candidate_pts":used})
        delivered.extend({"trial":key,"session":r["session_id"],**v} for v in got)
        observation={"trial":key,"goal":gid,"arm":r["arm"],"status":r["status"],"session_id":r.get("session_id"),"attempt":r["attempt"],
                     "N_visible":a["visible_count"],"n_repeat_pts":a["visible_count"]-len(by_pts),"positive_roles_visible":len(covered),"roles_grounded_in_answer":len(grounded),
                     "role_total":len(goalref),"complete_visible":len(covered)==len(goalref),"complete_answer":len(grounded)==len(goalref),
                     "first_positive_image":min((x["ordinal"] for x in covered.values()),default=None),
                     "first_complete_image":max((x["ordinal"] for x in covered.values()),default=None) if len(covered)==len(goalref) else None,
                     "wave_calls":len(wave_calls),"later_image_requests_referencing_earlier_wave_endpoints":len(guided),
                     "input_tokens":a.get("provider_usage_tokens",{}).get("inputTokens"),"output_tokens":a.get("provider_usage_tokens",{}).get("outputTokens"),
                     "host_prompt_text_bytes":a.get("host_prompt_text_bytes"),"tool_return_text_bytes":a.get("tool_return_text_bytes"),
                     "final_claims_with_unseen_or_missing_pts":sum(not x["all_citations_seen"] for x in review)}
        rows.append(observation)
        details[key]={"summary":observation,"positive_roles_visible":covered,"grounded_answer_roles":grounded,"facts_review":review,"matched_wave_before_delivery":guided,
                      "final_answer":answer,"final_answer_text":a.get("final_answer_text"),"model_route_verified":r["provider_route_verified"]}
        aggregate[r["arm"]].update(sessions=1,completed=int(r["status"]=="completed"),images=a["visible_count"],visible_roles=len(covered),grounded_roles=len(grounded),
                                   catalog_navigation_use=len(wave_calls),completed_role_sets=int(len(grounded)==len(goalref)))
    csv_write(out/"agent_observation_scores.csv",rows)
    jsonl(out/"agent_visible_stream_receipts.jsonl",delivered)
    jsonl(out/"agent_navigation_query_receipts.jsonl",nav)
    jsonl(out/"agent_independent_delivery_decisions.jsonl",tool_decisions)
    commit(out/"agent_e0_review.json",{"analysis":details,"aggregate":aggregate,"grounding_rule":"Native received PTS in locked role window plus explicit positive image-description concepts. Conservative proxy; no resolution of inferred physical causality."})
    found=sum(r["status"]=="completed" and r["N_visible"]>0 for r in rows)
    commit(out/"model_use_audit.json",{"status":"DSH_EXECUTED" if found else "DSH_NOT_EXECUTED","provider":PROTO["route"],"accepted_formal_sessions":found,
           "planned_pairs":len(PROTO["agent_historical"])+len(PROTO["agent_new"]),"by_arm":aggregate,"formal_sessions":use,"attempts_retained":attempts,
           "validated_visual_preflight":ref(root/"preflight/account_flash_max/result.json"),"native_mcp_discovery_probe":ref(root/"native_mcp_preflight/A0/attempt_1/result.json"),
           "actual_transport":"DSH sessions proactively produce action JSON; host dispatches read-only tools and supplies verified RGB-native PNG blocks to the same DSH SDK conversation",
           "native_mcp_registered_in_formal_run":False,"model_turn_receipts_and_final_answers_saved":True})


def run(out,mode):
    pub=read(out/"frozen_p223_candidate_and_wave_manifest.json")
    lock=read(out/"E0_reference_lock.json")
    truth=checked(lock["strict"])["goals"]
    broad=checked(lock["broad"])["goals"]
    if mode in ("offline","all"):score_o(out,pub,truth,broad)
    if mode in ("agent","all"):score_a(out,pub,truth)
    modefiles=[]
    for s in ("offline","agent"):
        file=out/(s+"_navigation_query_receipts.jsonl")
        if file.exists():modefiles.append((s,file))
    jsonl(out/"navigation_query_receipts.jsonl",({"track":s,**r} for s,f in modefiles for r in read_jsonl(f)))
    modefiles=[]
    for s in ("offline","agent"):
        file=out/(s+"_delivery_decisions.jsonl" if s=="offline" else "agent_independent_delivery_decisions.jsonl")
        if file.exists():modefiles.append((s,file))
    jsonl(out/"independent_delivery_decisions.jsonl",({"track":s,**r} for s,f in modefiles for r in read_jsonl(f)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True,type=Path);p.add_argument("--mode",default="all",choices=("offline","agent","all"))
    args=p.parse_args();run(args.out.resolve(),args.mode)
