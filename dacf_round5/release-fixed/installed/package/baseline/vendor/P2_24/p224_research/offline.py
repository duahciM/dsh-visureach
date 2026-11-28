"""Goal-driven offline queries and independent release conditions on one frozen batch."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from if17_research.catalog import read,ref
from p216_research.navigate import checked_json
from p217_research.triggers import visual
from p218_research.observer import observe
from p219_research.score import checked
from p220_research.score import jsonl
from p220_research.uniform import run as uniform
from p223_research.delivery import route as earlier_d1

from .access import Video
from .prepare import PROTO


POL=PROTO["o_policy"]


def views(a,b):
    return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))


def read_low(video,r,cache):
    k=r["candidate_id"]
    if k not in cache:
        w=video.source["probe"]["width"];h=video.source["probe"]["height"]
        cache[k]=visual(r["image"]["path"],round(h*320/w))[0]
    return cache[k]


def patch_stats(video,a,b,region,cache):
    p=read_low(video,a,cache);q=read_low(video,b,cache)
    x,y,w,h=map(int,region["box"])
    left,right=p[y:y+h,x:x+w],q[y:y+h,x:x+w]
    if not left.size or left.shape!=right.shape:
        return None
    delta=np.mean(np.abs(left.astype(np.float32)-right))
    chroma=float(np.mean(right.max(axis=2)-right.min(axis=2)))
    return {"mean_abs_rgb_difference":float(delta),"candidate_roi_chroma":chroma,
            "region_analysis_xywh":region["box"],"region_roots":region["roots"],"compared_candidate":a["candidate_id"]}


def stage(video,base,use_complements):
    packet=video.packet;need=video.goal["text"].casefold();intent=video.goal["intent"]
    rows=video.by_id;axis=video.features;selected={r["candidate_id"]:r for r in base if r["candidate_id"] in rows}
    actions=[];requests=[];query=[];cache={};extra={}
    explanation={k:{"state":"deliver","reason":"existing visible goal-specific or comparable baseline observation"} for k in selected}
    if intent not in ("open","scene","recurrence") or not selected:
        return list(selected.values()),requests,[{"candidate":k,**explanation.get(k,{"state":"eligible","reason":"single, boundary, relation or grounded-abstention path unchanged"})} for k in rows],query

    def release(r,reason,measurement,relates):
        k=r["candidate_id"]
        observation={"candidate":k,"pts":r["pts"],"source_frame":r["source_frame"],"state":"deliver","reason":reason,"selection_evidence":measurement,"related_navigation_or_request":relates}
        explanation[k]=observation
        if k not in selected:
            proof,_=video.delivery_request([k],reason=reason)[0]
            requests.append(proof|{"related_navigation_or_request":relates,"selection_evidence":measurement,"native_verified_before_observer":True})
            selected[k]=r

    lo,hi=video.scope
    fixed=[lo+(hi-lo)*j/POL["coverage_zones"] for j in range(POL["coverage_zones"]+1)]
    for band,(a,b) in enumerate(zip(fixed,fixed[1:])):
        if intent!="open":
            continue
        # Natural competing local-activity hypotheses in each source interval.
        # A cue only creates a finite investigation list. Source-local RGB
        # comparison and an independent unmet visual reason decide release.
        before=len(selected)
        available=video.wave_query(a,b,"rise",page_size=30)
        receipt={"id":"wave-rise-"+str(band),"goal":video.goal["id"],"tool":"wave.query","requested_window":[a,b],
                 "returned":available,"images_bytes":0,"auto_delivered":False}
        query.append(receipt)
        cues=[c for c in video.cues if a<=c["suggested_window_seconds"][1]<=b and c["delta_R"]>0]
        supported=[]
        for cue in cues:
            p,q=(rows.get(k) for k in cue["source_candidates"])
            if p is None or q is None or q["candidate_id"] in selected:
                continue
            key=q["candidate_id"];f=axis[key];g=axis[p["candidate_id"]]
            explanation[key]={"state":"investigate","reason":"metadata rise is a place to compare, not a positive role","navigation_call":receipt["id"]}
            if set(cue["flags"])&{"unobserved_gap","transition_bracket"} or views(f["view_color_grid"],g["view_color_grid"])>POL["comparable_view_delta"]:
                continue
            for rg in f["candidate_regions"]:
                stats=patch_stats(video,p,q,rg,cache)
                if stats and stats["mean_abs_rgb_difference"]>=POL["wave_local_floor"]:
                    # Use the strongest actual local comparison within this
                    # public investigation, not the maximum waveform amplitude.
                    supported.append((q,stats,cue))
        if supported:
            winner,stats,cue=max(supported,key=lambda h:(h[1]["mean_abs_rgb_difference"],-h[0]["seconds"]))
            release(winner,"actual source-local pixels differ within a comparable view, for the public local-change question",stats,
                    {"query":receipt["id"],"source_pts":cue["source_pts"],"open_need":"inspect an independently checkable local state in this time entrance"})
        assert len(selected)>=before

    if use_complements:
        pre={k:r for k,r in selected.items()}
        legacy=[r for r in rows.values() if r["D0_order"] is not None and r["candidate_id"] not in selected and not r["in_catalog"]]
        needs_late=intent in ("scene","recurrence") and any(t in need for t in (" later","opening","return","early","different parts","separated"))
        if needs_late or intent=="open":
            # Target requirements request genuinely distinct approved originals.
            # Raw rooted comparisons remain eligible even when I/R is zero.
            for r in legacy:
                k=r["candidate_id"]
                t=r["seconds"]
                late=t>=fixed[-2]
                raw_roles={z.get("role",z.get("observation_role")) for z in r["legacy_reasons"]}
                if not raw_roles & {"D6_public_coverage_comparison_check","shadow_supported_coarse_episode","rooted_local_excursion_in_quiet_coarse_bracket"}:
                    continue
                if needs_late and not late:
                    continue
                f=axis[k]
                near=min(pre.values(),key=lambda v:abs(v["seconds"]-t))
                raw=read_low(video,r,cache);b=read_low(video,near,cache)
                patch=cv2.blur(np.max(cv2.absdiff(raw,b),axis=2).astype(np.float32),(14,10))
                _,maximum,_,center=cv2.minMaxLoc(patch)
                descriptor=views(f["view_color_grid"],axis[near["candidate_id"]]["view_color_grid"])
                eligible=maximum>=POL["local_floor"] and (descriptor>=POL["view_delta"] or maximum>=2*POL["wave_local_floor"])
                explanation[k]={"state":"investigate","reason":"approved cross-time endpoint considered independently of I/R","local_change":maximum,"view_difference":descriptor}
                if not eligible:
                    continue
                extra[k]=(r, {"raw_original_local_difference":maximum,"analysis_xy":list(center),"view_difference":descriptor,"other_pts":near["pts"],"rooted_raw_roles":sorted(v for v in raw_roles if v)})
            # Exact candidate frames are grouped only if their same-time,
            # same-view measurements overlap. Retain independent temporal
            # episodes without assuming any relation to the evaluation roles.
            accepted=[]
            for k,(r,evidence) in sorted(extra.items(),key=lambda kv:kv[1][0]["seconds"]):
                equivalent=any(abs(r["seconds"]-j["seconds"])<.85 and views(axis[k]["view_color_grid"],axis[j["candidate_id"]]["view_color_grid"])<POL["view_delta"]
                               for j in accepted)
                if not equivalent:
                    release(r,"source-grounded complementary reference for the public state or viewpoint comparison",evidence,
                            {"explicit_need":"different temporal or visual context","public_goal_supplies_later_request":needs_late})
                    accepted.append(r)
    return list(selected.values()),requests,[{"candidate":k,"pts":rows[k]["pts"],**explanation.get(k,{"state":"eligible","reason":"available for a later independent decision"})} for k in rows],query


def run(out):
    bundle=read(out/"frozen_p223_candidate_and_wave_manifest.json")
    def one(goal):
        gid=goal["id"];pointer=bundle["packets"][gid];p=checked(pointer);video=Video(p)
        batch=p["proposal"]["batch"]; source=video.source
        armdata={};counts=[];decisions=[];queries=[];native=[]
        prior=p["baseline"]
        if not prior:
            d0=sorted((r for r in batch if r["D0_order"] is not None),key=lambda r:r["D0_order"])
            d1,_=earlier_d1(goal,batch,list(p["features"].values()),[],source)
            for name,selected in (("O0",d0),("O1",d1)):
                home=out/"streams"/name/gid
                observe(home,goal,name,source,[r["image"] for r in selected],"new_source_reused_C4_or_D1_method")
                prior[name]=ref(home/"receipt.json")
        for name,pin in prior.items():
            r=checked(pin)
            armdata[name+"/"+gid]={"observer":pin,"frozen_history":not p["new_source"],"batch":pointer}
        old_d1=checked(prior["O1"])["items"]
        lookup={r["pts"]:r for r in batch}
        base=[lookup[v["pts"]] for v in old_d1]
        for arm in ("O2","O3"):
            selected,reads,trace,q=stage(video,base,arm=="O3")
            folder=out/"streams"/arm/gid
            r=observe(folder,goal,arm,source,[x["image"] for x in selected],"independent_goal_release_done" if selected else "no_grounded_offer")
            assert r["technical_abort"] or [x["pts"] for x in r["items"]]==[x["pts"] for x in selected]
            counts.append({"goal":goal,"arm":arm,"n":r["N_visible"],"public_range":r["public_range"],"source_sha256":r["source_sha256"],"technical_abort":r["technical_abort"]})
            trial=arm+"/"+gid
            path=folder/"independent_decision.json"
            trace_rows=[{"trial":trial,**row} for row in trace];query_rows=[{"trial":trial,**row} for row in q]
            checked_json(path,{"decision":trace_rows,"metadata_queries":query_rows,"source_revisit_receipts":reads,"actual_delivered_pts":[x["pts"] for x in r["items"]]})
            decisions.extend(trace_rows);queries.extend(query_rows);native.extend({"trial":trial,**x} for x in reads)
            armdata[trial]={"observer":ref(folder/"receipt.json"),"frozen_history":False,"decision":ref(path),"batch":pointer}
            print(trial,len(batch),r["N_visible"],flush=True)
        return armdata,counts,decisions,queries,native
    with ThreadPoolExecutor(max_workers=3) as threads:
        all_results=list(threads.map(one,bundle["goals"]))
    checked_json(out/"count_only.json",{"pairs":[p for x in all_results for p in x[1]],"explicit_only_counts":True})
    checked_json(out/"algorithm_lock.json",{"routes":{k:v for x in all_results for k,v in x[0].items()},"count_input":ref(out/"count_only.json")})
    for name,i in (("offline_delivery_decisions.jsonl",2),("offline_navigation_query_receipts.jsonl",3),("offline_source_revisit_receipts.jsonl",4)):
        jsonl(out/name,(row for x in all_results for row in x[i]))
    jsonl(out/"navigation_index.jsonl",({"goal":g,"packet":p,**checked(p)["navigation"]} for g,p in bundle["packets"].items()))
    checked_json(out/"inputs_and_goals.json",bundle)
    uniform(out)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True,type=Path);run(p.parse_args().out.resolve())
