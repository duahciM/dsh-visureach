"""Freeze upstream files and materialize the targeted public session packets."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

from if17_research.catalog import read,ref,sha
from p215_research.prepare import EXCLUDED_SHA256,ROOT,supplemental
from p215_research.semantic import Clip,indexed_case
from p216_research.navigate import checked_json
from p217_research.triggers import Case,make_candidates
from p219_research.score import checked
from p220_research.run import Exact
from p221_research.recover import RawGrid
from p222_research.extremes import transition
from p222_research.run import fresh
from p223_research.features import figure1,figure2,node_bundle
from p223_research.run import freeze_batch
from p223_research.timeline import timeline


HERE=Path(__file__).resolve().parent
OLD=ROOT/"p223_timeline_20261121"
PROTO=read(HERE/"protocol.json")
NEW={"road_fresh":{"case":"road_new","intent":"scene","text":"Compare the opening road layout with a later view of traffic or another visible change. Preserve useful spatial context and distinguish what is directly visible."},
     "notebook_scene":{"case":"notebook_new","intent":"scene","text":"Compare an early view of the object and a later visibly different configuration, preserving the surrounding spatial reference. Report visible state and uncertainty."}}
ASSETS={"road_new":{"file":"road.mp4","group":"Intel sample videos, fresh original source bytes","source_url":"https://github.com/intel-iot-devkit/sample-videos/blob/master/car-detection.mp4"},
        "notebook_new":{"file":"notebook.mp4","group":"Video2URDF phone recording, fresh original source bytes","source_url":"https://github.com/SungHyunC/video2urdf/blob/master/docs/models/notebook/phone.mp4"}}


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        yield from map(json.loads,f)


def freeze(out,card):
    old_lock=read(OLD/"protocol_lock.json")
    files=old_lock["files"].copy()
    manifest=read(OLD/"run_manifest.json")
    files.update({v["path"]:v["sha256"] for v in manifest["artifacts"].values()})
    for f in (OLD/"run_manifest.json",OLD/"input_receipt.json",OLD/"protocol_lock.json",HERE/"protocol.json",*[HERE/n for n in ("prepare.py","access.py","mcp_server.py","agent.py","offline.py")]):
        if f.is_file():
            files[str(f)]=sha(f)
    frozen=read(OLD/"candidate_batch_lock.json")["proposals"]
    routes=read(OLD/"algorithm_lock.json")["routes"]
    for gid in PROTO["offline_historical"]:
        files[frozen[gid]["path"]]=frozen[gid]["sha256"]
        for a in ("D0","D1","D2"):
            files.update({r["path"]:r["sha256"] for r in routes[a+"/"+gid].values()})
    checked_json(out/"protocol_lock.json",{"task_card":ref(card),"files":files,"rules":PROTO,"excluded_source_sha256":EXCLUDED_SHA256,
                 "no_E0_in_preparation_or_agent_packets":True,"held_out_sha_before_review":True})


def write_packet(out,goal,proposal,features,wave,baseline,new=False):
    folder=out/"public_packets"
    folder.mkdir(exist_ok=True)
    gid=goal["id"]
    quality={r["candidate"]:r for r in features}
    assert {x["candidate_id"] for x in proposal["batch"]}==set(quality)
    pkt={"goal":goal,"proposal":proposal,"features":quality,"navigation":wave,"baseline":baseline,"new_source":new}
    path=folder/(gid+".json")
    checked_json(path,pkt)
    return ref(path)


def historical(out):
    previous=read(OLD/"source_and_goal_manifest.json")
    wanted=set(PROTO["offline_historical"])
    goals={g["id"]:g for g in previous["goals"] if g["id"] in wanted}
    assert set(goals)==wanted
    p=read(OLD/"candidate_batch_lock.json")["proposals"]
    routes=read(OLD/"algorithm_lock.json")["routes"]
    waves={r["goal"]:r["F1_wave"] for r in jsonl(OLD/"information_relation_timeline.jsonl") if r["goal"] in wanted}
    cues={k:[] for k in wanted}
    for c in jsonl(OLD/"wave_navigation_trace.jsonl"):
        if c["goal"] in wanted and c["regime"]=="F1":
            cues[c["goal"]].append(c)
    public={}
    for gid,g in goals.items():
        upstream=checked(p[gid]); snapshot=checked(checked(routes["D1/"+gid]["trace"])["spatial_snapshot"])
        entry=previous["library"][g["case"]]
        assert entry["source"]["sha256"]!=EXCLUDED_SHA256 and sha(entry["source"]["path"])==entry["source"]["sha256"]
        data={"original_packet":p[gid],"batch":upstream["batch"],"source_read_permissions":upstream["source_read_permissions"],
              "transitions":upstream["transition_status_source"],"source":checked(entry["catalog"])["source"]}
        public[gid]=write_packet(out,g,data,snapshot["F1"],{"wave":waves[gid],"cues":cues[gid]},
                  {a:routes[d+"/"+gid]["observer"] for a,d in (("O0","D0"),("O1","D1"),("O4","D2"))})
    checked_json(out/"historical_packets.json",{"packets":public,"goals":list(goals.values()),"library":{g["case"]:previous["library"][g["case"]] for g in goals.values()}})


def fresh_sources(out):
    known=read(OLD/"source_and_goal_manifest.json")["library"]
    found={r["source"]["sha256"] for r in known.values()}|{EXCLUDED_SHA256}
    intake=out/"fresh_input_lock.json"
    pointers={c:ref(out/"assets"/m["file"]) for c,m in ASSETS.items()}
    assert not ({r["sha256"] for r in pointers.values()} & found) and len({r["sha256"] for r in pointers.values()})==len(pointers)
    for r in pointers.values():
        subprocess.run(["ffmpeg","-v","error","-xerror","-i",r["path"],"-map","0:v:0","-f","null","NUL"],check=True)
    checked_json(intake,{"originals":pointers,"sources":ASSETS,"sha_before_review":True})
    model=ROOT/"p215_discovery_20261121"
    with (model/"assets/openimages.csv").open(encoding="utf-8-sig",newline="") as f:
        words=sorted({a[1].casefold() for a in csv.reader(f) if len(a)==2})
    encoder=Clip(model)
    text=encoder.text([f"a photo of a {w}." for w in words])
    lib={};packets={};goals=[]
    for gid,need in NEW.items():
        case=need["case"]
        goal={"id":gid,**need,"cohort":"P2_24_new_source"}
        folder=out/"fresh_verified"/case
        entry=supplemental(folder,"fresh",Path(pointers[case]["path"]))
        sem=folder/"semantic";sem.mkdir()
        rows,_,errs=indexed_case(checked(entry["catalog"]),words,text,encoder)
        checked_json(sem/"candidates.json",{"rows":rows,"errors":errs,"scope":"unchanged C4 compatibility"})
        entry|={"semantic":str(sem),"public_provenance":ASSETS[case]}
        lib[case]=entry;goals.append(goal)
        book=Case(out,case,entry); exact=Exact(book); facts,missing=make_candidates(book)
        base,_,queries,decisions,_,(c4,reasons),revisits=fresh(book,goal,out,exact,facts,missing,RawGrid(book,exact))
        by= {d["pts"]:[{"observation_role":d.get("role","catalog review"),"root_ids":d.get("parents",[])}] for d in decisions if "pts" in d}
        candidates,perm=freeze_batch(book,goal,c4,reasons,{x["pts"] for x in base},queries,revisits,by)
        indexed={x["pts"]:x for x in book.rows}; axes=[]
        for r in candidates:
            n=node_bundle(book,r,indexed.get(r["pts"]));axes.append(figure1(n,figure2(n)))
        cuts=transition(book)
        wave,cues=timeline(candidates,axes,cuts["window_candidates"],book.source)
        packets[gid]=write_packet(out,goal,{"batch":candidates,"source_read_permissions":perm,"transitions":cuts,"source":book.source},axes,{"wave":wave,"cues":cues},{},True)
    checked_json(out/"new_packets.json",{"packets":packets,"library":lib,"goals":goals,"fresh_file_lock":ref(intake)})


def manifest(out):
    a=read(out/"historical_packets.json"); b=read(out/"new_packets.json")
    checked_json(out/"frozen_p223_candidate_and_wave_manifest.json",{"packets":a["packets"]|b["packets"],"library":a["library"]|b["library"],
        "goals":a["goals"]+b["goals"],"prior_source":ref(OLD/"source_and_goal_manifest.json"),"frozen_without_E0":True})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--phase",required=True,choices=("historical","fresh","manifest"));p.add_argument("--card",type=Path)
    a=p.parse_args();out=a.out.resolve()
    if a.phase=="historical":
        freeze(out,a.card.resolve());historical(out)
    elif a.phase=="fresh":
        fresh_sources(out)
    else:
        manifest(out)
