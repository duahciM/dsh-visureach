"""Audit frame addressing separately from manual visual and conclusion adjudication."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import re

from if17_research.catalog import read,ref,sha
from if18_research.verify import refs
from p1.io import dump

from .fixed import ADD
from .inputs import PROTOCOL


def integers(x):
    if isinstance(x,bool):
        return []
    if isinstance(x,int):
        return [x]
    if isinstance(x,str) and re.fullmatch(r"\s*\d+(?:\s*[,/ ]\s*\d+)*\s*",x):
        return [int(u) for u in re.findall(r"\d+",x)]
    if isinstance(x,list):
        return [i for u in x for i in integers(u)]
    return []


def evidence_fields(facts,actual):
    rows=[]
    for n,f in enumerate(facts,1):
        if not isinstance(f,dict):
            continue
        entries=f.get("source",[])
        entries=[entries] if isinstance(entries,dict) else entries if isinstance(entries,list) else []
        source_pts=[p for e in entries if isinstance(e,dict) for p in integers(e.get("pts"))]
        pts=list(dict.fromkeys(integers(f.get("pts"))+source_pts))
        absent=[p for p in pts if p not in actual]
        invalid=[]
        for e in [f,*entries]:
            if not isinstance(e,dict):
                continue
            once=integers(e.get("pts"))
            iam=e.get("source_frame")
            t=e.get("seconds",e.get("time_seconds"))
            if len(once)!=1 or once[0] not in actual:
                continue
            p=actual[once[0]]
            if iam is not None and isinstance(iam,(str,int)) and integers(iam)!=[p["source_frame"]]:
                invalid.append({"field":"source_frame","value":iam,"belongs_to_pts":once[0]})
            if t is not None:
                try:
                    if abs(float(t)-p["seconds"])>.0011:
                        invalid.append({"field":"seconds","value":t,"belongs_to_pts":once[0]})
                except (TypeError,ValueError):
                    pass
        described=bool(isinstance(f.get("visible_description"),str) and f["visible_description"].strip())
        supported=bool(isinstance(f.get("visual_support"),str) and f["visual_support"].strip())
        rows.append({"ordinal":n,"claim_id":f.get("claim_id"),"claim":f.get("claim"),"typed_pts":pts,
          "pts_not_previously_sent":absent,"source_mapping_conflicts":invalid,
          "typed_source_ref_valid":bool(pts) and not absent and not invalid,"visible_detail_field":described,
          "explicit_support_field":supported,
          "full_structural_record":bool(pts) and not absent and not invalid and described and supported,
          "semantic_status_not_structurally_scored":True})
    return rows


def tally(rows):
    return {"facts":len(rows),"addressed_by_valid_pts":sum(r["typed_source_ref_valid"] for r in rows),
            "without_integer_pts_field":sum(not r["typed_pts"] for r in rows),
            "wrong_pts_or_mapping":sum(bool(r["pts_not_previously_sent"] or r["source_mapping_conflicts"]) for r in rows),
            "description_fields":sum(r["visible_detail_field"] for r in rows),
            "support_fields":sum(r["explicit_support_field"] for r in rows),
            "structural_complete_records":sum(r["full_structural_record"] for r in rows)}


def audit_A(o):
    m=read(o/"A"/"manifest.json")
    summaries={}
    locks=defaultdict(list)
    for r in m["groups"]:
        if not r["accepted"]:
            continue
        p=Path(r["accepted"]["path"])
        doc=read(p)
        assert sha(p)==r["accepted"]["sha256"]
        assert doc["route"]==PROTOCOL["route"]
        book=read(o/"fixed_input"/r["case"]/"input.json")
        p_hash=book["image"]["sha256"]
        assert sha(book["image"]["path"])==p_hash and book["native_audit"]["pass"]
        known={f["pts"]:f for f in book["positions"]}
        wire=refs(p.parent/"runtime_events.jsonl")
        outputs=[]
        bases=[]
        for i,row in enumerate(doc["transcript"]):
            assert row["sent"] in wire
            assert [a["sha256"] for a in row["receipt"]["attachments"]]==([p_hash] if i==0 else [])
            text=row["sent"]
            if r["arm"]=="A1":
                assert text.endswith("\n"+ADD)
                text=text[:-len(ADD)-1]
            bases.append(text)
            z=row["receipt"]["result"]["reply"]
            stats=evidence_fields(z.get("facts",[]),known)
            outputs.append({"turn":i+1,"fact_audit":stats,"counts":tally(stats),
                 "reported_unresolved":z.get("unresolved"),"narration":z.get("final_narration")})
        locks[r["case"]].append(tuple(bases))
        summaries[f'{r["case"]}_{r["arm"]}_{r["replicate"]}']={"case":r["case"],"arm":r["arm"],
          "results":outputs,"source_packet_sha256":p_hash,"input_count":len(known),
          "usage":doc["usage"],"seconds":doc["seconds"],"transcript":ref(p)}
    assert all(len(set(v))==1 for v in locks.values())
    return summaries


def audit_B(o):
    manifest=read(o/"B"/"manifest.json")
    source=read(o/"B"/"transport_and_source_audit.json")["sessions"]
    observations={}
    for r in manifest["groups"]:
        p=r["accepted"]
        if not p:
            continue
        path=Path(p["path"])
        assert sha(path)==p["sha256"]
        session=path.parent.name
        obs=read(path)
        assert obs["route"]==PROTOCOL["route"]
        history={}
        rows=[]
        for t in obs["turns"]:
            for receipt in obs["calls"]:
                if receipt["returned_to_observer"] and receipt["returned_on_turn"]==t["turn"]:
                    for picture in receipt["frame_receipts"]:
                        history[picture["pts"]]=picture
            reply=t["response"]["result"]["reply"]
            fields=evidence_fields(reply.get("facts",[]),history)
            rows.append({"turn":t["turn"],"fact_audit":fields,"counts":tally(fields),
              "visually_received_count":len(history),"stop":reply.get("stop"),
              "narration":reply.get("final_narration"),"unresolved":reply.get("unresolved"),
              "update":reply.get("update")})
        observations[f'{r["case"]}_{r["condition"]}']={"case":r["case"],"arm":r["condition"],"results":rows,
          "session":session,"usage":obs["usage"],"seconds":obs["seconds"],"tool_stats":obs["query_stats"],
          "transport_source":source[session],"transcript":ref(path),"ended_on_safety_limit":obs["ended_on_safety_limit"]}
    by_clip=defaultdict(list)
    for x in observations.values():
        r=read(x["transcript"]["path"])
        p=r["turns"][0]["prompt"]
        if x["arm"]=="B1":
            assert p.endswith("\n"+ADD)
            p=p[:-len(ADD)-1]
        by_clip[x["case"]].append(p)
    assert all(len(set(v))==1 for v in by_clip.values())
    return observations


def arm_stats(all_cases):
    result={}
    for arm in sorted({r["arm"] for r in all_cases.values()}):
        rows=[r for r in all_cases.values() if r["arm"]==arm]
        result[arm]={"sessions":len(rows),"outputs":sum(len(r["results"]) for r in rows),
           "final_only":dict(sum((Counter(r["results"][-1]["counts"]) for r in rows),Counter())),
           "all_turns":dict(sum((Counter(t["counts"]) for r in rows for t in r["results"]),Counter())),
           "output_tokens":sum(r["usage"].get("outputTokens",0) for r in rows)}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--stage-a-only",action="store_true")
    x=p.parse_args()
    o=x.out.resolve()
    a=audit_A(o)
    b={} if x.stage_a_only else audit_B(o)
    metrics=arm_stats(a|b)
    dump(o/("A"/Path("structural_audit.json") if x.stage_a_only else Path("structural_audit.json")),
        {"A":a,"B":b,"metrics":metrics,"native_same_A_inputs":True,"wire_and_receipts_checked":True,
         "visibility_and_reasoning_scored_elsewhere":True})
    print(metrics,flush=True)


if __name__=="__main__":
    main()
