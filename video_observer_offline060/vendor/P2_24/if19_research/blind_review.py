"""Hide method labels and compare two independent source-image review sessions per case."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from traceback import format_exc

from if11_research.agent import CLI,Harness
from if17_research.catalog import read,ref,sha
from p1.io import dump

from .fixed import ROUTE,tokens
from .inputs import PROTOCOL,QUESTIONS

PROMPT="""You are an independent reviewer of four unnamed observer packets describing the same delivered source frames.
Use the attached images and source time/ordinal table. The calibration key describes just the agreed groundable image facts and ambiguity. All answers received these same frames. Check actual image pixels. Do not credit verbose reasons just for being verbose.
For each opaque observer and each checkpoint, list the core target roles stated correctly with actual visual support. Mark separate errors:
source: missing or wrong integer PTS/frame reference when a supplied pointer is asserted. Lack of pointer should be marked separately from false visual content.
observation: explicit image detail contradicts pixels.
inference: visible detail is correct but it does not support the conclusion.
cross_time: unknown gap, exact release or assumed identity presented as seen.
narration: a visibly correct intermediate fact disappears/is reversed, or unsupported assertions are added in final prose.
Answer only JSON, with facts as a list of four brief verdicts {id, first_valid_roles, final_valid_roles, mistakes:[{checkpoint,layer,pts,reason}], final_retains, unresolved_preserved}. Up to three important mistakes per packet. Use short phrases. No other actions, no tools. Stay under 430 words."""


def flatten(resp):
    return {"claims":[{k:v for k,v in m.items() if k in (
               "claim","pts","source","source_frame","time_seconds","visible_description","visual_support","status","open_questions")}
                      for m in resp.get("facts",[]) if isinstance(m,dict)],
            "unresolved":resp.get("unresolved",[]),"hypotheses":resp.get("hypotheses",[]),
            "final":resp.get("final_narration","")}


def packets(o,case,manifest):
    accepted=[p for p in manifest["groups"] if p["case"]==case and p["accepted"]]
    result={}
    order=[]
    for item in accepted:
        end=item["accepted"]
        a=read(end["path"])
        code=sha256(f'{case}-P3-10-opaque-{end["sha256"]}'.encode()).hexdigest()[:8]
        order.append(code)
        one,two=[z["receipt"]["result"]["reply"] for z in a["transcript"]]
        result[code]={"mapping":{"arm":item["arm"],"replicate":item["replicate"],"original":end},
                     "blinded":{"id":code,"first":flatten(one),"final":flatten(two)}}
    return result,sorted(order)


def native_review(d,case,first,table,answers,labels,order):
    d.mkdir(parents=True,exist_ok=False)
    output={"case":case,"input_order":order,"model":ROUTE,"turns":[]}
    text=PROMPT+"\nQuestion: "+QUESTIONS[case]+"\nSource time table: "+json.dumps(table)+"\nEvaluation-only key: "+json.dumps(labels)+"\nPackets:\n"+json.dumps([answers[k]["blinded"] for k in order],ensure_ascii=False)
    dump(d/"packet.json",{"opaque_order":order,"prompt":text,"attachment_sha256":sha(first)})
    h=None
    try:
        h=Harness(d,CLI)
        output["sdk"]=h.result(h.send("initialize",ROUTE|{"cwd":str(d)}))
        z=h.turn(text,[first],1)
        output["turns"].append({"receipt":z})
        ans=z["result"]["reply"]
        if not ans or z["result"]["tool_calls"] or not isinstance(ans.get("facts"),list):
            raise RuntimeError("Blank or malformed blind-review reply")
        if {str(v["id"]) for v in ans["facts"]}!=set(order):
            raise ValueError("Review did not score all opaque packets")
        assert z["attachments"][0]["sha256"]==sha(first)
        output.update(status="completed",verdicts=ans["facts"])
    except Exception:
        output.update(status="failed",error=format_exc())
    finally:
        if h:
            h.close()
        output["usage"]=tokens(output["turns"])
        dump(d/"result.json",output)
    print(case,d.name,output["status"],flush=True)
    return ref(d/"result.json")|{"status":output["status"]}


def each(o,c,n,book,order,rows):
    p=Path(book["image"]["path"])
    labels=read(o/"evaluation_only"/f"{c}_reference.json")["facts"]
    table=book["positions"]
    order=order if n=="R1" else order[::-1]
    candidates=[]
    for retry in (1,2):
        target=o/"review"/c/n/f"attempt_{retry}"
        existing=target/"result.json"
        if existing.is_file():
            record=read(existing)
            r=ref(existing)|{"status":record.get("status","logging_failure")}
        else:
            r=native_review(target,c,p,table,rows,labels,order)
        candidates.append(r)
        if r["status"]=="completed":
            break
    return {"case":c,"reviewer":n,"attempts":candidates,"accepted":r if r["status"]=="completed" else None}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",required=True,type=Path)
    a.add_argument("--workers",type=int,default=2)
    x=a.parse_args()
    o=x.out.resolve()
    p=o/"review"
    p.mkdir(exist_ok=True)
    m=read(o/"A"/"manifest.json")
    jobs=[]
    mapping={}
    for c in PROTOCOL["stage_A"]["cases"]:
        book=read(o/"fixed_input"/c/"input.json")
        out,order=packets(o,c,m)
        if len(order)!=4:
            raise ValueError("A packet population is incomplete")
        mapping[c]={k:v["mapping"] for k,v in out.items()}
        jobs.extend((c,n,book,order,out) for n in ("R1","R2"))
    dump(p/"offline_mapping.json",mapping)
    with ThreadPoolExecutor(max_workers=x.workers) as pool:
        result=[pool.submit(each,o,*job) for job in jobs]
        files=[f.result() for f in result]
    dump(p/"manifest.json",{"reviews":files,"reviewers_not_observers":True,
         "method_names_hidden":True,"answer_structure_visible":True,"reversed_packet_order":True})


if __name__=="__main__":
    main()

