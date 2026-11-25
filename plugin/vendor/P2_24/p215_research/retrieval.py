"""Source-bound cold-start discovery, ranked union and evidence-packet reading."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from math import floor
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from PIL import Image

from if17_research.browse import read_original_range
from if17_research.catalog import read, ref, sha
from p1.io import dump
from .prepare import RULES


LIMITS=RULES["limits"]
N=RULES["noise"]
DIMS={f"D{i}": name for i,name in enumerate(
    ("local region","local state difference","image relation","viewpoint","unknown or conflict","temporal coverage"),1)}
OBJECT_ARMS={"R2","R4","R5","R4_noise","R5_noise"}
GRAPH_ARMS={"R3","R4","R5","R4_noise","R5_noise"}
ROLE_ARMS={"R5","R5_no_semantic","R5_noise"}


def uniform(rows):
    if not rows:
        return []
    anchors=np.linspace(rows[0]["t"],rows[-1]["t"],min(len(rows),LIMITS["initial_catalog_reads"]))
    selected=list(dict.fromkeys(min(rows,key=lambda r:abs(r["t"]-a))["frame_id"] for a in anchors))
    return selected+[r["frame_id"] for r in rows if r["frame_id"] not in selected]


def goal_spec(text):
    q=text.casefold()
    terms=tuple(dict.fromkeys(re.findall(r'["“]([^"”]+)["”]',q)))
    if re.search(r"unseen|occlu|disappear|hidden|not visible",q):
        family,slots,dims="visibility_gap",["before","gap","after","reference"],["D5","D1","D2","D6"]
    elif re.search(r"reappear|recurr|return|again|different parts",q):
        family,slots,dims="recurrence",["before","after","reference","gap"],["D4","D1","D5","D6"]
    elif re.search(r"before|after|change|sequence|order|move|position.*differ",q):
        family,slots,dims="change_order",["before","middle","after","reference"],["D2","D1","D4","D6"]
    elif re.search(r"above|below|overlap|touch|near |next to|relative|separa|in relation",q):
        family,slots,dims="relation",["anchor","reference","gap"],["D3","D1","D5","D6"]
    elif terms:
        family,slots,dims="visibility",["anchor","reference"],["D1","D6","D5"]
    else:
        family,slots,dims="exploration",["before","middle","after","gap"],list(DIMS)
    predicates=["left_of","right_of","above","below","regions_overlap"]
    if "above" in q or "below" in q:
        predicates=["above","below"]
    elif "left" in q or "right" in q:
        predicates=["left_of","right_of"]
    return {"terms":terms,"family":family,"slots":slots,"dimension_queries":dims,"predicates":predicates}


def pseudo(*pieces):
    text=N["seed"]+"|"+"|".join(map(str,pieces))
    return int.from_bytes(sha256(text.encode()).digest()[:4],"big")/2**32


class Library:
    def __init__(self,out,case,query_vectors=None):
        self.out=out
        self.case=case
        self.paths=read(out/"library/manifest.json")[case]
        self.source_book=read(self.paths["catalog"]["path"])
        self.source=self.source_book["source"]
        self.rows=self.source_book["frames"]
        self.by={r["frame_id"]:r for r in self.rows}
        self.ledger=read(self.paths["index"]["path"])
        semantic_folder=out/"semantic"/case
        self.semantic_available=(semantic_folder/"metrics.json").exists()
        self.sem=read(semantic_folder/"candidates.json")["rows"] if self.semantic_available else []
        self.features=np.load(semantic_folder/"vectors.npy",allow_pickle=False) if self.semantic_available else None
        self.query_vectors=query_vectors or {}
        self.time_order=uniform(self.rows)
        self.known_pts={f["pts"] for f in self.rows}
        self.window_cache={}

    def overview(self,arm):
        summary={"span_seconds":[self.source["start"],self.source["end"]],
                 "source_sha256":self.source["sha256"],"time_base":self.source["time_base"],
                 "catalog_frames":len(self.rows),"native_resolution":[self.source["probe"]["width"],self.source["probe"]["height"]],
                 "initial_cursor":0,"route":"time browse","negative_semantics":"No-hit describes the index. Unseen and unverified intervals remain unknown."}
        if arm!="R0":
            summary["dimensions"]={key:{"description":desc,"indexed":len(self.ledger["filters"]["dimension"].get(key,[]))}
                                    for key,desc in DIMS.items()}
            summary["discovery_entrypoints"]=["dimension union","region","root","time","uncertainty","comparison frames"]
        if arm in OBJECT_ARMS:
            words=Counter(self.visible_word(w,r["frame_id"],arm) for r in self.sem for node in r["nodes"]
                          for w in node["visual_hypotheses"])
            words.pop(None,None)
            summary["automatic_candidate_vocabulary"]=dict(words.most_common())
        if arm in GRAPH_ARMS:
            summary["geometric_predicates"]=dict(Counter(e["predicate"] for r in self.sem for e in r["relations"]))
        return summary

    @staticmethod
    def visible_word(candidate,key,arm):
        word=candidate["word"]
        if not arm.endswith("_noise"):
            return word
        if pseudo(key,word,"drop")<N["object_drop"]:
            return None
        if pseudo(key,word,"corrupt")<N["wrong_label"]:
            return "unverified category "+str(int(100*pseudo(key,word,"alias")))
        return word

    def discovery_terms(self,arm):
        found=defaultdict(set)
        for row in self.sem:
            for node in row["nodes"]:
                for tag in node["visual_hypotheses"]:
                    word=self.visible_word(tag,row["frame_id"],arm)
                    if word:
                        found[word].add(row["frame_id"])
        candidates={word:len(frames) for word,frames in found.items() if len(frames)>=2}
        return sorted(candidates,key=lambda word:(abs(candidates[word]-max(2,len(self.rows)//4)),word))[:3]

    def search(self,arm,spec):
        channels={"time":self.time_order}
        if arm!="R0":
            for dim in spec["dimension_queries"]:
                values={}
                for r in self.rows:
                    scores=[float(n["priority_within_dimension"]) for n in r["nominations"] if n["dimension_id"]==dim]
                    if scores:
                        values[r["frame_id"]]=max(scores)
                channels[dim]=sorted(values,key=lambda k:(-values[k],self.by[k]["t"]))
            filters=self.ledger["filters"]
            if "D5" in spec["dimension_queries"]:
                channels["unknown"]=filters["uncertainty"].get("unknown_or_competing_explanations",[])
            if "D1" in spec["dimension_queries"] or "D3" in spec["dimension_queries"]:
                channels["region"]=list(dict.fromkeys(key for hits in filters["region_candidate"].values() for key in hits))
                channels["root"]=list(dict.fromkeys(hits[0] for hits in filters["root"].values() if hits))
        if arm in OBJECT_ARMS:
            for term in spec["terms"]:
                lexical={}
                cross={}
                vector=self.query_vectors.get(term)
                for row in self.sem:
                    key=row["frame_id"]
                    for node in row["nodes"]:
                        words={self.visible_word(w,key,arm) for w in node["visual_hypotheses"]}
                        if any(term in word or word in term for word in words if word and len(word)>2):
                            lexical[key]=max(lexical.get(key,0),max(w["cosine"] for w in node["visual_hypotheses"]))
                        if vector is not None and (not arm.endswith("_noise") or pseudo(key,node["candidate_id"],"vector")>=N["object_drop"]):
                            score=float(self.features[node["feature_row"]]@vector)
                            if score>=RULES["semantics"]["min_cosine"]:
                                cross[key]=max(score,cross.get(key,0))
                channels[f"object:{term}"]=sorted(lexical,key=lambda k:(-lexical[k],self.by[k]["t"]))
                channels[f"text_region:{term}"]=sorted(cross,key=lambda k:(-cross[k],self.by[k]["t"]))
        if arm in GRAPH_ARMS:
            relation_scores={}
            inverse={"above":"below","below":"above","left_of":"right_of","right_of":"left_of"}
            for row in self.sem:
                edges=[]
                for edge in row["relations"]:
                    value=edge["predicate"]
                    if arm.endswith("_noise") and pseudo(edge["evidence_root"],edge["subject_candidate_id"],edge["object_candidate_id"],"edge")<N["edge_reverse"]:
                        value=inverse.get(value,value)
                        if value=="regions_overlap":
                            value="right_of"
                    if value in spec["predicates"]:
                        edges.append(max(map(abs,edge["normalized_offset"])))
                if edges:
                    relation_scores[row["frame_id"]]=max(edges)
            channels["scene_graph"]=sorted(relation_scores,key=lambda k:(-relation_scores[k],self.by[k]["t"]))
        return channels

    def expand(self,seed):
        row=self.by[seed]
        exact=list(dict.fromkeys(p for obs in row["observations"] for p in obs["source_comparison_frames"]
                                if p in self.by and p!=seed))
        near=sorted((k for k in self.by if k!=seed and k not in exact),
                    key=lambda k:abs(self.by[k]["t"]-row["t"]))[:2]
        return [(key,"existing comparison frame") for key in exact[:4]]+[(key,"soft temporal neighbor") for key in near]

    def reject_noise_links(self):
        timeline=self.ledger["timeline"]
        if len(timeline)<2:
            return []
        delivered=[]
        for i,key in enumerate(timeline):
            if pseudo(key,"identity")<N["untrusted_identity_link"]:
                dest=timeline[(i+max(1,len(timeline)//2))%len(timeline)]
                delivered.append({"source_frame_id":key,"target_frame_id":dest,"proposal":"same_instance",
                                  "geometry_or_origin_evidence":[],
                                  "status":"rejected_as_unverified","reason":"no verified correspondence"})
        return delivered

    def read_window(self,start,end,rate):
        key=(start,end,rate)
        if key in self.window_cache:
            return self.window_cache[key],True
        slug=sha256(repr(key).encode()).hexdigest()[:14]
        folder=self.out/"raw"/self.case/slug
        file=folder/"read_receipt.json"
        if file.exists():
            receipt=read(file)
            cached=True
        else:
            receipt=read_original_range(self.source_book,folder,start,end,rate)
            cached=False
        self.window_cache[key]=receipt
        return receipt,cached


def ranked_union(channels):
    records=defaultdict(dict)
    for name,seq in channels.items():
        for i,key in enumerate(seq,1):
            records[key][name]=i
    # Multiple roots or repeated labels are never added as independent votes.
    def ranking(key):
        evidence=records[key]
        scores=[pos+(1 if c=="time" else 0) for c,pos in evidence.items()]
        return min(scores),evidence.get("time",10**8),key
    return sorted(records,key=ranking),records


def slot_schedule(book,spec,ranked,hits):
    start,end=book.source["start"],book.source["end"]
    duration=end-start
    result=[]
    slot_choices={}
    # Positional terms are used as allocation hypotheses only, not as video-time labels.
    for slot in spec["slots"]:
        def score(key):
            t=(book.by[key]["t"]-start)/duration
            signals=hits[key]
            if slot=="before":
                return .8*t + .025*min(signals.values())
            if slot=="after":
                return .8*(1-t) + .025*min(signals.values())
            if slot=="middle":
                return .6*abs(t-.5)+.015*min(signals.get("D2",60),signals.get("D1",60))+.01*min(signals.values())
            if slot=="gap":
                return .015*signals.get("D5",80)+.014*min(signals.values())
            if slot=="reference":
                return .02*min(signals.get("D4",55),signals.get("D3",55),signals.get("scene_graph",55))+.018*min(signals.values())
            return min(signals.values())
        candidate=next((p for p in sorted(ranked,key=score) if p not in result),None)
        if candidate:
            slot_choices[slot]=candidate
            result.append(candidate)
    paths=[]
    for p in result.copy():
        for q,kind in book.expand(p):
            if q not in result and q not in {x["target"] for x in paths}:
                paths.append({"seed":p,"target":q,"path_type":kind,"identity":"unknown"})
    return result+[x["target"] for x in paths]+ranked,slot_choices,paths


def proposed_windows(book,first,channels,structured):
    lo,hi=book.source["start"],book.source["end"]
    w=min(LIMITS["max_window_seconds"],max(.6,(hi-lo)/16))
    candidates=[]
    if structured:
        for channel in ("D5","D2"):
            candidates.extend(book.by[k]["t"] for k in channels.get(channel,[])[:2])
    seen=sorted({lo,hi,*[f["seconds"] for f in first]})
    candidates.extend((a+b)/2 for a,b in sorted(zip(seen,seen[1:]),key=lambda x:x[0]-x[1]))
    made=[]
    for center in candidates:
        start=round(max(lo,min(hi-w,lo+round((center-lo)/w-.5)*w)),6)
        end=min(hi,round(start+w,6))
        if end>start and (start,end) not in made:
            made.append((start,end))
    return made


def handoff(book,path,expected,stamp,prior):
    raw=Path(path).read_bytes()
    if sha256(raw).hexdigest()!=expected:
        raise ValueError("Handoff PNG failed digest check")
    with Image.open(path) as png:
        size=png.size
        if png.format!="PNG" or size!=(book.source["probe"]["width"],book.source["probe"]["height"]):
            raise ValueError("Expected source-sized single PNG")
        png.load()
    t=stamp["t"]
    return {"path":str(path),"sha256":expected,"bytes_opened":len(raw),"mime":"image/png","native_wh":list(size),
            "source_sha256":book.source["sha256"],"source_frame":stamp["source_frame"],"pts":stamp["pts"],
            "time_base":stamp["time_base"],"seconds":t,"previous_sent_seconds":prior,
            "open_interval_between_items":sorted([t,prior]) if prior is not None and prior!=t else None,
            "status":"returned_and_opened_by_local_consumer","presentation":"independent PNG, no crop or resize"}


def query(book,goal,arm):
    started=perf_counter()
    spec=goal_spec(goal["text"])
    discovered=book.discovery_terms(arm) if arm in OBJECT_ARMS and not spec["terms"] else []
    if discovered:
        spec["terms"]=tuple(discovered)
    structured=arm in ROLE_ARMS
    channels=book.search(arm,spec)
    ranked,independent=ranked_union(channels)
    if arm=="R0":
        ranked=book.time_order
    selected,slots,paths=slot_schedule(book,spec,ranked,independent) if structured else (ranked,{},[])
    received=[]
    failures=[]
    returned=set()
    def put(path,digest,row,**info):
        if row["pts"] in returned:
            return False
        try:
            ack=handoff(book,path,digest,row,received[-1]["seconds"] if received else None)
        except (OSError,ValueError) as exc:
            failures.append({"pts":row["pts"],"stage":"return_and_open","error":str(exc)})
            return False
        received.append(ack|info)
        returned.add(row["pts"])
        return True
    def catalog_read(key):
        row=book.by[key]
        src=row["source_image"]
        reasons=independent[key]
        return put(src["path"],src["sha256"],row,kind="catalog",frame_id=key,
                   channel_ranks=reasons,dimensions=row["image_dimensions"],
                   roots=sorted({root for g in row["independent_evidence_groups"] for root in g["roots"]}))
    for key in dict.fromkeys(selected):
        if len(received)>=LIMITS["initial_catalog_reads"]:
            break
        catalog_read(key)
    window_records=[]
    for i,(start,end) in enumerate(proposed_windows(book,received,channels,structured)):
        row={"proposal":[start,end],"rank":i+1,"status":"budget_unassessed","stages":{"proposed":True,"budgeted":False,"reader_ran":False,"returned":0}}
        window_records.append(row)
        if i>=LIMITS["windows"]:
            continue
        row["stages"]["budgeted"]=True
        try:
            rate=min(LIMITS["max_window_fps"],max(1.0,LIMITS["frames_per_window"]/(end-start)))
            result,cached=book.read_window(start,end,round(rate,6))
            row["stages"]["reader_ran"]=not cached
            row["stages"]["verified_cache_attach"]=cached
            row.update(decoded_candidates=len(result["returned"]),status="unsupported",requested_fps=rate)
            for frame in result["returned"][:LIMITS["frames_per_window"]]:
                if len(received)>=LIMITS["distinct_reads"]:
                    break
                if put(frame["native_image"],frame["native_image_sha256"],frame,
                       kind="direct_source_range",window=[start,end],in_catalog=frame["pts"] in book.known_pts):
                    row["stages"]["returned"]+=1
            if row["stages"]["returned"]:
                row["status"]="supported"
            row["duplicate_or_budget_not_delivered"]=row["decoded_candidates"]-row["stages"]["returned"]
        except (ValueError,RuntimeError,OSError) as exc:
            row.update(status="unassessable",error=str(exc))
    for key in dict.fromkeys(selected):
        if len(received)>=LIMITS["distinct_reads"]:
            break
        catalog_read(key)
    received_pts={r["pts"] for r in received}
    gap_points=sorted({book.source["start"],book.source["end"],*[r["seconds"] for r in received]})
    return {"goal":goal,"arm":arm,"case":book.case,"elapsed_seconds":perf_counter()-started,
            "initial_queries":[{"channel":k,"request":spec["terms"] if ":" in k else k,
                                "matches":len(v)} for k,v in channels.items()],
            "automatically_discovered_query_terms":discovered,
            "distinct_candidate_union":len(independent),"independent_hits":{k:len(v) for k,v in channels.items()},
            "ranked_catalog":ranked,"role_requirements":spec["slots"] if structured else [],
            "slot_candidates":{slot:{"frame_id":p,"actually_read":book.by[p]["pts"] in received_pts,
                  "verification":"unassessable","note":"Candidate selection needs source-image verification."} for slot,p in slots.items()},
            "source_links":paths,"injected_identity_rejections":book.reject_noise_links() if arm.endswith("_noise") else [],
            "windows":window_records,"delivered":received,"failures":failures,
            "open_unobserved_intervals":[[a,b] for a,b in zip(gap_points,gap_points[1:]) if b>a],
            "gap_status":"insufficient_until_original_observations_are_adjudicated", "absence_or_identity_claimed":False,
            "index_capability":{"semantic":book.semantic_available if arm in OBJECT_ARMS else None,
              "scene_graph":book.semantic_available if arm in GRAPH_ARMS else None,
              "arbitrary_text":bool(book.query_vectors) if arm in OBJECT_ARMS else None}}


def run(out):
    from .semantic import Clip
    started=perf_counter()
    goals=read(out/"public_goals.json")
    locked=read(out/"protocol_lock.json")
    assert sha(out/"public_goals.json")==locked["public_goals"]["sha256"]
    terms=sorted({term for goal in goals for term in goal_spec(goal["text"])["terms"]})
    text_error=None
    try:
        vectors=dict(zip(terms,Clip(out,vision=False).text([f"a photo of a {word}." for word in terms]),strict=True))
    except Exception as exc:
        vectors={}
        text_error=f"{type(exc).__name__}: {exc}"
    group=defaultdict(list)
    for goal in goals:
        group[goal["case"]].append(goal)
    destination=out/"retrieval"
    destination.mkdir(exist_ok=True)
    def one_case(case):
        book=Library(out,case,vectors)
        case_results={}
        for arm in RULES["arms"]:
            folder=destination/case/arm
            folder.mkdir(parents=True,exist_ok=True)
            dump(folder/"overview.json",book.overview(arm))
            for goal in group[case]:
                name=f'{case}/{arm}/{goal["id"]}'
                path=folder/f'{goal["id"]}.json'
                if not path.exists():
                    dump(path,query(book,goal,arm))
                case_results[name]=ref(path)
        print(case,len(case_results),flush=True)
        return case_results
    with ThreadPoolExecutor(max_workers=3) as pool:
        trials={key:value for found in pool.map(one_case,group) for key,value in found.items()}
    dump(destination/"manifest.json",{"trials":trials,"expected":len(goals)*len(RULES["arms"]),"text_encoder_failure":text_error,
          "clock_seconds":perf_counter()-started,"receipt_scope":"Real native consumer file opens; this offline study has no M9 model SDK handoff."})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.out.resolve())
