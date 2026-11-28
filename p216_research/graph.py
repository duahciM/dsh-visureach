"""Build source-bound navigation edges and deterministic, inspectable frontiers."""

from collections import defaultdict, deque
from statistics import median

import numpy as np

from if17_research.catalog import read
from p215_research.retrieval import Library, goal_spec

from .prepare import CFG


R_ROUTES={"U0":"R0","U1":"R1","U2":"R2","U3":"R3","U4":"R4","U5":"R5"}
OBJECT={"U2","U4","U5"}
GRAPH={"U3","U4","U5"}


class Book(Library):
    def __init__(self,out,case,paths,vectors):
        self.out=out
        self.case=case
        self.paths=paths
        self.source_book=read(paths["catalog"]["path"])
        self.source=self.source_book["source"]
        self.rows=self.source_book["frames"]
        self.by={r["frame_id"]:r for r in self.rows}
        self.ledger=read(paths["index"]["path"])
        self.pool=read(paths["coverage"]["path"])["candidate_frames"]
        semantic=read(paths["semantic"]+"/candidates.json")
        self.sem=semantic["rows"]
        if [r["frame_id"] for r in self.sem]!=list(self.by):
            raise ValueError("Frozen semantic index membership differs")
        self.semantic_available=True
        self.features=np.load(paths["semantic"]+"/vectors.npy",allow_pickle=False)
        self.query_vectors=vectors
        self.time_order=list(self.by)


def build(book):
    edges=defaultdict(list)
    missing=[]
    roots=book.ledger["filters"]["root"]
    groups=book.ledger["filters"]["time_group"]
    nodes={}
    windows={}
    seen=set()
    contract=book.source_book["coordinate_contract"]
    w,h=book.source["probe"]["width"],book.source["probe"]["height"]
    for key,row in book.by.items():
        windows[key]={o["analysis_window"] for o in row["observations"] if o["analysis_window"] is not None}
        regions=[]
        for obs in row["observations"]:
            if obs["region_candidate"]:
                x,y,rw,rh=obs["region_xywh_analysis"]
                sx,sy=w/contract["analysis_width"],h/contract["analysis_height"]
                regions.append({"node_type":"region","id":obs["region_candidate"],"root_ids":obs["evidence_root_ids"],
                                "region_original_pixels":[x*sx,y*sy,rw*sx,rh*sy]})
        nodes[key]={"node_type":"frame","id":key,"source_sha256":book.source["sha256"],"pts":row["pts"],
                    "source_frame":row["source_frame"],"time_base":row["time_base"],"seconds":row["t"],
                    "image_ref":row["source_image"],"regions":regions,"windows":sorted(windows[key]),
                    "root_ids":sorted({x for o in row["observations"] for x in o["evidence_root_ids"]})}

    def link(a,b,kind,root_ids=()):
        token=(a,b,kind,tuple(sorted(root_ids)))
        if a==b or token in seen:
            return
        seen.add(token)
        edge={"from_id":a,"to_id":b,"kind":kind,"root_ids":list(token[3]),"relation_status":"unknown",
              "reason":{"comparison":"original nomination comparison", "same_source_root":"same evidence family root",
                        "temporal_near":"consecutive in time group with a shared non-timing dimension"}[kind]}
        if b not in nodes:
            missing.append(edge|{"endpoint_status":"comparison_endpoint_outside_library"})
            return
        edge["cross_window"]=bool(windows[a] and windows[b] and not windows[a]&windows[b])
        edges[a].append(edge)

    for key,row in book.by.items():
        for obs in row["observations"]:
            for dest in obs["source_comparison_frames"]:
                link(key,dest,"comparison",obs["evidence_root_ids"])
    for root,members in roots.items():
        for a,b in zip(members,members[1:]):
            link(a,b,"same_source_root",[root])
            link(b,a,"same_source_root",[root])
    for members in groups.values():
        for a,b in zip(members,members[1:]):
            shared=set(book.by[a]["image_dimensions"])&set(book.by[b]["image_dimensions"])-{"D6"}
            if shared:
                link(a,b,"temporal_near")
                link(b,a,"temporal_near")
    return {"source_sha256":book.source["sha256"],"frames":nodes,"edges":dict(edges),"missing_endpoints":missing,
            "root_nodes":{root:{"node_type":"evidence_root","member_ids":members} for root,members in roots.items()},
            "semantic_nodes_reference":book.paths["semantic"]+"/candidates.json",
            "edge_scope":"Retrieval links. All physical identity, continuity, and causality remain unknown."}


def entries(book,goal,arm):
    spec=goal_spec(goal["text"])
    terms=list(spec["terms"])
    discovered=[]
    if arm in OBJECT and not terms:
        frames=defaultdict(set)
        for row in book.sem:
            for node in row["nodes"]:
                for candidate in node["visual_hypotheses"]:
                    frames[candidate["word"]].add(row["frame_id"])
        recurring={term for term,found in frames.items() if 2<=len(found)<len(book.rows)}
        first=[term for term in book.discovery_terms(R_ROUTES[arm]) if term in recurring]
        discovered=first+sorted(recurring-set(first))
        spec["terms"]=discovered
    if arm=="U0":
        return {"time:full_chronological":book.time_order}, spec, discovered, {}
    proposed=book.search(R_ROUTES[arm],spec)
    channels={name:seq for name,seq in proposed.items() if name not in ("time","region","root","unknown")}
    unknown=book.ledger["filters"]["uncertainty"].get("unknown_or_competing_explanations",[])
    channels["unknown"]=unknown
    anchors=book.ledger["filters"]["dimension"].get("D6",[])
    channels["time:frozen_D6_anchors"]=anchors
    slots={}
    if arm=="U5":
        dims=book.ledger["filters"]["dimension"]
        choices={"before":anchors,"after":list(reversed(anchors)),"middle":channels.get("D2",dims.get("D2",[])),
                 "gap":unknown,"reference":dims.get("D4",[]) or dims.get("D3",[]),
                 "anchor":dims.get("D1",[]) or anchors}
        slots={slot:choices[slot] for slot in spec["slots"]}
        channels={**{f"role:{slot}":seq for slot,seq in slots.items()},**channels}
    return channels,spec,discovered,slots


def agenda(channels,adjacency):
    """Enumerate every applicable metadata cursor, draining each wave's graph frontier."""
    width=CFG["transport_page_size"]
    parents={}
    order=[]
    pages=[]
    offered=defaultdict(list)
    initial=[]
    duplicate_edges=0
    longest=max(map(len,channels.values()),default=0)
    for name,ids in channels.items():
        if not ids:
            pages.append({"entry":name,"cursor":0,"next_cursor":None,"ids":[],"status":"empty_index_hit"})
    for offset in range(0,longest,width):
        pending=deque()
        for name,ids in channels.items():
            page=ids[offset:offset+width]
            if not page:
                continue
            end=offset+len(page)
            pages.append({"entry":name,"cursor":offset,"next_cursor":end if end<len(ids) else None,"ids":page})
            if offset==0:
                initial.extend(page)
            for key in page:
                offered[key].append(name)
                if key not in parents:
                    parents[key]={"entry":name,"cursor":offset,"preceding_edge":None}
                    pending.append(key)
        while pending:
            current=pending.popleft()
            order.append(current)
            for edge in adjacency.get(current,[]):
                dest=edge["to_id"]
                if dest in parents:
                    duplicate_edges+=1
                    continue
                parents[dest]={"entry":None,"preceding_edge":edge}
                pending.append(dest)
    return {"metadata_pages":pages,"first_wave_seeds":list(dict.fromkeys(initial)),"entry_hits":dict(offered),
            "generated_order":order,"parent_links":parents,"suppressed_graph_revisits":duplicate_edges}


def window_plan(book,goal,arm,graph,order):
    config=CFG["source"]
    lo,hi=book.source["start"],book.source["end"]
    coarse=sorted({p["t"] for p in book.pool})
    cadence=median([b-a for a,b in zip(coarse,coarse[1:])]) if len(coarse)>1 else hi-lo
    threshold=config["sparse_cadence_multiple"]*cadence
    proposals=[]
    def propose(a,b,reason,anchor=None):
        a,b=max(lo,a),min(hi,b)
        if b>a:
            proposals.append({"start":a,"end":b,"reason":reason,"anchor":anchor})

    def sparse(times,reason):
        vals=sorted({lo,hi,*times})
        for a,b in zip(vals,vals[1:]):
            if b-a>threshold+1e-6:
                propose(a,b,reason)

    sparse((row["t"] for row in book.rows),"catalog cadence gap")
    text=goal["text"].casefold()
    explicit="inspect between samples" in text or "inspect between keyframes" in text
    if explicit:
        propose(lo,hi,"explicit source detail request")
    if arm=="U5":
        sparse((book.by[key]["t"] for key in order),"retrieval path cadence gap")
        active=set(order)
        for key in book.ledger["filters"]["uncertainty"].get("unknown_or_competing_explanations",[]):
            if key in active:
                center=book.by[key]["t"]
                propose(center-cadence,center+cadence,"near a frozen unknown or conflict",key)
        time_by_pool_frame={r["source_frame"]:r["t"] for r in book.pool}
        for edge in graph["missing_endpoints"]:
            if edge["from_id"] not in active:
                continue
            video,index,*_=edge["to_id"].split(":")
            if video==book.source["sha256"] and int(index) in time_by_pool_frame:
                center=time_by_pool_frame[int(index)]
                propose(center-cadence,center+cadence,"direct comparison endpoint outside the catalog",edge["to_id"])
    unions=[]
    for row in sorted(proposals,key=lambda p:(p["start"],p["end"])):
        if unions and row["start"]<=unions[-1][1]:
            unions[-1][1]=max(row["end"],unions[-1][1])
        else:
            unions.append([row["start"],row["end"]])
    windows=[]
    for low,high in unions:
        pos=low
        while pos<high-1e-8:
            end=min(high,pos+config["chunk_seconds"])
            contributors=[i for i,p in enumerate(proposals) if p["start"]<end and p["end"]>pos]
            windows.append({"node_type":"source_window","start":pos,"end":end,
                            "fps":config["explicit_detail_rate"] if explicit else config["initial_rate"],
                            "proposal_indexes":contributors})
            pos=end
    return {"median_coarse_cadence_seconds":cadence,"proposals":proposals,"windows":windows}
