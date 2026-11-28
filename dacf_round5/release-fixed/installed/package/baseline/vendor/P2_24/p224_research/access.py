"""Pure metadata navigation and exact-source, reasoned native delivery."""

from collections import Counter
from math import isfinite
from pathlib import Path

from if20_research.direct import inspect_png


ALIASES={"rise":"_rise","fall":"_fall","plateau":"plateau","equal_height_change":"equal_count","peak":"_local_peak","gap":"unobserved_gap","transition":"transition_bracket"}


class Video:
    def __init__(self,packet):
        self.packet=packet
        self.goal=packet["goal"]
        self.source=packet["proposal"]["source"]
        lo,hi=self.source["start"],self.source["end"]
        self.scope=(lo+(hi-lo)*2/3 if self.goal.get("scope_hint")=="late" else lo,hi)
        self.by_id={r["candidate_id"]:r for r in packet["proposal"]["batch"] if self.scope[0]<=r["seconds"]<self.scope[1]}
        self.by_pts={r["pts"]:r for r in self.by_id.values()}
        self.ordered=sorted(self.by_id.values(),key=lambda r:(r["seconds"],r["source_frame"]))
        self.features=packet["features"]
        self.waves=packet["navigation"]["wave"]
        self.cues=packet["navigation"]["cues"]

    def bounds(self,start,end):
        a=self.scope[0] if start is None else float(start); b=self.scope[1] if end is None else float(end)
        if not (isfinite(a) and isfinite(b) and a<=b):
            raise ValueError("Invalid source interval")
        return max(self.scope[0],a),min(self.scope[1],b)

    def item(self,r):
        v=self.features[r["candidate_id"]]
        return {"frame_id":r["candidate_id"],"pts":r["pts"],"time_base":r["time_base"],"seconds":r["seconds"],
                "source_kind":"catalog_original" if r["in_catalog"] else "approved_source_original",
                "dimensions":r["dimensions"],"I_proxy":v["I"],"R_proxy":v["R"],"state":"eligible"}

    def catalog_search(self,start=None,end=None,dimension=None,mode="overview",cursor=0,page_size=12):
        lo,hi=self.bounds(start,end)
        rows=[r for r in self.ordered if lo<=r["seconds"]<=hi and (not dimension or dimension in r["dimensions"])]
        page=max(1,min(30,int(page_size)))
        offset=max(0,int(cursor))
        if mode=="overview" and len(rows)>page:
            targets=[rows[round(j*(len(rows)-1)/max(1,page-1))] for j in range(page)]
            next_page=None
        elif mode=="earliest" or mode=="overview":
            targets=rows[offset:offset+page];next_page=offset+page if offset+page<len(rows) else None
        elif mode=="latest":
            targets=rows[::-1][offset:offset+page];next_page=offset+page if offset+page<len(rows) else None
        else:
            raise ValueError("Unsupported catalogue sorting mode")
        return {"source_public_range":list(self.scope),"resolved_limits":[lo,hi],"eligible_count":len(rows),
                "catalogue_count":sum(bool(r["in_catalog"]) for r in rows),"next_cursor":next_page,"frames":[self.item(r) for r in targets],
                "metadata_not_images":True,"proxy_fields_not_fact":True}

    def wave_query(self,start=None,end=None,signal="all",cursor=0,page_size=12):
        lo,hi=self.bounds(start,end)
        if signal!="all" and signal not in ALIASES:
            raise ValueError("Unrecognized wave signal")
        matches=[]
        for c in self.cues:
            if not all(k in self.by_id for k in c["source_candidates"]):
                continue
            x,y=c["suggested_window_seconds"]; labels=c["flags"]
            if lo<=y and x<=hi and (signal=="all" or any(ALIASES[signal] in k for k in labels)):
                a,b=(self.by_id[k] for k in c["source_candidates"])
                roots=sorted(set(a["evidence_roots"]+b["evidence_roots"]+self.features[b["candidate_id"]]["information_roots"]))
                matches.append({"window":[x,y],"flags":labels,"delta_I":c["delta_I"],"delta_R":c["delta_R"],
                    "candidate_endpoints":[self.item(a),self.item(b)],"root_count":len(roots),"root_ids":roots,
                    "state":"investigate","note":"discrete candidates only; changes and objects unverified"})
        if signal=="all":
            def useful(q):
                return ("transition_bracket" in q["flags"],"unobserved_gap" in q["flags"],
                        "relation_signature_change_at_equal_count" not in q["flags"],-(abs(q["delta_I"])+abs(q["delta_R"])),q["window"][0])
            matches.sort(key=useful)
        offset=max(0,int(cursor));page=max(1,min(30,int(page_size)))
        return {"source_public_range":list(self.scope),"resolved_limits":[lo,hi],"signal":signal,"matched":len(matches),
                "next_cursor":offset+page if offset+page<len(matches) else None,"hints":matches[offset:offset+page],
                "unknown_intervals":[[max(lo,a),min(hi,b)] for a,b in self.waves["unknown_intervals"] if max(lo,a)<min(hi,b)],
                "bytes_of_images":0,"metadata_only":True,"delivery_receipts":0}

    def delivery_request(self,ids=None,pts=None,reason=""):
        if not isinstance(reason,str) or len(reason.strip())<8:
            raise ValueError("Give the observable goal requirement for requesting the original image")
        simple=reason.lower().strip().rstrip(".")
        if simple in {"wave rise","wave peak","wave changed","because the wave rises","波形上升","因为波形上升"}:
            raise ValueError("Wave flags permit investigation only; state what comparison or view the goal needs")
        requested=[self.by_id[k] for k in (ids or [])]+[self.by_pts[p] for p in (pts or [])]
        if not requested:
            raise ValueError("Use frame identifiers or exact integer PTS returned by the bound tools")
        if len(requested)!=len({r["candidate_id"] for r in requested}):
            raise ValueError("Each frame may appear once per request; repeat requests are tracked across calls")
        images=[]
        for r in requested:
            path=Path(r["image"]["path"])
            bits=path.read_bytes();native=inspect_png(bits)
            if native["sha256"]!=r["image"]["sha256"] or native["width"]!=self.source["probe"]["width"] or native["height"]!=self.source["probe"]["height"]:
                raise ValueError("Source native PNG identity or geometry mismatch")
            images.append((self.item(r)|{"observation_need":reason,"state":"deliver","native":native,"path":str(path),
                           "source_frame":r["source_frame"],"source_sha256":r["source_sha256"]},bits))
        return images
