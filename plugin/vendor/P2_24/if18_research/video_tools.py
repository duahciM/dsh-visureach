"""Model-facing catalog and original-source tools restricted to one bound video."""
from collections import Counter
from pathlib import Path
from math import isfinite
from time import perf_counter

from p1.io import dump
from if17_research.browse import as_pixels,rendered_page,read_original_range
from if17_research.catalog import read,sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PRIOR=ROOT/"if17_research_20261120"
PROTOCOL=read(HERE/"protocol.json")
CAP=PROTOCOL["caps"]
CASES={"g3":"new_6a8c74a6","short":"new_a31953d0","long":"new_4740e05f"}
LOW_LEVEL={"D1":"registered local residual","D2":"pairwise local difference",
    "D3":"local region to stable image-edge distance","D4":"registered image displacement",
    "D5":"detection gap or uncertain registration","D6":"timing anchor"}
MD_FIELDS={"dimension":"dimension","root":"root","region_candidate":"region_candidate",
           "relationship":"relationship","uncertainty":"uncertainty"}


def rgb_receipt(src,row):
    return {"source_sha256":src["sha256"],"pts":row["pts"],"source_frame":row["source_frame"],
           "time_base":row["time_base"],"seconds":row["t"],"native_width":src["probe"]["width"],
           "native_height":src["probe"]["height"],"orientation":"original view, unmirrored"}


def exact_integer(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,str)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


class VideoTools:
    def __init__(self,case,path,multidimensional,inject_read_failure=False):
        self.case=case
        self.path=path
        self.multidimensional=multidimensional
        self.inject_read_failure=inject_read_failure
        self.calls=[]
        self.events=[]
        self.delivered={}
        self.original_windows={}
        self.ranges_attempted=0
        self.last_new=[]
        d=PRIOR/"P2_13"/CASES[case]
        catalog=read(d/"catalog.json")
        self.source=catalog["source"]
        self.coordinate=catalog["coordinate_contract"]
        self.rows=catalog["frames"]
        self.by_pts={x["pts"]:x for x in self.rows}
        self.index=read(d/"index.json")
        self.old_pool={r["pts"]:r["in_catalog"] for r in read(d/"coverage.json")["candidate_frames"]}
        assert len(self.by_pts)==len(self.rows)
        self.lock={"catalog_sha256":sha(d/"catalog.json"),"index_sha256":sha(d/"index.json"),
                   "source_sha256":self.source["sha256"],"catalog_frames":len(self.rows),
                   "bound_case":case}

    def bounds(self,args):
        a=float(args.get("start",self.source["start"]))
        b=float(args.get("end",self.source["end"]))
        if not isfinite(a) or not isfinite(b) or b<a:
            raise ValueError("Require finite increasing seconds")
        return a,b

    def page(self,args,maximum):
        n=exact_integer(args.get("limit",min(6,maximum)),"limit")
        k=exact_integer(args.get("cursor",0),"cursor")
        if not 1<=n<=maximum or k<0:
            raise ValueError(f"Use limit 1–{maximum} and cursor >= 0")
        return k,n

    def overview(self,args,d):
        if args:
            raise ValueError("overview takes an empty args object")
        src=self.source
        a,b=src["start"],src["end"]
        step=30 if b-a>30 else b-a
        windows=[[a+i*step,min(b,a+(i+1)*step)] for i in range(max(1,int((b-a-1e-8)//step)+1))]
        overview={"source_span_seconds":[a,b],"source_sha256":src["sha256"],"time_base":src["time_base"],
             "archived_frames":len(self.rows),"source_frame_count":src["source_frame_count"],
             "time_browsing_windows":windows,"native_size":[src["probe"]["width"],src["probe"]["height"]],
             "coverage_note":"Catalog includes candidates. Visible states and physical identity remain unverified."}
        if self.multidimensional:
            overview["low_level_index"]={k:{"meaning":name,"matches":len(self.index["filters"]["dimension"].get(k,[]))}
                                        for k,name in LOW_LEVEL.items()}
            overview["coordinate_contract"]=self.coordinate
        return overview,[],[]

    def search(self,args,d):
        extra=set(args)-{"start","end","cursor","limit",*MD_FIELDS,"region"}
        if extra:
            raise ValueError(f"Unknown search fields: {sorted(extra)}")
        if not self.multidimensional and (set(args)&set(MD_FIELDS)|({"region"}&set(args))):
            raise ValueError("This session exposes the time directory only; start/end/cursor/limit are available")
        lo,hi=self.bounds(args)
        offset,size=self.page(args,CAP["catalog_page"])
        retain=set(self.index["timeline"])
        filters=self.index["filters"]
        for name,field in MD_FIELDS.items():
            if name in args:
                values=args[name] if isinstance(args[name],list) else [args[name]]
                retain.intersection_update({f for v in values for f in filters[field].get(str(v),[])})
        area=args.get("region")
        if area is not None:
            if not isinstance(area,list) or len(area)!=4:
                raise ValueError("region is [x,y,width,height] in analysis pixels")
            x,y,w,h=map(float,area)
            if not all(map(isfinite,(x,y,w,h))) or w<=0 or h<=0:
                raise ValueError("region must be a finite positive rectangle")
            def spatial_match(r):
                return any(a<x+w and x<a+c and b<y+h and y<b+d for a,b,c,d in
                           (z["region_xywh_analysis"] for z in r["observations"] if z["region_xywh_analysis"]))
        else:
            def spatial_match(r):
                return True
        found=[r for r in self.rows if lo<=r["t"]<=hi and r["frame_id"] in retain and spatial_match(r)]
        choose=found[offset:offset+size]
        result=[]
        for r in choose:
            item=rgb_receipt(self.source,r)
            if self.multidimensional:
                # Only native automatic geometry labels, never sealed evaluation roles or filenames.
                available=[o for o in r["observations"] if o["region_candidate"]]
                item.update(dimensions=r["image_dimensions"],low_level_regions=[
                   {"candidate":z["region_candidate"],"dimensions":z["dimension_id"],"xywh":z["region_xywh_analysis"],
                    "root":z["evidence_root_ids"][0]} for z in available[:2]])
            result.append(item)
        next_offset=offset+len(choose)
        return {"matches":len(found),"cursor":offset,"next_cursor":next_offset if next_offset<len(found) else None,
              "frames":result,"coordinate_space":self.coordinate if self.multidimensional else None,
              "images_returned":0,"note":"Metadata page only. No images observed from this search."},[],[]

    def atlas(self,rows,d):
        src=self.source
        pictures=[]
        out=[]
        prior=None
        for i,f in enumerate(sorted(rows,key=lambda f:f["pts"])):
            r=rgb_receipt(src,f)|{"ordinal":f'{d.name}-{i+1:02d}',"row":i//4+1,"column":i%4+1,
                "previous_displayed_seconds":prior,
                "open_gap_before_tile":[prior,f["t"]] if prior is not None else None}
            r["frozen_scope"]="indexed" if f["pts"] in self.by_pts else "sampled_not_nominated" if f["pts"] in self.old_pool else "not_in_frozen_pool"
            r["previously_delivered"]=f["pts"] in self.delivered
            out.append(r)
            pictures.append(f)
            prior=f["t"]
        if not out:
            return {"images_returned":0,"frames":[]},[],[]
        chart=d/"source_images.png"
        rendered_page(pictures,chart,out,0)
        file_hash=sha(chart)
        for r,f in zip(out,pictures):
            r["image_sheet_sha256"]=file_hash
            r["native_png_sha256"]=sha(f["native_image"])
        meta={"images_returned":len(out),"frames":out,
              "coordinate_space":"native display pixels unless a candidate is marked in analysis pixels",
              "observed_time_scope":"Only these discrete image times. The open intervals are gaps in this batch."}
        return meta,[chart],out

    def read_frames(self,args,d):
        if "range_token" in args:
            token=str(args["range_token"])
            if token not in self.original_windows:
                raise ValueError("Unknown source-range token in this session")
            offset,size=self.page(args,CAP["image_batch"])
            src_rows=self.original_windows[token]
            part=src_rows[offset:offset+size]
            meta,files,receipts=self.atlas(part,d)
            return meta|{"range_token":token,"next_cursor":offset+len(part) if offset+len(part)<len(src_rows) else None},files,receipts
        pts=args.get("pts")
        if not isinstance(pts,list) or not 1<=len(pts)<=CAP["image_batch"]:
            raise ValueError(f'pts must be an integer array of 1–{CAP["image_batch"]} items')
        ids=sorted({exact_integer(v,"pts") for v in pts})
        available=[self.by_pts[k] for k in ids if k in self.by_pts]
        absent=[k for k in ids if k not in self.by_pts]
        if available and self.inject_read_failure:
            self.inject_read_failure=False
            raise RuntimeError("Injected transient catalog attachment failure; no image was returned. Retry the same request.")
        meta,images,proof=self.atlas([as_pixels(r) for r in available],d)
        return meta|{"requested_pts":ids,"not_returned_pts":absent},images,proof

    def read_range(self,args,d):
        if self.ranges_attempted>=CAP["source_ranges"]:
            raise ValueError("Original-range safety limit reached; catalog or already-read source pages are still available")
        a,b=self.bounds(args)
        rate=float(args.get("fps",5))
        if not isfinite(rate) or not (self.source["start"]<=a<b<=self.source["end"]+1e-8
                and b-a<=CAP["max_range_s"]+1e-8 and 1<=rate<=CAP["max_fps"]):
            raise ValueError(f'Use range inside source with width <= {CAP["max_range_s"]} s and fps 1–{CAP["max_fps"]}')
        self.ranges_attempted+=1
        dumped=read_original_range({"source":self.source,"frames":self.rows},d/"original",a,min(b,self.source["end"]),rate)
        rows=sorted(dumped["returned"],key=lambda f:f["pts"])
        token=d.name
        self.original_windows[token]=rows
        part=rows[:CAP["image_batch"]]
        meta,images,proof=self.atlas(part,d)
        return meta|{"range_token":token,"requested_seconds":[a,b],"requested_fps":rate,
               "original_candidates":len(rows),"next_cursor":len(part) if len(part)<len(rows) else None,
               "verified_original_decode":True},images,proof

    def record(self,args,d):
        pts=args.get("pts",[])
        if not isinstance(pts,list) or not pts or any(exact_integer(x,"pts") not in self.delivered for x in pts):
            raise ValueError("Observation must cite already delivered integer PTS values")
        item={"claim":args.get("claim",""),"pts":pts,"status":"source references valid; visual claim still to assess"}
        self.events.append(item)
        return item,[],[]

    def dispatch(self,action,turn):
        tool=action.get("tool")
        args=action.get("args",{})
        why=action.get("why")
        n=len(self.calls)+1
        d=self.path/f"call_{n:02d}"
        d.mkdir()
        started=perf_counter()
        result={"call":n,"turn_requested":turn,"tool":tool,"why":why,"args":args,
                "source_sha256":self.source["sha256"]}
        fs,images=[],[]
        func={"catalog_overview":self.overview,"search_frames":self.search,"read_frames":self.read_frames,
              "read_source_range":self.read_range,"record_observation":self.record}.get(tool)
        try:
            if func is None or not isinstance(args,dict) or not isinstance(why,str) or not why.strip():
                raise ValueError("Provide a known tool, an args object and a concrete why")
            data,images,fs=func(args,d)
            result.update(status="returned",reply=data)
        except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
            message=str(error).replace(self.source["path"],"<bound video>").replace(Path(self.source["path"]).name,"<bound video>").replace(str(self.path),"<session>")
            result.update(status="error",internal_error=str(error),reply={"error":message,"images_returned":0})
        result.update(seconds=perf_counter()-started,
           queued_images=[{"path":str(p),"sha256":sha(p)} for p in images],frame_receipts=fs,
           returned_to_observer=False,returned_on_turn=None)
        self.calls.append(result)
        dump(d/"tool_receipt.json",result)
        return result,images

    def confirm(self,turn,reports,attachments):
        delivered=set(a["sha256"] for a in attachments)
        added=set()
        for r in reports:
            if not {a["sha256"] for a in r["queued_images"]}<=delivered:
                raise ValueError("Harness attachment receipt did not match queued tool reply")
            r["returned_to_observer"]=True
            r["returned_on_turn"]=turn
            r["new_pts_for_observer"]=[]
            for f in r["frame_receipts"]:
                pts=f["pts"]
                if pts not in self.delivered:
                    added.add(pts)
                    r["new_pts_for_observer"].append(pts)
                self.delivered[pts]=f
            dump(self.path/f'call_{r["call"]:02d}'/"tool_receipt.json",r)
        self.last_new.append(len(added))

    def summary(self):
        f=sorted(self.delivered.values(),key=lambda row:row["seconds"])
        receipts=[r for r in self.calls if r["returned_to_observer"]]
        actual=sum(len(r["frame_receipts"]) for r in receipts)
        times=[self.source["start"],*[r["seconds"] for r in f],self.source["end"]]
        return {"calls":len(self.calls),"tool_seconds":sum(r["seconds"] for r in self.calls),
                "tool_counts":dict(Counter(r["tool"] for r in self.calls)),
                "actual_read_batches":sum(bool(r["frame_receipts"]) for r in receipts),
                "distinct_read_frames":len(f),"repeat_frame_deliveries":actual-len(f),
                "source_range_attempts":self.ranges_attempted,
                "parts_not_returned":sum(len(r["reply"].get("not_returned_pts",[])) for r in self.calls),
                "delivered_pts":[v["pts"] for v in f],"delivered":f,
                "unobserved_open_intervals":[[a,b] for a,b in zip(times,times[1:]) if b>a],
                "evidence_gain_by_tool_round":self.last_new}

