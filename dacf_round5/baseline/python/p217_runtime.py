"""Frozen P2-17 comparison and feedback runtime, with archived intake imports removed."""
from collections import defaultdict
from hashlib import sha256
from math import hypot
from statistics import median
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

from if17_research.catalog import read
from p1.core import Parameters, pair_metrics, quality

CFG = json.loads((Path(__file__).resolve().parents[1] / 'vendor/P2_24/p217_research/protocol.json').read_text(encoding='utf-8'))
PARAMS = Parameters()

def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))


def visual(path,h):
    with Image.open(path) as image:
        rgb=np.asarray(image.convert("RGB").resize((320,h),Image.Resampling.BILINEAR))
    return rgb,cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)


def identifier(*parts):
    return sha256("|".join(map(str,parts)).encode()).hexdigest()[:20]


class Case:
    def __init__(self,out,name,entry):
        self.out,self.case,self.entry=out,name,entry
        self.source_book=read(entry["catalog"]["path"])
        self.source=self.source_book["source"]
        self.rows=self.source_book["frames"]
        self.by={r["frame_id"]:r for r in self.rows}
        self.coverage=read(entry["coverage"]["path"])["candidate_frames"]
        self.pool_by_index={r["source_frame"]:r for r in self.coverage}
        self.contract=self.source_book["coordinate_contract"]
        self.height=self.contract["analysis_height"]
        self.sem=None
        self.pixels={}
        self.pair_info={}
        cadence=sorted({r["t"] for r in self.coverage})
        self.cadence=median([b-a for a,b in zip(cadence,cadence[1:])]) if len(cadence)>1 else self.source["end"]-self.source["start"]

    def native_box(self,box):
        if box is None:
            return None
        sx=self.source["probe"]["width"]/320
        sy=self.source["probe"]["height"]/self.height
        return [round(v*s,3) for v,s in zip(box,(sx,sy,sx,sy))]

    def stamp(self,key,box=None):
        if key is None:
            return None
        row=self.by[key]
        return {"id":key,"pts":row["pts"],"time_base":row["time_base"],"source_frame":row["source_frame"],
                "t":row["t"],"image_ref":row["source_image"],"region_original_pixels":self.native_box(box)}

    def image(self,key):
        if key not in self.pixels:
            self.pixels[key]=visual(self.by[key]["source_image"]["path"],self.height)
        return self.pixels[key]

    def compare(self,left,right):
        key=(left,right)
        if key in self.pair_info:
            return self.pair_info[key]
        ar,a=self.image(left)
        cr,c=self.image(right)
        m=pair_metrics(a,c,PARAMS)
        q=[quality(a)["sharpness"],quality(c)["sharpness"]]
        delta=(c.astype(np.float32)-a)/255
        exposure=float(np.median(delta))
        consistency=float(np.mean(np.sign(exposure)*delta>=CFG["exposure_median_abs_min"]/2))
        exposure_like=abs(exposure)>=CFG["exposure_median_abs_min"] and consistency>=CFG["exposure_agreement"]
        competing=[]
        shift=hypot(m["dx"],m["dy"])
        if min(q)<PARAMS.min_sharpness:
            competing.append("low anchor sharpness")
        if not m["registration_valid"]:
            competing.append("registration unsupported; viewpoint, edit or ambiguous texture")
        if shift>CFG["registered_shift_limit_at_320_px"] or m["motion_supported"]:
            competing.append("viewpoint or wide translation may explain the difference")
        if exposure_like:
            competing.append("global signed brightness shift")
        label="no" if exposure_like or m["motion_supported"] else "unknown" if min(q)<PARAMS.min_sharpness or not m["registration_valid"] else "partial" if competing else "yes"
        result=m|{"quality":q,"exposure_median":exposure,"exposure_agreement":consistency,"comparable":label,
                  "competing_explanations":competing,"unaligned_rgb_mad":float(np.mean(np.abs(cr.astype(np.float32)-ar))/255)}
        self.pair_info[key]=result
        return result


def clip_box(box,h):
    if box is None:
        return None
    x,y,w,b=box
    x1,y1=max(0,round(x)),max(0,round(y))
    x2,y2=min(320,round(x+w)),min(h,round(y+b))
    return [x1,y1,x2-x1,y2-y1] if x2>x1 and y2>y1 else None


def shifted(box,dx,dy,h):
    return clip_box([box[0]+dx,box[1]+dy,*box[2:]],h) if box else None


def translated(img,dx,dy):
    h,w=img.shape[:2]
    return cv2.warpAffine(img,np.float32([[1,0,dx],[0,1,dy]]),(w,h),borderMode=cv2.BORDER_REPLICATE)


def roi(img,box):
    x,y,w,h=box
    return img[y:y+h,x:x+w].astype(np.float32)


def distance(a,c,box):
    return float(np.mean(np.abs(roi(a,box)-roi(c,box)))/255)


def dim_rows(book):
    output={d:defaultdict(list) for d in ("D2","D3","D4","D5")}
    missing=[]
    ordered={r["frame_id"]:r["t"] for r in book.rows}
    for row in book.rows:
        src=row["frame_id"]
        for n in row["nominations"]:
            d=n["dimension_id"]
            if d not in output:
                continue
            others=[x for x in n["comparison_frame_ids"] if x!=src]
            if (d=="D5" and n["role"]=="sampled_detection_gap" and len(n["comparison_frame_ids"])==2):
                pairs=[tuple(n["comparison_frame_ids"])]
            else:
                pairs=[(src,x) for x in others]
            for a,c in pairs:
                if a==c:
                    continue
                if a not in ordered or c not in ordered:
                    missing.append({"dimension":d,"observed_id":src,"comparison_ids":[a,c],"root_ids":n["evidence_root_ids"],
                                    "state":"comparison_endpoint_outside_catalog"})
                    continue
                a,c=sorted((a,c),key=ordered.get)
                if d=="D3" and ordered[c]-ordered[a]>CFG["relation_bracket_max_seconds"]+1e-6:
                    continue
                entry=output[d][(a,c)]
                if (src,n) not in entry:
                    entry.append((src,n))
    return output,missing


def kind_support(book,pair,channel,records,measure,d3_by):
    a,c=pair
    ar,_=book.image(a)
    cr,_=book.image(c)
    dx,dy=measure["dx"],measure["dy"]
    ar_reg=translated(ar,dx,dy) if measure["registration_valid"] else ar
    if channel=="D3":
        opts=[]
        for u in d3_by[a]:
            for v in d3_by[c]:
                if "context_edge_xy" not in u or "context_edge_xy" not in v:
                    continue
                ax,ay=u["context_edge_xy"]
                cx,cy=v["context_edge_xy"]
                context=hypot(ax+dx-cx,ay+dy-cy)
                earlier=shifted(u["applicable_region"],dx,dy,book.height)
                later=clip_box(v["applicable_region"],book.height)
                delta=abs(u["distance_px"]-v["distance_px"])
                if earlier and later and overlap(earlier,later)>0 and delta>=CFG["relation_distance_delta_px"] and context<=CFG["near_relation_context_tolerance_px"]:
                    opts.append((context,-delta,u,v,later,earlier))
        if not opts:
            return "relative_relation_difference",None,None,[],{},"no corroborated pair of local context-edge measurements"
        context,_,u,v,bc,ba=min(opts,key=lambda x:x[:2])
        return "relative_relation_difference",ba,bc,[(a,u),(c,v)],{"image_edge_distances_px":[u["distance_px"],v["distance_px"]],
               "context_locus_delta_px":context,"crop_mad":distance(ar_reg,cr,bc)},None
    if channel=="D2":
        local=[(origin,n) for origin,n in records if n["applicable_region"]]
        if not local:
            return "comparable_state_difference",None,None,records,{},"no local footprint"
        origin,n=max(local,key=lambda x:(x[0]==c,x[1]["priority_within_dimension"]))
        box=clip_box(n["applicable_region"],book.height)
        bc=box if origin==c else shifted(box,dx,dy,book.height)
        ba=shifted(bc,-dx,-dy,book.height)
        if not ba or not bc:
            return "comparable_state_difference",ba,bc,records,{},"no shared registered crop"
        return "comparable_state_difference",ba,bc,records,{"crop_mad":distance(ar_reg,cr,bc)},None
    named_gap=channel=="D5" and any(n["role"]=="sampled_detection_gap" for _,n in records)
    return ("conflicting_observations" if named_gap else "unexplained_structure"),None,None,records,{},None


def make_candidates(book):
    channels,missing=dim_rows(book)
    d3=defaultdict(list)
    for row in book.rows:
        d3[row["frame_id"]]=[n for n in row["nominations"] if n["dimension_id"]=="D3"]
    facts=[]
    for channel,pairs in channels.items():
        for pair,records in pairs.items():
            a,c=pair
            m=book.compare(a,c)
            kind,ba,bc,basis,metrics,reason=kind_support(book,pair,channel,records,m,d3)
            roots=sorted({r for _,n in basis for r in n["evidence_root_ids"]})
            if min(m["quality"])<PARAMS.min_sharpness:
                reason="low anchor sharpness"
            if kind=="comparable_state_difference" and not reason and metrics["crop_mad"]<PARAMS.activity_mad:
                reason="no independently measurable registered crop difference"
            strict=kind in ("comparable_state_difference","relative_relation_difference") and m["comparable"]!="yes"
            if channel in ("D4","D5") and not reason and m["raw_mad"]<PARAMS.activity_mad and kind!="conflicting_observations":
                reason="no measurable image difference"
            facts.append({"id":identifier(book.case,channel,*pair),"dimension":channel,"trigger_kind":kind,
                "source_sha256":book.source["sha256"],"observation_a":book.stamp(a,ba),"observation_c":book.stamp(c,bc),
                "a_id":a,"c_id":c,"crop_a":ba,"crop_c":bc,"comparison":m|metrics|{"same_region_candidate":bool(ba and bc),
                     "basis_root_ids":roots,"observed_difference":kind,"physical_instance":"unknown"},
                "support_nominations":[{"frame":origin,"root_ids":n["evidence_root_ids"],"role":n["role"]} for origin,n in basis],
                "base_abstention":reason,"strict_abstention":"local comparability insufficient" if strict else None})
    # Record what was visible for an input with no usable frozen comparison. These are diagnoses only.
    if not facts:
        for a,c in zip(book.rows,book.rows[1:]):
            m=book.compare(a["frame_id"],c["frame_id"])
            facts.append({"id":identifier(book.case,"absent",a["pts"],c["pts"]),"dimension":None,"trigger_kind":"no_frozen_visual_trigger",
                          "source_sha256":book.source["sha256"],"observation_a":book.stamp(a["frame_id"]),"observation_c":book.stamp(c["frame_id"]),
                          "comparison":m|{"basis_root_ids":[]},"base_abstention":"no frozen visual comparison nomination"})
    return facts,missing


def feedback(book,t,rows,samples):
    kind=t["trigger_kind"]
    comparable=t["comparison"].get("comparable","unknown")
    bc=t.get("crop_c")
    new=[]
    if kind in ("coverage_probe","explicit_source_request"):
        return {"result":"remains_unknown","reason":"coverage/explicit request did not assert a change", "observations":[]}
    if not rows:
        return {"result":"no_new_evidence","reason":"no new sampled source PTS", "observations":[]}
    a,_=book.image(t["a_id"])
    c,cg=book.image(t["c_id"])
    m=t["comparison"]
    if m["registration_valid"]:
        a=translated(a,m["dx"],m["dy"])
    for row in rows:
        key=row["pts"]
        if key not in samples:
            samples[key]=visual(row["path"],book.height)
        b,bg=samples[key]
        ac=pair_metrics(bg,cg,PARAMS)
        if kind=="unexplained_structure" and ac["motion_supported"]:
            result="clarifies_viewpoint"
            data={"source_relative_residual_ratio":ac["residual_ratio"]}
        elif bc and comparable=="yes" and ac["registration_valid"] and hypot(ac["dx"],ac["dy"])<=CFG["registered_shift_limit_at_320_px"] and quality(bg)["sharpness"]>=PARAMS.min_sharpness:
            aligned=translated(b,ac["dx"],ac["dy"])
            measures=[distance(a,aligned,bc),distance(c,aligned,bc)]
            result="supports_difference" if min(measures)>=CFG["feedback_crop_mad"] else "no_new_evidence"
            data={"distance_to_anchor_crops":measures}
        else:
            result="remains_unknown"
            data={"source_relative_registration":ac["registration_valid"]}
        new.append({"pts":key,"source_frame":row["source_frame"],"source_image":row["path"],"root_id":identifier(book.source["sha256"],key),
                    "result":result,"basis":"pixel-only candidate; no continuity or causal finding"}|data)
    priority=["clarifies_viewpoint","supports_difference","remains_unknown","no_new_evidence"]
    given={r["result"] for r in new}
    return {"result":next(k for k in priority if k in given),"observations":new,
            "new_pixel_root_ids":list(dict.fromkeys(r["root_id"] for r in new)),"physical_causation":"unknown"}
