"""Source-linked local comparisons, coverage probes, and concrete investigation gaps."""

from collections import defaultdict
from hashlib import sha256
from math import hypot
from statistics import median

import cv2
import numpy as np
from PIL import Image

from if17_research.catalog import read
from p1.core import Parameters, pair_metrics, quality
from p215_research.semantic import overlap

from .prepare import CFG


PARAMS=Parameters()
cv2.setNumThreads(1)
WORDS={
    "comparable_state_difference":("intermediate_state","Does an additional view in this region show a state distinct from the sampled endpoints?"),
    "relative_relation_difference":("interaction_clue","How does the image-space spacing from the local reference edge look between the samples?"),
    "conflicting_observations":("viewpoint_control","Does an intervening image show the area clearly, obscure it, or leave the correspondence uncertain?"),
    "unexplained_structure":("viewpoint_control","Does a source sample show the same structure under a comparable viewpoint, or is background/registration changing?"),
    "coverage_probe":("source_coverage","What is visible in this ordinary timestamp sample?"),
    "explicit_source_request":("intermediate_state","Read between the archived samples as explicitly requested by the user.")
}


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


def gap(book,goal,trigger,lo,hi):
    kind=trigger["trigger_kind"]
    missing,question=WORDS[kind]
    anchors=[item for item in (trigger.get("observation_a"),trigger.get("observation_c")) if item]
    known=sorted({lo,hi,*[r["t"] for r in book.rows if lo<r["t"]<hi]})
    goal_clause="the stated observation" if goal["text"] else "open exploration"
    return {"query_context":goal,"question":question,"missing_observation_type":missing,
            "observation_scope":goal_clause,"source_interval_seconds":[lo,hi],"time_base":book.source["time_base"],
            "anchor_refs":anchors,"already_catalogued_inner_pts":[r["pts"] for r in book.rows if lo<r["t"]<hi],
            "unsampled_catalog_intervals":[[a,c] for a,c in zip(known,known[1:]) if c>a],
            "epistemic_status":"not_yet_observed","finite_scope":"single measured footprint or stated source range"}


def trigger_with_interval(book,goal,t,lo,hi):
    return t|{"investigation_gap":gap(book,goal,t,lo,hi),"interval":[lo,hi],"decision":"proposed"}


def semantic_hints(book,t):
    if t["crop_c"] is None:
        return []
    if book.sem is None:
        path=book.entry["semantic"]+"/candidates.json"
        book.sem={r["frame_id"]:r for r in read(path)["rows"]}
    row=book.sem[t["c_id"]]
    box=book.native_box(t["crop_c"])
    hits=[]
    for node in row["nodes"]:
        part=node["region"]["bbox"]
        shared=overlap(box,part)
        iou=shared/max(1,box[2]*box[3]+part[2]*part[3]-shared)
        if iou>=CFG["semantic_region_min_iou"] and node.get("region",{}).get("proposal_type")!="full_frame":
            hits.append({"candidate_id":node["candidate_id"],"original_box":part,"iou_with_difference":iou,
                         "hypotheses":node["visual_hypotheses"],"scope":"crop query hint; no action claim"})
    ids={n["candidate_id"] for n in hits}
    edges=[n for n in row["relations"] if n["subject_candidate_id"] in ids or n["object_candidate_id"] in ids]
    return {"object_region_candidates":hits,"geometric_edges_touching_proposals":edges}


def route(book,goal,arm,facts,missing):
    number=int(arm[1:])
    enable={1:{"D2"},2:{"D2","D3"},3:{"D2","D3","D4","D5"}}
    using=enable.get(min(3,number),set())
    anchors=[r for r in book.rows if "D6" in r["image_dimensions"]]
    if not anchors:
        anchors=[book.rows[0],book.rows[-1]]
    proposals=[]
    assessed=[]
    root_seen=set()
    if number==0:
        for a,c in zip(anchors,anchors[1:]):
            left,right=a["t"],c["t"]
            mid=(left+right)/2
            lo,hi=max(left,mid-book.cadence/2),min(right,mid+book.cadence/2)
            t={"id":identifier(book.case,"coverage",a["pts"],c["pts"]),"dimension":"D6","trigger_kind":"coverage_probe",
               "a_id":a["frame_id"],"c_id":c["frame_id"],"observation_a":book.stamp(a["frame_id"]),"observation_c":book.stamp(c["frame_id"]),
               "comparison":{"comparable":"unknown","basis_root_ids":[a["frame_id"],c["frame_id"]]},"source_sha256":book.source["sha256"]}
            proposals.append(trigger_with_interval(book,goal,t,lo,hi))
    else:
        for t in facts:
            if t["dimension"] is not None and t["dimension"] not in using:
                continue
            why=t["base_abstention"] or (t.get("strict_abstention") if number>=3 else None)
            if why:
                assessed.append({"trigger":t["id"],"decision":"abstained","reason":why})
                continue
            lo,hi=t["observation_a"]["t"],t["observation_c"]["t"]
            token=(t["trigger_kind"],tuple(t["comparison"]["basis_root_ids"]),lo,hi)
            if token in root_seen:
                assessed.append({"trigger":t["id"],"decision":"deduplicated_correlated_root"})
                continue
            root_seen.add(token)
            item=trigger_with_interval(book,goal,t,lo,hi)
            if number>=4:
                item["semantic_navigation_hints"]=semantic_hints(book,t)
            proposals.append(item)
            assessed.append({"trigger":t["id"],"decision":"proposed","proposal_id":t["id"]})
    text=goal["text"].casefold()
    if "inspect between samples" in text or "inspect between keyframes" in text:
        explicit={"id":identifier(book.case,"explicit",goal["id"]),"dimension":None,"trigger_kind":"explicit_source_request",
                  "source_sha256":book.source["sha256"],"comparison":{"basis_root_ids":[],"comparable":"unknown"}}
        proposals.append(trigger_with_interval(book,goal,explicit,book.source["start"],book.source["end"]))
    regions=[]
    if number==5:
        if "viewpoints" in text:
            preferred="viewpoint_control"
        elif "relative" in text:
            preferred="interaction_clue"
        else:
            preferred="intermediate_state"
        for t in proposals:
            hypothesis=t["investigation_gap"]
            hypothesis["goal_relevance"]="primary" if hypothesis["missing_observation_type"]==preferred else "alternative_or_control"
            hypothesis["competing_interpretations"]=t["comparison"].get("competing_explanations",[])+["local image difference does not establish a physical identity or action"]
        proposals.sort(key=lambda t:t["investigation_gap"]["goal_relevance"]!="primary")
        active=[t for t in proposals if t["trigger_kind"] in ("comparable_state_difference","relative_relation_difference")]
        for a,c in zip(anchors,anchors[1:]):
            viable=[t for t in active if a["t"]<=t["observation_c"]["t"]<=c["t"]]
            if viable:
                ordered=sorted(viable,key=lambda t:t["observation_c"]["t"])
                entry=list(dict.fromkeys([ordered[0]["a_id"],ordered[-1]["c_id"]]))
                regions.append({"path_type":"independent temporal entrance anchored by visual differences","time_range":[a["t"],c["t"]],
                                "anchor_ids":entry,"origin_trigger_ids":list(dict.fromkeys([ordered[0]["id"],ordered[-1]["id"]]))})
    open_ids=list(dict.fromkeys([key for r in regions for key in r["anchor_ids"]]+
                                 [key for t in proposals for key in (t.get("a_id"),t.get("c_id")) if key]))
    return {"goal":goal,"arm":arm,"candidate_status":assessed,"unresolved_comparison_endpoints":missing if number else [],
            "independent_time_entrances":regions,"anchor_ids":open_ids,"proposals":proposals,"archive_sha256":book.entry["catalog"]["sha256"],
            "coverage_not_change":True,"new_claims_of_identity_or_cause":[]}


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
