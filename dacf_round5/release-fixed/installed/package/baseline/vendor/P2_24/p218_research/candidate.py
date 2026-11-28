"""Independent A3 shadow repairs. No time or source literals from the evaluator."""

from collections import defaultdict
import re

import numpy as np

from p217_research.triggers import PARAMS, feedback as old_feedback, identifier, route as frozen_route, trigger_with_interval
from p217_research.prepare import CFG as ORIGINAL


def anchor_trigger(book,goal,a,c,kind,roots,basis,comparison,channel):
    item={"id":identifier(book.case,"p218",channel,a,c,*roots),"source_sha256":book.source["sha256"],
          "a_id":a,"c_id":c,"observation_a":book.stamp(a),"observation_c":book.stamp(c),"crop_a":None,"crop_c":None,
          "trigger_kind":kind,"dimension":channel,"comparison":comparison|{"basis_root_ids":roots,
          "observed_difference":basis,"same_region_candidate":False,"physical_instance":"unknown"},
          "support_nominations":[],"candidate_patch":True}
    return trigger_with_interval(book,goal,item,book.by[a]["t"],book.by[c]["t"])


def neighbors_for_view(book,goal,proposals):
    """Continue a genuine D5 view ambiguity to its next measured or D6-backed anchor."""
    position={r["frame_id"]:i for i,r in enumerate(book.rows)}
    offered=[]
    tried=[]
    for trigger in proposals:
        if trigger.get("dimension")!="D5" or trigger["trigger_kind"]!="unexplained_structure":
            continue
        c=trigger["c_id"]
        next_key=None
        for frame in book.rows[position[c]+1:]:
            key=frame["frame_id"]
            measure=book.compare(c,key)
            if measure["comparable"]=="yes" or "D6" in frame["image_dimensions"]:
                next_key=key
                cause="measured registrable following view" if measure["comparable"]=="yes" else "next independent D6 time witness; comparison remains uncertain"
                break
        if next_key is None:
            tried.append({"trigger":trigger["id"],"status":"no_following_catalog_anchor"})
            continue
        original_roots=trigger["comparison"]["basis_root_ids"]
        tag=anchor_trigger(book,goal,c,next_key,"unexplained_structure",original_roots,
                "Follow a grounded D5 ambiguity into the adjacent unobserved bracket: "+cause,measure,"D5_following_bracket")
        tag["parent_trigger_id"]=trigger["id"]
        tag["investigation_gap"]["question"]="What view or visibility evidence is present after the last ambiguous observation and before this next anchor?"
        tag["investigation_gap"]["selection_basis"]=cause
        offered.append(tag)
        tried.append({"trigger":trigger["id"],"proposal":tag["id"],"range":tag["interval"],"status":"proposed"})
    return offered,tried


def image_difference(book,goal):
    """Compare D6-only source images using the already deployed P1 metric family."""
    if any(set(row["image_dimensions"])-{"D6"} for row in book.rows):
        return [],[]
    offered=[]
    audit=[]
    for a,c in zip(book.rows,book.rows[1:]):
        A,C=a["frame_id"],c["frame_id"]
        m=book.compare(A,C)
        if m["raw_mad"]>=PARAMS.activity_mad and m["comparable"]=="yes":
            root=identifier(a["source_image"]["sha256"],c["source_image"]["sha256"],"P1_pair_metrics")
            item=anchor_trigger(book,goal,A,C,"unexplained_structure",[root],
                    "Measured image-space difference across a registered D6 pair; source mechanism unknown",m,"D6_with_P1_pixels")
            offered.append(item)
            audit.append({"a":A,"c":C,"state":"measured_registered_difference","root":root})
        else:
            audit.append({"a":A,"c":C,"state":"no_qualified_difference","mad":m["raw_mad"],"comparability":m["comparable"]})
    # A bounded candidate cross-fade check may corroborate otherwise unregistered
    # endpoints. It uses the three already visible originals only.
    for a,b,c in zip(book.rows,book.rows[1:],book.rows[2:]):
        ids=[r["frame_id"] for r in (a,b,c)]
        m=book.compare(ids[0],ids[2])
        af,bf,cf=[book.image(key)[1].astype(np.float32)/255 for key in ids]
        delta=cf-af
        if min(m["quality"])<PARAMS.min_sharpness or float(np.mean(np.abs(delta)))<PARAMS.activity_mad:
            continue
        alpha=float(np.sum((bf-af)*delta)/max(1e-8,float(np.sum(delta*delta))))
        error=float(np.mean(np.abs(bf-(af+alpha*delta))))
        closest=min(float(np.mean(np.abs(bf-af))),float(np.mean(np.abs(bf-cf))))
        if not .15<=alpha<=.85 or error>=min(PARAMS.blend_error,closest*PARAMS.trend_fraction):
            continue
        root=identifier(*(row["source_image"]["sha256"] for row in (a,b,c)),"P1_blend_proxy")
        verified=m|{"blend_coefficient":alpha,"blend_mad":error,"nearest_endpoint_mad":closest,
                    "third_actual_observation":book.stamp(ids[1])}
        item=anchor_trigger(book,goal,ids[0],ids[2],"unexplained_structure",[root],
                  "Three archived observations support a pixel mixture proxy; blend mechanism remains a candidate",verified,"D6_with_P1_blend_proxy")
        offered.append(item)
        audit.append({"a":ids[0],"b":ids[1],"c":ids[2],"state":"measured_three_image_mixture_proxy","root":root})
    return offered,audit


def independent_catalog_points(book,goal):
    if not re.search(r"different (parts|time regions)|separate time (areas|ranges|regions)",goal["text"].casefold()):
        return [],[]
    pts=sorted({book.source["start"],book.source["end"],*[f["t"] for f in book.rows if "D6" in f["image_dimensions"]]})
    groups=[]
    for lo,hi in zip(pts,pts[1:]):
        actual=[r for r in book.rows if lo<=r["t"]<hi and {"D1","D3"}.intersection(r["image_dimensions"])]
        items=[{"frame_id":r["frame_id"],"pts":r["pts"],"time_base":r["time_base"],"image_ref":r["source_image"],
                "basis":[o for o in r["observations"] if o["dimension_id"] in ("D1","D3")]} for r in actual]
        groups.append({"region":[lo,hi],"entries":items,"selection":"independent public time-region D1/D3 evidence; no motion edge, identity or cross-region traversal required"})
    by_region=[iter(group["entries"]) for group in groups]
    order=[]
    while by_region:
        active=[]
        for region in by_region:
            row=next(region,None)
            if row:
                order.append(row["frame_id"])
                active.append(region)
        by_region=active
    return order,groups


def route(book,goal,arm,facts,missing):
    plan=frozen_route(book,goal,"T5",facts,missing)
    extensions,decisions=neighbors_for_view(book,goal,plan["proposals"])
    lowlevel,comparisons=image_difference(book,goal)
    entrance,groups=independent_catalog_points(book,goal)
    more=[*extensions,*lowlevel]
    allowed={p["id"] for p in plan["proposals"]}
    for p in more:
        if p["id"] not in allowed:
            plan["proposals"].append(p)
            allowed.add(p["id"])
    plan["anchor_ids"]=list(dict.fromkeys([*entrance,*plan["anchor_ids"],*[p[k] for p in more for k in ("a_id","c_id")]]))
    plan["arm"]=arm
    plan["candidate_repairs"]={"independent_catalog_regions":groups,"view_continuations":decisions,"D6_pixel_comparisons":comparisons,
                "version":"A3-v1","status":"shadow","feedback_semantics":"scoped interval and prior-state confirmation"}
    return plan


def feedback(book,trigger,rows,samples):
    original=old_feedback(book,trigger,rows,samples)
    a=trigger.get("observation_a")
    c=trigger.get("observation_c")
    comparison=trigger["comparison"]
    threshold=ORIGINAL["feedback_crop_mad"]
    result=defaultdict(list)
    annotated=[]
    for sample in original["observations"]:
        change=sample["result"]
        if "distance_to_anchor_crops" in sample and comparison.get("crop_mad",0)>=threshold and comparison.get("comparable")=="yes":
            dm,dn=sample["distance_to_anchor_crops"]
            if dm<threshold<=dn:
                change="comparable_earlier_state_confirmed"
            elif dn<threshold<=dm:
                change="comparable_later_state_confirmed"
            elif max(dm,dn)<threshold:
                change="no_distinguishable_new_sample"
            else:
                change="new_distinguishable_crop_state"
        result[change].append(sample["pts"])
        annotated.append(sample|{"typed_evidence":change})
    early=result["comparable_earlier_state_confirmed"]
    later=result["comparable_later_state_confirmed"]
    bound=None
    if a and c and (early or later):
        lo=max([a["pts"],*early])
        hi=min([c["pts"],*later])
        bound={"previous_inclusive_pts":[a["pts"],c["pts"]],"updated_observed_bracket_pts":[lo,hi] if lo<hi else None,
               "time_base":a["time_base"],"result":"observation_interval_tightened" if lo<hi else "contradictory_observed_order",
               "scope":"Only two comparable sampled states. Exact timing, continuity and physical causation remain unknown."}
    return original|{"observations":annotated,"typed_source_evidence":dict(result),"observed_state_bracket":bound}
