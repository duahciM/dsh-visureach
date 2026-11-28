"""Positive pixel witnesses and independent complementary time entrances."""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from if17_research.catalog import read
from p217_research.triggers import PARAMS, identifier

from .prepare import CFG


P=CFG["pixel_rules"]


def describe(book,items):
    """A view descriptor derived from pixels; it does not identify the scene."""
    return {k:cv2.resize(book.image(k)[0],(8,5),interpolation=cv2.INTER_AREA).astype(np.float32)/255 for k in items}


def compact_observations(rgb,bg):
    height,width=rgb.shape[:2]
    delta=np.max(np.abs(rgb.astype(np.int16)-bg),axis=2)
    mask=np.uint8(delta>=P["pixel_residual"])
    count,_,rows,_=cv2.connectedComponentsWithStats(mask)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edge=cv2.Canny(gray,36,92)
    offers=[]
    for x,y,w,h,area in rows[1:count]:
        if area<P["minimum_component"] or w*h>P["max_local_fraction"]*width*height or min(w,h)<3:
            continue
        if x<2 or y<2 or x+w>width-2 or y+h>height-2:
            continue
        strength=float(delta[y:y+h,x:x+w].mean()/255)
        edges=int(np.count_nonzero(edge[y:y+h,x:x+w]))
        if strength<P["minimum_mean_residual"] or edges<4:
            continue
        offers.append({"box":[int(x),int(y),int(w),int(h)],"strength":strength,"edge_pixels":edges,"pixels":int(area),
                       "bounding_aspect":max(w/h,h/w)})
    return offers


def candidate_parts(book,rows,background,goal):
    family=goal["text"].casefold()
    output={}
    for row in rows:
        key=row["frame_id"]
        rgb,_=book.image(key)
        parts=compact_observations(rgb,background)
        if "fastener" in family or "bottle" in family:
            parts=[p for p in parts if p["bounding_aspect"]>=P["minimum_shape_ratio"]]
        if "left frame edge" in family:
            # Edge approach is an image-region cue; a padded crop can still be visible.
            parts=[p for p in parts if p["box"][0]<rgb.shape[1]/2]
        if not parts and {"D1","D3"}.intersection(row["image_dimensions"]) and "fastener" not in family and "bottle" not in family:
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            for n in row["observations"]:
                box=n["region_xywh_analysis"]
                if n["dimension_id"] not in ("D1","D3") or not box:
                    continue
                x,y,w,h=map(int,box)
                crop=gray[y:y+h,x:x+w]
                if crop.size>=P["minimum_component"] and min(crop.shape)>=3 and np.ptp(crop)>=40:
                    parts.append({"box":list(map(int,box)),"strength":float(np.std(crop)/255),"edge_pixels":int(np.count_nonzero(cv2.Canny(crop,36,92))),
                                  "pixels":int(crop.size),"candidate_via":"indexed local footprint with visible pixel contrast"})
        output[key]=sorted(parts,key=lambda p:(-p["strength"],-p["pixels"],p["box"]))
    return output


def visibility(book,scope,baseline):
    rows=scope["eligible"]
    if not rows:
        return {},[]
    median=np.median(np.stack([book.image(r["frame_id"])[0] for r in book.rows]),axis=0).astype(np.int16)
    checks=candidate_parts(book,rows,median,scope["goal"])
    plausible={k:v for k,v in checks.items() if v}
    trace=[{"observation":book.stamp(k),"positive_region_candidates":[p|{"region_original_px":book.native_box(p["box"])} for p in parts],
            "pixel_state":"local_visible_footprint" if parts else "uncertain_from_available_view",
            "goal":"one visible object and the required surrounding context","baseline_proposal":k in baseline} for k,parts in checks.items()]
    if not plausible:
        return {},trace
    existing=next((k for k in baseline if k in plausible),None)
    key=existing if existing is not None else max(plausible,key=lambda k:(plausible[k][0]["strength"],plausible[k][0]["pixels"]))
    packet=plausible[key][0]
    roots=[identifier(key,"spatial-local-residual",packet["box"])]
    return {key:[{"observation_role":"single_view_with_visible_local_candidate","reason":"Public request for one view, with a locally visible measured footprint",
                  "region_original_px":book.native_box(packet["box"]),"root_ids":roots,"candidate_role_only":True}]},trace


def ball_context(rgb):
    values=rgb.astype(np.int16)
    colored=np.uint8(values.max(2)-values.min(2)>=P["color_delta"])
    height,width=colored.shape
    _,_,shapes,centres=cv2.connectedComponentsWithStats(colored)
    offers=[]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    for (x,y,w,h,n),(cx,cy) in zip(shapes[1:],centres[1:]):
        if not 14<=n<=.018*width*height or min(w,h)<4 or min(w,h)/max(w,h)<P["minimum_roundness"]:
            continue
        if n<.52*w*h or y<2 or x<2 or y+h>=height-1 or x+w>=width-1:
            continue
        radius=int(max(w,h)*P["comparison_radius_in_ball_widths"])
        lx,hx=max(0,int(cx-w)),min(width,int(cx+w)+1)
        ly=max(0,y-radius)
        above=gray[ly:y,lx:hx]
        beneath=colored[min(y+h,height-1):min(height,y+h+2*radius),max(x-radius,0):min(width,x+w+radius)]
        if above.size and beneath.size and np.mean(above<82)>=.12 and np.mean(beneath)>=.03:
            offers.append({"box":[int(x),int(y),int(w),int(h)],"relative_projection":"compact round chromatic region with dark upper context and larger coloured lower context",
                           "centre":[float(cx),float(cy)],"area":int(n),"upper_dark_fraction":float(np.mean(above<82)),
                           "lower_color_fraction":float(np.mean(beneath))})
    return offers


def find_archived_complements(book,scope,baseline):
    goal=scope["goal"]
    if goal["intent"] not in ("open","scene","recurrence"):
        return {},[]
    items=scope["eligible"]
    if not items:
        return {},[]
    descriptors=describe(book,[p["frame_id"] for p in items])
    if len(items)<2 or all(book.compare(a["frame_id"],c["frame_id"])["raw_mad"]<PARAMS.activity_mad for a,c in zip(items,items[1:])):
        return {},[]
    proposals=defaultdict(list)
    review=[]
    region=scope["region"]
    def add(key,role,root,note,**more):
        proposals[key].append({"observation_role":role,"root_ids":[root],"reason":note,"independent_time_entry":region(book.by[key]),**more})
    if goal["intent"] in ("scene","open"):
        tracked=defaultdict(list)
        for key in baseline:
            if key in descriptors:
                tracked[region(book.by[key])].append(descriptors[key])
        for row in items:
            key=row["frame_id"]
            z=region(row)
            if not ({"D4","D6"}.intersection(row["image_dimensions"])):
                continue
            marker=descriptors[key]
            separate=not tracked[z] or all(float(np.mean(np.abs(marker-v)))>=P["scene_descriptor_mad"] for v in tracked[z])
            review.append({"key":key,"region":z,"role":"view_or_temporal_context","independent_view":separate})
            if separate:
                root=identifier(key,"low-frequency-image-view")
                add(key,"independent_view_or_temporal_context",root,"An indexed D4/D6 witness has a different measured picture context in this time entrance.")
                tracked[z].append(marker)
    if goal["intent"]=="recurrence":
        if "ball" in goal["text"].casefold():
            # Preserve individually localized candidate relations when supported
            # by positive pixels. Inclusion does not rely on one winner per epoch.
            for row in items:
                key=row["frame_id"]
                if not {"D1","D3"}.intersection(row["image_dimensions"]):
                    continue
                rgb,_=book.image(key)
                offers=ball_context(rgb)
                review.append({"key":key,"region":region(row),"role":"localized_projected_ball_relation","pixel_candidates":offers})
                if offers:
                    add(key,"complementary_projected_region_state",identifier(key,"ball-picture-context"),
                        "Positive localized projection candidate within its own time entrance; does not establish a physical identity or contact.",
                        regions_original_px=[book.native_box(x["box"]) for x in offers])
        else:
            # A temporally independent D6 anchor is possible even when no names
            # or long chains are available; only visible-change scopes get these.
            for row in items:
                key=row["frame_id"]
                if "D6" in row["image_dimensions"] and any(d!="D6" for d in row["image_dimensions"]):
                    add(key,"independent_comparison_anchor",identifier(key,"endorsed-independent-time"),"Time witness with its own local evidence at this source PTS.")
    return dict(proposals),review


def coarse_episodes(book,scope):
    if scope["goal"]["intent"]!="recurrence" or any(set(r["image_dimensions"])-{"D6"} for r in book.rows):
        return [],[]
    frozen=Path(book.entry["catalog"]["path"]).parent/"pool/frames.json"
    if not frozen.is_file():
        return [],[]
    src=read(frozen)
    assert src["source"]["sha256"]==book.source["sha256"]
    members={r["source_frame"] for r in book.coverage}
    raw=sorted((r for r in src["frames"] if r["source_frame"] in members),key=lambda x:x["t"])
    seen=[]
    for row in raw:
        bgr=cv2.imread(row["image"])
        if bgr is None:
            return [],[]
        img=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        height=round(img.shape[0]*320/img.shape[1])
        seen.append(cv2.resize(img,(320,height),interpolation=cv2.INTER_AREA))
    if len(seen)<2:
        return [],[]
    baseline=np.median(np.stack(seen),axis=0).astype(np.int16)
    rows=[]
    spans=[]
    buffer=[]
    def push():
        if buffer:
            best=max(buffer,key=lambda x:(x[1][0]["pixels"],x[1][0]["strength"]))
            row,parts=best
            spans.append({"aligned_coarse_span":[buffer[0][0]["t"],buffer[-1][0]["t"]],"source_frame":row["source_frame"],"pts":row["pts"],
                          "region_original_px":book.native_box(parts[0]["box"]),"coarse_comparisons":[r[0]["pts"] for r in buffer],
                          "root_id":identifier(book.source["sha256"],row["pts"],"coarse-median-foreground-run"),
                          "reason":"An independently separated episode has a positively measured foreground footprint in frozen coarse pixels."})
    for row,image in zip(raw,seen,strict=True):
        parts=sorted(compact_observations(image,baseline),key=lambda x:(-x["pixels"],-x["strength"]))
        rows.append({"pts":row["pts"],"source_frame":row["source_frame"],"measured_regions":parts})
        if parts:
            buffer.append((row,parts))
        elif buffer:
            push()
            buffer=[]
    push()
    return spans,rows
