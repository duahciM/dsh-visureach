"""Independent nomination channels over source-frame image and motion evidence."""
from collections import defaultdict
from itertools import combinations
from math import hypot, sqrt

import cv2
import numpy as np

from if09_research import subjects
from p1.core import Parameters, pair_metrics
from .common import COMPONENTS, RULES, frame_key


def proposal(dimension, role, sample, roots, family, comparison, strength=0., region=None, **details):
    return {"dimension_id":dimension, "role":role, "sample":sample,
            "evidence_root_ids":list(dict.fromkeys(roots)), "evidence_family":family,
            "confidence_scope":"observed 2D image measurements",
            "applicable_region":region, "comparison_frame_ids":list(comparison),
            "status":"nominated", "cost":{"additional_source_decodes":0},
            "priority_within_dimension":round(float(strength),7)} | details


def position(src,f):
    return {"source_frame":f["source_frame"],"pts":f["pts"],"time_base":f["time_base"],"t":f["t"],
            "frame_id":frame_key(src,f)}


def charts(fs):
    ps=Parameters()
    return {i:pair_metrics(a["gray"],b["gray"],ps) for i,(a,b) in enumerate(zip(fs,fs[1:]),1)}


def register(reference, target, taper, max_shift, config):
    offset, response=cv2.phaseCorrelate(reference.astype(np.float32),target.astype(np.float32),taper)
    if not np.isfinite([*offset,response]).all() or response<config["min_phase"] or hypot(*offset)>max_shift:
        return None
    return np.float32([[1,0,offset[0]],[0,1,offset[1]]])


def regions(delta, cfg):
    found,state=subjects.candidates(delta,np.zeros_like(delta),COMPONENTS)
    h,w=delta.shape
    bounds=cfg["border_px"]
    compact=[]
    for q in found:
        x,y,rw,rh=q["bbox"]
        occupancy=q["area"]/(rw*rh)
        if (x<bounds or y<bounds or x+rw>w-bounds or y+rh>h-bounds
            or occupancy<cfg["minimum_compactness"] or q["area"]>h*w*cfg["max_region_fraction"]):
            continue
        values=delta[y:y+rh,x:x+rw]
        score=float(values[values>=COMPONENTS["delta"]].mean())*occupancy*sqrt(q["area"])
        compact.append(subjects.public_candidate(q)|{"compactness":occupancy,"local_strength":score})
    chosen=sorted(compact,key=lambda x:(-x["local_strength"],x["bbox"]))[:cfg["maximum_regions_per_frame"]]
    return chosen, state if chosen or not found else "components_rejected_by_shape_gate"


def foreground(fs, keys, cfg):
    n=len(fs)
    h,w=fs[0]["gray"].shape
    taper=cv2.createHanningWindow((w,h),cv2.CV_32F)
    colors=[cv2.cvtColor(f["rgb"],cv2.COLOR_BGR2YCrCb).astype(np.float32)/255 for f in fs]
    out=[]
    for i,f in enumerate(fs):
        if f["sharpness"]<8:
            out.append({"i":i,"state":"quality_abstention","candidates":[],"comparisons":[]})
            continue
        warps=[]
        compared=[]
        for j in range(max(0,i-cfg["neighbor_offset"]),min(n,i+cfg["neighbor_offset"]+1)):
            if i==j:
                continue
            warp=register(fs[j]["gray"],f["gray"],taper,cfg["max_registration_px"],cfg)
            if warp is None:
                continue
            warped=cv2.warpAffine(colors[j],warp,(w,h),borderMode=cv2.BORDER_REPLICATE)
            valid=cv2.warpAffine(np.ones((h,w),np.float32),warp,(w,h))>.999
            warps.append((warped,valid))
            compared.append({"frame_id":keys[j],"sample":j,"matrix":warp.tolist()})
        root=f'{keys[i]}:registered_neighborhood'
        if len(warps)<cfg["minimum_comparisons"]:
            out.append({"i":i,"state":"registration_abstention","candidates":[],"comparisons":compared,"root":root})
            continue
        valid=np.logical_and.reduce([m for _,m in warps])
        bg=np.median(np.stack([im for im,_ in warps]),axis=0)
        residual=np.max(np.abs(colors[i]-bg),axis=2)
        residual[~valid]=0
        cs,state=regions(residual,cfg)
        stable=np.uint8(np.clip(bg[:,:,0]*255,0,255))
        edges=cv2.Canny(stable,48,124)>0
        for c in cs:
            x,y,rw,rh=c["bbox"]
            pad=cfg["context_edge_distance_px"]
            ys,xs=np.nonzero(edges[max(0,y-pad):min(h,y+rh+pad),max(0,x-pad):min(w,x+rw+pad)])
            xx=xs+max(0,x-pad)
            yy=ys+max(0,y-pad)
            external=(xx<x-3)|(xx>=x+rw+3)|(yy<y-3)|(yy>=y+rh+3)
            if external.any():
                cx,cy=c["center"]
                ds=np.hypot(xx[external]-cx,yy[external]-cy)
                nearest=int(np.argmin(ds))
                if ds[nearest]<=pad:
                    c.update(context_edge_xy=[int(xx[external][nearest]),int(yy[external][nearest])],
                             context_image_distance=float(ds[nearest]))
        out.append({"i":i,"state":state,"root":root,"comparisons":compared,"candidates":cs})
    return out


def object_observations(out, fs, keys, fg, cfg):
    for row in fg:
        i=row["i"]
        for c in row["candidates"]:
            near=[q["i"] for q in fg[max(0,i-8):i+9] if q["i"]!=i and
                  abs(fs[q["i"]]["t"]-fs[i]["t"])<=cfg["nearby_seconds"] and
                  any(subjects.dist_box(c["bbox"],other["bbox"])>=.35 for other in q["candidates"])]
            rank=c["local_strength"]/sqrt(1+len(near))
            support=[r["frame_id"] for r in row["comparisons"]]
            out["D1"].append(proposal("D1","localized_visible_residual",i,[row["root"]],"registered_neighborhood",
                          support,rank,c["bbox"],nearby_overlap_samples=near,component=c,
                          identity=None,explanation="Image-space component measured against a registered temporal median."))
            if "context_edge_xy" in c:
                out["D3"].append(proposal("D3","component_near_repeatable_edge",i,[row["root"]],
                              "registered_neighborhood",support,rank/(1+c["context_image_distance"]/28),
                              c["bbox"],context_edge_xy=c["context_edge_xy"],
                              distance_px=c["context_image_distance"],physical_relation=None))
    # Visibility gaps are proposals with directly visible endpoints and explicit abstention.
    qualifying=[r for r in fg if r["candidates"]]
    for left,right in zip(qualifying,qualifying[1:]):
        i,j=left["i"],right["i"]
        if j<=i+1 or fs[j]["t"]-fs[i]["t"]>cfg["bracket_max_s"]:
            continue
        mid=fg[i+1:j]
        valid=[m for m in mid if len(m["comparisons"])>=cfg["minimum_comparisons"]]
        if not valid:
            continue
        gap=valid[len(valid)//2]
        a,b=left["candidates"][0],right["candidates"][0]
        if subjects.dist_box(a["bbox"],b["bbox"])>=.35:
            continue
        s=min(a["local_strength"],b["local_strength"])
        roots=[left["root"],right["root"],gap["root"]]
        for k,role,c in [(i,"visible_before_candidate_gap",a),(gap["i"],"sampled_detection_gap",None),
                         (j,"visible_after_candidate_gap",b)]:
            out["D5"].append(proposal("D5",role,k,roots,"registered_neighborhood",[keys[i],keys[j]],s,
                c["bbox"] if c else None, alternative_explanations=["undetected component","visibility loss","different region"],
                gap_source_frame_ids=[keys[r["i"]] for r in valid], physical_identity=None))


def pair_observations(out,fs,keys,old,motion,cfg):
    h,w=fs[0]["gray"].shape
    for i,m in motion.items():
        a,b=fs[i-1:i+1]
        if min(a["sharpness"],b["sharpness"])<8:
            continue
        root=f'{keys[i-1]}|{keys[i]}:phase_motion'
        related=[keys[i-1],keys[i]]
        prior=old.get(b["source_frame"],{})
        row=prior.get("motion",m)
        shift=hypot(m["dx"],m["dy"])
        if row.get("brightness",{}).get("supported"):
            continue
        if m["registration_valid"] and shift<=cfg["max_registration_px"]:
            H=np.float32([[1,0,m["dx"]],[0,1,m["dy"]]])
            for x,y,M,dest,role in [(a,b,H,i,"local_state_comparison_later"),
                                  (b,a,cv2.invertAffineTransform(H),i-1,"local_state_comparison_earlier")]:
                aligned=cv2.warpAffine(x["gray"],M,(w,h)).astype(np.float32)
                mask=cv2.warpAffine(np.ones((h,w),np.float32),M,(w,h))>.999
                r=np.abs(aligned-y["gray"])/255
                r[~mask]=0
                best,_=regions(r,cfg)
                if best:
                    c=best[0]
                    out["D2"].append(proposal("D2",role,dest,[root],"registered_pair",related,
                                      c["local_strength"],c["bbox"],bbox_side="destination",
                                      warning="Difference footprint; the component need not belong to a single object."))
        if m["motion_supported"] or (prior.get("state")=="specific_difference" and shift>0):
            out["D4"].append(proposal("D4","registered_view_shift",i,[root],"registered_pair",related,
                          shift, image_shift_xy=[m["dx"],m["dy"]], world_pose=None))
        if m["raw_mad"]>=.065 and not m["registration_valid"]:
            out["D5"].append(proposal("D5","wide_change_unregistered",i,[root],"registered_pair",related,
                              m["raw_mad"], alternative_explanations=["camera or parallax","photometry","edit or local change"]))


def nominate(fs,src,old=None,disabled=()):
    keys=[frame_key(src,f) for f in fs]
    cfg=RULES["local"]
    proposals=defaultdict(list)
    fg=foreground(fs,keys,cfg)
    motion=charts(fs)
    object_observations(proposals,fs,keys,fg,cfg)
    pair_observations(proposals,fs,keys,old or {},motion,cfg)
    for i in sorted({round(q*(len(fs)-1)/7) for q in range(8)}):
        part="endpoint" if i in (0,len(fs)-1) else "coverage_anchor"
        proposals["D6"].append(proposal("D6",part,i,[keys[i]],"source_time",[keys[i]],0.,
                             bounded_scope="current analysis span"))
    for values in proposals.values():
        for r in values:
            r["frame_id"]=keys[r["sample"]]
    return {d:proposals.get(d,[]) if d not in disabled else [] for d in RULES["modules"]},fg,motion

