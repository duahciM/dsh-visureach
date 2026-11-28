"""Evaluate immutable candidate evidence without time, using an independent spatial graph."""

from collections import Counter
from itertools import combinations
from math import hypot

import cv2
import numpy as np

from p217_research.triggers import identifier, visual

from .prepare import PROTO


CFG = PROTO["features"]


def overlap(a, b):
    x = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return x*y/max(1, a[2]*a[3]+b[2]*b[3]-x*y)


def visible(rgb, box):
    h, w = rgb.shape[:2]
    x, y, dx, dy = map(round, box)
    a, b, c, d = max(0, x), max(0, y), min(w, x+dx), min(h, y+dy)
    if min(c-a, d-b) < 3 or (c-a)*(d-b) > CFG["max_region_frame_fraction"]*h*w:
        return None
    crop = rgb[b:d, a:c]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    contrast = float(np.percentile(gray, 90)-np.percentile(gray, 10))
    sat = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = sat[:, :, 1] >= 62
    if contrast < CFG["min_local_contrast"] and mask.mean() < .22:
        return None
    if mask.any():
        hist = np.bincount(sat[:, :, 0][mask]//15, minlength=12)
        color = "color-bin-"+str(int(hist.argmax()))
    else:
        color = "achromatic"
    shp = "horizontal" if dx >= 1.5*dy else "vertical" if dy >= 1.5*dx else "compact"
    return {"box": [a,b,c-a,d-b], "contrast": contrast, "appearance": color+"/"+shp}


def node_bundle(book, item, archived):
    """Only spatial observations and their attachment, with no PTS field exposed to Figure2."""
    rgb = book.image(archived["frame_id"])[0] if archived else visual(item["image"]["path"], book.height)[0]
    raw = []
    anchors = []
    if archived:
        for obs in archived["observations"]:
            box = obs["region_xywh_analysis"]
            if obs["dimension_id"] in ("D1", "D2", "D3", "D5") and box:
                raw.append((box, obs["evidence_root_ids"], obs["role"]))
        for n in archived["nominations"]:
            xy = n.get("context_edge_xy") or (n.get("component") or {}).get("context_edge_xy")
            if n["dimension_id"] == "D3" and xy and n["applicable_region"]:
                x,y = xy
                anchors.append((n["applicable_region"], [x-4, y-4, 9, 9], n["evidence_root_ids"]))
    else:
        for why in item["legacy_reasons"]:
            box = why.get("region_original_px") or why.get("measurement", {}).get("region_original_px")
            if box and len(box) == 4:
                sx = 320/book.source["probe"]["width"]
                sy = book.height/book.source["probe"]["height"]
                raw.append(([box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy], evidence_roots(why), "old_source_local_proof"))
    nodes = []
    for box, roots, role in raw:
        found = visible(rgb, box)
        if found is None or not roots:
            continue
        comparable = next((r for r in nodes if overlap(found["box"], r["box"]) >= CFG["min_iou_same_region"]), None)
        if comparable:
            comparable["roots"] = sorted(set(comparable["roots"]) | set(roots))
            comparable["roles"] = sorted(set(comparable["roles"]) | {role})
        else:
            nodes.append(found | {"id": str(len(nodes)), "roots": sorted(set(roots)), "roles": [role]})
    references = []
    for box, near, parents in anchors:
        anchor = next((r for r in nodes if overlap(r["box"], box) >= CFG["min_iou_same_region"]), None)
        feature = visible(rgb, near)
        if anchor and feature and parents and overlap(anchor["box"], feature["box"]) < .20:
            references.append({"subject": anchor["id"], "edge_box": feature["box"], "roots": sorted(set(parents))})
    return {"candidate": item["candidate_id"], "regions": nodes, "spatial_reference_observations": references,
            "view_color_grid": (cv2.resize(rgb, (12,8), interpolation=cv2.INTER_AREA).astype(np.float32)/255).reshape(-1).tolist(),
            "unknown_human_object_labels": True}


def evidence_roots(why):
    return sorted(set(why.get("root_ids", [])+why.get("parents", [])+([why["evidence_root"]] if why.get("evidence_root") else [])))


def graph(nodes, edge_observations):
    """Build degree and distinct pair counts from confirmed image-space evidence inputs."""
    lookup = {r["id"]: r for r in nodes}
    pairs = {}
    for edge in edge_observations:
        a, b = edge["ends"]
        if a == b or a not in lookup or b not in lookup or not edge["roots"]:
            continue
        key = tuple(sorted((a,b)))
        if key not in pairs:
            pairs[key] = {"ends": list(key), "types": [], "roots": []}
        prior = pairs[key]
        prior["types"] = sorted(set(prior["types"]) | {edge["type"]})
        prior["roots"] = sorted(set(prior["roots"]) | set(edge["roots"]))
    degree = Counter({k: 0 for k in lookup})
    for a,b in pairs:
        degree.update((a,b))
    signatures = [sorted((lookup[a]["appearance"], lookup[b]["appearance"]))+r["types"] for (a,b), r in pairs.items()]
    return {"nodes": list(lookup.values()), "relations": list(pairs.values()), "distinct_pairs": len(pairs),
            "degree_by_object_or_region": dict(degree), "degree_distribution": {str(k):v for k,v in Counter(degree.values()).items()},
            "projected_signature": sorted(signatures), "physical_identity_or_contact": "unknown"}


def figure2(spatial):
    nodes = spatial["regions"].copy()
    edges = []
    for src in spatial["spatial_reference_observations"]:
        box = src["edge_box"]
        partner = next((r for r in nodes if r["id"] != src["subject"] and overlap(r["box"], box) >= .18), None)
        if partner is None:
            partner = {"id": "edge-"+str(len(nodes)), "appearance": "context-edge", "box": box,"roots": src["roots"],"roles": ["localized_reference_edge"]}
            nodes.append(partner)
        edges.append({"ends": [src["subject"], partner["id"]], "type":"measured_near_context_edge", "roots":src["roots"]})
    for a,b in combinations(spatial["regions"],2):
        if set(a["roots"]) & set(b["roots"]):
            continue
        x,y,w,h = a["box"]
        xx,yy,ww,hh = b["box"]
        gap = hypot(max(x-(xx+ww), xx-(x+w), 0), max(y-(yy+hh), yy-(y+h),0))
        if gap <= CFG["max_near_gap_box_fraction"]*min(max(w,h),max(ww,hh)) and overlap(a["box"],b["box"]) < .3:
            edges.append({"ends":[a["id"],b["id"]],"type":"measured_projected_proximity","roots": sorted(set(a["roots"]+b["roots"]))})
    return graph(nodes, edges) | {"candidate": spatial["candidate"], "confidence": "image-space evidence only"}


def figure1(spatial, auxiliary=None):
    regions = spatial["regions"]
    references = spatial["spatial_reference_observations"]
    simple = {(r["subject"], tuple(r["edge_box"])) for r in references}
    baseline = {"R": len(simple), "relation_states": sorted(["local_reference_edge",r["appearance"]] for r in regions if any(x[0] == r["id"] for x in simple)),
                "relation_regime": "existing_annotation_only"}
    mapped = {"R": auxiliary["distinct_pairs"], "relation_states": auxiliary["projected_signature"],
              "relation_regime": "figure2_structural_assistance", "degree": auxiliary["degree_by_object_or_region"]} if auxiliary else baseline
    return {"candidate": spatial["candidate"], "I": len(regions), **mapped, "information_roots": sorted({k for r in regions for k in r["roots"]}),
            "region_appearances": sorted(r["appearance"] for r in regions), "candidate_regions": regions,
            "view_color_grid": spatial["view_color_grid"], "semantic_labels": "unknown",
            "confidence": "localized measurable support" if regions else "no confirmed region from this source proposal"}
