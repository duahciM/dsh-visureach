"""Explore two soft-priority lanes and compare local evidence independently."""
from collections import defaultdict
from itertools import zip_longest

import cv2
import numpy as np

from p221_research.recover import distance, view

from .prepare import PROTO


ROLE_DIMS = {"temporal_context": {"D4", "D6"}, "locally_distinct_state": {"D1", "D2", "D3", "D5"},
             "recurring_projected_context": {"D1", "D3", "D4"}}


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    shared = max(0, x1-x0)*max(0, y1-y0)
    return shared/max(1, a[2]*a[3]+b[2]*b[3]-shared)


def crop_for(book, frame, box):
    rgb, gray = book.image(frame["frame_id"])
    x, y, w, h = box
    nx, ny = book.source["probe"]["width"], book.source["probe"]["height"]
    ih, iw = gray.shape
    l, r = max(0, round(x*iw/nx)), min(iw, round((x+w)*iw/nx))
    t, b = max(0, round(y*ih/ny)), min(ih, round((y+h)*ih/ny))
    patch = rgb[t:b, l:r]
    if min(patch.shape[:2]) < 3:
        return None
    span = float(np.percentile(patch, 90) - np.percentile(patch, 10))
    if span < PROTO["selection"]["min_root_patch_contrast"]:
        return None
    proxy = cv2.resize(patch, (20, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return proxy, span


def localized(book, frame, candidate):
    role = candidate["role"]
    if role not in ("locally_distinct_state", "recurring_projected_context"):
        return []
    evidence = candidate["measurements"]
    boxes = []
    for o in frame["observations"]:
        if o["dimension_id"] in ROLE_DIMS[role] and o["region_xywh_analysis"]:
            boxes.append((book.native_box(o["region_xywh_analysis"]), o["evidence_root_ids"]))
    discovered = evidence.get("projection_boxes", []) if role == "recurring_projected_context" else evidence.get("foreground_regions_original_px", [])
    boxes.extend((box, candidate["parents"]) for box in discovered)
    result = []
    seen = set()
    for box, parents in boxes:
        if not parents or tuple(box) in seen:
            continue
        info = crop_for(book, frame, box)
        if info:
            result.append({"box": box, "roots": sorted(set(parents)), "crop": info[0], "contrast": info[1]})
            seen.add(tuple(box))
    return result


def supports(book, f, s):
    arr, _ = book.image(f["frame_id"])
    brr, _ = book.image(s["frame_id"])
    p = cv2.resize(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), (80, 48)).astype(np.float32)
    q = cv2.resize(cv2.cvtColor(brr, cv2.COLOR_RGB2GRAY), (80, 48)).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(p, q, cv2.createHanningWindow((80, 48), cv2.CV_32F))
    return {"shift_x": float(dx), "shift_y": float(dy), "response": float(resp), "valid": bool(np.isfinite(resp) and resp >= PROTO["selection"]["min_registration_response"] and abs(dx) <= 3 and abs(dy) <= 3)}


def compare_local(book, test, test_frame, accepted):
    rules = PROTO["selection"]
    candidates = []
    for frame, known in accepted:
        substitutions = []
        for a in test:
            for b in known:
                overlap = iou(a["box"], b["box"])
                if overlap >= rules["min_crop_iou"]:
                    substitutions.append({"candidate_box": a["box"], "reference_box": b["box"], "IoU": overlap,
                                          "crop_mad": distance(a["crop"], b["crop"]), "candidate_roots": a["roots"], "reference_roots": b["roots"]})
        if not substitutions:
            continue
        passed = {tuple(x["candidate_box"]) for x in substitutions if x["crop_mad"] <= rules["max_crop_mad"]}
        reg = supports(book, test_frame, frame) if passed else None
        equivalent = len(passed) == len(test) and bool(reg and reg["valid"])
        candidates.append({"against_pts": frame["pts"], "regional_comparisons": substitutions, "registration": reg,
                           "all_regions_equivalent": equivalent})
        if equivalent:
            return True, candidates
    return False, candidates


def routes(book, goal, baseline, discovered, scores, flags):
    if goal["intent"] not in PROTO["selection"]["local_intents"]:
        return [], [], []
    by_pts = {r["pts"]: r for r in book.rows}
    descriptors = {}
    def context(pts):
        if pts not in descriptors:
            descriptors[pts] = view(book.image(by_pts[pts]["frame_id"])[0])
        return descriptors[pts]
    initial = {r["pts"] for r in baseline}
    rank = defaultdict(list)
    for r in scores:
        if r["time_scale"] == PROTO["primary_scale_s"]:
            rank[r["pts"]].append(r)
    grouped = defaultdict(list)
    for n, row in enumerate(discovered):
        if row.get("role") in ROLE_DIMS and "measurements" in row and row["pts"] in by_pts:
            grouped[(row["zone"], row["role"])].append((n, row))
    offered, priorities, local_trace = [], [], []
    for (zone, role), bucket in grouped.items():
        lanes = []
        metrics = defaultdict(dict)
        for name in flags:
            if name == "local":
                continue
            key, method = name[0], "clean" if name.endswith("clean") else "raw"
            ordered = []
            for n, row in bucket:
                nomination = [r for r in rank[row["pts"]] if r["dimension"] in ROLE_DIMS[role] and r["scores"][method][key] is not None]
                if nomination:
                    best = max(nomination, key=lambda r: r["scores"][method][key])
                    value = best["scores"][method][key]
                    metrics[n][name] = {"value": value, "nomination": best["nomination"], "roots": best["root_ids"],
                                        "support": best["scores"][method]["support"], "segment": best["segment_id"],
                                        "transition_status": best["transition_status"]}
                    ordered.append((n, value))
            lanes.append([n for n, _ in sorted(ordered, key=lambda x: (-x[1], x[0]))])
        if lanes:
            ordered = list(dict.fromkeys(n for pair in zip_longest(*lanes) for n in pair if n is not None))
        else:
            ordered = []
        values = dict(bucket)
        ordered.extend(n for n, _ in bucket if n not in set(ordered))
        old = [r["pts"] for _, r in bucket if r["pts"] in initial]
        accepted = [context(pts) for pts in old]
        verified = [(by_pts[r["pts"]], localized(book, by_pts[r["pts"]], r)) for _, r in bucket if r["pts"] in initial] if "local" in flags else []
        for pos, n in enumerate(ordered, 1):
            row = values[n]
            pts = row["pts"]
            frame = by_pts[pts]
            desc = context(pts)
            whole = min((distance(desc, x) for x in accepted), default=None)
            threshold = .018 if role == "recurring_projected_context" else .037
            reject_global = whole is not None and whole < threshold
            keep = not reject_global
            decision = "novel whole view" if keep else "whole-picture representative" 
            check = {}
            if "local" in flags and pts not in initial:
                boxes = localized(book, frame, row)
                if boxes:
                    substitute, check_list = compare_local(book, boxes, frame, verified)
                    keep = not substitute
                    decision = "local role cannot be substituted" if keep else "all local regions substitutable in a comparable view"
                    check = {"regions": [{k: v for k, v in b.items() if k != "crop"} for b in boxes],
                             "against": check_list, "substituted": substitute, "rescued_from_global": reject_global and keep}
                    if keep:
                        verified.append((frame, boxes))
                else:
                    check = {"regions": [], "withheld_local_judgment": True}
            if pts in initial:
                keep = False
                decision = "bytes already present in common historical prefix"
            if keep:
                accepted.append(desc)
                offered.append({"pts": pts, "source_frame": frame["source_frame"], "role": role, "zone": zone, "path": decision,
                                "root_ids": row["parents"], "lanes": metrics[n], "local_rescued": check.get("rescued_from_global", False)})
            summary = {"pts": pts, "zone": zone, "role": role, "priority_ordinal": pos, "lane_scores": metrics[n],
                       "whole_view_distance": whole, "whole_gate_would_fold": reject_global, "selected": keep,
                       "decision": decision, "root_ids": row["parents"], "true_in_catalog": True}
            priorities.append(summary)
            if check or "local" in flags:
                local_trace.append(summary | check)
    return offered, priorities, local_trace
