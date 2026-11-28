"""Reference-free feature probes. Scene IDs and annotation times never enter."""
import hashlib

import cv2
import numpy as np


def histogram(rgb, cfg):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, cfg["bins"], cfg["ranges"]).ravel().astype(np.float64)
    return hist / max(float(hist.sum()), 1.)


def hull_coverage(points, shape):
    if len(points) < 3:
        return 0.
    h, w = shape
    return float(cv2.contourArea(cv2.convexHull(np.float32(points)))) / max((w - 1) * (h - 1), 1)


def sift_pair(query, ref, query_shape, ref_shape, cfg, seed_key):
    qkp, qdesc = query
    rkp, rdesc = ref
    out = {"distance": cfg["failed_match_distance_cap"], "verified": False,
           "query_keypoints": len(qkp), "reference_keypoints": len(rkp),
           "ratio_forward_count": 0, "ratio_backward_count": 0, "mutual_count": 0,
           "inlier_count": 0, "inlier_fraction": 0., "query_hull_coverage": 0., "reference_hull_coverage": 0.,
           "reason": "no_descriptors", "homography": None, "inlier_pairs": []}
    if qdesc is None or rdesc is None or len(qdesc) < 2 or len(rdesc) < 2:
        return out
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(qdesc, rdesc, k=2)
    backward = matcher.knnMatch(rdesc, qdesc, k=2)
    fwd = {(p[0].queryIdx, p[0].trainIdx) for p in forward if len(p) == 2 and p[0].distance < cfg["ratio_test"] * p[1].distance}
    bwd = {(p[0].trainIdx, p[0].queryIdx) for p in backward if len(p) == 2 and p[0].distance < cfg["ratio_test"] * p[1].distance}
    mutual = sorted(fwd & bwd)
    out.update(ratio_forward_count=len(fwd), ratio_backward_count=len(bwd), mutual_count=len(mutual), reason="insufficient_mutual_matches")
    if len(mutual) < cfg["minimum_mutual_matches"]:
        return out
    qpts = np.float32([qkp[i].pt for i, _ in mutual])
    rpts = np.float32([rkp[j].pt for _, j in mutual])
    seed = (cfg["seed"] + int(hashlib.sha256(seed_key.encode()).hexdigest()[:7], 16)) % (2 ** 31 - 1)
    cv2.setRNGSeed(seed)
    hom, mask = cv2.findHomography(qpts, rpts, cv2.RANSAC, cfg["ransac_threshold_px"],
                                   maxIters=cfg["ransac_max_iterations"], confidence=cfg["ransac_confidence"])
    if hom is None or mask is None or not np.isfinite(hom).all():
        out["reason"] = "geometry_failed"
        return out
    keep = mask.ravel().astype(bool)
    count = int(keep.sum())
    fraction = count / len(mutual)
    qcov, rcov = hull_coverage(qpts[keep], query_shape), hull_coverage(rpts[keep], ref_shape)
    out.update(distance=1. - fraction, inlier_count=count, inlier_fraction=fraction,
               query_hull_coverage=qcov, reference_hull_coverage=rcov, homography=hom.ravel().tolist(),
               inlier_pairs=[{"query_xy": a.tolist(), "reference_xy": b.tolist()} for a, b in zip(qpts[keep], rpts[keep])])
    if count < cfg["minimum_inliers"]:
        out["reason"] = "too_few_inliers"
    elif fraction < cfg["minimum_inlier_ratio"]:
        out["reason"] = "low_inlier_fraction"
    elif min(qcov, rcov) < cfg["minimum_hull_coverage_each_image"]:
        out["reason"] = "insufficient_spatial_coverage"
    else:
        out.update(verified=True, reason="geometrically_verified_local_correspondence")
    return out


def feature_pairs(gray, rgb, prototype_samples, cfg):
    """Compute once per source/prototype; K1/K3 reuse identical measurements."""
    sift = cv2.SIFT_create(nfeatures=cfg["sift"]["nfeatures"])
    descriptors = [sift.detectAndCompute(g, None) for g in gray]
    hists = [histogram(x, cfg["histogram"]) for x in rgb]
    normal = gray.astype(np.float32) / 255.
    pairs = {"gray": {}, "hsv": {}, "sift": {}}
    for i in range(len(gray)):
        for j in prototype_samples:
            key = f"{i}:{j}"
            pairs["gray"][key] = {"distance": float(np.mean(np.abs(normal[i] - normal[j]))), "verified": True, "reason": "raw_grayscale_MAD"}
            bc = float(np.sum(np.sqrt(hists[i] * hists[j])))
            pairs["hsv"][key] = {"distance": float(np.sqrt(max(0., 1. - min(bc, 1.)))), "verified": True, "reason": "L1_HSV_Bhattacharyya"}
            pairs["sift"][key] = sift_pair(descriptors[i], descriptors[j], gray[i].shape, gray[j].shape, cfg["sift"], key)
    return pairs


def reduce_group(sample, prototypes, pairs, feature, cfg):
    scored = [{"prototype_id": p["id"], "prototype_sample": p["sample"], "prototype_image": p["rgb_image"],
               **pairs[f"{sample}:{p['sample']}"]} for p in prototypes]
    verified = [s for s in scored if s["verified"]]
    eligible = sorted(verified if verified else scored, key=lambda s: (s["distance"], s["prototype_id"]))
    nearest = eligible[0]
    supported = bool(verified) and nearest["distance"] <= cfg["absolute_distance_caps"][feature]
    distance = nearest["distance"] if feature != "sift" or verified else cfg["sift"]["failed_match_distance_cap"]
    return {"distance": distance, "observed_best_attempt_distance": nearest["distance"], "absolute_supported": supported,
            "nearest_prototype_id": nearest["prototype_id"], "nearest_prototype_sample": nearest["prototype_sample"],
            "nearest_prototype_image": nearest["prototype_image"], "nearest_evidence": nearest,
            "second_nearest_distance": eligible[1]["distance"] if len(eligible) > 1 else None,
            "second_nearest_prototype_id": eligible[1]["prototype_id"] if len(eligible) > 1 else None,
            "all_pairs": scored}


def probe(feature, k, frames, prototypes, pair_table, cfg):
    # Strip scene IDs explicitly: this routine receives view references only.
    refs = [{key: p[key] for key in ("id", "sample", "group", "single_reference", "rgb_image")} for p in prototypes]
    selected = [p for p in refs if k == 3 or p["single_reference"]]
    common_exclusion = {p["sample"] for p in refs}
    rows = []
    for i, frame in enumerate(frames):
        a = reduce_group(i, [p for p in selected if p["group"] == "A"], pair_table, feature, cfg)
        b = reduce_group(i, [p for p in selected if p["group"] == "B"], pair_table, feature, cfg)
        da, db = a["distance"], b["distance"]
        relative = (da - db) / (da + db + cfg["epsilon"])
        qa, qb = a["absolute_supported"], b["absolute_supported"]
        both_strong = qa and qb and max(da, db) <= cfg["absolute_distance_caps"][feature] * cfg["both_strong_abstention_fraction_of_distance_cap"]
        label, reason = "unknown", "relative_margin_ambiguous"
        if frame["sharpness"] < cfg["preprocessing"]["minimum_sharpness"]:
            reason = "low_sharpness_or_texture"
        elif not qa and not qb:
            reason = "both_groups_unsupported"
        elif both_strong:
            reason = "both_groups_strong_no_unique_evidence"
        elif qa and relative <= -cfg["minimum_absolute_relative_margin"]:
            label, reason = "A", "supported_relative_A_view_evidence"
        elif qb and relative >= cfg["minimum_absolute_relative_margin"]:
            label, reason = "B", "supported_relative_B_view_evidence"
        rows.append({"sample": i, "t": frame["t"], "pts": frame.get("pts"), "pts_time": frame.get("pts_time"),
                     "source_frame": frame["source_frame"], "image": frame["rgb_image"], "sharpness": frame["sharpness"],
                     "d_A": da, "d_B": db, "S": relative, "A": a, "B": b,
                     "absolute_support_A": qa, "absolute_support_B": qb, "group_evidence": label,
                     "abstention_or_support_reason": reason, "prototype_self_in_this_arm": i in {p["sample"] for p in selected},
                     "excluded_common_prototype_union": i in common_exclusion,
                     "scene_change_decision": "unknown"})
    return rows


def decode(rows, cfg):
    """Full-timeline readout. No manual transition boundaries or identity IDs."""
    runs, current = [], []
    def flush():
        nonlocal current
        if current:
            runs.append({"label": current[0]["group_evidence"], "start": current[0]["t"], "end": current[-1]["t"],
                         "samples": [r["sample"] for r in current],
                         "confirmed": len(current) >= cfg["minimum_run_samples"] and current[-1]["t"] - current[0]["t"] + 1e-6 >= cfg["minimum_run_span_s"]})
        current = []
    for row in rows:
        if row["excluded_common_prototype_union"]:
            continue
        if row["group_evidence"] == "unknown":
            flush()
            continue
        if current and (current[-1]["group_evidence"] != row["group_evidence"] or row["t"] - current[-1]["t"] > cfg["maximum_sample_gap_s"] + 1e-6):
            flush()
        current.append(row)
    flush()
    intervals, last_a = [], None
    for run in runs:
        if not run["confirmed"]:
            continue
        if run["label"] == "A":
            last_a = run
        elif last_a is not None and run["start"] > last_a["end"]:
            intervals.append({"start": last_a["end"], "end": run["start"], "left_evidence_samples": last_a["samples"],
                              "right_evidence_samples": run["samples"], "kind": "view_transition_candidate", "scene_change_decision": "unknown"})
            last_a = None
    return {"intervals": intervals, "primary_interval": intervals[0] if intervals else None, "runs": runs,
            "no_candidate_reason": None if intervals else "no_confirmed_A_run_followed_by_B_run",
            "scene_change_decision": "unknown", "interpretation": "A/B view-evidence transition only, no physical-scene identity inference"}
