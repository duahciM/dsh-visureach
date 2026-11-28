"""Local and excluded-context correspondences with descriptor holdout before fitting."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


RULES = json.loads((Path(__file__).with_name("if14_anchor.json")).read_text(encoding="utf-8"))


def prepare(path):
    with Image.open(path) as image:
        width, height = image.size
        shape = (320, max(8, round(height * 320 / width)))
        rgb = np.asarray(image.convert("RGB").resize(shape, Image.Resampling.BILINEAR))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keypoints, desc = cv2.SIFT_create(nfeatures=600).detectAndCompute(gray, None)
    return gray, (width, height), np.array([key.pt for key in keypoints], dtype=np.float32).reshape(-1, 2), desc


def box_at(box, prepared):
    gray, (width, height), _, _ = prepared
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("Region must stay within the original frame.")
    return np.array([x, y, w, h], dtype=float) * np.tile(np.array(gray.shape[::-1]) / (width, height), 2)


def in_box(points, box):
    return np.all((points >= box[:2]) & (points < box[:2] + box[2:]), axis=1)


def correspond(a, b):
    if a is None or b is None or min(len(a), len(b)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratios(x, y):
        return {first.queryIdx: first.trainIdx for match in matcher.knnMatch(x, y, k=2) if len(match) == 2
                for first, rival in [match] if first.distance < .75 * rival.distance}

    reverse = ratios(b, a)
    return sorted((i, j) for i, j in ratios(a, b).items() if reverse.get(j) == i)


def fit(a, b, qualify):
    selected = np.where(qualify)[0]
    held = selected[::RULES["held_every"]]
    train = np.setdiff1d(selected, held)
    result = {"raw_count": len(selected), "fit_count": len(train), "held_count": len(held),
              "fit_points": np.c_[a[train], b[train]].tolist(), "held_points": np.c_[a[held], b[held]].tolist()}
    if len(train) < RULES["min_outside_train"] or len(held) < RULES["min_outside_held"]:
        return result | {"state": "insufficient_direct_descriptors"}, None
    if min(cv2.contourArea(cv2.convexHull(x[train])) for x in (a, b)) <= 1:
        return result | {"state": "degenerate_fit_domain"}, None
    matrix, support = cv2.estimateAffine2D(a[train], b[train], method=cv2.RANSAC, ransacReprojThreshold=RULES["median_px"])
    if matrix is None or np.linalg.det(matrix[:, :2]) <= 0 or int(support.sum()) < RULES["min_outside_train"]:
        return result | {"state": "fit_unavailable"}, None
    error = np.linalg.norm(a[held] @ matrix[:, :2].T + matrix[:, 2] - b[held], axis=1)
    median, p90 = float(np.median(error)), float(np.quantile(error, .9))
    passed = median <= RULES["median_px"] and p90 <= RULES["p90_px"]
    return result | {"state": "heldout_supported" if passed else "heldout_disagrees", "median_px": median, "p90_px": p90,
                     "affine": matrix.tolist(), "held_errors_px": error.tolist(), "fit_inliers": int(support.sum())}, matrix if passed else None


def pair(left, right, native_a, native_b):
    gray_a, _, a, da = left
    gray_b, _, b, db = right
    box_a, box_b = box_at(native_a, left), box_at(native_b, right)
    matches = correspond(da, db)
    ia, ib = (np.array([m[i] for m in matches], dtype=int) for i in (0, 1))
    a, b = a[ia], b[ib]
    qa, qb = in_box(a, box_a), in_box(b, box_b)
    local, _ = fit(a, b, qa & qb)
    context, transform = fit(a, b, ~qa & ~qb)
    residual = {"state": "unassessable_no_heldout_context"}
    if transform is not None:
        h, w = gray_b.shape
        aligned = cv2.warpAffine(gray_a.astype(np.float32), transform, (w, h))
        coverage = cv2.warpAffine(np.ones(gray_a.shape, dtype=np.float32), transform, (w, h)) > .999
        y, x = np.indices(gray_b.shape)
        roi = ((x >= box_b[0]) & (y >= box_b[1]) & (x < box_b[0] + box_b[2]) & (y < box_b[1] + box_b[3]))
        usable, other = coverage & roi, coverage & ~roi
        if usable.any() and other.any():
            delta = gray_b.astype(np.float32) - aligned
            offset = float(np.median(delta[other]))
            abs_delta = np.abs(delta - offset)
            residual = {"state": "measured_on_overlap", "scope": "second ROI in context-projected image; 7 luma is a diagnostic bin",
                        "usable_pixels": int(usable.sum()), "roi_pixels": int(roi.sum()), "median_background_offset_8bit": offset,
                        "target_mad_8bit": float(abs_delta[usable].mean()), "context_mad_8bit": float(abs_delta[other].mean()),
                        "target_fraction_over_7_luma": float(np.mean(abs_delta[usable] > 7)),
                        "context_fraction_over_7_luma": float(np.mean(abs_delta[other] > 7))}
        if (qa & qb).any():
            errors = np.linalg.norm(a[qa & qb] @ transform[:, :2].T + transform[:, 2] - b[qa & qb], axis=1)
            residual["target_projection_median_px"] = float(np.median(errors))
    return {"local": local, "excluded_context": context, "residual": residual,
            "raw_descriptor_matches": len(matches), "boundary_crossing_pairs": np.c_[a[qa != qb], b[qa != qb]].tolist(),
            "coordinate_space": {"width": gray_b.shape[1], "height": gray_b.shape[0]}}


def inspect(args):
    prototypes = [(prepare(f["path"]), f["native_xywh"]) for f in args["prototypes"]]
    rows = []
    for frame in args["samples"]:
        image = prepare(frame["path"])
        probes = [pair(ref, image, box, frame["native_xywh"]) for ref, box in prototypes]
        rows.append({"comparisons": probes})
    return {"status": "completed", "rows": rows}
