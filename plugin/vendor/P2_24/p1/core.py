"""Two stable anchors + translation compensation, with conservative abstention."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from .io import common, reference


@dataclass(frozen=True)
class Parameters:
    coarse_fps: float = 5.
    width: int = 320
    max_frames: int = 5000
    activity_mad: float = .003
    tile_mad: float = .006
    global_tile_fraction: float = .35
    pixel_change: float = .05
    hard_cut_mad: float = .10
    stable_mad: float = .003
    anchor_dwell_s: float = .4
    anchor_search_s: float = 2.
    min_interval_s: float = .2
    merge_gap_s: float = .25
    max_stable_s: float = 30.
    max_gap_s: float = 1.
    min_phase_response: float = .15
    min_overlap: float = .6
    max_residual_ratio: float = .45
    motion_support_fraction: float = .65
    min_motion_px: float = .5
    min_sharpness: float = 8.
    anchor_distance: float = .04
    blend_error: float = .015
    trend_fraction: float = .8

    def validate(self):
        values = asdict(self)
        if not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError("All parameters must be positive and finite")
        for key in ("global_tile_fraction", "min_overlap",
                    "max_residual_ratio", "motion_support_fraction", "trend_fraction"):
            if values[key] > 1:
                raise ValueError(key + " must be <= 1")
        if self.width < 16 or self.max_frames < 2:
            raise ValueError("width >= 16 and max_frames >= 2 required")


def pair_metrics(a, b, p):
    """Shift maps a -> b; warp a and a binary support mask into b coordinates.

    Mask excludes borders AND interpolated edge pixels. Residual is normalized
    8-bit MAD on valid overlap; low overlap never counts as motion support.
    """
    af, bf = a.astype(np.float32), b.astype(np.float32)
    diff = np.abs(af - bf) / 255.
    h, w = a.shape
    tiles = [float(np.mean(x)) for row in np.array_split(diff, 4) for x in np.array_split(row, 6, axis=1)]
    raw = float(diff.mean())
    shift, response = cv2.phaseCorrelate(af.copy(), bf.copy(), cv2.createHanningWindow((w, h), cv2.CV_32F))
    dx, dy = map(float, shift)
    finite = all(math.isfinite(v) for v in (dx, dy, response))
    if not finite:
        dx, dy, response = 0., 0., 0.
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(af, matrix, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ones_like(af), matrix, (w, h), flags=cv2.INTER_LINEAR) > .999
    overlap = float(mask.mean())
    residual = float(np.mean(np.abs(warped[mask] - bf[mask])) / 255) if mask.any() else None
    ratio = residual / max(raw, 1e-6) if residual is not None else None
    moving = math.hypot(dx, dy) >= p.min_motion_px
    support = bool(finite and response >= p.min_phase_response and overlap >= p.min_overlap
                   and ratio is not None and ratio <= p.max_residual_ratio and moving)
    return {"raw_mad": raw, "changed_fraction": float(np.mean(diff > p.pixel_change)),
            "tile_fraction": float(np.mean(np.array(tiles) > p.tile_mad)), "tile_mad": tiles,
            "dx": dx, "dy": dy, "phase_response": float(response), "overlap": overlap,
            "residual_mad": residual, "residual_ratio": ratio, "motion_supported": support,
            "registration_valid": bool(finite and response >= p.min_phase_response and overlap >= p.min_overlap)}


def quality(gray):
    h, w = gray.shape
    # Ignore the outer 10% so a sharp caption/border does not hide a blurred
    # main view. This remains a heuristic, not a readability model.
    center = gray[max(1, h // 10):h - max(1, h // 10), max(1, w // 10):w - max(1, w // 10)]
    return {"sharpness": float(cv2.Laplacian(center, cv2.CV_32F).var()),
            "full_frame_sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "contrast": float(gray.std() / 255)}


def find_anchor(frames, metrics, qualities, edge, direction, p):
    # For the left anchor all dwell links precede the boundary; for the right
    # they follow it. Clip edges without a complete stable dwell are unknown.
    candidates = range(edge - 1, -1, -1) if direction < 0 else range(edge, len(frames))
    for j in candidates:
        if abs(frames[j]["t"] - frames[edge]["t"]) > p.anchor_search_s:
            break
        if qualities[j]["sharpness"] < p.min_sharpness:
            continue
        k = j
        while 0 <= k + direction < len(frames):
            nxt = k + direction
            m = metrics[max(k, nxt)]
            if m["raw_mad"] > p.stable_mad or m["gap_s"] > p.max_gap_s:
                break
            k = nxt
            if abs(frames[k]["t"] - frames[j]["t"]) + 1e-9 >= p.anchor_dwell_s:
                return j
    return None


def anchor_curves(frames, left, right, a, b):
    if left is None or right is None:
        return [], None, None, None
    l = frames[left]["gray"].astype(np.float32) / 255.
    r = frames[right]["gray"].astype(np.float32) / 255.
    delta = r - l
    denom = float(np.sum(delta * delta))
    rows = []
    for i in range(a - 1, b + 1):
        x = frames[i]["gray"].astype(np.float32) / 255.
        alpha = float(np.sum((x - l) * delta) / max(denom, 1e-9))
        mix_error = float(np.mean(np.abs(x - (l + np.clip(alpha, 0, 1) * delta))))
        dl, dr = float(np.mean(np.abs(x - l))), float(np.mean(np.abs(x - r)))
        rows.append({"sample": i, "t": frames[i]["t"], "d_left": dl, "d_right": dr,
                     "relative_to_right": dl / max(dl + dr, 1e-9),
                     "blend_alpha": alpha, "blend_error": mix_error})
    diffs = np.diff([x["relative_to_right"] for x in rows])
    trend = float(np.mean(diffs >= -.02)) if len(diffs) else 0.
    error = float(np.mean([x["blend_error"] for x in rows]))
    return rows, float(np.mean(np.abs(delta))), trend, error


def analyze(frames, source, p=Parameters()):
    p.validate()
    from .io import validate
    validate(frames, source["end"])
    if source["start"] != frames[0]["t"]:
        raise ValueError("Source start must equal first sample PTS")
    n = len(frames)
    qualities = [quality(f["gray"]) for f in frames]
    metrics = [{"sample": 0, "t": frames[0]["t"], "raw_mad": 0., "gap_s": 0.,
                "tile_fraction": 0., "changed_fraction": 0., "motion_supported": False}]
    active, local_events = [], []
    for i in range(1, n):
        m = pair_metrics(frames[i - 1]["gray"], frames[i]["gray"], p)
        m.update(sample=i, t=frames[i]["t"], from_t=frames[i - 1]["t"],
                 gap_s=frames[i]["t"] - frames[i - 1]["t"])
        metrics.append(m)
        global_change = m["raw_mad"] >= p.activity_mad and m["tile_fraction"] >= p.global_tile_fraction
        gap = m["gap_s"] > p.max_gap_s
        low_quality = min(qualities[i - 1]["sharpness"], qualities[i]["sharpness"]) < p.min_sharpness
        if global_change or m["motion_supported"] or gap or low_quality:
            active.append(i)
        elif m["raw_mad"] >= p.activity_mad or m["changed_fraction"] > .005:
            local_events.append({"start": frames[i - 1]["t"], "end": frames[i]["t"],
                                 "frame_left": reference(frames[i - 1]), "frame_right": reference(frames[i]),
                                 "raw_mad": m["raw_mad"], "changed_fraction": m["changed_fraction"],
                                 "reason": "spatially_limited_change_candidate; no semantic label"})
    if n == 1:
        return {"schema": "p1.visual_continuity.v1", "source": source, "parameters": asdict(p),
                "segments": [{"start": source["start"], "end": source["end"], "kind": "uncertain",
                              "anchor_left": None, "anchor_right": None, "motion_type": "unknown",
                              "quality": {"sample_count": 1}, "evidence": {"reason": "insufficient_samples"}}],
                "local_event_candidates": [], "frames": [{**reference(frames[0]), **qualities[0]}],
                "pair_metrics": metrics, "anchor_curves": []}
    groups = []
    for i in active:
        if groups and frames[i - 1]["t"] - frames[groups[-1][-1]]["t"] <= p.merge_gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    segments, transitions, curves = [], [], []
    cursor = source["start"]

    def stable(start, end):
        while end - start > 1e-9:
            finish = min(end, start + p.max_stable_s)
            ids = [i for i, f in enumerate(frames) if start <= f["t"] < finish]
            first, last = (ids[0], ids[-1]) if ids else (None, None)
            segments.append({"start": start, "end": finish, "kind": "stable",
                             "anchor_left": reference(frames[first]) if first is not None else None,
                             "anchor_right": reference(frames[last]) if last is not None else None,
                             "motion_type": "unknown", "quality": {"sample_count": len(ids)},
                             "evidence": {"reason": "no_global_visual_candidate_at_sampled_resolution",
                                          "coverage_only": finish < end,
                                          "sample_times": [frames[i]["t"] for i in ids]}})
            start = finish

    for group in groups:
        a, b = group[0], group[-1]
        left = find_anchor(frames, metrics, qualities, a, -1, p)
        right = find_anchor(frames, metrics, qualities, b, 1, p)
        rows, distance, trend, mix_error = anchor_curves(frames, left, right, a, b)
        cid = len(transitions)
        curves.extend({"candidate": cid, **row} for row in rows)
        links = metrics[a:b + 1]
        support = float(np.mean([m["motion_supported"] for m in links]))
        gap = any(m["gap_s"] > p.max_gap_s for m in links)
        blur = any(qualities[i]["sharpness"] < p.min_sharpness for i in range(a - 1, b + 1))
        anchor_ok = left is not None and right is not None
        hard = len(links) == 1 and links[0]["raw_mad"] >= p.hard_cut_mad and not links[0]["motion_supported"]
        gradual = (len(links) >= 2 and distance is not None and distance >= p.anchor_distance
                   and trend >= p.trend_fraction)
        reason, kind, motion = "unexplained_visual_change", "uncertain", "unknown"
        if gap:
            reason = "sampling_gap"
        elif blur:
            reason = "low_sharpness_or_low_texture"
        elif support >= p.motion_support_fraction:
            reason, kind, motion = "translation_explains_overlap", "transition", "continuous_motion"
        elif anchor_ok and hard:
            reason, kind, motion = "isolated_abrupt_global_change", "transition", "likely_edit"
        elif anchor_ok and gradual and mix_error <= p.blend_error:
            reason, kind, motion = "gradual_candidate_with_low_linear_blend_error", "transition", "likely_edit"
        elif not anchor_ok:
            reason = "stable_anchor_unavailable"
        elif gradual:
            reason = "gradual_candidate_motion_unexplained"
        start, end = frames[a - 1]["t"], frames[b]["t"]
        if b == n - 1 and right is None:
            end = source["end"]
        # Expand very short candidates without moving or deleting source PTS.
        if end - start < p.min_interval_s:
            end = min(source["end"], start + p.min_interval_s)
        # Source-frame-sized final candidates can overlap after expansion.
        start = max(cursor, start)
        if start > cursor:
            stable(cursor, start)
        segment = {"start": start, "end": max(start, end), "kind": kind,
                   "anchor_left": reference(frames[left]) if left is not None else None,
                   "anchor_right": reference(frames[right]) if right is not None else None,
                   "motion_type": motion,
                   "quality": {"anchors_available": anchor_ok, "translation_support_fraction": support,
                               "anchor_distance": distance, "trend_fraction": trend,
                               "linear_blend_error": mix_error, "has_sampling_gap": gap,
                               "has_low_sharpness": blur},
                   "evidence": {"candidate": cid, "reason": reason, "hard_cut_candidate": hard,
                                "gradual_candidate": gradual, "sample_indices": list(range(a - 1, b + 1)),
                                "sample_times": [frames[i]["t"] for i in range(a - 1, b + 1)],
                                "metric_indices": list(range(a, b + 1))}}
        if segment["end"] > start:
            segments.append(segment)
            transitions.append(segment)
            cursor = segment["end"]
    stable(cursor, source["end"])
    return {"schema": "p1.visual_continuity.v1", "source": source, "parameters": asdict(p),
            "interval_convention": "[start,end); seconds in source time; final end uses source frame duration",
            "interpretation": "visual continuity only; unknown is intentional; scores are uncalibrated quality signals",
            "segments": segments, "local_event_candidates": local_events,
            "frames": [{**reference(f), **q} for f, q in zip(frames, qualities)],
            "pair_metrics": metrics, "anchor_curves": curves}
