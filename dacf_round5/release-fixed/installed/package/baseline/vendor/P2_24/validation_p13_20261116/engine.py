"""Experimental dataflow only: no reference labels, file paths or P1.2 inputs."""
from copy import deepcopy
import importlib.util
import sys

import numpy as np

from p1.core import anchor_curves, find_anchor, Parameters
from p1.io import reference


def local_states(frames, metrics, qualities, p, config):
    """A link's bad quality cannot mark an entire original group as bad."""
    links = metrics[1:]
    valid = [min(qualities[i - 1]["sharpness"], qualities[i]["sharpness"]) >= p.min_sharpness
             and m["gap_s"] <= p.max_gap_s for i, m in enumerate(metrics[1:], 1)]
    out = []
    half = config["support_window_seconds"] / 2
    for j, m in enumerate(links):
        center = (m["from_t"] + m["t"]) / 2
        lo, hi = max(frames[0]["t"], center - half), min(frames[-1]["t"], center + half)
        weights = [max(0., min(hi, link["t"]) - max(lo, link["from_t"])) for link in links]
        good = sum(w for w, ok in zip(weights, valid) if ok)
        fraction = good / (hi - lo)
        support = sum(w for w, ok, link in zip(weights, valid, links) if ok and link["motion_supported"]) / good if good else None
        if not valid[j]:
            state = "local_quality_or_gap"
        elif fraction < config["minimum_valid_window_fraction"]:
            state = "insufficient_local_quality"
        elif support >= p.motion_support_fraction:
            state = "motion_context"
        else:
            state = "unexplained_context"
        out.append({"sample": j + 1, "from_t": m["from_t"], "t": m["t"], "state": state,
                    "window_start": lo, "window_end": hi, "valid_window_fraction": fraction,
                    "local_support_fraction": support, "link_quality_valid": valid[j]})
    return out


def _classify_original_group(frames, metrics, qualities, p, a, b):
    """Original P1 conditions and their order, applied to a supplied local group.

    No thresholds/features/anchor-search changes. New grouping determines only
    which links reach these original decisions and how local quality is scoped.
    """
    left = find_anchor(frames, metrics, qualities, a, -1, p)
    right = find_anchor(frames, metrics, qualities, b, 1, p)
    rows, distance, trend, mix_error = anchor_curves(frames, left, right, a, b)
    links = metrics[a:b + 1]
    support = float(np.mean([m["motion_supported"] for m in links]))
    gap = any(m["gap_s"] > p.max_gap_s for m in links)
    blur = any(qualities[i]["sharpness"] < p.min_sharpness for i in range(a - 1, b + 1))
    anchor_ok = left is not None and right is not None
    hard = len(links) == 1 and links[0]["raw_mad"] >= p.hard_cut_mad and not links[0]["motion_supported"]
    gradual = (len(links) >= 2 and distance is not None and distance >= p.anchor_distance and trend >= p.trend_fraction)
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
    quality = {"anchors_available": anchor_ok, "translation_support_fraction": support, "anchor_distance": distance,
               "trend_fraction": trend, "linear_blend_error": mix_error, "has_sampling_gap": gap, "has_low_sharpness": blur}
    return {"kind": kind, "motion_type": motion, "quality": quality,
            "anchor_left": reference(frames[left]) if left is not None else None,
            "anchor_right": reference(frames[right]) if right is not None else None,
            "evidence": {"reason": reason, "hard_cut_candidate": hard, "gradual_candidate": gradual,
                         "sample_indices": list(range(a - 1, b + 1)), "metric_indices": list(range(a, b + 1)),
                         "sample_times": [frames[i]["t"] for i in range(a - 1, b + 1)]}}, rows


def arm2(frames, original, config):
    p = Parameters(**original["parameters"])
    metrics, qualities = original["pair_metrics"], original["frames"]
    states = local_states(frames, metrics, qualities, p, config)
    groups = []
    for row in states:
        if groups and groups[-1]["state"] == row["state"]:
            groups[-1]["b"] = row["sample"]
        else:
            groups.append({"state": row["state"], "a": row["sample"], "b": row["sample"]})
    result = deepcopy(original)
    result.update(schema="p1.p13.arm2.local_group_quality", experiment_arm="arm2", segments=[], anchor_curves=[])
    for cid, group in enumerate(groups):
        a, b = group["a"], group["b"]
        segment, rows = _classify_original_group(frames, metrics, qualities, p, a, b)
        segment.update(start=frames[a - 1]["t"], end=frames[b]["t"] if b < len(frames) - 1 else original["source"]["end"])
        segment["evidence"].update(candidate=cid, grouping_state=group["state"])
        result["segments"].append(segment)
        result["anchor_curves"].extend({**row, "candidate": cid} for row in rows)
    analysis = []
    for g, segment in zip(groups, result["segments"]):
        if g["state"] == "motion_context":
            continue
        # Preserve fine segment classification/quality; this union is an explicit
        # analysis extent, not a new semantic transition label.
        if analysis and abs(analysis[-1]["end"] - segment["start"]) < 1e-9:
            analysis[-1]["end"] = segment["end"]
            analysis[-1]["segment_indices"].append(segment["evidence"]["candidate"])
        else:
            analysis.append({"start": segment["start"], "end": segment["end"], "segment_indices": [segment["evidence"]["candidate"]]})
    primary = max(analysis, key=lambda x: x["end"] - x["start"]) if analysis else None
    result.update(local_grouping_config=config, local_support_series=states, analysis_intervals=analysis,
                  automatic_primary_analysis_interval=primary,
                  interpretation="Localized unexplained/quality analysis extents; no manual anchors or reference labels; uncertain is abstention.")
    return result


def forced_anchor_frozen(frames, source, p, left, right, frozen_core_path):
    """Load an isolated, byte-unchanged core; dependency injection stays local."""
    name = "p1._p13_isolated_frozen_core"
    spec = importlib.util.spec_from_file_location(name, frozen_core_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        mod.find_anchor = lambda frames_, metrics_, qualities_, edge_, direction_, params_: left if direction_ < 0 else right
        return mod.analyze(frames, source, p)
    finally:
        sys.modules.pop(name, None)


def anchor_feature_probe(frames, left, right, config):
    # Reuse the original exact pixel features on the whole source timeline.
    rows, distance, trend, mix_error = anchor_curves(frames, left, right, 1, len(frames) - 1)
    bracket = [row for row in rows if left <= row["sample"] <= right]
    lower, upper = config["lower_relative_threshold"], config["upper_relative_threshold"]
    start_row = next((r for r in bracket if r["sample"] > left and r["relative_to_right"] >= lower), None)
    end_row = next((r for r in bracket if start_row is not None and r["t"] >= start_row["t"] and r["relative_to_right"] >= upper), None)
    interval = {"start": start_row["t"], "end": end_row["t"]} if start_row and end_row and end_row["t"] > start_row["t"] else None
    interior = [r for r in bracket if left < r["sample"] < right]
    values = [r["relative_to_right"] for r in interior]
    return {"feature_source": "unchanged p1.core.anchor_curves", "anchor_left": reference(frames[left]),
            "anchor_right": reference(frames[right]), "curves": rows,
            "full_curve_original_quality": {"anchor_distance": distance, "trend_fraction": trend, "linear_blend_error": mix_error},
            "probe_readout": {"rule": "diagnostic first passage 10%-90% within known anchor span", "thresholds": [lower, upper],
                              "interval": interval, "start_crossing_at_left_anchor": bool(start_row and start_row["sample"] == left),
                              "end_crossing_at_right_anchor": bool(end_row and end_row["sample"] == right),
                              "interior_reaches_upper_threshold": any(v >= upper for v in values),
                              "interior_relative_min": min(values) if values else None,
                              "interior_relative_max": max(values) if values else None,
                              "interior_trend_fraction": float(np.mean(np.diff(values) >= -.02)) if len(values) > 1 else None},
            "anchor_span_only_baseline": {"start": frames[left]["t"], "end": frames[right]["t"]},
            "interpretation": "Oracle-anchor feature diagnostic only; readout does not replace frozen decision, prove semantic transition or count as automatic performance."}
