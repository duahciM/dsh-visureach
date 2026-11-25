"""Execute frozen V functions with original budgets/NMS, without weakening V."""
import sys
import json

import cv2

from .io import ROOT, _load, common

sys.path.insert(0, str(ROOT / "vendor"))
v = _load("p1_frozen_select_v", ROOT / "vendor/select_v.py")


def run_v(frames):
    cfg = json.loads((ROOT / "vendor/experiment.json").read_text(encoding="utf-8"))
    radius = cfg["sampling"]["change_nms_seconds"]
    budgets = cfg["design"]["initial_budgets"]
    thumbs = [{"i": i, "t": f["t"], "source_frame": f.get("source_frame"),
               "buf": cv2.resize(f["gray"], (96, 54), interpolation=cv2.INTER_AREA).tobytes()}
              for i, f in enumerate(frames)]
    diffs = v.diffs_for(thumbs)
    selections = {}
    for budget in budgets:
        ids = v.nms_top(thumbs, diffs, budget, radius)
        selections[str(budget)] = {"sample_indices": ids,
                                   "times": [frames[i]["t"] for i in ids],
                                   "source_frames": [frames[i].get("source_frame") for i in ids],
                                   "diff_scores": [round(diffs[i], 6) for i in ids]}
    # This is the exact diagnostics logic of select_v.main, including its
    # zero-difference ties. Do not add floors to make the control look worse.
    threshold = common.percentile(diffs[1:], 90)
    peaks = [i for i in range(1, len(frames) - 1)
             if diffs[i] >= threshold and diffs[i] >= diffs[i - 1] and diffs[i] >= diffs[i + 1]]
    peaks.sort(key=lambda i: (-diffs[i], frames[i]["t"]))
    kept = []
    for i in peaks:
        if all(abs(frames[i]["t"] - frames[j]["t"]) > radius for j in kept):
            kept.append(i)
    return {"method": "frozen_real_observation_v2_V", "nms_seconds": radius,
            "thumb_dimensions": [96, 54], "parameters_changed": False,
            "diff_series": [round(d, 6) for d in diffs], "times": [f["t"] for f in frames],
            "diagnostic_threshold_p90": threshold,
            "change_peaks_t": [frames[i]["t"] for i in kept[:8]], "budgets": selections,
            "note": "V is a keyframe selector, not a segment classifier. Native diagnostic peaks include flat ties."}
