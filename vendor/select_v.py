"""V: pure visual-change initial-frame baseline (deterministic).

V only uses adjacent-frame mean absolute grayscale difference and non-maximum
suppression.  It has no coverage, semantic, diversity or causal term.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (json_dump, json_load, load_config, mean_abs_diff,
                    percentile, read_pgm)  # noqa: E402


def _t(c):
    return float(c.get("source_t", c["t"]))


def load_thumbs(cands):
    out = []
    for c in cands:
        w, h, buf = read_pgm(c["thumb"])
        out.append({"i": c["i"], "t": _t(c), "nominal_t": c["t"],
                    "source_frame": c.get("source_frame"), "buf": buf})
    return out


def diffs_for(frames):
    out = [0.0] * len(frames)
    for i in range(1, len(frames)):
        out[i] = mean_abs_diff(frames[i - 1]["buf"], frames[i]["buf"])
    return out


def nms_top(frames, diffs, n, radius):
    order = sorted(range(len(frames)), key=lambda i: (-diffs[i], frames[i]["t"]))
    picked = []
    for i in order:
        t = frames[i]["t"]
        if all(abs(t - frames[j]["t"]) > radius for j in picked):
            picked.append(i)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for i in order:
            if i not in picked:
                picked.append(i)
            if len(picked) >= n:
                break
    return sorted(picked, key=lambda i: frames[i]["t"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    budgets = cfg["design"]["initial_budgets"]

    ids = []
    if args.all:
        manifest = json_load(ROOT / "materials" / "manifest.json")
        if not manifest:
            raise SystemExit("缺少 materials/manifest.json")
        ids = [v["video_id"] for v in manifest["videos"]]
    elif args.id:
        ids = [args.id]
    else:
        raise SystemExit("需要 --id 或 --all")

    for vid in ids:
        work = ROOT / "work" / vid
        cand = json_load(work / "candidates.json")
        if not cand:
            raise SystemExit("缺少 %s，先运行 prepare_candidates.py" % (work / "candidates.json"))
        frames = load_thumbs(cand["candidates"])
        diffs = diffs_for(frames)
        radius = float(cfg["sampling"]["change_nms_seconds"])
        out = {
            "video_id": vid,
            "method": "V",
            "candidate_fps": cand["candidate_fps"],
            "note": "Adjacent-frame mean absolute grayscale difference; NMS; no semantic term.",
            "nms_seconds": radius,
            "diff_series": [round(d, 6) for d in diffs],
            "change_peaks_t": [],
            "budgets": {},
            "time_basis": "source_t" if all("source_t" in c for c in cand["candidates"]) else "nominal_t",
        }
        # local maxima for diagnostics only
        vals = diffs[1:]
        if vals:
            thr = percentile(vals, 90)
            peaks = [i for i in range(1, len(frames) - 1)
                     if diffs[i] >= thr and diffs[i] >= diffs[i - 1] and diffs[i] >= diffs[i + 1]]
            peaks.sort(key=lambda i: (-diffs[i], frames[i]["t"]))
            kept = []
            for i in peaks:
                if all(abs(frames[i]["t"] - frames[j]["t"]) > radius for j in kept):
                    kept.append(i)
            out["change_peaks_t"] = [frames[i]["t"] for i in kept[:8]]
        for n in budgets:
            picked = nms_top(frames, diffs, n, radius)
            out["budgets"][str(n)] = {
                "times": [frames[i]["t"] for i in picked],
                "source_frames": [frames[i].get("source_frame") for i in picked],
                "nominal_times": [frames[i]["nominal_t"] for i in picked],
                "candidate_ids": [frames[i]["i"] for i in picked],
                "diff_scores": [round(diffs[i], 6) for i in picked],
            }
        json_dump(work / "selections_v.json", out)
        print("[V] %s %s" % (vid, {k: v["times"] for k, v in out["budgets"].items()}))


if __name__ == "__main__":
    main()
