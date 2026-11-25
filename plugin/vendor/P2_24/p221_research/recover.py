"""Independent catalog, coarse-pool and source-only positive candidate routes."""

from collections import defaultdict
from pathlib import Path
from statistics import median
import subprocess

import cv2
import numpy as np

from if17_research.catalog import read
from p217_research.triggers import identifier
from p220_research.visual import ball_context


CFG = read(Path(__file__).with_name("protocol.json"))["sampling"]
SIZE = CFG["analysis_width"], CFG["analysis_height"]


def view(rgb):
    return cv2.resize(rgb, (12, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def distance(a, b):
    return float(np.abs(a - b).mean())


def native_box(box, shape, book):
    x, y, w, h = box
    return [round(v * m / n) for v, m, n in zip((x, y, w, h), (book.source["probe"]["width"], book.source["probe"]["height"]) * 2, (shape[1], shape[0]) * 2)]


def root(book, layer, role, pts, parents):
    return identifier(book.source["sha256"], layer, role, pts, *parents)


def offer(book, layer, role, stamp, parents, measurement, zone=None):
    return {"layer": layer, "role": role, "source_frame": stamp["source_frame"], "pts": stamp["pts"],
            "seconds": stamp["t"], "zone": zone, "evidence_root": root(book, layer, role, stamp["pts"], parents),
            "parents": parents, "measurement": measurement, "proposed": True, "identity_confirmed": False}


def public_scope(book, goal):
    a, b = book.source["start"], book.source["end"]
    return (a + (b - a) * 2 / 3 if goal.get("scope_hint") == "late" else a), b


def entrance(t, lo, hi):
    n = CFG["catalog_zones"]
    return min(n - 1, max(0, int(n * (t - lo) / max(hi - lo, 1e-6))))


def components(rgb, bg):
    delta = np.max(np.abs(rgb.astype(np.int16) - bg.astype(np.int16)), axis=2)
    mask = np.uint8(delta >= CFG["coarse_high_residual"])
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    result = []
    for x, y, cw, ch, area in stats[1:]:
        if area < CFG["foreground_min_area"] or cw * ch > CFG["foreground_max_fraction"] * w * h or min(cw, ch) < 2:
            continue
        strength = float(delta[y:y + ch, x:x + cw].mean()) / 255
        if strength >= CFG["local_foreground"]:
            result.append({"box": [int(v) for v in (x, y, cw, ch)], "pixels": int(area), "strength": strength,
                           "strength_area": float(area * strength)})
    return sorted(result, key=lambda r: -r["strength_area"])


def sharpness(rgb):
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def catalog(book, goal, indexes, initial):
    if goal["intent"] not in ("scene", "open", "recurrence"):
        return [], [{"decision": "unchanged public goal type"}]
    lo, hi = public_scope(book, goal)
    items = [r for r in indexes["eligible"] if lo <= r["t"] < hi]
    if len(items) < 2:
        return [], [{"decision": "insufficient catalog candidates"}]
    desc = {r["pts"]: view(book.image(r["frame_id"])[0]) for r in items}
    first = desc[items[0]["pts"]]
    if max(distance(first, x) for x in desc.values()) < CFG["quiet_tail_context"]:
        return [], [{"decision": "quiet whole-source catalog; no extra positive candidate"}]
    baseline = np.median(np.stack([book.image(r["frame_id"])[0] for r in items]), axis=0)
    selected, trace = [], []
    occupied = {x["pts"] for x in initial}
    by_zone = defaultdict(list)
    for r in items:
        by_zone[entrance(r["t"], lo, hi)].append(r)
    for zone in range(CFG["catalog_zones"]):
        rows = by_zone[zone]
        if not rows:
            trace.append({"zone": zone, "decision": "unknown: region has no catalog candidate"})
            continue
        supported = defaultdict(list)
        for row in rows:
            rgb, _ = book.image(row["frame_id"])
            quality = sharpness(rgb)
            parts = components(rgb, baseline)
            dims = set(row["image_dimensions"])
            roots = sorted({v for o in row["observations"] for v in o["evidence_root_ids"]}) or [row["frame_id"]]
            boxes = [native_box(x["box"], rgb.shape, book) for x in parts]
            context = {"dims": sorted(dims), "sharpness": quality, "foreground_regions_original_px": boxes,
                       "distance_from_opening": distance(first, desc[row["pts"]]), "pixel_support": bool(parts)}
            if quality < CFG["visible_sharpness"]:
                trace.append({"zone": zone, "pts": row["pts"], "decision": "visible detail uncertain", "measurements": context})
                continue
            if "D6" in dims or "D4" in dims:
                supported["temporal_context"].append((row, roots, context))
            if parts and dims & {"D1", "D2", "D3", "D4", "D6"}:
                supported["locally_distinct_state"].append((row, roots, context | {"local_score": parts[0]["strength_area"]}))
            if goal["intent"] == "recurrence" and "ball" in goal["text"].casefold() and dims & {"D1", "D3"}:
                localized = ball_context(rgb)
                if localized:
                    rec = context | {"projection_boxes": [native_box(x["box"], rgb.shape, book) for x in localized]}
                    supported["recurring_projected_context"].append((row, roots, rec))
        for role, records in supported.items():
            priority = {
                "temporal_context": sorted(records, key=lambda x: (-x[2]["sharpness"], x[0]["t"])),
                "locally_distinct_state": sorted(records, key=lambda x: (-x[2].get("local_score", 0), x[0]["t"])),
                "recurring_projected_context": sorted(records, key=lambda x: x[0]["t"]),
            }[role]
            accepted = [desc[r["pts"]] for r, _, _ in records if r["pts"] in occupied]
            for row, parents, measures in priority:
                signal = desc[row["pts"]]
                min_novelty = min((distance(signal, known) for known in accepted), default=None)
                threshold = CFG["quiet_tail_context"] if role == "recurring_projected_context" else CFG["context_novelty"]
                same_role = min_novelty is not None and min_novelty < threshold
                trace.append({"zone": zone, "pts": row["pts"], "role": role, "parents": parents, "measurements": measures,
                              "distance_to_accepted_same_role": min_novelty, "decision": "existing or redundant measured view" if same_role else "independent pixel witness"})
                if not same_role:
                    selected.append(offer(book, "M7_catalog", role, row, parents, measures, zone))
                    accepted.append(signal)
    return selected, trace


class RawGrid:
    def __init__(self, book, source):
        self.book = book
        self.source = source
        command = ["ffmpeg", "-v", "error", "-noautorotate", "-i", book.source["path"], "-map", "0:v:0",
                   "-vf", f"scale={SIZE[0]}:{SIZE[1]}:flags=area,format=rgb24", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
        raw = subprocess.check_output(command)
        self.frames = np.frombuffer(raw, np.uint8).reshape(-1, SIZE[1], SIZE[0], 3)
        if len(self.frames) != len(source.ledger):
            raise ValueError("Original PTS to low-resolution measurements mismatch")
        self.times = np.array([float(x["pts_time"]) for x in source.ledger])
        self.pool = {r["source_frame"]: r for r in book.coverage}
        self.archive = {r["source_frame"] for r in book.rows}
        self.coarse_background = np.median(self.frames[sorted(self.pool)], axis=0)
        self.context = np.stack([view(rgb) for rgb in self.frames])

    def stamp(self, i):
        return {"source_frame": i, "pts": self.source.ledger[i]["pts"], "t": float(self.times[i])}

    def roots(self, indices):
        return [identifier(self.book.source["sha256"], self.source.ledger[i]["pts"], "frozen_coarse_point") for i in indices]


def coarse(book, goal, grid):
    if goal["intent"] not in ("scene", "recurrence", "open"):
        return [], [{"decision": "unchanged public goal type"}]
    lo, hi = public_scope(book, goal)
    measured, episodes, active = [], [], []
    for i in sorted(grid.pool):
        if not lo <= grid.times[i] < hi:
            continue
        rgb = grid.frames[i]
        parts = components(rgb, grid.coarse_background)
        if not parts or i in grid.archive:
            if active:
                episodes.append(active)
                active = []
        elif sharpness(rgb) >= CFG["visible_sharpness"]:
            active.append((i, parts[0]))
        measured.append({"pts": grid.source.ledger[i]["pts"], "in_archive": i in grid.archive, "foreground": parts,
                         "M7_shadow_candidate": bool(parts) and i not in grid.archive})
    if active:
        episodes.append(active)
    result = []
    for episode in episodes:
        ids = [i for i, _ in episode]
        anchors = grid.roots(ids)
        peak = max(p["strength_area"] for _, p in episode)
        last = None
        for i, p in episode:
            zone = entrance(grid.times[i], lo, hi)
            span = [float(grid.times[ids[0]]), float(grid.times[ids[-1]])]
            distinct = last is None or distance(grid.context[i], grid.context[last]) >= CFG["quiet_tail_context"]
            edge = i in (ids[0], ids[-1])
            if (edge or distinct) and p["strength_area"] >= CFG["recurrence_strength_ratio"] * peak:
                result.append(offer(book, "coarse_candidate_not_catalog", "shadow_supported_coarse_episode",
                                    grid.stamp(i), anchors, p | {"aligned_coarse_span": span, "episode_endpoint": edge,
                                    "region_original_px": native_box(p["box"], grid.frames[i].shape, book)}, zone))
                last = i
    if not result:
        measured.append({"decision": "no supported off-catalog coarse run"})
    return result, measured


def local_delta(grid, i, a):
    residual = cv2.absdiff(grid.frames[i], grid.frames[a]).max(axis=2)
    patch = cv2.blur(residual.astype(np.float32), (12, 9))
    return float(patch.max() / 255)


def outside(book, goal, grid, initial):
    if goal["intent"] not in ("scene", "recurrence", "open"):
        return [], [{"decision": "unchanged public goal type"}]
    lo, hi = public_scope(book, goal)
    anchors = [i for i in sorted(grid.pool) if lo <= grid.times[i] < hi]
    if len(anchors) < 2:
        return [], [{"decision": "no frozen D6/coarse bracket"}]
    result, trace = [], []
    coarse_cadence = median(grid.times[b] - grid.times[a] for a, b in zip(anchors, anchors[1:]))
    legacy = [r["seconds"] for r in initial]
    for a, b in zip(anchors, anchors[1:]):
        span = [float(grid.times[a]), float(grid.times[b])]
        if b <= a + 1:
            continue
        loc = local_delta(grid, b, a)
        quiet = loc < CFG["source_local_delta"]
        got = any(span[0] < t < span[1] for t in legacy)
        roots = grid.roots([a, b])
        # A parity return in one coarse bracket merits one actual changed-state
        # excursion; scanning proxies does not itself deliver an original.
        if quiet:
            vals = [(i, local_delta(grid, i, a)) for i in range(a + 1, b) if i not in grid.pool and i not in grid.archive]
            runs = []
            active = []
            for i, strength in vals:
                if strength >= CFG["source_local_delta"]:
                    active.append((i, strength))
                elif active:
                    runs.append(active)
                    active = []
            if active:
                runs.append(active)
            for group in runs:
                i, strength = max(group, key=lambda row: row[1])
                p = offer(book, "source_search_outside_coarse", "rooted_local_excursion_in_quiet_coarse_bracket", grid.stamp(i), roots,
                          {"window": span, "endpoint_local_delta": loc, "measured_local_delta": strength,
                           "distinct_excursion_pts": [grid.stamp(x)["pts"] for x, _ in group], "representative_rule": "one measured peak for contiguous same excursion"})
                result.append(p)
            if not runs:
                trace.append({"window": span, "root_ids": roots, "decision": "D6 comparison had no above-threshold local excursion"})
            continue
        # Protect intermediate positions only for an uncovered comparison window.
        nearest_seen = min((abs(t - sum(span) / 2) for t in legacy), default=float("inf"))
        if not got and nearest_seen >= coarse_cadence * CFG["neighborhood_in_coarse_steps"]:
            i = min(range(a + 1, b), key=lambda x: abs(grid.times[x] - sum(span) / 2))
            if i not in grid.pool and i not in grid.archive and sharpness(grid.frames[i]) >= CFG["visible_sharpness"]:
                result.append(offer(book, "source_search_outside_coarse", "D6_public_coverage_comparison_check", grid.stamp(i), roots,
                                    {"window": span, "endpoint_local_delta": loc, "cadence": float(coarse_cadence),
                                     "criterion": "midpoint in an active bracket lacking nearby delivered observations"}))
        else:
            trace.append({"window": span, "root_ids": roots, "decision": "legacy interface nearby; no second D6 probe"})
    return result, trace
