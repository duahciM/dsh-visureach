"""Measure both temporal extremes on source-linked nomination/role records."""
from collections import Counter, defaultdict
from math import exp, sqrt

import numpy as np

from if17_research.catalog import read, ref
from p1.core import analyze
from p1.io import read_video

from .prepare import LEGACY, PROTO


DIMENSIONS = tuple(f"D{i}" for i in range(1, 7))


def transition(book):
    rules = PROTO["transition"]
    start, end = book.source["start"], book.source["end"]
    candidates = []
    for info in LEGACY.glob("*/input_manifest.json"):
        if read(info).get("source_sha256") != book.source["sha256"]:
            continue
        source = info.parent / "A/segments_refined.json"
        if not source.is_file():
            continue
        for d in read(source)["diagnostics"]:
            interval = d.get("transition_range")
            if interval and d.get("transition_accepted"):
                candidates.append({"start": interval[0], "end": interval[1], "status": "confirmed", "anchor": ref(source),
                                   "reason": "frozen SHA-matched v08 two-sided source visual acceptance"})
    frames, declared = read_video(book.source["path"], fps=rules["fallback_fps"], width=128, max_frames=5000)
    assert declared["sha256"] == book.source["sha256"]
    baseline = analyze(frames, declared)
    edges = baseline["pair_metrics"]
    def known(a, b):
        return any(x["start"] <= b and a <= x["end"] for x in candidates)
    for i, link in enumerate(edges[1:], 1):
        if link["raw_mad"] < rules["abrupt_min_rgb_mad"] or link["motion_supported"]:
            continue
        side = rules["side_samples"]
        neighborhood = edges[max(1, i-side):i] + edges[i+1:i+side+1]
        if len(neighborhood) < 2 * side:
            continue
        support = max(x["raw_mad"] for x in neighborhood)
        if link["raw_mad"] < rules["abrupt_min_ratio"] * max(.005, support):
            continue
        a, b = link["from_t"], link["t"]
        if not known(a, b):
            candidates.append({"start": a, "end": b, "status": "confirmed", "reason": "isolated native abrupt contrast, stable same-side observations and no translation match",
                               "pts_left": frames[i-1]["pts"], "pts_right": frames[i]["pts"], "raw_gray_mad": link["raw_mad"], "max_same_side_mad": support})
    for item in baseline["segments"]:
        a, b = item["start"], item["end"]
        evidence = item["evidence"]["reason"]
        uncertain = item["motion_type"] == "likely_edit" or evidence in ("sampling_gap", "gradual_candidate_motion_unexplained")
        if uncertain and not known(a, b):
            candidates.append({"start": a, "end": b, "status": "suspected", "reason": evidence, "P1": item})
    candidates.sort(key=lambda r: (r["start"], r["end"], r["status"]))
    boundaries = sorted({start, end, *((a["start"] + a["end"]) / 2 for a in candidates if a["status"] == "confirmed" and start < a["start"] < end)})
    return {"source_sha256": book.source["sha256"], "window_candidates": candidates,
            "segments": [{"id": i, "start": a, "end": b} for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))],
            "evaluated_P1_links": len(edges)-1, "confirmed_mask_margin_s": rules["mask_margin_s"],
            "no_source_role_labels_read": True}


def status(t, evidence):
    margin = evidence["confirmed_mask_margin_s"]
    active = [r["status"] for r in evidence["window_candidates"] if r["start"]-margin <= t <= r["end"]+margin]
    return "confirmed" if "confirmed" in active else "suspected" if active else "continuous"


def roots(book, evidence):
    output = []
    sections = evidence["segments"]
    for frame in book.rows:
        t = frame["t"]
        matches = [j for j in sections if j["start"] <= t < j["end"]]
        segment = matches[0] if matches else sections[-1]
        for i, n in enumerate(frame["observations"]):
            output.append({"nomination": f'{frame["frame_id"]}:{i}', "frame_id": frame["frame_id"],
                          "source_frame": frame["source_frame"], "pts": frame["pts"], "time_base": frame["time_base"], "seconds": t,
                          "dimension": n["dimension_id"], "independent_role": n["role"], "root_ids": sorted(set(n["evidence_root_ids"])),
                          "segment_id": segment["id"], "segment_range": [segment["start"], segment["end"]], "transition_status": status(t, evidence),
                          "source_time_only": n["related_evidence_family"] == "source_time", "analysis_box": n["region_xywh_analysis"],
                          "comparison_frames": n["source_comparison_frames"]})
    return output


def kernel(delta, tau):
    return exp(-(delta/tau)**2)


def archive_scores(records):
    by_segment = defaultdict(list)
    for n in records:
        by_segment[n["segment_id"]].append(n)
    rules = PROTO["extremes"]
    output = []
    for n in records:
        own_roots = set(n["root_ids"])
        t = n["seconds"]
        potential = [r for r in by_segment[n["segment_id"]] if r["dimension"] != n["dimension"]]
        delta = sorted(((abs(t-r["seconds"]), r["nomination"], r) for r in potential), key=lambda x: (x[0], x[1]))
        for tau in PROTO["scales_s"]:
            measures = {}
            for mode in ("unmasked", "raw", "dedup", "clean", "clean_mask_suspected"):
                masked = mode != "unmasked"
                remove_roots = mode in ("dedup", "clean", "clean_mask_suspected")
                density = mode.startswith("clean")
                hide = {"confirmed", "suspected"} if mode == "clean_mask_suspected" else {"confirmed"}
                other = [x for x in delta if not (masked and x[2]["transition_status"] in hide) and not (density and x[2]["source_time_only"])]
                used_roots, chosen = own_roots.copy(), {}
                can_priority = not (masked and n["transition_status"] in hide) and bool(own_roots)
                for dist, _, r in (other if can_priority else []):
                    rr = set(r["root_ids"])
                    d = r["dimension"]
                    if d in chosen or (remove_roots and (not rr or rr & used_roots)):
                        continue
                    local_density = len({tuple(k["root_ids"]) for dd, _, k in other if k["dimension"] == d and dd <= tau and k["root_ids"]})
                    weight = 1 / sqrt(max(1, local_density)) if density else 1
                    chosen[d] = {"nomination": r["nomination"], "pts": r["pts"], "delta_s": dist,
                                 "roots": r["root_ids"], "source_time_only": r["source_time_only"], "density_roots": local_density,
                                 "kernel_weighted": kernel(dist, tau) * weight}
                    used_roots |= rr
                eligible_dims = {x[2]["dimension"] for x in other}
                div = max(1, len(eligible_dims) if density else len(chosen))
                c = sum(v["kernel_weighted"] for v in chosen.values()) / div if chosen else None
                real = {d: r for d, r in chosen.items() if not r["source_time_only"]}
                both_independent = len(real) >= rules["minimum_independent_peer_dimensions_for_O"]
                root_count = len({tuple(x[2]["root_ids"]) for x in other if not x[2]["source_time_only"] and set(x[2]["root_ids"]) - own_roots})
                a, b = n["segment_range"]
                reason = None
                if not can_priority:
                    reason = "masked_transition_or_no_evidence_root"
                elif n["source_time_only"] or not both_independent or root_count < rules["minimum_root_points_for_O"]:
                    reason = "insufficient_non_D6_independent_coverage"
                elif n["transition_status"] == "suspected":
                    reason = "suspected_transition_no_isolation_claim"
                elif b-a < rules["minimum_segment_tau"]*tau or min(t-a, b-t) < rules["edge_tau"]*tau:
                    reason = "short_section_or_edge"
                o = None if reason else 1-kernel(min(r["delta_s"] for r in real.values()), tau)
                measures[mode] = {"C": c, "O": o, "support": chosen, "O_uncertainty": reason,
                                  "eligible_dimension_count": len(eligible_dims), "independent_non_D6_root_sets": root_count}
            output.append(n | {"time_scale": tau, "scores": measures})
    return output


def fixtures():
    toy = {"A": [1, 2, 3], "B": [1.5, 2.5, 4], "C": [1.4, 4.1, 9]}
    gaps = {d: [min(abs(t-u) for k, vals in toy.items() if k != d for u in vals) for t in xs] for d, xs in toy.items()}
    means = {d: sum(vals)/len(vals) for d, vals in gaps.items()}
    assert min(means, key=means.get) == "B" and max(max(v) for v in gaps.values()) == gaps["C"][-1] == 5
    def note(dim, t, rr, label="continuous"):
        return {"nomination": f"{dim}:{t}", "frame_id": f"sample:{t}", "seconds": t, "pts": int(t*10),
                "time_base": "1/10", "segment_id": 0, "segment_range": [0, 12], "dimension": dim,
                "source_time_only": False, "transition_status": label, "root_ids": [rr]}
    distinct = [note("D1", 2, "a"), note("D2", 2, "a"), note("D3", 2, "a")]
    trap = [note("D1", 3, "a1", "confirmed"), note("D2", 3, "b1", "confirmed"), note("D3", 3, "c1", "confirmed")]
    out = [x for x in archive_scores(distinct + trap) if x["time_scale"] == 1]
    cases = {r["nomination"]: r["scores"] for r in out}
    assert cases["D1:2"]["raw"]["C"] == 1 and cases["D1:2"]["dedup"]["C"] is None
    assert cases["D1:3"]["unmasked"]["C"] == 1 and cases["D1:3"]["raw"]["C"] is None
    real = [note(f"D{i+1}", t, f"{d}{t}") for i, (d, vals) in enumerate(toy.items()) for t in vals]
    metric = [x for x in archive_scores(real) if x["time_scale"] == 1 and x["nomination"] == "D3:9"][0]
    assert metric["scores"]["clean"]["O"] > .99
    dense = [note("D1", 3, "case"), note("D2", 3, "b1"), note("D2", 3.2, "b2"), note("D2", 3.4, "b3"), note("D3", 3, "d1")]
    density = next(x for x in archive_scores(dense) if x["nomination"] == "D1:3" and x["time_scale"] == 1)
    assert density["scores"]["clean"]["C"] < density["scores"]["dedup"]["C"]
    return {"toy_distances_s": gaps, "means_s": means, "nearest_dimension": "B", "isolated": "C at 9s",
            "root_duplicate_and_transition": cases, "C9_scored": metric, "density_pair": density, "all_assertions_passed": True}


def profile(scores):
    supports = defaultdict(set)
    for entry in scores:
        if entry["time_scale"] != PROTO["primary_scale_s"]:
            continue
        for key in ("C", "O"):
            value = entry["scores"]["clean"][key]
            if value is not None and value >= .5:
                supports[(entry["dimension"], key)].add((entry["segment_id"], int(entry["seconds"]), tuple(entry["root_ids"])))
    return {d: {k: len(supports[(d, k)]) for k in ("C", "O")} for d in DIMENSIONS}
