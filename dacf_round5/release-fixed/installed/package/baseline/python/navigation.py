"""Derive discrete navigation cues from a catalogue.

Mirrors the frozen p223 timeline contract: adjacent catalogued frames only, no
interpolation, no event identity, and every window it suggests stays a
suggestion until the agent reads the original frames.

Two axes are reported per frame. Both are counts over observations that were
actually made, never over time:

  I  distinct qualified region tokens visible at that frame
  S  distinct relation structures at that frame

S used to count distinct D3 role strings, which collapsed everything: in the
frozen chain every D3 observation carries the same role
(component_near_repeatable_edge), so the count was really a region count in
disguise and different relation structures became indistinguishable. S now keys
each relation by its region token *and* its evidence family, the two fields that
actually separate one relation structure from another here. A structure change
therefore stays visible even when the region count is unchanged.

The frozen chain calls the second axis R and builds it from Figure2 graph pairs.
This port has no graph edges, so the field stays named S and its value is a
distinct-structure count, not a pair count.
"""
import json
from fractions import Fraction
from math import ceil
from pathlib import Path

# p223_research/protocol.json features.uncertain_gap_in_cadences
GAP_IN_CADENCES = 2.5


def frame_axes(row):
    regions = set()
    structures = set()
    for observation in row.get("observations", []):
        region = observation.get("region_candidate")
        if region:
            regions.add(region)
        if observation.get("dimension_id") == "D3":
            # Region plus evidence family, so two structurally different
            # relations on the same region stay countable, and the same
            # structure repeated on many regions does not inflate into many.
            structures.add((
                region or "no-region",
                observation.get("related_evidence_family") or "unlabelled-family",
            ))
    return len(regions), len(structures), sorted("|".join(item) for item in structures)


def time_of(row):
    """Source seconds from PTS and time base, never from a frame index."""
    return float(Fraction(int(row["pts"])) * Fraction(row["time_base"]))


def corpus(rows, source):
    """Catalogue rows plus every sampled frame, so gaps stay distinguishable."""
    frames = []
    for row in rows:
        i, s, signature = frame_axes(row)
        frames.append({
            "frame_id": row["frame_id"],
            "pts": row["pts"],
            "source_frame": row["source_frame"],
            "seconds": time_of(row),
            "I": i,
            "S": s,
            "signature": signature,
            "dimensions": sorted({o["dimension_id"] for o in row.get("observations", [])}),
            "catalogued": True,
        })
    frames.sort(key=lambda f: f["pts"])
    return frames


def build(rows, source, manifest):
    frames = corpus(rows, source)
    start = float(source["start"])
    end = float(source["end"])
    if not frames:
        return {
            "points": [],
            "cues": [],
            "unknown_intervals": [[start, end]],
            "sampling": manifest.get("sampling", {}),
            "axes": "I=distinct region tokens, S=distinct relation signatures",
            "note": "no catalogued frame; nothing was measured",
        }

    spacing = [b["seconds"] - a["seconds"] for a, b in zip(frames, frames[1:])]
    cadence = sorted(spacing)[len(spacing) // 2] if spacing else (end - start)
    detail_gap = cadence * GAP_IN_CADENCES

    unknown = []
    if frames[0]["seconds"] - start > detail_gap:
        unknown.append([start, frames[0]["seconds"]])

    cues = []
    for index in range(1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        flags = []
        if current["seconds"] - previous["seconds"] > detail_gap:
            flags.append("unobserved_gap")
            unknown.append([previous["seconds"], current["seconds"]])
        for axis in ("I", "S"):
            delta = current[axis] - previous[axis]
            if delta > 0:
                flags.append(f"{axis}_rise")
            elif delta < 0:
                flags.append(f"{axis}_fall")
            else:
                flags.append(f"{axis}_same")
        if current["S"] > 0 and previous["S"] > 0 and current["signature"] != previous["signature"]:
            flags.append("relation_signature_change")
            if previous["S"] == current["S"]:
                flags.append("relation_signature_change_at_equal_count")
        if current["S"] > 0 and current["S"] == previous["S"] and "relation_signature_change" not in flags:
            flags.append("observed_S_plateau")
        cues.append({
            "window": [previous["seconds"], current["seconds"]],
            "flags": flags,
            "delta_I": current["I"] - previous["I"],
            "delta_S": current["S"] - previous["S"],
            "endpoints": [previous["frame_id"], current["frame_id"]],
            "endpoint_seconds": [previous["seconds"], current["seconds"]],
            "root_ids": sorted(set(_roots(rows, previous["frame_id"]) + _roots(rows, current["frame_id"]))),
        })

    for index in range(1, len(frames) - 1):
        for axis in ("I", "S"):
            value = frames[index][axis]
            if value > frames[index - 1][axis] and value >= frames[index + 1][axis]:
                cues.append({
                    "window": [frames[index - 1]["seconds"], frames[index + 1]["seconds"]],
                    "flags": [f"{axis}_local_peak"],
                    "delta_I": 0,
                    "delta_S": 0,
                    "endpoints": [frames[index - 1]["frame_id"], frames[index + 1]["frame_id"]],
                    "endpoint_seconds": [frames[index - 1]["seconds"], frames[index + 1]["seconds"]],
                    "root_ids": _roots(rows, frames[index]["frame_id"]),
                })

    if end - frames[-1]["seconds"] > detail_gap:
        unknown.append([frames[-1]["seconds"], end])

    merged = []
    for low, high in sorted(unknown):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])

    return {
        "points": frames,
        "cues": cues,
        "unknown_intervals": merged,
        "sampling": manifest.get("sampling", {}),
        "axes": "I=distinct region tokens, S=distinct relation signatures",
        "gap_threshold_seconds": detail_gap,
        "note": "Adjacent catalogued frames only. No interpolation, and a flag is a navigation "
                "hint, not an observed action, contact or cause.",
    }


def _roots(rows, frame_id):
    for row in rows:
        if row["frame_id"] == frame_id:
            return [root for observation in row.get("observations", [])
                    for root in (observation.get("evidence_root_ids") or [])]
    return []


def write(catalog_path, out_path):
    catalog_path = Path(catalog_path)
    packet = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads((catalog_path.parent / "manifest.json").read_text(encoding="utf-8"))
    result = build(packet, packet["source"], manifest)
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
