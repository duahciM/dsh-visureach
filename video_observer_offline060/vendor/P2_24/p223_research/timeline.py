"""Join a frozen Figure1 snapshot to native PTS and derive discrete navigation cues."""

from fractions import Fraction
from statistics import median

from .prepare import PROTO


def time_of(row):
    return row["pts"]*Fraction(row["time_base"])


def timeline(batch, figure1_rows, transitions, source):
    by_id = {r["candidate"]:r for r in figure1_rows}
    assert set(by_id) == {p["candidate_id"] for p in batch}
    ordered = sorted(batch, key=time_of)
    if not ordered:
        return {"points": [], "unknown_intervals": [[source["start"],source["end"]]], "usable_as_continuous_interpolation": False}, []
    def status(t):
        labels = [r["status"] for r in transitions if r["start"] <= t <= r["end"]]
        return "confirmed" if "confirmed" in labels else "suspected" if labels else "unmarked"
    wave = []
    cues = []
    cadence = median([float(time_of(b)-time_of(a)) for a,b in zip(ordered,ordered[1:])]) if len(ordered)>1 else source["end"]-source["start"]
    detail_gap = cadence*PROTO["features"]["uncertain_gap_in_cadences"]
    unknown = []
    first_t = float(time_of(ordered[0]))
    if first_t > source["start"] and (len(ordered) == 1 or first_t-source["start"] > detail_gap):
        unknown.append([source["start"],first_t])
    for index, p in enumerate(ordered):
        t = float(time_of(p))
        x = by_id[p["candidate_id"]]
        w = {"candidate": p["candidate_id"], "pts": p["pts"], "time_base": p["time_base"], "seconds": t,
             "I": x["I"], "R": x["R"], "relation_states": x["relation_states"], "information_roots": x["information_roots"],
             "transition_status": status(t), "confidence": x["confidence"], "read_between_points": False, "flags": []}
        if index:
            previous = wave[-1]
            lo = previous["seconds"]
            if t-lo > detail_gap:
                w["flags"].append("unobserved_gap")
                unknown.append([lo,t])
            if any(q["start"] <= t and q["end"] >= lo for q in transitions):
                w["flags"].append("transition_bracket")
            for axis in ("I","R"):
                d = w[axis]-previous[axis]
                w["flags"].append(axis+"_rise" if d>0 else axis+"_fall" if d<0 else axis+"_same_measurement")
            relation_shift = x["relation_states"] and previous["relation_states"] and x["relation_states"] != previous["relation_states"]
            if relation_shift:
                w["flags"].append("projected_relation_signature_change")
                if previous["R"] == w["R"]:
                    w["flags"].append("relation_signature_change_at_equal_count")
            if w["R"]>0 and w["R"]==previous["R"] and not relation_shift and "transition_bracket" not in w["flags"]:
                w["flags"].append("observed_R_plateau")
            cues.append({"source_candidates": [previous["candidate"], w["candidate"]], "source_pts": [previous["pts"],w["pts"]],
                         "source_time_base": w["time_base"], "suggested_window_seconds": [lo,t], "flags":w["flags"],
                         "delta_I":w["I"]-previous["I"], "delta_R":w["R"]-previous["R"],
                         "actual_points_only": True})
        wave.append(w)
    for i in range(1,len(wave)-1):
        for axis in ("I","R"):
            v=wave[i][axis]
            if v>wave[i-1][axis] and v>=wave[i+1][axis]:
                wave[i]["flags"].append(axis+"_local_peak")
    last_t = wave[-1]["seconds"]
    if source["end"] > last_t and (len(ordered) == 1 or source["end"]-last_t > detail_gap):
        unknown.append([last_t,source["end"]])
    return {"points": wave, "unknown_intervals": unknown, "no_time_value_reward": True, "usable_as_continuous_interpolation": False,
            "gap_threshold_seconds_from_batch": detail_gap}, cues
