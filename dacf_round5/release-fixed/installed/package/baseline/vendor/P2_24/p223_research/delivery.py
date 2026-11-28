"""Target-driven sender, separate from candidate generation and Figure1/2."""

from collections import defaultdict

import numpy as np


def kind(row):
    return {why.get("role", why.get("observation_role")) for why in row["legacy_reasons"]}


def distance(a,b):
    return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))


def dominates(a,b):
    return a["I"] >= b["I"] and a["R"] >= b["R"] and (a["I"] > b["I"] or a["R"]>b["R"])


def route(goal, batch, evaluations, cues, source):
    candidates = {c["candidate_id"]:c for c in batch}
    features = {r["candidate"]:r for r in evaluations}
    assert candidates.keys() == features.keys()
    policy = goal["intent"]
    lo,hi = source["start"],source["end"]
    declared_lo = lo+(hi-lo)*2/3 if goal.get("scope_hint")=="late" else lo
    key = lambda c: c["candidate_id"]
    chosen, decisions, why, pending = [], [], {}, defaultdict(list)

    def accept(c, reason, support=None):
        if key(c) not in why:
            chosen.append(c)
            why[key(c)] = {"decision":"send", "reason":reason, "support":support}

    legacy = sorted((c for c in batch if c["D0_order"] is not None),key=lambda x:x["D0_order"])
    if policy in ("single","boundary","relation"):
        for c in legacy:
            accept(c,"explicit single or previously protected comparable relation/boundary witnesses", {"original_order":c["D0_order"]})
    else:
        # Existing C0 witnesses preserve a tested observation lineage. New raw and
        # catalogue evidence has a separate decision path and may be held unread.
        for c in legacy:
            if c["historical_core"] and c["seconds"] >= declared_lo:
                accept(c,"retained comparable or direct historical cue", {"historical_root":c["candidate_id"]})
        for c in legacy:
            roles = kind(c)
            if c["seconds"] < declared_lo:
                continue
            if roles & {"rooted_local_excursion_in_quiet_coarse_bracket","shadow_supported_coarse_episode"} and c["available_source_original"]:
                accept(c,"source already supplied measured local episode relevant to the broad comparison", {"source_roles":sorted(x for x in roles if x)})
        # An active named query may need the opening context even at low I/R.
        # Quiet controls with no seeded evidence do not open a catalogue stream.
        if chosen and policy in ("open","scene"):
            options = [c for c in batch if c["seconds"]>=declared_lo and c["in_catalog"] and ("D4" in c["dimensions"] or "D6" in c["dimensions"])]
            if options:
                opening=min(options,key=lambda x:x["seconds"])
                accept(opening,"public comparison benefits from an actual opening view", {"dims":opening["dimensions"]})
        if chosen:
            for c in batch:
                cid=key(c); f=features[cid]; types=kind(c)
                if cid in why or c["seconds"]<declared_lo:
                    continue
                # Public-range membership is available to T0. Adjacent slopes and
                # projected state changes are only available to the wave branch.
                band=min(2,max(0,int(3*(c["seconds"]-declared_lo)/max(hi-declared_lo,1e-6))))
                previous_region = [features[key(x)] for x in chosen if declared_lo+(hi-declared_lo)*band/3 <= x["seconds"] < declared_lo+(hi-declared_lo)*(band+1)/3]
                if policy in ("scene","recurrence") and "D6_public_coverage_comparison_check" in types:
                    if not previous_region or all(distance(f["view_color_grid"],x["view_color_grid"])>.045 for x in previous_region):
                        pending[(band,"readable comparison from authorized source window")].append((c,f))
                if c["in_catalog"] and f["I"] and f["R"]:
                    pattern=tuple(sorted({tag for pair in f["relation_states"] for tag in pair if tag.startswith("color-bin-")}))
                    if not pattern:
                        continue
                    prior_context=[x for x in previous_region if x["R"] and tuple(sorted({tag for p in x["relation_states"] for tag in p if tag.startswith("color-bin-")}))==pattern]
                    if not prior_context and policy in ("scene","recurrence"):
                        pending[(band,"new visible projected participant pattern",pattern)].append((c,f))
            for bucket, group in pending.items():
                # Equal coordinates are one indistinguishable evidence class under
                # this feature contract. Distinct non-dominated coordinates survive.
                admitted=set()
                for c,f in group:
                    point=f["I"],f["R"]
                    if point not in admitted and not any(dominates(x,f) for _,x in group):
                        family=[list(part) if isinstance(part,tuple) else part for part in bucket]
                        accept(c,"independent I/R-supported role entry", {"family":family,"I_R":list(point),"same_role_candidates":len(group)})
                        admitted.add(point)

        explored = set()
        if cues and chosen and policy in ("open","scene","recurrence"):
            for event in cues:
                flags=set(event["flags"])
                if flags & {"unobserved_gap","transition_bracket"} or event["delta_R"] < 0:
                    continue
                previous, current = (candidates[x] for x in event["source_candidates"])
                f=features[key(current)]
                a=features[key(previous)]
                supported_trigger = event["delta_R"]>0 and bool(f["relation_states"]) or "relation_signature_change_at_equal_count" in flags
                if not supported_trigger or not f["I"] or not f["information_roots"] or key(current) in why or current["seconds"]<declared_lo:
                    continue
                # Lower confidence from a changing whole view does not make a
                # relation transition observable by itself. Inspect stable context.
                if distance(a["view_color_grid"],f["view_color_grid"])>.055:
                    continue
                compact=tuple(sorted({tag for pair in f["relation_states"] for tag in pair if tag.startswith("color-bin-")}))
                if not compact:
                    continue
                band=min(2,max(0,int(3*(current["seconds"]-declared_lo)/max(hi-declared_lo,1e-6))))
                pattern=(band,compact, "same-count-change" if "relation_signature_change_at_equal_count" in flags else "rise")
                if pattern in explored:
                    continue
                explored.add(pattern)
                accept(current,"navigable local relation rise or projected subject change with stable comparison context", {
                    "wave":event, "information_roots":f["information_roots"], "I_R":[f["I"],f["R"]]})
                if key(previous) not in why and a["I"] and distance(a["view_color_grid"],f["view_color_grid"])<=.055:
                    accept(previous,"actual preceding comparison image for the selected wave cue", {"wave":event,"information_roots":a["information_roots"]})
    for c in batch:
        decisions.append({"candidate": key(c), "pts": c["pts"], "source_frame": c["source_frame"],
                          **why.get(key(c),{"decision":"hold","reason":"unknown or no independent selected observation role","support":None})})
    return chosen, decisions
