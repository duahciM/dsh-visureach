"""Score only sealed positive roles and record concrete failures by stage."""

import argparse
import json
from collections import Counter
from pathlib import Path

from if17_research.catalog import read, ref
from p216_research.navigate import commit
from p219_research.score import checked, csv_write, e0, grade

from .prepare import OLD


def jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def role_stage(evidence, found, trace):
    pts = set(evidence["source_pts_in_reviewed_scope"])
    raw = pts.intersection(r["pts"] for r in trace["new_exact_source_checks"])
    for query in trace["frozen_source_queries"]:
        raw.update(pts.intersection(query["internally_opened_pts"]))
    selected = pts.intersection(trace["selected_pts_in_order"])
    if found:
        stage = "visible_and_verified"
    elif selected:
        stage = "M8_selected_but_not_verified_on_delivery"
    elif raw:
        stage = "M8_original_read_without_visible_delivery"
    elif evidence["archive_pts"]:
        stage = "M8_archived_role_not_offered"
    elif evidence["coarse_pts"]:
        stage = "M7_missing_from_archive_and_M8_no_successful_revisit"
    elif pts:
        stage = "M0_coarse_sampling_missed_and_M8_no_successful_revisit"
    else:
        stage = "source_pts_unverified_in_sealed_windows"
    return {"phase": stage, "raw_read_pts": sorted(raw), "selected_pts": sorted(selected),
            "coarse_pts": evidence["coarse_pts"], "archive_pts": evidence["archive_pts"]}


def metrics(scored):
    return {"roles": scored["roles_found"], "complete": scored["full"],
            "first_positive": min((r["ordinal"] for r in scored["hits"].values()), default=None),
            "first_complete": scored["first_complete_k"]}


def stream_rows(trial, receipt, reasons):
    for item in receipt["items"]:
        yield {"trial": trial, "goal": receipt["goal"]["id"], "arm": receipt["arm"],
               "goal_intent": receipt["goal"]["intent"], "evidence_reasons": reasons.get(str(item["pts"]), []), **item}


def summarize(records):
    groups = {}
    for row in records:
        for cohort in (row["cohort"], "all"):
            c = groups.setdefault(row["arm"], {}).setdefault(cohort, Counter())
            c.update(goals=1, comparable_goals=int(row["comparable"]), with_ref=int(row["required_roles"] > 0),
                     images=row["native_images"], roles=row["algorithm_roles"], role_target=row["required_roles"],
                     complete=int(bool(row["algorithm_complete"])), no_offer=int(row["native_images"] == 0),
                     repeats=row["repeats"], global_complete=int(bool(row["global_complete"])),
                     scope_complete=int(bool(row["public_scope_complete"])))
            for label in ("global", "public_scope"):
                delta = row[label + "_delta"]
                if delta is not None:
                    c[label + "_role_delta_sum"] += delta
                    c[label + ("_wins" if delta > 0 else "_losses" if delta < 0 else "_ties")] += 1
    return groups


def run(out):
    directory = out / "evaluation_only"
    lock = read(directory / "reference_lock.json")
    truth = checked(lock["roles"])["goals"]
    public = read(out / "inputs_and_goals.json")
    ceiling = e0(out, public, truth)
    commit(out / "E0_catalog_and_source_upper.json", {"goals": ceiling, "read_by_predictor": False})
    old_review = OLD / "evaluation_only/blind_visual_adjudication.json"
    explicit = {}
    for row in read(old_review)["confirmed_noncontributors"]:
        explicit.setdefault(row["goal"], {})[row["pts"]] = "confirmed_noncontributing_sample"
    routes = read(out / "algorithm_lock.json")["routes"]
    glob = read(out / "paired_uniform_global.json")["pairs"]
    scope = read(out / "paired_uniform_public_scope.json")["pairs"]
    assert routes.keys() == glob.keys() == scope.keys()
    rows, complementary, observations, visible, candidates, checks, revisit = [], [], [], [], [], [], []
    details, stages, streams, branches = {}, {}, {}, {}
    from_b0 = {}
    for key, route in routes.items():
        track = checked(route["candidate"])
        index = checked(route["index"])
        triplet = {"algorithm": checked(route["observer"]), "global": checked(glob[key]["observer"]),
                   "public_scope": checked(scope[key]["observer"])}
        algo = triplet["algorithm"]
        goal = algo["goal"]
        gid = goal["id"]
        scores = {}
        for label, receipt in triplet.items():
            assert all(receipt[field] == algo[field] for field in ("goal", "source_sha256", "public_range"))
            trial = key if label == "algorithm" else label + "/" + key
            ptr = route["observer"] if label == "algorithm" else (glob if label == "global" else scope)[key]["observer"]
            streams[trial] = ptr
            scores[label], obs = grade(receipt, truth[gid], explicit.get(gid))
            observations.extend({"trial": trial, **row} for row in obs)
            reasons = track["offer_reasons_by_pts"] if label == "algorithm" else {str(im["pts"]): [
                {"observation_role": "fixed_uniform_phase", "reason": (glob if label == "global" else scope)[key]["basis"]}]
                for im in receipt["items"]}
            visible.extend(stream_rows(trial, receipt, reasons))
        comparable = all(not r["technical_abort"] and r["N_visible"] == algo["N_visible"] for r in triplet.values())
        summary = {"trial": key, "case": goal["case"], "goal": gid, "intent": goal["intent"], "cohort": goal["cohort"],
                   "arm": algo["arm"], "native_images": algo["N_visible"], "unique_pts": algo["N_unique_pts"], "repeats": algo["N_repeated"],
                   "uniform_global_images": triplet["global"]["N_visible"], "uniform_scope_images": triplet["public_scope"]["N_visible"],
                   "required_roles": len(truth[gid]["roles"]), "comparable": comparable,
                   "public_scope": scope[key]["basis"], "uniform_public_start": scope[key]["resolved_scope_seconds"][0],
                   "sdk_acks": 0}
        for label, value in scores.items():
            summary.update({label + "_" + k: v for k, v in metrics(value).items()})
        if algo["arm"] == "B0":
            from_b0[gid] = summary["algorithm_roles"]
        summary["delta_vs_frozen_B0"] = summary["algorithm_roles"] - from_b0[gid]
        for label in ("global", "public_scope"):
            summary[label + "_delta"] = summary["algorithm_roles"] - summary[label + "_roles"] if comparable else None
        rows.append(summary)
        details[key] = {"summary": summary, "scores": scores}
        stages[key] = {role: role_stage(evidence, role in scores["algorithm"]["hits"], track)
                       for role, evidence in ceiling[gid]["roles"].items()}
        if summary["required_roles"] > 1:
            for label, value in scores.items():
                complementary.append({"trial": key, "goal": gid, "selection": label, "images": triplet[label]["N_visible"],
                                      "positive_roles": ";".join(value["hits"]), "unverified": ";".join(value["unverified"]),
                                      "distinct_pts_jointly_verified": value["complementary_observation"], **metrics(value)})
        for case in track["one_view_selection"]["verified_positive_checks"] if track["one_view_selection"] else []:
            checks.append({"trial": key, "check_source": "one_view_pixel_proxy", "candidate": case})
        for case in track["complementary_checks"]:
            checks.append({"trial": key, "check_source": "view_or_projected_context", "candidate": case})
        for role, point in scores["algorithm"]["hits"].items():
            checks.append({"trial": key, "check_source": "independent_positive_role_at_actual_interface", "role": role, "point": point})
        for kind, group in (("frozen_comparison", track["frozen_source_queries"]), ("new_exact_read", track["new_exact_source_checks"])):
            revisit.extend({"trial": key, "kind": kind, "receipt": record} for record in group)
        candidates.append({"trial": key, "M7_archive": route["index"], "M8_checks_detailed": route["candidate"], "native_delivery": route["observer"],
                           "common_M8_unread_eligible_count": index["all_scope_candidates"], "comparison_groups": index["historical_representative_groups"],
                           "selected_pts": track["selected_pts_in_order"], "actual_delivered_pts": track["delivered_pts_in_order"],
                           "offer_reasons_by_pts": track["offer_reasons_by_pts"]})
        if algo["arm"] == "B0":
            branches[gid] = {"catalogue": len(index["entries"]), "visible": algo["N_visible"],
                             "comparison_groups": index["historical_representative_groups"], "coarse_and_archive_positive_pts": ceiling[gid]["roles"],
                             "source_queries": track["frozen_source_queries"], "goal_intent": goal["intent"]}
    for gid, ptr in read(out / "natural_D6.json").items():
        streams["D6_natural/" + gid] = ptr
        receipt = checked(ptr)
        natural, obs = grade(receipt, truth[gid])
        details["D6_natural/" + gid] = natural
        visible.extend(stream_rows("D6_natural/" + gid, receipt, {}))
        observations.extend({"trial": "D6_natural/" + gid, **row} for row in obs)
    csv_write(out / "positive_role_scores.csv", rows)
    csv_write(out / "complementary_role_results.csv", complementary)
    csv_write(out / "role_by_visible_image.csv", observations)
    jsonl(out / "candidate_trace.jsonl", candidates)
    jsonl(out / "positive_visibility_checks.jsonl", checks)
    jsonl(out / "source_revisit_receipts.jsonl", revisit)
    jsonl(out / "visible_stream_receipts.jsonl", visible)
    commit(out / "goal_results.json", {"summary": summarize(rows), "trials": details, "explicit_review_source": ref(old_review),
                                      "unjudged_counts_are_not_false_positives": True})
    commit(out / "failure_attribution.json", {"trials": stages, "source_role_provenance": ref(out / "E0_catalog_and_source_upper.json"),
                                          "stage_is_evidence_attestation_not_causal_proof": True})
    commit(out / "representative_branch_audit.json", {"goals": branches, "global_fixed_cap": None, "scope":
        "S2 single max; D6-only contiguous-region representative; one source comparison per observed region/channel in frozen index. See candidate trace for members, nomination and selected anchors."})
    commit(out / "final_stream_index.json", streams)
    print(f"Scored {len(rows)} paired routes and {len(streams)} native streams", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
