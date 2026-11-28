"""Run C0–C4 and write real native-PNG receipts with per-mechanism provenance."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p216_research.navigate import checked_json, delivery
from p217_research.triggers import Case, make_candidates
from p218_research.observer import observe
from p219_research.choose import index
from p219_research.run import for_observer
from p219_research.score import checked
from p220_research.run import Exact, actual_read, s2_for_fresh
from p220_research.visual import coarse_episodes, find_archived_complements, visibility

from .prepare import EXCLUDED_SHA256, OLD, PROTOCOL
from .recover import RawGrid, catalog, coarse, outside


def fresh_b3(book, source, goal, scope):
    images, reasons, queries = s2_for_fresh(book, source, scope)
    base = {r["frame_id"]: reasons[str(r["pts"])] for r in book.rows if str(r["pts"]) in reasons}
    if goal["intent"] == "single":
        proposals, _ = visibility(book, scope, base)
        images = [delivery(book, k, None) | {"in_catalog": True, "in_coarse_pool": True} for k in proposals]
        return images, {str(r["pts"]): proposals[r["frame_id"]] for r in images}, queries
    offers, _ = find_archived_complements(book, scope, base)
    episodes, _ = coarse_episodes(book, scope)
    present = {r["pts"] for r in images}
    for key, proof in offers.items():
        item = delivery(book, key, None) | {"in_catalog": True, "in_coarse_pool": True}
        reasons.setdefault(str(item["pts"]), []).extend(proof)
        if item["pts"] not in present:
            present.add(item["pts"])
            images.append(item)
    for item, proof in actual_read(book, source, episodes):
        if item["pts"] not in present:
            present.add(item["pts"])
            images.append(item)
        reasons.setdefault(str(item["pts"]), []).append(proof)
    return images, reasons, queries


def previous(route):
    images = checked(route["observer"])
    decision = checked(route["candidate"])
    return [for_observer(r) for r in images["items"]], decision["offer_reasons_by_pts"], decision["frozen_source_queries"]


def run(out):
    input_receipt = read(out / "input_receipt.json")
    frozen = checked(input_receipt["protocol"])
    for path, digest in frozen["files"].items():
        if Path(path).parent == Path(__file__).resolve().parent:
            assert sha(path) == digest
    public = checked(input_receipt["public"])
    historical = read(OLD / "algorithm_lock.json")["routes"]
    by_case = defaultdict(list)
    for goal in public["goals"]:
        by_case[goal["case"]].append(goal)

    def one(job):
        case, goals = job
        book = Case(out, case, public["library"][case])
        assert sha(book.source["path"]) == book.source["sha256"] != EXCLUDED_SHA256
        source = Exact(book)
        facts, missing = make_candidates(book)
        grid = None
        routes, counts = {}, []
        for goal in goals:
            gid = goal["id"]
            trace = out / "traces" / gid
            trace.mkdir(exist_ok=True, parents=True)
            scope = index(book, goal, facts, missing)
            record = historical.get("B3/" + gid)
            initial, initial_reasons, previous_queries = previous(record) if record else fresh_b3(book, source, goal, scope)
            supported = goal["intent"] in ("scene", "open", "recurrence")
            if grid is None and supported:
                grid = RawGrid(book, source)
            channels = {"catalog": catalog(book, goal, scope, initial),
                        "coarse": coarse(book, goal, grid), "outside": outside(book, goal, grid, initial)}
            by_pts = {r["pts"]: r for r in book.rows}
            all_revisit = {r["source_frame"] for key in ("coarse", "outside") for r in channels[key][0]}
            source.ensure(sorted(all_revisit))
            checked_json(trace / "mechanisms.json", {"goal": goal, "catalog_index": scope["entries"],
                        "frozen_scope_partitions": scope["partitions"], "inherited_B3_trace": record,
                        "mechanisms": {k: {"proposals": offers, "decisions": decisions} for k, (offers, decisions) in channels.items()}})
            for arm, enabled in PROTOCOL["mechanisms"].items():
                current = initial.copy()
                reasons = {k: list(v) for k, v in initial_reasons.items()}
                seen = {r["pts"] for r in current}
                offered, revisits = [], []
                for kind in enabled:
                    for row in channels[kind][0]:
                        pts = row["pts"]
                        duplicate = pts in seen
                        reasons.setdefault(str(pts), []).append(row)
                        if kind == "catalog":
                            frame = by_pts[pts]
                            item = delivery(book, frame["frame_id"], None) | {"in_catalog": True, "in_coarse_pool": True}
                        else:
                            item, proof = actual_read(book, source, [row | {"region_original_px": row["measurement"].get("region_original_px")}])[0]
                            assert item["in_coarse_pool"] == (kind == "coarse") and not item["in_catalog"]
                            revisits.append(proof)
                        if not duplicate:
                            current.append(item)
                            seen.add(pts)
                        offered.append(row | {"already_present_same_source_frame": duplicate})
                stream = out / "streams" / arm / gid
                receipt = observe(stream, goal, arm, book.source, current, "positive_candidates_offered" if current else "no_grounded_offer")
                path = trace / f"{arm}.json"
                checked_json(path, {"arm": arm, "goal": goal, "proposals": offered, "original_image_revisits": revisits,
                                    "inherited_source_queries": previous_queries, "reasons_by_pts": reasons,
                                    "selected_pts": [x["pts"] for x in current], "actual_pts": [x["pts"] for x in receipt["items"]],
                                    "local_receiver_only": True})
                routes[f"{arm}/{gid}"] = {"observer": ref(stream / "receipt.json"), "candidate": ref(path),
                                          "mechanisms": ref(trace / "mechanisms.json")}
                counts.append({"goal": goal, "arm": arm, "n": receipt["N_visible"], "public_range": receipt["public_range"],
                               "source_sha256": receipt["source_sha256"], "technical_abort": receipt["technical_abort"]})
                print(arm, gid, receipt["N_visible"], flush=True)
        return routes, counts

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(one, by_case.items()))
    checked_json(out / "count_only.json", {"pairs": [x for _, rows in results for x in rows], "source_role_or_algorithm_pts_exposed": False})
    checked_json(out / "algorithm_lock.json", {"routes": {k: v for part, _ in results for k, v in part.items()}, "count_input": ref(out / "count_only.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
