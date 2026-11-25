"""Run frozen W0 and the W/L ablations on native observer entry points."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p216_research.navigate import checked_json
from p216_research.navigate import delivery as deliver
from p217_research.triggers import Case, make_candidates
from p218_research.observer import observe
from p219_research.choose import index
from p219_research.run import for_observer
from p219_research.score import checked
from p220_research.run import Exact, actual_read
from p220_research.score import jsonl
from p221_research.recover import RawGrid, catalog, coarse, outside
from p221_research.run import fresh_b3

from .extremes import archive_scores, fixtures, profile, roots, transition
from .prepare import EXCLUDED_SHA256, OLD, PROTO
from .routing import routes


def fresh(book, goal, out, source, facts, missing, grid):
    idx = index(book, goal, facts, missing)
    base, reasons, queries = fresh_b3(book, source, goal, idx)
    by_pts = {r["pts"]: r for r in book.rows}
    cats, decisions = catalog(book, goal, idx, base)
    pts = {r["pts"] for r in base}
    start = base.copy()
    c1_reasons = {k: list(v) for k, v in reasons.items()}
    for proposal in cats:
        stamp = proposal["pts"]
        c1_reasons.setdefault(str(stamp), []).append(proposal)
        if stamp not in pts:
            row = by_pts[stamp]
            start.append(deliver(book, row["frame_id"], None) | {"in_catalog": True, "in_coarse_pool": True})
            pts.add(stamp)
    c4 = start.copy()
    c4_reasons = {k: list(v) for k, v in c1_reasons.items()}
    new_reads = []
    for proposals, _ in (coarse(book, goal, grid), outside(book, goal, grid, base)):
        source.ensure(sorted({r["source_frame"] for r in proposals}))
        for item, proof in actual_read(book, source, [r | {"region_original_px": r["measurement"].get("region_original_px")} for r in proposals]):
            new_reads.append(proof)
            c4_reasons.setdefault(str(item["pts"]), []).append(proof)
            if item["pts"] not in pts:
                pts.add(item["pts"])
                c4.append(item)
    return base, reasons, queries, decisions, (start, c1_reasons), (c4, c4_reasons), new_reads


def previous(old, gid):
    receipts = {arm: checked(old[f"{arm}/{gid}"]["observer"]) for arm in ("C0", "C1", "C4")}
    decisions = {arm: checked(old[f"{arm}/{gid}"]["candidate"]) for arm in receipts}
    def items(arm):
        return [for_observer(r) for r in receipts[arm]["items"]]
    base = items("C0")
    trace = checked(old[f"C1/{gid}"]["mechanisms"])["mechanisms"]["catalog"]["decisions"]
    return (base, decisions["C0"]["reasons_by_pts"], decisions["C0"]["inherited_source_queries"], trace,
            (items("C1"), decisions["C1"]["reasons_by_pts"]), (items("C4"), decisions["C4"]["reasons_by_pts"]), decisions["C4"]["original_image_revisits"])


def run(out):
    fixture = fixtures()
    checked_json(out / "small_sample_fixture.json", fixture)
    input_receipt = read(out / "input_receipt.json")
    frozen = checked(input_receipt["protocol"])
    for path, digest in frozen["files"].items():
        if Path(path).parent == Path(__file__).resolve().parent:
            assert sha(path) == digest
    public = checked(input_receipt["public"])
    old = read(OLD / "algorithm_lock.json")["routes"]
    by_case = defaultdict(list)
    for g in public["goals"]:
        by_case[g["case"]].append(g)

    def execute(job):
        case, goals = job
        book = Case(out, case, public["library"][case])
        assert sha(book.source["path"]) == book.source["sha256"] != EXCLUDED_SHA256
        folder = out / "case_audit" / case
        folder.mkdir(parents=True, exist_ok=True)
        segmentation = transition(book)
        records = roots(book, segmentation)
        scores = archive_scores(records)
        jsonl(folder / "candidate_dimension_roots.jsonl", records)
        jsonl(folder / "cross_dimension_extremes.jsonl", scores)
        checked_json(folder / "segments.json", segmentation | {"dimension_profile": profile(scores)})
        print(f"Indexed {case}: {len(records)} nominations", flush=True)
        counts, receipts = [], {}
        is_new = not any(f'C0/{g["id"]}' in old for g in goals)
        if is_new:
            source = Exact(book)
            facts, missing = make_candidates(book)
            grid = RawGrid(book, source)
        indexed = {r["pts"]: r for r in book.rows}
        for goal in goals:
            gid = goal["id"]
            baseline, reasons, queries, candidates, c1, c4, rereads = (fresh(book, goal, out, source, facts, missing, grid) if is_new else previous(old, gid))
            audit = out / "traces" / gid
            audit.mkdir(exist_ok=True, parents=True)
            pulse = {r["pts"]: r for r in rereads if r.get("role") == "rooted_local_excursion_in_quiet_coarse_bracket"}
            for arm, flags in PROTO["arms"].items():
                counts_before = len(baseline)
                if arm.startswith("W0_"):
                    selected, labels = c1 if arm == "W0_C1" else c4
                    priorities, local_log = [], []
                    paths = []
                    actual_rereads = rereads if arm == "W0_C4" else []
                else:
                    selected = baseline.copy()
                    labels = {k: list(v) for k, v in reasons.items()}
                    present = {r["pts"] for r in selected}
                    paths, priorities, local_log = routes(book, goal, baseline, candidates, scores, flags)
                    for addition in paths:
                        pts = addition["pts"]
                        labels.setdefault(str(pts), []).append(addition)
                        if pts not in present:
                            present.add(pts)
                            frame = indexed[pts]
                            selected.append(deliver(book, frame["frame_id"], None) | {"in_catalog": True, "in_coarse_pool": True})
                    actual_rereads = []
                    for item in c4[0]:
                        pts = item["pts"]
                        if pts in pulse and pts not in present:
                            row = pulse[pts]
                            selected.append(item)
                            present.add(pts)
                            actual_rereads.append(row)
                            labels.setdefault(str(pts), []).append(row | {"path": "identical inherited D6-rooted excursion"})
                target = out / "streams" / arm / gid
                receipt = observe(target, goal, arm, book.source, selected, "research_route_end" if selected else "no_grounded_offer")
                actual = [r["pts"] for r in receipt["items"]]
                assert receipt["technical_abort"] or actual == [r["pts"] for r in selected]
                path = audit / f"{arm}.json"
                checked_json(path, {"goal": goal, "arm": arm, "prefix_images": counts_before, "proposals": paths, "candidate_priority": priorities,
                                    "local_substitution": local_log, "inherited_source_queries": queries, "actual_revisit_receipts": actual_rereads,
                                    "evidence_by_pts": labels, "actual_pts": actual, "computed_weights_do_not_modify_M7": True})
                receipts[arm + "/" + gid] = {"observer": ref(target / "receipt.json"), "trace": ref(path)}
                counts.append({"goal": goal, "arm": arm, "n": receipt["N_visible"], "source_sha256": receipt["source_sha256"],
                               "technical_abort": receipt["technical_abort"], "public_range": receipt["public_range"]})
                print(arm, gid, receipt["N_visible"], flush=True)
        return receipts, counts, case, segmentation

    with ThreadPoolExecutor(max_workers=3) as workers:
        done = list(workers.map(execute, by_case.items()))
    for name in ("candidate_dimension_roots.jsonl", "cross_dimension_extremes.jsonl"):
        with (out / name).open("wb") as output:
            for _, _, case, _ in done:
                output.write((out / "case_audit" / case / name).read_bytes())
    checked_json(out / "video_segment_and_transition_audit.json", {case: data for _, _, case, data in done})
    checked_json(out / "count_only.json", {"pairs": [r for _, rows, _, _ in done for r in rows], "scope": "delivered count only"})
    checked_json(out / "algorithm_lock.json", {"routes": {k: v for part, _, _, _ in done for k, v in part.items()}, "count_input": ref(out / "count_only.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.out.resolve())
