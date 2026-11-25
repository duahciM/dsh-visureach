"""Adapters from the running source catalogue to frozen spatial research functions."""

from pathlib import Path
import json

import cv2
from p217_runtime import Case, make_candidates, feedback, visual
from p223_runtime import figure1, figure2, node_bundle
from p14_scene_prototypes.features import sift_pair


class LocalBook(Case):
    def __init__(self, args):
        self.source_book = json.loads(Path(args["catalog"]).read_text(encoding="utf-8"))
        self.source = self.source_book["source"]
        self.case = self.source["sha256"]
        self.contract = self.source_book["coordinate_contract"]
        self.height = self.contract["analysis_height"]
        root = Path(args["analysis"])
        self.rows = []
        for row in self.source_book["frames"]:
            receipt = row["source_image"] | {"path": str(root / f'{row["source_frame"]}.png'), "presentation": "analysis_pixels"}
            self.rows.append(row | {"source_image": receipt})
        self.by = {row["frame_id"]: row for row in self.rows}
        self.pixels = {}
        self.pair_info = {}


def research(args):
    book = LocalBook(args)
    operation = args["action"]
    if operation == "figure2":
        row = next((r for r in book.rows if r["pts"] == args["pts"]), None)
        if row is None:
            return {"status": "unavailable", "scope": "PTS has no frozen catalogue nomination; use direct original access."}
        spatial = node_bundle(book, {"candidate_id": row["frame_id"]}, row)
        relations = figure2(spatial)
        return {"status": "completed", "pts": row["pts"], "time_base": row["time_base"], "scope": relations["confidence"],
                "summary": {"regions": len(relations["nodes"]), "measured_pairs": relations["distinct_pairs"],
                            "degree_distribution": relations["degree_distribution"], "edge_types": sorted({x for e in relations["relations"] for x in e["types"]})},
                "figure2": relations, "figure1": figure1(spatial, relations), "coordinates": book.contract}
    if operation == "triggers":
        candidates, missing = make_candidates(book)
        lo, hi = args["start"], args["end"]
        candidates = [c for c in candidates if c["observation_a"]["t"] <= hi and c["observation_c"]["t"] >= lo]
        viable = sum(not (c["base_abstention"] or c["strict_abstention"]) for c in candidates)
        return {"status": "completed", "scope": "P2-17 measured change prompts; unassessed endpoints are retained. No images were delivered.",
                "candidate_count": len(candidates), "supported_for_investigation": viable,
                "items": [{"trigger_id": c["id"], "kind": c["trigger_kind"], "dimension": c["dimension"],
                           "start": c["observation_a"]["t"], "end": c["observation_c"]["t"],
                           "abstention": c["base_abstention"] or c["strict_abstention"]} for c in candidates],
                "candidates": candidates, "outside_catalogue": missing}
    if operation == "feedback":
        trigger = args["trigger"]
        result = feedback(book, trigger, args["frames"], {})
        return {"status": "completed", "trigger_id": trigger["id"], "scope": "P2-17 source readback; comparability and local visual difference only.",
                "recheck": result, "actual_source_pts": [{k: a[k] for k in ("pts", "source_frame", "time_base", "seconds")} for a in args["frames"]]}
    if operation == "appearance":
        features = json.loads((Path(__file__).resolve().parent.parent / "vendor/P2_24/p14_scene_prototypes/protocol.json").read_text(encoding="utf-8"))["sift"]
        detector = cv2.SIFT_create(nfeatures=features["nfeatures"])
        _, anchor_gray = visual(args["anchor"]["path"], book.height)
        first = detector.detectAndCompute(anchor_gray, None)
        comparisons, hits = [], []
        for row in book.rows:
            if row["pts"] == args["anchor"]["pts"] or not args["start"] <= row["t"] <= args["end"]:
                continue
            _, gray = book.image(row["frame_id"])
            result = sift_pair(first, detector.detectAndCompute(gray, None), anchor_gray.shape, gray.shape,
                               features, f'{book.case}:{args["anchor"]["pts"]}:{row["pts"]}')
            stamp = {"pts": row["pts"], "seconds": row["t"], "frame_id": row["frame_id"], "time_base": row["time_base"]}
            comparisons.append(stamp | {"geometry": result})
            if result["verified"]:
                hits.append(stamp | {"verified_inliers": result["inlier_count"], "hull_coverage": min(result["query_hull_coverage"], result["reference_hull_coverage"])})
        return {"status": "completed", "scope": "Direct appearance evidence from the given anchor across the requested catalogue. No transitive merge; no match is unknown.",
                "examined": len(comparisons), "verified_matches": len(hits), "items": hits, "comparisons": comparisons,
                "anchor": {k: args["anchor"][k] for k in ("pts", "time_base", "seconds", "source_frame", "png_sha256")}, "coordinates": book.contract}
    raise ValueError(operation)
