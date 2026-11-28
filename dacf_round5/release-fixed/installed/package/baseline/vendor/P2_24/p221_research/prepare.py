"""Freeze experiments and build public inputs with the unchanged M7 intake."""

import argparse
import csv
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p215_research.prepare import EXCLUDED_SHA256, ROOT, RULES, supplemental
from p215_research.semantic import Clip, indexed_case
from p216_research.navigate import checked_json


HERE = Path(__file__).resolve().parent
OLD = ROOT / "p220_positive_20261121"
PROTOCOL = read(HERE / "protocol.json")
HOLDOUTS = {
    "breakfast": {"filename": "so101-breakfast.mp4", "page": "https://github.com/ljt228/pi05-so101-finetune#sample-videos",
                  "acquisition": "https://raw.githubusercontent.com/ljt228/pi05-so101-finetune/main/assets/sample_breakfast_front.mp4",
                  "publisher": "ljt228, real SO-101 project demonstration"},
    "local_pan": {"filename": "单场景-摄像机来回平移一次.mp4", "directory": "C:/Users/Administrator/Desktop/视觉插件/素材",
                  "publisher": "user supplied local camera footage, not in the P2-20 goal/source manifest"},
    "local_return": {"filename": "双场景-从A到B再回A.mp4", "directory": "C:/Users/Administrator/Desktop/视觉插件/素材",
                     "publisher": "user supplied local camera footage, not in the P2-20 goal/source manifest"},
}
GOALS = [
    ("breakfast_scene", "breakfast", "scene", "Compare the opening tabletop arrangement with a later visible arrangement. Preserve the arm and tabletop references."),
    ("breakfast_recur", "breakfast", "recurrence", "Show the arm and visible objects at separated parts of the demonstration. Keep object identity and contact uncertain."),
    ("pan_scene", "local_pan", "scene", "Compare opening and later viewpoints. Preserve shared structures and show what becomes visible.") ,
    ("pan_recur", "local_pan", "recurrence", "Show visually comparable opening and later views. Keep physical scene identity uncertain."),
    ("return_scene", "local_return", "scene", "Compare an opening view, a different view in the middle, and a later reference. Use visible spatial structure."),
    ("return_recur", "local_return", "recurrence", "Show comparable appearances of the opening and later views. Keep physical scene identity uncertain."),
]


def freeze(out, card):
    reference = read(OLD / "run_manifest.json")
    former = read(OLD / "protocol_lock.json")
    artifacts = former["files"].copy()
    for file in (OLD / "protocol_lock.json", OLD / "run_manifest.json", OLD / "algorithm_lock.json", OLD / "inputs_and_goals.json",
                 *[HERE / name for name in ("protocol.json", "prepare.py", "recover.py", "run.py", "uniform.py")]):
        artifacts[str(file)] = sha(file)
    for key, row in read(OLD / "algorithm_lock.json")["routes"].items():
        if key.startswith("B3/"):
            artifacts.update({v["path"]: v["sha256"] for v in row.values()})
    artifacts.update({v["path"]: v["sha256"] for v in reference["artifacts"].values()})
    checked_json(out / "protocol_lock.json", {"files": artifacts, "task_card": ref(card), "rules": PROTOCOL,
                 "verified_before_holdout_image_review": True, "excluded_source_sha256": EXCLUDED_SHA256})


def public(out):
    frozen = read(out / "protocol_lock.json")
    for name in ("protocol.json", "prepare.py", "recover.py", "run.py", "uniform.py"):
        assert frozen["files"][str(HERE / name)] == sha(HERE / name)
    prior = read(OLD / "inputs_and_goals.json")
    library = prior["library"]
    seen = {item["source"]["sha256"] for item in library.values()} | {EXCLUDED_SHA256}
    model = ROOT / "p215_discovery_20261121"
    with (model / "assets/openimages.csv").open(encoding="utf-8-sig", newline="") as handle:
        lexicon = sorted({r[1].casefold() for r in csv.reader(handle) if len(r) == 2})
    encoder = Clip(model)
    text = encoder.text([f"a photo of a {word}." for word in lexicon])
    for key, provenance in HOLDOUTS.items():
        source = Path(provenance["directory"]) / provenance["filename"] if "directory" in provenance else out / "assets" / provenance["filename"]
        source_ref = ref(source)
        assert source_ref["sha256"] not in seen, key
        seen.add(source_ref["sha256"])
        dest = out / "holdouts" / key
        cached = dest / "library/fresh"
        if cached.is_dir():
            assert read(cached / "catalog.json")["source"]["sha256"] == source_ref["sha256"]
            entry = {n: ref(cached / f"{n}.json") for n in ("catalog", "index", "coverage")}
            entry.update(source=source_ref, candidate_provenance=source_ref, basis=RULES["added_cases"]["fresh"])
        else:
            entry = supplemental(dest, "fresh", source)
        sem = dest / "semantic"
        sem.mkdir(exist_ok=True)
        if not (sem / "candidates.json").exists():
            generated, _, errors = indexed_case(read(entry["catalog"]["path"]), lexicon, text, encoder)
            checked_json(sem / "candidates.json", {"rows": generated, "errors": errors, "scope": "unchanged P2-20 local CLIP compatibility index"})
        library[key] = entry | {"public_provenance": provenance, "semantic": str(sem), "semantic_sha256": sha(sem / "candidates.json")}
    goals = prior["goals"] + [{"id": k, "case": c, "intent": i, "text": t, "cohort": "P2_21_new_source"} for k, c, i, t in GOALS]
    path = out / "source_and_goal_manifest.json"
    checked_json(path, {"library": library, "goals": goals, "source_ranges_from_native_catalog": True,
                        "scope": PROTOCOL["scope"], "E0_readable_by_predictor": False})
    checked_json(out / "input_receipt.json", {"public": ref(path), "protocol": ref(out / "protocol_lock.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--card", type=Path)
    args = p.parse_args()
    folder = args.out.resolve()
    if args.card:
        freeze(folder, args.card.resolve())
    else:
        public(folder)
