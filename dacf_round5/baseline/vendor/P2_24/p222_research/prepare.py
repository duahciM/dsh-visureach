"""Lock protocol, old archives and exact historical W0 streams; prepare new sources."""
import argparse
import csv
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p215_research.prepare import EXCLUDED_SHA256, ROOT, supplemental
from p215_research.semantic import Clip, indexed_case
from p216_research.navigate import checked_json

HERE = Path(__file__).resolve().parent
OLD = ROOT / "p221_recovery_20261121"
PROTO = read(HERE / "protocol.json")
LEGACY = ROOT / "v08_first_run_20261117"
SOURCES = {
    "overhead": {"file": "traffic.mp4", "acquisition_group": "Supervisely sample project", "ref": "https://github.com/supervisely-ecosystem/oriented-bbox-video-sample-annotated", "use": "only original video member, no annotation accessed"},
    "turn": {"file": "单场景-摄像机旋转360°.mp4", "directory": "C:/Users/Administrator/Desktop/视觉插件/素材", "acquisition_group": "user's local originals"},
    "edited": {"file": "controlled-cut-and-hold.mp4", "acquisition_group": "controlled derivative, not independent footage", "use": "8s original overhead plus 8s user turn, followed by 6s held last image; 480x270 20fps; editorial timeline is evaluator/fixture metadata only"},
}
GOALS = (
    ("overhead_scene", "overhead", "scene", "Compare the fixed road layout and a later view with road activity. Show the visible lane context."),
    ("overhead_recur", "overhead", "recurrence", "Show vehicle activity in distinct parts of the clip. Do not assume instance identity."),
    ("turn_scene", "turn", "scene", "Compare opening direction, the view during the turn, and the later view. Preserve recognizable spatial context."),
    ("turn_recur", "turn", "recurrence", "Show comparable opening and later viewpoints when visible, without claiming physical identity."),
    ("edited_scene", "edited", "scene", "Show the views on each side of a strong scene change and the later settled view if visible."),
    ("edited_boundary", "edited", "boundary", "Find a checkable view on each side of the apparent source cut, with actual source times."),
)


def freeze(out, card):
    before = read(OLD / "protocol_lock.json")["files"].copy()
    for path in (OLD / "protocol_lock.json", OLD / "E0_reference_lock.json", OLD / "source_and_goal_manifest.json", OLD / "algorithm_lock.json", OLD / "run_manifest.json", *[HERE / n for n in ("prepare.py", "protocol.json", "extremes.py", "routing.py", "run.py", "uniform.py")]):
        before[str(path)] = sha(path)
    before.update({v["path"]: v["sha256"] for v in read(OLD / "run_manifest.json")["artifacts"].values()})
    for key, pointers in read(OLD / "algorithm_lock.json")["routes"].items():
        if key.split("/")[0] in ("C0", "C1", "C4"):
            before.update({v["path"]: v["sha256"] for v in pointers.values()})
    for meta in LEGACY.glob("*/input_manifest.json"):
        for path in (meta, meta.parent / "A/segments_refined.json"):
            if path.is_file():
                before[str(path)] = sha(path)
    source_inputs = {k: ref(Path(v["directory"]) / v["file"] if "directory" in v else out / "assets" / v["file"]) for k, v in SOURCES.items()}
    checked_json(out / "protocol_lock.json", {"files": before, "task_card": ref(card), "rules": PROTO, "new_media": source_inputs,
                 "pre_holdout_review": True, "excluded_source_sha256": EXCLUDED_SHA256})


def public(out):
    seal = read(out / "protocol_lock.json")
    for path, digest in seal["files"].items():
        if Path(path).parent == HERE:
            assert sha(path) == digest
    prior = read(OLD / "source_and_goal_manifest.json")
    library = prior["library"]
    seen = {entry["source"]["sha256"] for entry in library.values()} | {EXCLUDED_SHA256}
    model = ROOT / "p215_discovery_20261121"
    with (model / "assets/openimages.csv").open(encoding="utf-8-sig", newline="") as f:
        vocabulary = sorted({x[1].casefold() for x in csv.reader(f) if len(x) == 2})
    clip = Clip(model)
    text = clip.text([f"a photo of a {label}." for label in vocabulary])
    for case, metadata in SOURCES.items():
        media = seal["new_media"][case]
        source = Path(media["path"])
        assert sha(source) == media["sha256"] and media["sha256"] not in seen
        seen.add(media["sha256"])
        folder = out / "holdouts" / case
        item = supplemental(folder, "fresh", source)
        semantic = folder / "semantic"
        semantic.mkdir()
        rows, _, errors = indexed_case(read(item["catalog"]["path"]), vocabulary, text, clip)
        checked_json(semantic / "candidates.json", {"rows": rows, "errors": errors, "scope": "unchanged P2-21 compatibility index"})
        provenance = {k: v for k, v in metadata.items() if k != "use"}
        library[case] = item | {"public_provenance": provenance, "semantic": str(semantic), "semantic_sha256": sha(semantic / "candidates.json")}
    # The prediction manifest exposes no cue that names the constructed cut or hold time.
    goals = prior["goals"] + [{"id": g, "case": c, "intent": i, "text": q, "cohort": "P2_22_new_source"} for g, c, i, q in GOALS]
    manifest = out / "source_and_goal_manifest.json"
    checked_json(manifest, {"library": library, "goals": goals, "E0_independent": True, "scope": "late = last third of publicly allowed source"})
    checked_json(out / "input_receipt.json", {"public": ref(manifest), "protocol": ref(out / "protocol_lock.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--card", type=Path)
    a = p.parse_args()
    freeze(a.out.resolve(), a.card.resolve()) if a.card else public(a.out.resolve())
