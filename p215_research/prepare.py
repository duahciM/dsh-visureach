"""Freeze original M7 ledgers and replay unchanged M7 on the two supplemental inputs."""
import argparse
from collections import defaultdict
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw

from if16_research.common import frame_key, sequence
from if16_research.nominate import nominate
from if16_research.select import merge
from if17_research.catalog import build_catalog, native_references, read, ref, sha
from p1.io import dump
from v05.engine import load_source


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RULES = read(HERE / "protocol.json")
OLD = ROOT / RULES["baseline"]
CASES = {"new_6a8c74a6": "g3", "new_a31953d0": "short", "new_4740e05f": "long",
         "new_7e08d03a": "g1", "new_a32f9e1f": "g4", "room_corridor_dev": "room"}
F01 = ROOT / "if09_research_20261119/IF09/F01/frames.json"
EXCLUDED_SHA256 = "7f74580c0426420011267f3ad0c68bfa8a53c91b2ae74fa168912a72d4a201cf"


def supplemental(out, case, fresh):
    folder = out / "library" / case
    folder.mkdir(exist_ok=False, parents=True)
    if case == "f01":
        original = read(F01)
        pool, source = original["frames"], original["source"]
        windows = [pool]
        provenance = ref(F01)
    else:
        _, source = load_source(str(fresh), folder / "pool", fps=2, limit=300)
        pool = read(folder / "pool/frames.json")["frames"]
        windows = [[f for f in pool if a <= f["t"] < a + 30] for a in range(0, ceil(source["end"]), 30)]
        provenance = ref(fresh)
    if sha(source["path"]) != source["sha256"]:
        raise ValueError("Original source changed")
    combined = defaultdict(list)
    positions = {frame_key(source, row): i for i, row in enumerate(pool)}
    for i, window in enumerate(windows):
        if not window:
            continue
        channels, _, _ = nominate(sequence(window), source)
        for key, items in channels.items():
            combined[key].extend(r | {"sample": positions[r["frame_id"]], "analysis_window": i} for r in items)
    library = merge(pool, source, combined)
    native = native_references(pool, source, library, folder / "native")
    build_catalog(library, source, native, folder)
    owned = {r["frame_id"] for r in library}
    dump(folder / "coverage.json", {"native_source_frame_count": source["source_frame_count"],
         "candidate_frames": [{k: f[k] for k in ("source_frame","pts","t")} |
                              {"in_catalog": frame_key(source, f) in owned} for f in pool]})
    return {"catalog": ref(folder / "catalog.json"), "index": ref(folder / "index.json"),
            "coverage": ref(folder / "coverage.json"), "source": ref(Path(source["path"])),
            "basis": RULES["added_cases"][case], "candidate_provenance": provenance}


def prepare(out, fresh):
    if sha(fresh) == EXCLUDED_SHA256:
        raise ValueError("The user-excluded video cannot be used")
    stage = out / "library"
    stage.mkdir(exist_ok=False, parents=True)
    catalog_refs = {}
    for path in sorted(OLD.glob("*/catalog.json")):
        folder = path.parent
        catalog = read(path)
        source = ref(Path(catalog["source"]["path"]))
        assert source["sha256"] == catalog["source"]["sha256"]
        key = CASES.get(folder.name, folder.name)
        catalog_refs[key] = {"catalog": ref(path), "index": ref(folder / "index.json"),
            "coverage": ref(folder / "coverage.json"), "source": source, "basis": "frozen P2-13"}
    prior_hashes = {value["source"]["sha256"] for value in catalog_refs.values()}
    if sha(fresh) in prior_hashes or sha(fresh) == read(F01)["source"]["sha256"]:
        raise ValueError("Fresh clip duplicates an existing source")
    for key in ("f01", "fresh"):
        catalog_refs[key] = supplemental(out, key, fresh)
    dump(stage / "manifest.json", catalog_refs)
    print({key: read(value["catalog"]["path"])["frame_count"] for key, value in catalog_refs.items()}, flush=True)


def overview(out):
    manifest = read(out / "library/manifest.json")
    target = out / "reference_review"
    target.mkdir(exist_ok=True)
    guides = {"g3":[0,4.4,4.6,4.8,5.2,5.4,5.5], "short":[.124,2.924,4.624,5.624,6.624,7.124,10.124],
              "long":[.5,5.083,6,27,31,48,62,91,112.458,119.5], "g1":[1,4,10,13,18,23.75,34,48,66],
              "g4":[.1,4,7,12,15,21,25,28,30], "f01":[1,4,6,8,14,18,21,23,24.2,26],
              "fresh":[0,7,25,42,59,70,79,83,90,100,110,118],
              "room":[1,4.8,8,10.42,12,14.6,18],"ppt":[0,11,15.8,16.2,18,30]}
    reviews = {}
    for case, times in guides.items():
        book = read(manifest[case]["catalog"]["path"])
        rows = book["frames"]
        picks = list({row["pts"]: row for t in times for row in [min(rows, key=lambda r: abs(t-r["t"]))]}.values())
        sheet = Image.new("RGB", (1280, ((len(picks)+3)//4)*204))
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(picks):
            x,y=n%4*320,n//4*204
            with Image.open(row["source_image"]["path"]) as image:
                image.thumbnail((320,180))
                sheet.paste(image,(x,y))
            draw.text((x+5,y+182),f'{row["t"]:.3f}s  PTS {row["pts"]}',fill="white")
        sheet.save(target/f"{case}.jpg")
        reviews[case]=[{k:r[k] for k in ("t","pts","source_frame","source_image")} for r in picks]
    dump(target/"inspected_sample_map.json",reviews)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--fresh",type=Path,required=True)
    args=p.parse_args()
    prepare(args.out.resolve(), args.fresh.resolve())
    overview(args.out.resolve())
