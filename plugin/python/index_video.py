"""Build a reusable keyframe catalog for one arbitrary local video.

Reuses the frozen P2-24 nomination/merge/catalog chain, but derives every input
from the video file itself. Nothing here reads historical experiment batches.
"""
import argparse
import json
from shutil import copyfile
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vendor" / "P2_24"))

from v05.engine import CFG as SOURCE_CONFIG  # noqa: E402
from if16_research.common import frame_key  # noqa: E402
from if16_research.select import merge  # noqa: E402
from if17_research.catalog import build_catalog  # noqa: E402
from resumable import scan

import navigation as wave_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    # The frozen base configuration samples at 5 fps (v05/protocol.json
    # sample_fps, if15_research pool.fps). Density is a property of the video,
    # not a hidden budget: 0 means "every frame this rate implies".
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = no cap. A positive value only bounds pair work and is always reported.")
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--keep-analysis", action="store_true")
    parser.add_argument("--interrupt-after-windows", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--disable-dimensions", nargs="*", choices=[f"D{i}" for i in range(1, 7)], default=[])
    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.window_seconds < 1 or args.fps <= 0:
        raise ValueError("Window seconds and sampling rate must be positive")
    frames, source, combined, window_count, resumed = scan(video, out, args.fps, args.limit, SOURCE_CONFIG["width"], args.window_seconds,
                                                            args.disable_dimensions, args.interrupt_after_windows)
    library = merge(frames, source, combined)

    # Native pixels are produced on demand at read time, so the catalogue keeps
    # full sampling density without storing a PNG for every nominated frame.
    native = {
        frame_key(source, row): {
            "source_path": source["path"],
            "source_frame": row["source_frame"],
            "pts": row["pts"],
            "time_base": source["time_base"],
            "width": source["probe"]["width"],
            "height": source["probe"]["height"],
            "materialized": False,
        }
        for row in library
    }
    catalog = build_catalog(library, source, native, out)

    # Discrete navigation cues over the catalogue, following the frozen p223
    # timeline contract. Zero images; it only proposes windows to investigate.
    wave = wave_module.build(catalog, source, {"sampling": {"fps": args.fps}})
    (out / "wave.json").write_text(
        json.dumps(wave, ensure_ascii=False), encoding="utf-8"
    )

    # Index the full sampled pool so the agent can tell "sampled but not nominated"
    # apart from "never sampled" — a distinction the delivery contract requires.
    nominated = {row["frame_id"] for row in catalog}
    analysis = None
    if args.keep_analysis:
        analysis = out / "analysis"
        analysis.mkdir(exist_ok=True)
        for frame in frames:
            if frame_key(source, frame) in nominated:
                copyfile(frame["image"], analysis / f'{frame["source_frame"]}.png')
    pool = [
        {
            "frame_id": frame_key(source, f),
            "source_frame": f["source_frame"],
            "pts": f["pts"],
            "seconds": f["t"],
            "in_catalog": frame_key(source, f) in nominated,
        }
        for f in frames
    ]

    dimensions = defaultdict(int)
    for row in catalog:
        for nomination in row["nominations"]:
            dimensions[nomination["dimension_id"]] += 1

    manifest = {
        "video_sha256": source["sha256"],
        "video_path": str(video),
        "bytes": video.stat().st_size,
        "probe": source["probe"],
        "time_base": source["time_base"],
        "start": source["start"],
        "end": source["end"],
        "source_frame_count": source["source_frame_count"],
        "duration_seconds": source["end"] - source["start"],
        "sampled_frame_count": len(frames),
        "catalog_frame_count": len(catalog),
        "analysis_windows": window_count,
        "window_receipts_reused": resumed,
        "nomination_counts": dict(sorted(dimensions.items())),
        "sampling": {
            "fps": args.fps,
            "configured_limit": args.limit,
            "applied_limit": args.limit or None,
            "frame_interval_seconds": 1.0 / args.fps,
            "downgrade": source.get("sampling_downgrade"),
            "note": "Every frame implied by this rate is indexed unless downgrade is not None.",
        },
        "pool": pool,
        "catalog_path": str(out / "catalog.json"),
        "wave_path": str(out / "wave.json"),
        "wave_summary": {
            "points": len(wave["points"]),
            "cues": len(wave["cues"]),
            "unknown_intervals": len(wave["unknown_intervals"]),
        },
        "index_path": str(out / "index.json"),
        "analysis_path": str(analysis) if analysis else None,
        "disabled_dimensions": args.disable_dimensions,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(json.dumps({"ok": True, "out": str(out), "catalog": len(catalog), "sampled": len(frames)}))


if __name__ == "__main__":
    main()
