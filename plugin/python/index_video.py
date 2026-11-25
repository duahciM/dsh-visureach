"""Build a reusable keyframe catalog for one arbitrary local video.

Reuses the frozen P2-24 nomination/merge/catalog chain, but derives every input
from the video file itself. Nothing here reads historical experiment batches.
"""
import argparse
import json
import sys
from collections import defaultdict
from hashlib import sha256
from math import ceil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vendor" / "P2_24"))

from v05.engine import load_source  # noqa: E402
from if16_research.common import frame_key, sequence  # noqa: E402
from if16_research.nominate import nominate  # noqa: E402
from if16_research.select import merge  # noqa: E402
from if17_research.catalog import build_catalog  # noqa: E402
from vendor.common import probe_video  # noqa: E402

import navigation as wave_module


def probe_duration(path):
    """Ask ffprobe for the real duration so the frame budget follows the video."""
    info = probe_video(Path(path))
    if info.get("duration_s"):
        return float(info["duration_s"])
    raise SystemExit(f"cannot determine duration of {path}")


def file_sha(path):
    h = sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def windowed_nominations(frames, source, window_seconds):
    """Run the unchanged nominator per time window, keyed by original frame."""
    end = ceil(source["end"])
    starts = range(0, max(end, 1), window_seconds)
    windows = [[f for f in frames if a <= f["t"] < a + window_seconds] for a in starts]
    positions = {frame_key(source, row): i for i, row in enumerate(frames)}
    combined = defaultdict(list)
    produced = 0
    for index, window in enumerate(windows):
        if not window:
            continue
        produced += 1
        channels, _, _ = nominate(sequence(window), source)
        for key, items in channels.items():
            combined[key].extend(
                row | {"sample": positions[row["frame_id"]], "analysis_window": index}
                for row in items
            )
    return combined, produced


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
    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Probe the real duration first so the frame budget follows the video
    # instead of a constant that silently thins long clips.
    duration = probe_duration(video)
    implied = max(1, int(duration * args.fps) + 2)
    limit = args.limit if args.limit > 0 else implied + 1

    frames, source = load_source(str(video), out / "pool", fps=args.fps, limit=limit)
    combined, window_count = windowed_nominations(frames, source, args.window_seconds)
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
        "nomination_counts": dict(sorted(dimensions.items())),
        "sampling": {
            "fps": args.fps,
            "configured_limit": args.limit,
            "applied_limit": limit,
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
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # The analysis-resolution pool is an intermediate: everything the tools need
    # now lives in manifest.json and the native frames. Keeping it double the
    # cache for no reader, so drop it after the catalog is written.
    pool_dir = out / "pool"
    if pool_dir.is_dir():
        for leftover in pool_dir.rglob("*"):
            if leftover.is_file():
                leftover.unlink()
        for folder in sorted(pool_dir.rglob("*"), reverse=True):
            if folder.is_dir():
                folder.rmdir()
        pool_dir.rmdir()

    print(json.dumps({"ok": True, "out": str(out), "catalog": len(catalog), "sampled": len(frames)}))


if __name__ == "__main__":
    main()
