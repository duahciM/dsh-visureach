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
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vendor" / "P2_24"))

from v05.engine import CFG as SOURCE_CONFIG, select_expression  # noqa: E402
from if16_research.common import frame_key, sequence  # noqa: E402
from if16_research.nominate import nominate  # noqa: E402
from if16_research.select import merge  # noqa: E402
from if17_research.catalog import build_catalog  # noqa: E402
from p1.io import command, read_video  # noqa: E402

import navigation as wave_module


def windowed_nominations(frames, source, window_seconds, disabled=()):
    """Run the unchanged nominator per time window, keyed by original frame."""
    windows = defaultdict(list)
    for frame in frames:
        windows[int(frame["t"] // window_seconds)].append(frame)
    positions = {frame_key(source, row): i for i, row in enumerate(frames)}
    combined = defaultdict(list)
    produced = 0
    for index, window in windows.items():
        produced += 1
        channels, _, _ = nominate(sequence(window), source, disabled=disabled)
        for key, items in channels.items():
            combined[key].extend(
                row | {"sample": positions[row["frame_id"]], "analysis_window": index}
                for row in items
            )
    return combined, produced


def scan_source(video, folder, fps, explicit_limit):
    """Reuse the exact source reader without the research adapter's 5,000-frame ceiling."""
    frames, source = read_video(video, fps=fps, width=SOURCE_CONFIG["width"], max_frames=sys.maxsize)
    if explicit_limit > 0 and len(frames) > explicit_limit:
        indices = sorted(set(np.linspace(0, len(frames) - 1, explicit_limit).round().astype(int)))
        frames = [frames[index] for index in indices]
        source["sampling_downgrade"] = f"uniformly bounded to {explicit_limit} candidates"
    height, width = frames[0]["gray"].shape
    with TemporaryDirectory(prefix="observer_rgb_") as scratch:
        expression = Path(scratch) / "select.txt"
        filtergraph = f"select='{select_expression([f['source_frame'] for f in frames])}',scale={width}:{height},format=rgb24"
        expression.write_text(filtergraph, encoding="ascii")
        raw = command(["ffmpeg", "-v", "error", "-xerror", "-noautorotate", "-i", video, "-map", "0:v:0",
                       "-/filter:v", expression, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    if len(raw) != len(frames) * height * width * 3:
        raise ValueError("RGB/source-PTS join mismatch")
    rgb = np.frombuffer(raw, np.uint8).reshape(len(frames), height, width, 3)
    folder.mkdir(parents=True, exist_ok=True)
    for i, (frame, pixels) in enumerate(zip(frames, rgb)):
        path = folder / f"{i:04d}.png"
        Image.fromarray(pixels).save(path)
        frame.update(sample=i, time_base=source["time_base"], image=str(path.resolve()))
    source["sampled_frame_count"] = len(frames)
    return frames, source


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
    parser.add_argument("--disable-dimensions", nargs="*", choices=[f"D{i}" for i in range(1, 7)], default=[])
    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.window_seconds < 1 or args.fps <= 0:
        raise ValueError("Window seconds and sampling rate must be positive")
    frames, source = scan_source(video, out / "pool", args.fps, args.limit)
    combined, window_count = windowed_nominations(frames, source, args.window_seconds, args.disable_dimensions)
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
