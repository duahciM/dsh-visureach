"""Window receipts for the unchanged offline nominator; retain only one window of pixels."""
import json
import os
from collections import defaultdict
from pathlib import Path
from subprocess import PIPE, Popen
from tempfile import TemporaryFile

import numpy as np
from PIL import Image
from if16_research.common import frame_key, sequence
from if16_research.nominate import nominate
from p1.io import command, common
from v05.engine import select_expression


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def describe(video, root, fps, limit, width, window, disabled):
    sha = common.sha256_file(video)
    signature = {"sha256": sha, "fps": fps, "limit": limit, "width": width, "window_seconds": window, "disabled": sorted(disabled)}
    path = root / "source.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["signature"] != signature:
            raise ValueError("Checkpoint source or parameters changed")
        cached["source"]["path"] = str(video)
        return cached["source"], cached["frames"]
    info = common.probe_video(video)
    parsed = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                                 "-show_entries", "frame=pts,pts_time,duration_time,pkt_duration_time:stream=time_base", "-of", "json", video]))
    original = parsed["frames"]
    if not original or any("pts_time" not in f or "pts" not in f for f in original):
        raise ValueError("Missing original PTS")
    times = [float(row["pts_time"]) for row in original]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Non-monotonic/duplicate source PTS")
    chosen, next_t = [], times[0]
    for i, t in enumerate(times):
        if t + 1e-9 >= next_t:
            chosen.append(i)
            next_t = times[0] + len(chosen) / fps
            while next_t <= t + 1e-9:
                next_t += 1 / fps
    if chosen[-1] != len(original) - 1:
        chosen.append(len(original) - 1)
    downgrade = None
    if limit and len(chosen) > limit:
        chosen = [chosen[n] for n in sorted(set(np.linspace(0, len(chosen)-1, limit).round().astype(int)))]
        downgrade = f"uniformly bounded to {limit} candidates"
    duration = float(original[-1].get("duration_time", original[-1].get("pkt_duration_time", 0)) or 0)
    if duration <= 0:
        raise ValueError("Missing last-frame duration")
    time_base = parsed["streams"][0]["time_base"]
    source = {"path": str(video), "sha256": sha, "time_basis": "source_pts_seconds", "time_base": time_base,
              "source_frame_count": len(original), "sampled_frame_count": len(chosen), "start": times[0], "end": times[-1] + duration,
              "probe": info, "decode_note": "Exact source selection with durable completed analysis windows"}
    if downgrade:
        source["sampling_downgrade"] = downgrade
    frames = [{"sample": rank, "source_frame": i, "pts": original[i]["pts"], "pts_time": original[i]["pts_time"], "t": times[i], "time_base": time_base}
              for rank, i in enumerate(chosen)]
    atomic(path, {"signature": signature, "source": source, "frames": frames})
    return source, frames


def scan(video, out, fps, limit, width, window_seconds, disabled, interrupt_after=0):
    root = out / "checkpoints"
    root.mkdir(exist_ok=True)
    source, frames = describe(video, root, fps, limit, width, window_seconds, disabled)
    positions = {frame_key(source, row): i for i, row in enumerate(frames)}
    groups = defaultdict(list)
    for frame in frames:
        groups[int(frame["t"] // window_seconds)].append(frame)
    combined, missing, resumed = defaultdict(list), {}, 0

    def absorb(rows):
        for key, reasons in rows.items():
            combined[key].extend(reasons)

    for key, window in groups.items():
        receipt = root / f"window-{key}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if all((root / f"{idx}.png").is_file() for idx in saved["nominated_indices"]):
                absorb(saved["channels"])
                resumed += 1
                continue
        missing[key] = window

    if missing:
        height = max(8, round(source["probe"]["height"] * width / source["probe"]["width"]))
        script = root / "select.ffilter"
        distinct = [frame["source_frame"] for window in missing.values() for frame in window]
        script.write_text(f"select='{select_expression(distinct)}',scale={width}:{height},format=rgb24", encoding="ascii")
        argv = ["ffmpeg", "-v", "error", "-xerror", "-noautorotate", "-i", str(video), "-map", "0:v:0",
                "-/filter:v", str(script), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        size = width * height * 3
        with TemporaryFile() as errors, Popen(argv, stdout=PIPE, stderr=errors) as reader:
            try:
                for written, (key, window) in enumerate(missing.items(), 1):
                    local = []
                    for original in window:
                        i = original["source_frame"]
                        pixels = reader.stdout.read(size)
                        if len(pixels) != size:
                            raise ValueError(f"RGB/source-PTS join failed at original {i}")
                        path = root / f"{i}.png"
                        temporary = root / f"{i}.pending.png"
                        Image.frombytes("RGB", (width, height), pixels).save(temporary)
                        os.replace(temporary, path)
                        local.append(original | {"image": str(path)})
                    channels, _, _ = nominate(sequence(local), source, disabled=disabled)
                    rows = {name: [r | {"sample": positions[r["frame_id"]], "analysis_window": key} for r in found]
                            for name, found in channels.items()}
                    wanted = {frames[r["sample"]]["source_frame"] for group in rows.values() for r in group}
                    for row in window:
                        if row["source_frame"] not in wanted:
                            (root / f'{row["source_frame"]}.png').unlink(missing_ok=True)
                    atomic(root / f"window-{key}.json", {"nominated_indices": sorted(wanted), "channels": rows})
                    absorb(rows)
                    if interrupt_after and written >= interrupt_after:
                        raise InterruptedError("Acceptance-only interruption after durable window")
                if reader.stdout.read(1) or reader.wait():
                    raise ValueError("Unexpected RGB decoder output after selected PTS")
            except BaseException:
                if reader.poll() is None:
                    reader.kill()
                reader.wait()
                errors.seek(0)
                if detail := errors.read():
                    print(detail.decode("utf-8", errors="replace"))
                raise
    # Restore original iteration ordering even when an earlier receipt needs rebuild.
    for rows in combined.values():
        rows.sort(key=lambda r: r["analysis_window"])
    for row in frames:
        row["image"] = str(root / f'{row["source_frame"]}.png')
    return frames, source, combined, len(groups), resumed
