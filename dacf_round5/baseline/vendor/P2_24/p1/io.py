"""PTS-preserving adapters. No nominal-FPS timestamp reconstruction."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = _load("p1_frozen_common", ROOT / "vendor/common.py")


def dump(path, obj):
    # Reject NaN/Inf rather than emitting invalid JSON.
    json.dumps(obj, allow_nan=False)
    common.json_dump(path, obj)


def command(args):
    proc = subprocess.run([str(x) for x in args], capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[-4000:]}")
    return proc.stdout


def validate(frames, end):
    if not frames:
        raise ValueError("No frames")
    times = [f["t"] for f in frames]
    if not all(math.isfinite(t) for t in times + [end]):
        raise ValueError("Non-finite timestamp")
    if any(b <= a for a, b in zip(times, times[1:])) or end <= times[-1]:
        raise ValueError("Require strictly increasing PTS and end > final PTS")
    shapes = {f["gray"].shape for f in frames}
    if len(shapes) != 1:
        raise ValueError("Frame dimensions differ")


def read_video(path, fps=5., width=320, max_frames=5000):
    """Decode sampled frame indices with ffmpeg; join with ffprobe frame PTS.

    Full metadata and full video decode are performed, even at low coarse FPS.
    This is offline reference code, not a seek-optimized or streaming reader.
    """
    path = Path(path).resolve()
    info = common.probe_video(path)
    data = json.loads(command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=pts,pts_time,duration_time,pkt_duration_time:stream=time_base",
        "-of", "json", path]))
    source = data["frames"]
    if not source or any("pts_time" not in f for f in source):
        raise ValueError("Missing source PTS; refusing nominal-FPS fallback")
    times = [float(f["pts_time"]) for f in source]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Non-monotonic/duplicate source PTS")
    chosen, next_t = [], times[0]
    for i, t in enumerate(times):
        if t + 1e-9 >= next_t:
            chosen.append(i)
            next_t = times[0] + len(chosen) / fps
            # Do not fill missing sampling ticks with duplicated frames.
            while next_t <= t + 1e-9:
                next_t += 1 / fps
        if len(chosen) > max_frames:
            raise ValueError("Sample limit exceeded; increase --max-frames explicitly")
    # Always retain the final source frame to close the sampled timeline.
    if chosen[-1] != len(source) - 1:
        chosen.append(len(source) - 1)
    if len(chosen) > max_frames:
        raise ValueError("Sample limit exceeded including final frame")
    height = max(8, round(info["height"] * width / info["width"]))
    def balanced_sum(terms):
        if len(terms) == 1:
            return terms[0]
        mid = len(terms) // 2
        return "(" + balanced_sum(terms[:mid]) + "+" + balanced_sum(terms[mid:]) + ")"
    expr = balanced_sum(["eq(n\\,%d)" % i for i in chosen])
    # Filter script avoids Windows command-line limits for many frame indices.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="p1_select_") as tmp:
        script = Path(tmp) / "select.txt"
        script.write_text(f"select='{expr}',scale={width}:{height},format=gray", encoding="utf-8")
        raw = command(["ffmpeg", "-v", "error", "-noautorotate", "-i", path,
                       "-/filter:v", script, "-fps_mode", "passthrough",
                       "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
    pixels = width * height
    if len(raw) != len(chosen) * pixels:
        raise ValueError("Decoded count differs from PTS index; aborting untrusted timestamp join")
    arr = np.frombuffer(raw, np.uint8).reshape(-1, height, width)
    frames = []
    for j, i in enumerate(chosen):
        frames.append({"t": times[i], "pts": source[i].get("pts"),
                       "pts_time": source[i]["pts_time"], "source_frame": i,
                       "path": str(path), "gray": arr[j].copy()})
    last = source[-1]
    duration = float(last.get("duration_time", last.get("pkt_duration_time", 0)) or 0)
    if duration <= 0:
        raise ValueError("Missing last-frame duration; use an explicit frame manifest with end")
    end = times[-1] + duration
    validate(frames, end)
    return frames, {"path": str(path), "sha256": common.sha256_file(path),
                    "time_basis": "source_pts_seconds", "time_base": data["streams"][0]["time_base"],
                    "source_frame_count": len(source), "sampled_frame_count": len(frames),
                    "start": times[0], "end": end, "probe": info,
                    "decode_note": "Full sequential decode; sampled pixels only retained; no autorotate"}


def read_manifest(path, width=320):
    """{end, frames:[{path,t,source_frame?,pts?,...}]} paths relative to manifest.

    t is required, is preserved verbatim numerically, and must be source time.
    An explicit end avoids inventing the duration after the final sample.
    Extra caller metadata is retained in each frame reference.
    """
    path = Path(path).resolve()
    doc = json.loads(path.read_text(encoding="utf-8"))
    frames = []
    for item in doc["frames"]:
        p = Path(item["path"])
        if not p.is_absolute():
            p = path.parent / p
        if p.suffix.lower() == ".pgm":
            w, h, buf = common.read_pgm(p)
            gray = np.frombuffer(buf, np.uint8).reshape(h, w)
        else:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Unreadable frame: {p}")
        h, w = gray.shape
        gray = cv2.resize(gray, (width, max(8, round(h * width / w))))
        frames.append({**item, "path": str(p.resolve()), "t": float(item["t"]), "gray": gray})
    end = float(doc["end"])
    validate(frames, end)
    return frames, {"path": str(path), "sha256": common.sha256_file(path),
                    "time_basis": "caller_source_t", "start": frames[0]["t"], "end": end,
                    "sampled_frame_count": len(frames), "source": doc.get("source")}


def reference(frame):
    return {k: v for k, v in frame.items() if k != "gray"}
