"""Shared helpers for real-observation-v2.

Stdlib + PIL only. ffmpeg/ffprobe are required for video work and are checked
explicitly so the pipeline fails loudly instead of silently degrading.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.json"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(path, obj) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, p)
    return p


def json_load(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def run(cmd, capture=True, cwd=None, timeout=None):
    """Run a command, return (returncode, stdout, stderr) as text."""
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def require_tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise SystemExit(
            "缺少必需工具 %r。请先安装 ffmpeg 并确保它在 PATH 中；"
            "当前阶段不抽帧、不渲染。" % name
        )
    return exe


def probe_video(path) -> dict:
    require_tool("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    code, out, err = run(cmd)
    if code != 0:
        raise SystemExit("ffprobe 失败：%s\n%s" % (path, err[-2000:]))
    data = json.loads(out)
    duration = float(data.get("format", {}).get("duration", 0.0))
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise SystemExit("文件没有视频流：%s" % path)
    fps = 0.0
    rate = video.get("r_frame_rate", "0/1")
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    return {
        "duration_s": round(duration, 6),
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "fps": fps,
        "codec": video.get("codec_name"),
        "size_bytes": int(data.get("format", {}).get("size", 0) or 0),
    }


def probe_first_pts(video):
    """Return the first video frame's PTS in seconds, or None."""
    require_tool("ffprobe")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time", "-read_intervals", "%+#1",
        "-of", "json", str(video),
    ]
    code, out, _ = run(cmd)
    if code != 0:
        return None
    try:
        frames = json.loads(out).get("frames", [])
        if frames and frames[0].get("pts_time") not in (None, "N/A"):
            return float(frames[0]["pts_time"])
    except Exception:
        pass
    return None


def extract_candidates(video, out_dir, fps, width, height, thumb_width, thumb_height) -> list[dict]:
    """Extract shared candidate pool: display JPEG + grayscale PGM.

    Timestamps are i/fps on the fixed grid. All methods and blind annotation
    use exactly this pool.
    """
    out_dir = ensure_dir(out_dir)
    for pattern in ("c_*.jpg", "c_*.pgm"):
        for old in Path(out_dir).glob(pattern):
            old.unlink()
    require_tool("ffmpeg")

    jpg_pattern = str(out_dir / "c_%05d.jpg")
    code, _, err = run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vf", "fps=%s,scale=%d:%d" % (fps, width, height),
        "-fps_mode", "passthrough", "-start_number", "0",
        "-q:v", "2", jpg_pattern,
    ])
    if code != 0:
        raise SystemExit("抽候选显示帧失败：%s\n%s" % (video, err[-2000:]))

    pgm_pattern = str(out_dir / "c_%05d.pgm")
    code, _, err = run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vf",
        "fps=%s,scale=%d:%d,format=gray" % (fps, thumb_width, thumb_height),
        "-fps_mode", "passthrough", "-start_number", "0",
        "-f", "image2", "-pix_fmt", "gray", pgm_pattern,
    ])
    if code != 0:
        raise SystemExit("抽候选缩略帧失败：%s\n%s" % (video, err[-2000:]))

    jpgs = sorted(out_dir.glob("c_*.jpg"))
    pgms = sorted(out_dir.glob("c_*.pgm"))
    n = min(len(jpgs), len(pgms))
    if n <= 0:
        raise SystemExit("没有抽到候选帧：%s" % video)
    items = []
    for i in range(n):
        t = round(i / float(fps), 3)
        items.append({
            "i": i,
            "t": t,
            "display": str(jpgs[i].resolve()),
            "thumb": str(pgms[i].resolve()),
            "display_sha256": sha256_file(jpgs[i]),
            "thumb_sha256": sha256_file(pgms[i]),
        })
    return items


def extract_frame(video, t, out_path, width=640):
    """Extract one display frame at time t (seconds) as JPEG."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    require_tool("ffmpeg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-ss", "%.3f" % max(0.0, float(t)),
        "-frames:v", "1", "-vf", "scale=%d:-2" % width, "-q:v", "2",
        str(out_path),
    ]
    code, _, err = run(cmd)
    if code != 0 or not out_path.exists():
        raise SystemExit("取帧失败 t=%.3f：%s\n%s" % (t, video, err[-1000:]))


def read_pgm(path):
    """Read a binary P5 PGM or P6 PPM into (width, height, bytes)."""
    data = Path(path).read_bytes()
    magic = data[:2]
    if magic not in (b"P5", b"P6"):
        raise ValueError("not P5/P6: %s" % path)
    idx = 2
    tokens = []
    while len(tokens) < 3:
        while idx < len(data) and data[idx:idx + 1].isspace():
            idx += 1
        if idx < len(data) and data[idx:idx + 1] == b"#":
            while idx < len(data) and data[idx:idx + 1] != b"\n":
                idx += 1
            continue
        start = idx
        while idx < len(data) and not data[idx:idx + 1].isspace():
            idx += 1
        tokens.append(int(data[start:idx]))
    idx += 1
    w, h, maxv = tokens
    if maxv != 255:
        raise ValueError("only 8-bit PGM/PPM supported")
    channels = 3 if magic == b"P6" else 1
    buf = data[idx:idx + w * h * channels]
    if len(buf) != w * h * channels:
        raise ValueError("truncated pixel buffer: %s" % path)
    return w, h, buf


def mean_abs_diff(a: bytes, b: bytes) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    total = 0
    step = max(1, n // 20000)
    count = 0
    for i in range(0, n, step):
        total += abs(a[i] - b[i])
        count += 1
    return (total / float(count)) / 255.0


def percentile(values, p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))[:120]


def contact_sheet(image_paths, out_path, cols=3, rows=2,
                  panel_w=560, panel_h=315, labels=None):
    """Build a labeled contact sheet from candidate frames.

    The label is only the timestamp / panel index. No truth, method, object
    names or selection rationale is written onto the annotation input.
    """
    from PIL import Image, ImageDraw, ImageFont

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    canvas = Image.new("RGB", (cols * panel_w, rows * panel_h), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for idx in range(cols * rows):
        r, c = divmod(idx, cols)
        x0, y0 = c * panel_w, r * panel_h
        if idx >= len(image_paths):
            d.rectangle([x0, y0, x0 + panel_w - 1, y0 + panel_h - 1],
                        outline=(210, 210, 210), width=2)
            continue
        try:
            img = Image.open(image_paths[idx]).convert("RGB")
        except Exception:
            d.rectangle([x0, y0, x0 + panel_w - 1, y0 + panel_h - 1],
                        fill=(230, 220, 220), outline=(180, 80, 80), width=3)
            d.text((x0 + 12, y0 + 12), "IMAGE ERROR", fill=(120, 0, 0), font=font)
            continue
        img = img.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        canvas.paste(img, (x0, y0))
        label = labels[idx] if labels and idx < len(labels) else "panel %d" % idx
        # bottom strip keeps the timestamp readable without covering the scene
        d.rectangle([x0, y0 + panel_h - 34, x0 + panel_w, y0 + panel_h],
                    fill=(0, 0, 0))
        d.text((x0 + 10, y0 + panel_h - 30), label, fill=(255, 255, 255), font=font)
        d.rectangle([x0, y0, x0 + panel_w - 1, y0 + panel_h - 1],
                    outline=(80, 80, 80), width=2)
    canvas.save(out_path, "PNG")
    return out_path


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def copy_file(src, dst):
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copyfile(src, dst)
    return dst


def _pil_version():
    try:
        import PIL
        return PIL.__version__
    except Exception:
        return None


if __name__ == "__main__":
    print(json.dumps({
        "root": str(ROOT),
        "python": sys.version,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "pil": _pil_version(),
    }, ensure_ascii=False, indent=2))
