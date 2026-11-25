"""Optional source-bound pixel, CLIP and text channels for the installed tools."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "vendor" / "P2_24"
sys.path.insert(0, str(VENDOR))
from if16_research.nominate import regions
from if16_research.common import RULES
from p1.core import Parameters, pair_metrics, quality
from p1.photometric import brightness_metrics
from p14_scene_prototypes.features import histogram, sift_pair

FEATURES = json.loads((VENDOR / "p14_scene_prototypes/protocol.json").read_text(encoding="utf-8"))


def file_hash(path):
    with Path(path).open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def inspect_pair(args):
    raw = [Image.open(p).convert("RGB") for p in args["paths"]]
    native = raw[0].size
    width, height = 320, max(8, round(320 * native[1] / native[0]))
    colors = [np.asarray(image.resize((width, height), Image.Resampling.BILINEAR)) for image in raw]
    first, second = (cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in colors)
    params = Parameters()
    motion = pair_metrics(first, second, params)
    light = brightness_metrics(first, second, motion, params)
    detector = cv2.SIFT_create(nfeatures=FEATURES["sift"]["nfeatures"])
    try:
        sift = sift_pair(detector.detectAndCompute(first, None), detector.detectAndCompute(second, None),
                         first.shape, second.shape, FEATURES["sift"], args["root"])
    except cv2.error as exc:
        sift = {"verified": False, "reason": str(exc), "inlier_pairs": []}
    hue = [histogram(c, FEATURES["histogram"]) for c in colors]
    edge = [cv2.Canny(gray, 48, 124) > 0 for gray in (first, second)]
    local = {"state": "unassessable", "reason": "no supported 2D alignment", "regions": []}
    if motion["registration_valid"]:
        warp = np.float32([[1, 0, motion["dx"]], [0, 1, motion["dy"]]])
        aligned = cv2.warpAffine(first.astype(np.float32), warp, (width, height))
        valid = cv2.warpAffine(np.ones_like(aligned), warp, (width, height)) > .999
        if light["supported"]:
            aligned = np.clip(light["gain"] * aligned + 255 * light["offset"], 0, 255)
        residual = np.abs(aligned - second) / 255
        residual[~valid] = 0
        changes, reason = regions(residual, RULES["local"])
        local = {"state": "supported" if changes else "unsupported", "reason": reason,
                 "regions": changes, "valid_pixels": int(valid.sum()), "total_pixels": width * height,
                 "overlap_mad": float(np.mean(residual[valid])), "region_frame": "second", "outside_overlap": "unassessable"}
    measured = {"sift": sift, "hsv_total_variation": float(np.abs(hue[0] - hue[1]).sum() / 2),
                "edge_xor_fraction_unregistered": float(np.mean(edge[0] != edge[1])), "phase": motion,
                "quality": [quality(first), quality(second)], "exposure_mean": [float(a.mean() / 255) for a in (first, second)],
                "photometry": light, "local": local}
    verdicts = [
        {"claim": "direct_pair_local_appearance_correspondence", "state": "supported" if sift["verified"] else "unassessable", "channel": "SIFT",
         "reason": sift["reason"]},
        {"claim": "2d_image_displacement", "state": "supported" if motion["motion_supported"] else "unsupported" if motion["registration_valid"] else "unassessable", "channel": "phase"},
        {"claim": "locally_different_image_state_on_valid_overlap", "state": local["state"], "channel": "aligned_residual"},
        {"claim": "photometric_alternative", "state": "supported" if light["supported"] else "unassessable" if light["validation_pixels"] == 0 else "unsupported", "channel": "gain_offset"},
    ]
    return {"channels": measured, "judgements": [row | {"evidence_roots": [args["root"]]} for row in verdicts],
            "coordinate_space": {"analysis_width": width, "analysis_height": height, "native_width": native[0], "native_height": native[1],
                                 "mapping": "(analysis_pixel + 0.5) * native_extent / analysis_extent - 0.5"},
            "unsupported_claims": ["physical identity", "world pose", "continuous motion", "contact", "causality"],
            "provenance": "computed metadata from exact original source frames; zero images"}


def semantic(args):
    assets = Path(args["assets"])
    for dependency in args.get("python_paths", []):
        sys.path.append(dependency)
    from p215_research.semantic import Clip, regions as semantic_regions, tensor

    names = [assets / "assets" / name for name in ("text.onnx", "vision.onnx", "tokenizer.json")]
    signature = sha256("|".join(file_hash(p) for p in names).encode()).hexdigest()
    model = Clip(assets)
    vector = model.text([f"a photo of {args['query']}."])[0]
    scores = []
    compared_regions = []
    for row in args["rows"]:
        image = Image.open(row["path"]).convert("RGB")
        boxes = semantic_regions(image, row["observations"], args["coordinates"])
        values = model.vision([tensor(image, b) for b, _ in boxes]) @ vector
        compared_regions.append({"frame_id": row["frame_id"], "pts": row["pts"],
                                  "regions": [{"analysis_xywh": list(box), "region_kind": k, "cosine": float(score)}
                                              for (box, k), score in zip(boxes, values)]})
        i = int(np.argmax(values))
        box, kind = boxes[i]
        scores.append({"frame_id": row["frame_id"], "pts": row["pts"], "time_base": row["time_base"], "seconds": row["t"],
                       "cosine": float(values[i]), "analysis_xywh": list(box), "region_kind": kind,
                       "source_image_sha256": file_hash(row["path"]), "evidence_roots": row["roots"]})
    scores.sort(key=lambda row: (-row["cosine"], row["seconds"]))
    offset, count = args["cursor"], args["page_size"]
    return {"status": "completed", "channel": "CLIP ViT-B/32 regional retrieval", "model_asset_signature": signature,
            "query": args["query"], "items": scores[offset:offset + count], "total_scored": len(scores),
            "all_scores": scores, "compared_regions": compared_regions,
            "next_cursor": offset + count if offset + count < len(scores) else None,
            "scope": "Analysis-resolution image similarities, not identity, absence, actions or evidence citations."}


@contextmanager
def text_workdir():
    with TemporaryDirectory(prefix="observer_text_") as folder:
        yield Path(folder)


def command(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, encoding="utf-8", errors="replace", **kwargs).stdout


def ocr(args):
    exe = args.get("executable") or shutil.which("tesseract")
    if exe:
        cmd = [exe, args["path"], "stdout"]
        if args.get("language"):
            cmd += ["-l", args["language"]]
        result = []
        for line in command(cmd + ["tsv"]).splitlines()[1:]:
            fields = line.split("\t", 11)
            if len(fields) == 12 and fields[11].strip() and float(fields[10]) >= 0:
                result.append({"text": fields[11], "native_xywh": [int(v) for v in fields[6:10]], "engine_confidence": float(fields[10])})
        return {"status": "completed", "channel": "OCR", "engine": "tesseract", "items": result}
    if sys.platform == "win32":
        raw = command(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "RemoteSigned", "-File",
                       str(HERE / "ocr_windows.ps1"), args["path"], args.get("language") or ""])
        return {"status": "completed", "channel": "OCR", "engine": "Windows.Media.Ocr", "items": json.loads(raw)}
    return {"status": "dependency_missing", "channel": "OCR", "reason": "Configure a Tesseract executable and installed language data."}


def get_text_track(args):
    ffmpeg = args["ffmpeg"]
    ffprobe = args["ffprobe"]
    kind = args["channel"]
    selector = "s:0" if kind == "subtitle" else "a:0"
    source_info = json.loads(command([ffprobe, "-v", "error", "-select_streams", selector, "-show_entries",
                                      "stream=index,start_time,time_base,codec_name:format=start_time", "-of", "json", args["source"]]))
    if not source_info.get("streams"):
        return {"status": "unavailable", "channel": kind, "reason": "No corresponding source stream."}
    stream = source_info["streams"][0]
    offset = float((source_info.get("format") or {}).get("start_time", 0))
    with text_workdir() as scratch:
        target = scratch / "text.srt"
        if kind == "subtitle":
            command([ffmpeg, "-v", "error", "-i", args["source"], "-map", "0:s:0", "-c:s", "srt", str(target)])
        else:
            if not args.get("model") or not Path(args["model"]).is_file():
                return {"status": "dependency_missing", "channel": kind, "reason": "Configure a local whisper.cpp model (GGML)."}
            # Relative scratch paths avoid FFmpeg filter escaping of Windows drive letters.
            try:
                (scratch / "model.bin").hardlink_to(Path(args["model"]).resolve())
            except OSError:
                shutil.copyfile(args["model"], scratch / "model.bin")
            command([ffmpeg, "-v", "error", "-i", args["source"], "-map", "0:a:0", "-af",
                     "whisper=model=model.bin:destination=text.srt:format=srt:queue=30:use_gpu=false", "-f", "null", "-"], cwd=scratch)
        srt = target.read_text(encoding="utf-8-sig") if target.exists() else ""
    def second(stamp):
        hours, minutes, tail = stamp.replace(",", ".").split(":")
        return offset + 3600 * int(hours) + 60 * int(minutes) + float(tail)

    blocks = []
    time = r"(\d+:\d+:\d+[,.]\d+)"
    for cue in re.finditer(rf"(?m)^{time} --> {time}[^\n]*\n(.*?)(?=\n\n|\Z)", srt, re.S):
        lo, hi = second(cue.group(1)), second(cue.group(2))
        if lo <= args["end"] and hi >= args["start"]:
            blocks.append({"start_seconds": lo, "end_seconds": hi, "text": cue.group(3).strip()})
    return {"status": "completed", "channel": kind, "stream": stream, "items": blocks, "source_time_offset": offset,
            "model_sha256": file_hash(args["model"]) if kind == "asr" else None,
            "scope": "Timestamped text track excerpts. ASR times are estimated, and neither track is visual evidence."}


def crop(args):
    with Image.open(args["path"]) as source:
        x, y, width, height = args["native_xywh"]
        if x < 0 or y < 0 or width < 1 or height < 1 or x + width > source.width or y + height > source.height:
            raise ValueError("Crop must be inside the native image.")
        output = source.convert("RGB").crop((x, y, x + width, y + height))
    if args["scale"] != 1:
        output = output.resize((max(1, round(width * args["scale"])), max(1, round(height * args["scale"]))), Image.Resampling.LANCZOS)
    output.save(args["out"], format="PNG")
    return {"status": "completed", "width": output.width, "height": output.height}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("pair", "semantic", "ocr", "text", "crop", "research"))
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.operation == "research":
        from research_bridge import research
        fn = research
    else:
        fn = {"pair": inspect_pair, "semantic": semantic, "ocr": ocr, "text": get_text_track, "crop": crop}[args.operation]
    try:
        result = fn(request)
    except (ImportError, FileNotFoundError) as exc:
        result = {"status": "dependency_missing", "reason": str(exc)}
    except subprocess.CalledProcessError as exc:
        result = {"status": "failed", "reason": (exc.stderr or str(exc))[-600:]}
    print(json.dumps(result, allow_nan=False, ensure_ascii=False))


if __name__ == "__main__":
    main()
