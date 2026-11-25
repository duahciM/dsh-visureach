"""Private simulator: deliver actual video frames into a committed-only SQLite inbox."""
import argparse
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import time
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS frames (pts INTEGER PRIMARY KEY, source_frame INTEGER UNIQUE, seconds REAL,
   width INTEGER, height INTEGER, sha TEXT, png BLOB, information REAL, delta_mad REAL, exposure REAL);
CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY, event TEXT, committed_pts INTEGER, document TEXT);
"""
PNG = b"\x89PNG\r\n\x1a\n"
WITHDRAWN = "7f74580c0426420011267f3ad0c68bfa8a53c91b2ae74fa168912a72d4a201cf"


def digest(path):
    with path.open("rb") as handle:
        return __import__("hashlib").file_digest(handle, "sha256").hexdigest()


def png_frame(stream):
    first = stream.read(8)
    if not first:
        return None
    if first != PNG:
        raise OSError("Expected an exact PNG frame from decoder")
    blocks = [first]
    while True:
        header = stream.read(8)
        if len(header) != 8:
            raise OSError("Truncated PNG header")
        count, kind = struct.unpack(">I4s", header)
        data = stream.read(count+4)
        if len(data) != count+4:
            raise OSError("Truncated PNG payload")
        blocks.extend((header, data))
        if kind == b"IEND":
            return b"".join(blocks)


def image_stats(bits, previous):
    with Image.open(BytesIO(bits)) as image:
        dims = image.size
        preview = np.asarray(image.convert("L").resize((160, max(8, round(160*dims[1]/dims[0])))))
    information = float(np.mean(cv2.Canny(preview, 48, 124) > 0))
    delta = float(np.mean(np.abs(preview.astype(np.int16)-previous))) if previous is not None else None
    return dims, preview, information, delta, float(preview.mean())


def run(args):
    source = args.video.resolve()
    source_sha = digest(source)
    if source_sha == WITHDRAWN:
        raise ValueError("Source withdrawn by the operator")
    info = json.loads(subprocess.check_output([args.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
                                              "frame=pts:stream=time_base", "-of", "json", str(source)]))
    pts = [int(r["pts"]) for r in info["frames"]]
    if not pts or any(b <= a for a,b in zip(pts,pts[1:])):
        raise ValueError("Missing or non-monotonic source PTS")
    time_base = info["streams"][0]["time_base"]
    tick = Fraction(time_base)
    args.db.resolve().parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db, timeout=4)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(SCHEMA)

    try:
        meta = dict(db.execute("SELECT key,value FROM metadata"))
        saved = int(meta.get("last_source_frame", -1))
        if meta and (meta["source_sha256"] != source_sha or meta["time_base"] != time_base):
            raise ValueError("A reopened stream must keep its source SHA and time base")
        if meta.get("closed") == "true":
            print(json.dumps({"status": "already_closed", "committed_source_frame": saved}))
            return
        stream_id = meta.get("stream_id", uuid4().hex)
        seq = db.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM events").fetchone()[0]

        def put(key, value):
            db.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", (key,str(value)))

        def emit(kind, committed, **fields):
            nonlocal seq
            doc = {"protocol": "video-observer-stream/v1", "stream_id": stream_id, "epoch": 0,
                   "sequence": seq, "event": kind, "committed_through_pts": committed, "time_base": time_base} | fields
            db.execute("INSERT INTO events VALUES(?,?,?,?)", (seq,kind,committed,json.dumps(doc)))
            seq += 1

        with db:
            if not meta:
                for key, value in {"stream_id": stream_id, "source_sha256": source_sha, "time_base": time_base, "closed": "false"}.items():
                    put(key, value)
                emit("opened", None)
            else:
                emit("discontinuity", pts[saved] if saved >= 0 else None, reason="producer resumed from durable same-source cursor")

        tail = db.execute("SELECT source_frame,png FROM frames ORDER BY source_frame DESC LIMIT 1").fetchone()
        previous = image_stats(tail[1], None)[1] if tail is not None and tail[0] == saved else None
        start_wall = time.monotonic()
        first = saved+1
        last = min(len(pts)-1, first+args.frames-1) if args.frames else len(pts)-1
        drops = set(args.drop_frame)
        command = [args.ffmpeg, "-v", "error", "-xerror", "-noautorotate", "-i", str(source), "-map", "0:v:0", "-vf",
                   "format=rgb24", "-fps_mode", "passthrough", "-frames:v", str(last+1), "-c:v", "png", "-f", "image2pipe", "-"]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as decoder:
            try:
                for index in range(last+1):
                    bits = png_frame(decoder.stdout)
                    if bits is None:
                        raise OSError("Source decoder ended before the indexed frame")
                    if index < first:
                        continue
                    if args.speed:
                        target = float((pts[index]-pts[first])*tick)/args.speed
                        time.sleep(max(0, start_wall+target-time.monotonic()))
                    with db:
                        # The cursor update and image or explicit gap commit together.
                        actual = db.execute("SELECT value FROM metadata WHERE key='last_source_frame'").fetchone()
                        if int(actual[0] if actual else -1) != index - 1:
                            raise RuntimeError("Another producer advanced this stream")
                        stamp = pts[index]
                        if index in drops:
                            emit("gap", stamp, range_pts=[stamp, stamp], reason="explicit simulator input loss")
                            previous = None
                        else:
                            (w,h), preview, information, delta, exposure = image_stats(bits, previous)
                            checksum = sha256(bits).hexdigest()
                            db.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?,?)",
                                       (stamp,index,float(stamp*tick),w,h,checksum,bits,information,delta,exposure))
                            emit("frame_committed", stamp, pts=stamp, original_sha256=checksum)
                            previous = preview
                        put("last_source_frame", index)
                    if args.crash_after and index-first+1 == args.crash_after:
                        os._exit(79)
                if decoder.wait():
                    raise OSError("Source decoder failed")
            finally:
                if decoder.poll() is None:
                    decoder.kill()
        with db:
            if last == len(pts)-1:
                put("closed", "true")
                emit("closed", pts[last])
            else:
                emit("draining", pts[last])
        print(json.dumps({"stream_id": stream_id, "committed_source_frame": last, "closed": last == len(pts)-1}))
    finally:
        db.close()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--video", type=Path, required=True)
    cli.add_argument("--db", type=Path, required=True)
    cli.add_argument("--ffmpeg", default="ffmpeg")
    cli.add_argument("--ffprobe", default="ffprobe")
    cli.add_argument("--frames", type=int, default=0, help="Producer arrivals in this invocation, 0 to end")
    cli.add_argument("--speed", type=float, default=1, help="Simulated source playback multiplier, 0 for fast test input")
    cli.add_argument("--drop-frame", type=int, action="append", default=[])
    cli.add_argument("--crash-after", type=int, default=0, help=argparse.SUPPRESS)
    args = cli.parse_args()
    if min(args.frames, args.speed, args.crash_after) < 0:
        cli.error("Frames, speed and simulated failure point must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
