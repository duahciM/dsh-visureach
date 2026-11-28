"""P3-11 single-frame observation, closed to labels and tools."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter
from traceback import format_exc

from PIL import Image

from if11_research.agent import CLI
from if15_research.audit import checks_for
from if17_research.catalog import read, ref, sha
from if19_research.inputs import native_rows
from if19_research_max.replay import LongTurnHarness
from p1.io import dump


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RULES = read(HERE / "protocol.json")


def inspect_png(data):
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.n_frames != 1:
            raise ValueError("Expected a single PNG frame")
        image.load()
        return {"sha256": sha256(data).hexdigest(), "mimeType": "image/png",
                "width": image.width, "height": image.height, "bytes": len(data),
                "rgb_sha256": sha256(image.convert("RGB").tobytes()).hexdigest()}


def prompt_audit(folder, sent, prompt, session):
    events = [json.loads(row) for row in (folder / "runtime_events.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [row["params"]["event"] for row in events if row.get("method") == "session.event"
              and row["params"].get("sessionId") == session]
    candidates = [event["data"]["content"] for event in events if event["type"] == "user/message"]
    matching = [items for items in candidates if any(item["type"] == "text" and item["text"] == prompt for item in items)]
    attachments = [[item["attachment"] for item in items if item["type"] == "image"] for items in matching]
    wanted = [{"attachmentId": "sha256:" + meta["sha256"], "mediaType": meta["mimeType"],
               **{key: meta[key] for key in ("width", "height", "bytes")}} for meta in sent]
    replies = [event["data"] for event in events if event["type"] == "assistant/message"]
    content = replies[-1]["message"]["content"] if replies else []
    return {"sdk_user_event_verified": bool(matching) and all(items == wanted for items in attachments),
            "sdk_attachment_records": attachments, "sdk_attachment_matches": len(matching),
            "visible_reply": "\n".join(item["text"] for item in content if item["type"] == "text").strip(),
            "reasoning_blocks": sum(item["type"].startswith("reasoning") for item in content),
            "reasoning_nonempty_blocks": sum(item["type"].startswith("reasoning") and bool(item.get("text")) for item in content),
            "termination": [event["data"] for event in events if event["type"] == "turn/end"],
            "usage": replies[-1].get("usage", {}) if replies else {},
            "provider_context": [event["data"].get("provider") for event in events if event["type"] == "request/context"],
            "model_context": [event["data"].get("model") for event in events if event["type"] == "request/context"],
            "sdk_retries": sum(event["type"] == "llm/retry" for event in events)}


class AuditedHarness(LongTurnHarness):
    def __init__(self, out):
        super().__init__(out, CLI)
        self.sent = []

    def send(self, method, params=None):
        if method == "session/prompt":
            images = [inspect_png(base64.b64decode(item["data"], validate=True))
                      for item in params["contentBlocks"] if item["type"] == "image"]
            self.sent.append(images)
        return super().send(method, params)


def prepare(out):
    root = out / "input"
    root.mkdir(parents=True, exist_ok=False)
    targets = {}
    for case, pts in RULES["target_pts"].items():
        source, rows, prior = native_rows(case)
        selected = [next(row for row in rows if row["pts"] == p) for p in pts]
        checked = checks_for(source, selected, [], [], [], len(selected))
        if not checked["pass"] or sha(source["path"]) != source["sha256"]:
            raise ValueError(f"Independent source audit failed for {case}")
        for row in selected:
            frame_id = f'{case}_{row["pts"]}'
            path = Path(row["native_image"])
            data = path.read_bytes()
            metadata = inspect_png(data)
            if (metadata["width"], metadata["height"], metadata["sha256"]) != (
                    source["probe"]["width"], source["probe"]["height"], row["native_image_sha256"]):
                raise ValueError(f"Native image mismatch for {frame_id}")
            target = root / f"{frame_id}.png"
            with target.open("xb") as stream:
                stream.write(data)
            targets[frame_id] = {"case": case, "source": ref(Path(source["path"])), "historical_session": prior,
                                 "source_png": ref(path), "historical_attachment_sha256": row["prior_attachment_sha256"],
                                 "historical_call": row["prior_call"], "png": ref(target), "image": metadata,
                                 **{field: row[field] for field in ("pts", "time_base", "source_frame", "t")}}
        dump(root / f"{case}_source_audit.json", checked)
    dump(root / "frames.json", targets)
    dump(out / "protocol_lock.json", {"protocol": ref(HERE / "protocol.json"), "rules": RULES,
                                     "code": ref(Path(__file__)), "frames": ref(root / "frames.json")})


def trial(out, stage, trial_id, prompt, images):
    folder = out / stage / trial_id
    folder.mkdir(exist_ok=False, parents=True)
    route = RULES["route"]
    wanted = [inspect_png(path.read_bytes()) for path in images]
    result = {"id": trial_id, "stage": stage, "route": route, "prompt": prompt,
              "prompt_inputs": [ref(path) for path in images], "expected_attachments": wanted}
    dump(folder / "input_lock.json", result)
    started = perf_counter()
    harness = None
    try:
        harness = AuditedHarness(folder)
        result["session_id"] = harness.session
        result["initialize"] = harness.result(harness.send("initialize", route | {"cwd": str(folder)}))
        turn = harness.turn(prompt, images, 1)
        result["transport"] = {"receipt": turn["receipt"], "event_count": turn["event_count"],
                               "legacy_termination": turn["result"]["termination"], "seconds": turn["seconds"]}
        audit = prompt_audit(folder, harness.sent[0], prompt, harness.session)
        result["actual_sent"] = harness.sent[0]
        result["audit"] = audit
        if (result["actual_sent"] != wanted or not audit["sdk_user_event_verified"]
                or turn["result"]["tool_calls"] or not audit["visible_reply"]
                or not audit["reasoning_nonempty_blocks"]
                or any(model != route["model"] for model in audit["model_context"])
                or any(provider != route["provider"] for provider in audit["provider_context"])
                or [r["reason"]["kind"] for r in audit["termination"]] != ["completed"]):
            raise ValueError("Output or attachment verification failed")
        result["status"] = "completed"
    except Exception:
        result.update(status="failed", error=format_exc())
        if harness and harness.sent and "audit" not in result:
            result["actual_sent"] = harness.sent[0]
            result["audit"] = prompt_audit(folder, harness.sent[0], prompt, harness.session)
    finally:
        if harness:
            harness.close()
        result["seconds"] = perf_counter() - started
        dump(folder / "result.json", result)
    print(stage, trial_id, result["status"], round(result["seconds"], 1), flush=True)
    return result


def run_stage(out, stage, jobs, workers):
    done = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for trial_id, prompt, images in jobs:
            prior = out / stage / trial_id / "result.json"
            if prior.exists():
                done[trial_id] = read(prior)
            else:
                futures.append(pool.submit(trial, out, stage, trial_id, prompt, images))
        for future in futures:
            row = future.result()
            done[row["id"]] = row
    manifest = {"results": {key: ref(out / stage / key / "result.json") for key in sorted(done)},
                "status_counts": dict(Counter(row["status"] for row in done.values()))}
    dump(out / stage / "manifest.json", manifest)


def s0(out, workers):
    packet = read(out / "protocol_lock.json")
    if packet["rules"] != RULES or sha(out / "input/frames.json") != packet["frames"]["sha256"]:
        raise ValueError("Locked protocol or inputs changed")
    frames = read(out / "input/frames.json")
    jobs = []
    for key, row in frames.items():
        path = Path(row["png"]["path"])
        if sha(path) != row["png"]["sha256"]:
            raise ValueError(f"Input PNG changed: {key}")
        jobs.extend((f"{key}_r{rep}", RULES["s0_prompt"], [path])
                    for rep in range(1, RULES["s0_repeats"] + 1))
    run_stage(out, "S0", jobs, workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("prepare", "s0"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.out.resolve())
    else:
        s0(args.out.resolve(), args.workers)
