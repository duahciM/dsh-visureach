"""Bounded image observation through the installed DeepSeek Harness SDK, on the account route."""
import argparse
import base64
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Thread
from time import monotonic
from uuid import uuid4

from identity_catalog_matching.common import read, sha
from if10_research.p3_live import contact, query_requests
from p1.io import dump

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "if10_research_20261119"
CLI = Path("D:/dsh-tulpa/node_modules/@deepseek-ai/dsh/lib/bin.js")
FIRST = """You are an independent observer of one silent video. Each tile is an original
video frame at the stamped time and PTS. Use only the images attached here. Do not
use tools. Consider: (1) an early visual reference and its distinguishing cues,
(2) what region becomes exposed later while some earlier cues remain, (3) when a
visually distinct view can first be directly supported by delivered images,
(4) whether the camera physically crossed an identifiable plane, or whether that
is indeterminate. Report observations and hypotheses separately. A planned read
is no evidence. Independently request up to two original-source windows in
[{start}, {end}], each 0.2-1.1 seconds long. Pick windows motivated by gaps in
the actual images, without estimating exact unseen event times. JSON only:
{{"facts":[{{"role":"early|new_region|distinct_view|physical_crossing","claim":"...",
"times":[0.0],"grounding":"direct|inference|unknown"}}],"uncertain":["..."],
"retrieval":[{{"start":0.0,"end":1.0,"why":"..."}}]}}"""
FOLLOW = """Use only the first image and the newly provided original source montages.
Here is the independent reader receipt, with actual frame times: {receipt}.
Revisit the same four questions, give the specific evidence added or withdrawn,
and a final short narration. Distinguish direct observation, inference and
unknown. Do not call tools or request further images. JSON only:
{{"facts":[{{"role":"early|new_region|distinct_view|physical_crossing","claim":"...",
"times":[0.0],"grounding":"direct|inference|unknown"}}],
"uncertain":["..."],"changes":["..."],"final_narration":"..."}}"""


class Harness:
    def __init__(self, out, cli):
        self.inbox = Queue()
        self.console = (out / "runtime.stderr.log").open("w", encoding="utf-8")
        self.wire = (out / "runtime_events.jsonl").open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            ["node", str(cli), "--profile", "sdk", "--patch", str(HERE / "dsh_patch.yml")],
            cwd=out, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.console,
            encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW,
            env=os.environ | {"DSH_HOME": str(Path.home() / ".dsh")},
        )
        Thread(target=self.pump, daemon=True).start()
        self.index = 0
        self.session = str(uuid4())

    def pump(self):
        for line in self.proc.stdout:
            try:
                self.inbox.put(json.loads(line))
            except ValueError:
                continue
        self.inbox.put(None)

    def send(self, method, params=None):
        self.index += 1
        msg = {"jsonrpc": "2.0", "id": self.index, "method": method}
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        return self.index

    def receive(self, deadline):
        try:
            msg = self.inbox.get(timeout=max(0, deadline - monotonic()))
        except Empty:
            raise TimeoutError("DeepSeek Harness SDK response timed out") from None
        if msg is None:
            raise RuntimeError(f"DeepSeek Harness exited before reply ({self.proc.poll()})")
        event = msg.get("params", {})
        if not event.get("sessionId") or event["sessionId"] == self.session:
            self.wire.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.wire.flush()
        return msg

    def result(self, token, seconds=35):
        deadline = monotonic() + seconds
        while True:
            msg = self.receive(deadline)
            if msg.get("id") == token:
                if "error" in msg:
                    raise RuntimeError(str(msg["error"]))
                return msg.get("result")

    def turn(self, text, images, ordinal):
        blocks = [{"type": "text", "text": text}]
        for img in images:
            blocks.append({"type": "image", "mimeType": "image/png",
                           "data": base64.b64encode(img.read_bytes()).decode("ascii")})
        started = monotonic()
        token = self.send("session/prompt", {"sessionId": self.session, "contentBlocks": blocks})
        events = []
        receipt = None
        deadline = started + 150
        end = None
        while True:
            msg = self.receive(deadline)
            if msg.get("id") == token:
                if "error" in msg:
                    raise RuntimeError(str(msg["error"]))
                receipt = msg.get("result")
            p = msg.get("params", {})
            if p.get("sessionId") != self.session:
                continue
            if msg.get("method") == "session.event":
                event = p["event"]
                events.append(event)
                kind = event.get("type") or event.get("event", {}).get("type")
                if kind in ("turn/end", "turn.end"):
                    end = event
            if msg.get("method") == "session.status" and p.get("status") == "idle" and (end or events):
                break
        return {"turn": ordinal, "seconds": monotonic() - started, "receipt": receipt,
                "attachments": [{"path": str(p), "sha256": sha(p)} for p in images],
                "result": unpack(events), "event_count": len(events)}

    def close(self):
        try:
            if self.proc.poll() is None:
                self.result(self.send("shutdown"), 5)
                self.proc.wait(timeout=5)
        except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=10)
        self.wire.close()
        self.console.close()


def walk(x):
    if isinstance(x, dict):
        yield x
        for y in x.values():
            yield from walk(y)
    elif isinstance(x, list):
        for y in x:
            yield from walk(y)


def unpack(events):
    replies = []
    usages = []
    calls = []
    errors = []
    for event in events:
        for node in walk(event):
            kind = node.get("type")
            if kind in ("tool-call", "tool_call", "tool/use", "tool_use"):
                calls.append(node.get("name") or node.get("toolName"))
            if kind in ("turn/end", "turn.end", "llm/error", "agent/error"):
                errors.append(node.get("data", node))
            if kind in ("request/usage", "step/usage") and isinstance(node.get("data"), dict):
                usages.append(node["data"])
            if isinstance(node.get("usage"), dict):
                usages.append(node["usage"])
            if node.get("role") == "assistant":
                content = node.get("content", node.get("contentBlocks", []))
                candidate = "\n".join(x.get("text", "") for x in content if x.get("type") == "text") if isinstance(content, list) else content
                if isinstance(candidate, str):
                    a, b = candidate.find("{"), candidate.rfind("}")
                    if a >= 0 and b > a:
                        try:
                            item = json.loads(candidate[a:b + 1])
                            if isinstance(item, dict) and "facts" in item:
                                replies.append(item)
                        except ValueError:
                            continue
    return {"reply": replies[-1] if replies else None, "usage": usages, "tool_calls": calls, "termination": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cli", type=Path, default=CLI)
    ap.add_argument("--recovery", action="store_true")
    args = ap.parse_args()
    root = args.out.resolve()
    target = root / ("P3_DSH_recovered" if args.recovery else "P3_DSH")
    target.mkdir(exist_ok=False)
    rules = read(HERE / "protocol.json")["p3"]
    frozen = read(OLD / "cases" / rules["seen_case"] / "selection.json")
    that = OLD / "cases" / rules["seen_case"]
    metadata = read(that / "frames.json")
    src = metadata["source"]
    first_im = target / "initial.png"
    initial = frozen["arms"][rules["initial_arm"]]
    contact(initial, first_im, "Initial source observations | development")
    specs = {"cwd": str(target), "provider": rules["provider"], "model": rules["model"],
             "maxTokens": rules["max_output_tokens"]}
    if args.recovery:
        specs["reasoningEffort"] = "off"
    first_prompt = FIRST.format(start=src["start"], end=src["end"])
    rec = {"route": specs, "case": rules["seen_case"], "independent_holdout": False,
           "initial_source_frames": initial, "first_prompt": first_prompt,
           "followup_template": FOLLOW, "rules": rules, "patch_sha256": sha(HERE / "dsh_patch.yml"),
           "code_sha256": sha(Path(__file__)), "initial_sha256": sha(first_im), "status": "started"}
    dump(target / "launch_lock.json", rec)
    live = None
    try:
        live = Harness(target, args.cli)
        rec["session_id"] = live.session
        rec["initialize"] = live.result(live.send("initialize", specs))
        rec["first"] = live.turn(first_prompt, [first_im], 1)
        rec["status"] = "first_turn_returned"
        dump(target / "result.json", rec)
        answer = rec["first"]["result"]["reply"]
        if answer is None or rec["first"]["result"]["tool_calls"]:
            raise RuntimeError("Initial reply missing, unparseable, or used a disallowed tool")
        reads, images = query_requests(answer, target, src, metadata)
        rec["reads"] = reads
        more = FOLLOW.format(receipt=json.dumps(reads, ensure_ascii=True))
        rec["followup"] = live.turn(more, images, 2)
        if rec["followup"]["result"]["reply"] is None or rec["followup"]["result"]["tool_calls"]:
            raise RuntimeError("Follow-up reply missing, unparseable, or used a disallowed tool")
        rec["status"] = "complete_with_source_reads" if images else "complete_without_source_reads"
    except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
        rec["status"] = "technical_failure"
        rec["error"] = str(exc)
    finally:
        if live:
            live.close()
        dump(target / "result.json", rec)
    print(rec["status"], rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main()

