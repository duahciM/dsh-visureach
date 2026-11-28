"""Native DSH A/B runner. The SDK loads the real registered bundle in each arm."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic

from p1.io import dump
from p224_research.agent import NativeHarness, json_answer, parse_events, strip_private

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "comparison_protocol.json").read_text(encoding="utf-8"))
ROUTE = {k: PROTOCOL[k] for k in ("provider", "model", "reasoningEffort", "maxTokens")}
SYSTEM_WIDE_TOOLS = (ROOT / "if11_research/dsh_patch.yml").read_text(encoding="utf-8")
FORMAT = ('Return a JSON object with "facts" as a list of {"claim": string, "kind": "direct" or "inference", '
          '"evidence_ids": [string]}, "unknown" as a list of specific remaining questions, and "summary" as a short narrative. '
          'Cite the original image evidence IDs you actually received. Mark any point you could not establish.')
FOLLOW = ('If any material uncertainty can be resolved with the available tools, revisit it as you judge useful. '
          'Give your final answer with the useful findings you have retained, any corrections, and remaining uncertainty. ' + FORMAT)
REF = re.compile(r"(?m)^\d+\. .*?\bevidence_id=(ev-[a-f0-9]+)")


def native_artifacts(events):
    calls, images, tool_receipts = [], [], []
    replies = {i for i, event in enumerate(events) if event["type"] == "assistant/message"}
    for index, event in enumerate(events):
        d = event["data"]
        if event["type"] == "tool/call":
            calls.append({k: d.get(k) for k in ("name", "callId", "arguments", "step")})
        if event["type"] != "tool/result":
            continue
        message = d.get("message", {})
        content = message.get("content", [])
        value = "\n".join(a.get("text", "") for a in content if a.get("type") == "text")
        refs = REF.findall(value)
        found = [a["attachment"] for a in content if a.get("type") == "image"]
        later = any(n > index for n in replies)
        tool_receipts.append({"call": d.get("callId") or message.get("toolUseId") or message.get("callId"),
                              "event_index": index, "metadata_bytes": len(value.encode()), "text": value,
                              "error": d.get("error", False) or message.get("isError", False), "images": len(found), "followed_by_model_reply": later})
        for i, image in enumerate(found):
            images.append({"ordinal": len(images) + 1, "event_index": index, "attachment": image,
                           "evidence_id": refs[i] if len(refs) == len(found) else None, "followed_by_model_reply": later})
    return {"calls": calls, "images": images, "tool_results": tool_receipts}


def run_case(assignment, attempt):
    case, arm = assignment.split("/")
    folder = HERE / "dsh" / case / arm / f"attempt_{attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        return json.loads((folder / "result.json").read_text(encoding="utf-8"))
    clip = PROTOCOL["goals"][case]
    source = HERE / "inputs" / clip["file"]
    isolated = folder / "workspace"
    isolated.mkdir(exist_ok=True)
    plugin = (HERE / "baseline" if arm == "base" else ROOT / "video_observer_plugin") / "index.js"
    settings = {"cacheDir": str(folder / "cache"), "pythonPath": str(ROOT / ".venv/Scripts/python.exe"),
                "sampleFps": 5, "purposeGate": True, "unobservedAxis": True,
                "semanticAssets": str(ROOT / "p215_discovery_20261121"),
                "semanticPythonPaths": [str(ROOT / "if13_research_20261120/runtime_onnx")],
                "whisperModel": str(HERE / "dependencies/whisper.bin")}
    native_bundle = {"insert": [{"id": "video-integration-under-test", "name": plugin.as_uri(), "config": settings}]}
    patch = folder / "profile-overlay.yml"
    patch.write_text(SYSTEM_WIDE_TOOLS + "\n- " + json.dumps(native_bundle) + "\n", encoding="utf-8")
    prompt = ('Use the native video observation tools to answer this question from the source frames you judge useful. '
              f'Local video: {source.as_posix()}\n{clip["question"]}\n{FORMAT}')
    result = {"case": case, "arm": arm, "attempt": attempt, "route_requested": ROUTE,
              "source_path": str(source), "source_sha256": sha256(source.read_bytes()).hexdigest(),
              "plugin": str(plugin), "prompts": [prompt, FOLLOW], "rounds": []}
    start = monotonic()
    h = None
    try:
        h = NativeHarness(folder, patch)
        result["sdk_session_id"] = h.session
        h.result(h.send("initialize", ROUTE | {"cwd": str(isolated)}), 60)
        for question in (prompt, FOLLOW):
            started = monotonic()
            _, events, stop = h.visual_run(question)
            p = parse_events(events, [])
            round_data = native_artifacts(events) | {"stop": stop, "provider": p["provider_context"],
                "effort": p["declared_reasoning_effort_in_headers"], "usage": p["provider_usage_tokens"],
                "answer": p["final_answer"], "answer_text": p["final_answer_text"],
                "prompt_utf8_bytes": len(question.encode()), "seconds": round(monotonic() - started, 3)}
            result["rounds"].append(round_data)
            dump(folder / f"round_{len(result['rounds'])}_events.json", strip_private(events))
            dump(folder / "checkpoint.json", result)
            if not stop or stop.get("kind") != "completed":
                break
        routes = [r for stage in result["rounds"] for r in stage["provider"]]
        effort = {e for stage in result["rounds"] for e in stage["effort"]}
        result["verified_route"] = bool(routes and effort == {ROUTE["reasoningEffort"]}
            and all(r["provider"] == ROUTE["provider"] and r["model"] == ROUTE["model"] for r in routes))
        result["status"] = "completed" if result["verified_route"] and len(result["rounds"]) == 2 and all(
            t["stop"].get("kind") == "completed" and isinstance(t["answer"], dict) for t in result["rounds"]) else "partial"
    except (TimeoutError, RuntimeError, ValueError) as exc:
        result.update(status="technical_error", failure=str(exc))
    finally:
        if h:
            h.close()
            (folder / "raw_runtime_private.jsonl").unlink(missing_ok=True)
        result["wall_seconds"] = round(monotonic() - start, 3)
        result["native_images"] = sum(len(t["images"]) for t in result["rounds"])
        result["native_tools"] = dict(Counter(c["name"] for t in result["rounds"] for c in t["calls"]))
        dump(folder / "result.json", result)
        print(assignment, attempt, result["status"], result["native_images"], result["wall_seconds"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", action="append", choices=PROTOCOL["counterbalanced_start_order"])
    parser.add_argument("--attempt", type=int, default=1)
    a = parser.parse_args()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(lambda key: run_case(key, a.attempt), a.only or PROTOCOL["counterbalanced_start_order"]))


if __name__ == "__main__":
    main()
