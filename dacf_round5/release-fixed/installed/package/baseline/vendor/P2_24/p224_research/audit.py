"""Verify frozen files, exact source pixels, and separate O and DSH receipts."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

from if17_research.catalog import read, sha
from if20_research.direct import inspect_png
from p216_research.navigate import commit
from p218_research.audit import audit_source
from p219_research.score import checked


def run(out):
    lock = read(out / "protocol_lock.json")
    frozen = lock["files"] | {lock["task_card"]["path"]: lock["task_card"]["sha256"]}
    changed = [p for p, digest in frozen.items() if not Path(p).is_file() or sha(p) != digest]
    old = read(out / "historical_packets.json")
    fresh = read(out / "new_packets.json")
    fresh_lock = read(out / "fresh_input_lock.json")
    for pointer in fresh_lock["originals"].values():
        assert sha(pointer["path"]) == pointer["sha256"]
    packets = {gid: checked(pointer) for gid, pointer in (old["packets"] | fresh["packets"]).items()}
    assert all(set(packet) == {"goal", "proposal", "features", "navigation", "baseline", "new_source"} for packet in packets.values())
    sources = {p["goal"]["case"]: p["proposal"]["source"] for p in packets.values()}
    for source in sources.values():
        assert source["sha256"] != lock["excluded_source_sha256"] and sha(source["path"]) == source["sha256"]
    assert len({p["sha256"] for p in fresh_lock["originals"].values()}) == 2
    assert not {sources[p["goal"]["case"]]["sha256"] for p in packets.values() if not p["new_source"]} & {p["sha256"] for p in fresh_lock["originals"].values()}
    frames = defaultdict(lambda: defaultdict(dict))
    pngs = {}
    counts = Counter()

    def remember(case, row, attachment):
        path = row["path"]
        if path not in pngs:
            pngs[path] = sha(path)
        assert pngs[path] == attachment["sha256"]
        frames[case][row["source_frame"]][path] = row | {"attachment": attachment}

    for route, pointer in read(out / "final_offline_stream_index.json").items():
        receipt = checked(pointer)
        case = receipt["goal"]["case"]
        assert receipt["source_sha256"] == sources[case]["sha256"] and not receipt["technical_abort"]
        page_items = [item for page in receipt["page_receipts"] for item in checked(page)["items"]]
        assert page_items == receipt["items"] and len(page_items) == receipt["N_visible"]
        assert sum(row["repeated_visible_pts"] for row in page_items) == receipt["N_repeated"]
        counts.update(offline_streams=1, offline_items=len(page_items))
        if route.startswith(("O0/", "O1/", "O4/")) and receipt["goal"]["id"] in old["packets"]:
            arm, gid = route.split("/")
            assert packets[gid]["baseline"][arm] == pointer
            counts.update(historical_reference_streams=1)
        for row in page_items:
            assert row["delivered_to_interface"] and not row["sdk_acknowledged_by_model"]
            assert row["source_sha256"] == sources[case]["sha256"] and row["format"] == "full_native_png"
            remember(case, row, row["attachment"])

    model = read(out / "model_use_audit.json")
    sessions = set()
    for pointer in model["formal_sessions"].values():
        result = checked(pointer)
        gid, arm = result["goal"], result["arm"]
        case = packets[gid]["goal"]["case"]
        packet_ref = (old["packets"] | fresh["packets"])[gid]
        assert result["packet"] == packet_ref and result["provider_route_verified"]
        assert result["session_id"] and result["session_id"] not in sessions
        sessions.add(result["session_id"])
        assert result["observe_route"] == model["provider"]
        checked(result["turn_receipts"])
        assert sha(result["model_events"]["path"]) == result["model_events"]["sha256"]
        originals = result["audit"]["received_originals"]
        assert len(originals) == result["audit"]["visible_count"]
        candidates = {row["candidate_id"]: row for row in packets[gid]["proposal"]["batch"]}
        tool_log = result["tool_log"]
        assert sha(tool_log["path"]) == tool_log["sha256"]
        with Path(tool_log["path"]).open(encoding="utf-8") as file:
            calls = {row["call_ref"]: row for row in map(json.loads, file)}
        for call in calls.values():
            if call["tool"] == "wave.query":
                assert arm == "A1"
                if not call["error"]:
                    answer = call["returned_metadata"]
                    assert answer["metadata_only"] and answer["bytes_of_images"] == 0 and answer["delivery_receipts"] == 0
                    counts.update(metadata_only_dsh_queries=1)
        for row in originals:
            native = row["native"]
            sdk_input = row.get("sdk_input", native)
            att = row["sdk_attachment"]
            candidate = candidates[row["frame_id"]]
            assert row["source_sha256"] == sources[case]["sha256"]
            assert row["same_turn_provider_replied"] and row["actual_dsh_session_id"] == result["session_id"]
            assert row.get("original_source_rgb_preserved", True) and sdk_input["rgb_sha256"] == native["rgb_sha256"]
            assert all(att[k] == sdk_input[k] == native[k] for k in ("width", "height"))
            assert att["mediaType"] == "image/png" and att["attachmentId"] == "sha256:" + sdk_input["sha256"]
            assert candidate["image"]["sha256"] == native["sha256"] and candidate["pts"] == row["pts"]
            assert all(candidate[k] == row[k] for k in ("time_base", "source_frame", "seconds"))
            assert calls[row["call_ref"]]["tool"] == "delivery.request" and not calls[row["call_ref"]]["error"]
            counts.update(dsh_visible_originals=1, dsh_lossless_repack=int(sdk_input["sha256"] != native["sha256"]))
            remember(case, row, native)
    counts["dsh_sessions"] = len(sessions)

    e0_lock = read(out / "E0_reference_lock.json")
    checked(e0_lock["strict"])
    checked(e0_lock["broad"])
    for gid, examples in checked(e0_lock["native_examples"]).items():
        case = packets[gid]["goal"]["case"]
        for sample in examples:
            path = sample["native_image"]
            content = inspect_png(Path(path).read_bytes())
            assert content["sha256"] == sample["native_image_sha256"]
            remember(case, sample | {"path": path, "seconds": sample["t"]}, content)

    def decode(case):
        result = audit_source(sources[case], frames[case])
        print(case, result["source_rgb_frames"], result["pass"], flush=True)
        return case, result

    with ThreadPoolExecutor(max_workers=3) as workers:
        pixels = dict(workers.map(decode, sorted(frames)))
    passed = not changed and all(result["pass"] for result in pixels.values())
    commit(out / "source_and_history_audit.json", {
        "pass": passed, "locked_files_including_task_card": len(frozen), "changed_or_missing": changed,
        "receipt_counts": dict(counts), "source_sha256": {case: row["sha256"] for case, row in sources.items()},
        "excluded_source_sha256": lock["excluded_source_sha256"], "fresh_originals": fresh_lock,
        "validated_png_paths": len(pngs), "distinct_source_frame_checks": sum(v["source_rgb_frames"] for v in pixels.values()),
        "native_pts_rgb_checks": pixels,
        "dsh_receipt_scope": "Same-session SDK attachment hash and source RGB, followed by provider response. Provider network request bytes are unavailable.",
    })
    print("Audit", passed, dict(counts), flush=True)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    raise SystemExit(not run(parser.parse_args().out.resolve()))
