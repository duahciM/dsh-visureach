"""Render evidence thumbnails and post-hoc role acquisition curves."""

import argparse
from collections import defaultdict
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont

from if17_research.catalog import read
from p219_research.score import checked


def plot(out):
    goals = {"street_open_semantics": "Street", "long": "Long G2", "bolt_scene": "Fastener",
             "turn_scene": "Camera turn", "road_fresh": "New road", "notebook_scene": "New laptop"}
    rows = defaultdict(list)
    with (out / "offline_role_acquisition.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            rows[row["trial"]].append((row["ordinal"], row["cumulative_roles"]))
    stats = read(out / "offline_result.json")["scores"]
    comparisons = [("O1", "Frozen O1", "#555e70", "-"), ("O2", "Navigation O2", "#2962a8", "-"),
                   ("O3", "Complement O3", "#cf572a", "-"), ("O4", "Frozen O4", "#644b8d", "-"),
                   ("global/O3", "O3 uniform global", "#408272", "--"),
                   ("public_scope/O3", "O3 uniform public scope", "#ae7c27", ":")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    handles = {}
    for ax, (goal, title) in zip(axes.flat, goals.items()):
        for arm, label, color, style in comparisons:
            pointer = stats.get(arm.split("/")[-1] + "/" + goal)
            if pointer is None:
                continue
            trial = arm + "/" + goal
            points = [(0, 0), *rows[trial]]
            points.append((pointer["summary"]["N_visible"], points[-1][1]))
            x, y = zip(*points)
            line, = ax.step(x, y, where="post", color=color, linestyle=style, linewidth=1.8, label=label)
            handles[label] = line
            ax.plot(x[-1], y[-1], ".", color=color, markersize=5)
        ax.set(title=title, ylim=(-0.1, stats["O3/" + goal]["summary"]["required_roles"] + 0.2), xlabel="Images actually received")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Strict positive roles found")
    axes[1, 0].set_ylabel("Strict positive roles found")
    fig.suptitle("IF-30 | Offline acquisition by actual image ordinal", fontsize=15)
    axes[0, 0].legend(handles.values(), handles.keys(), loc="upper left", fontsize=7, framealpha=0.93)
    fig.savefig(out / "offline_role_acquisition.png", dpi=150)
    plt.close(fig)


def sheet(out):
    sessions = read(out / "model_use_audit.json")["formal_sessions"]
    offline = read(out / "offline_result.json")["scores"]

    def agent(trial, pts, label):
        receipt = checked(sessions[trial])["audit"]["received_originals"]
        row = next(row for row in receipt if row["pts"] == pts)
        return row["path"], label, row["seconds"], pts

    def historical(trial, role, label):
        row = offline[trial]["scores"]["algorithm"]["hits"][role]
        return row["path"], label, row["seconds"], row["pts"]

    frames = [historical("O4/street_open_semantics", "central_activity", "Street O4: exact positive frame"),
              agent("street_open_semantics/A1", 72, "Street A1: nearby sample, strict miss"),
              agent("bolt_scene/A0", 802802, "Fastener A0: late surface reference"),
              agent("turn_scene/A0", 11855, "Turn A0: late door reference"),
              agent("ppt_boundary/A0", 15900, "Slide A0: blank lower-right field"),
              agent("ppt_boundary/A1", 16000, "Slide A1: first visible mark"),
              agent("notebook_scene/A1", 0, "Laptop A1: lid upright"),
              agent("notebook_scene/A1", 76800, "Laptop A1: near-horizontal lid"),
              historical("O2/road_fresh", "later_road_occupancy", "Road O2: strict vehicle sample"),
              agent("road_fresh/A1", 400000, "Road A1: vehicles outside strict window"),
              historical("O1/long", "late_elevated", "Long G2 O1: exact late elevated frame"),
              agent("long/A0", 1388032, "Long G2 A0: similar outside window")]
    size = 420, 288
    img = Image.new("RGB", (size[0] * 3, size[1] * 4), "#f4f6fa")
    draw = ImageDraw.Draw(img)
    label_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size=18)
    detail_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size=15)
    for index, (name, label, seconds, pts) in enumerate(frames):
        left, top = index % 3 * size[0], index // 3 * size[1]
        with Image.open(name) as original:
            tile = original.convert("RGB")
            tile.thumbnail((396, 225), Image.Resampling.LANCZOS)
            img.paste(tile, (left + (size[0] - tile.width) // 2, top + (235 - tile.height) // 2))
        draw.text((left + 12, top + 237), label, font=label_font, fill="#182138")
        draw.text((left + 12, top + 261), f"Source time {seconds:.6f} s  |  PTS {pts}", font=detail_font, fill="#4a5975")
    img.save(out / "evidence_contact_sheet.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    path = parser.parse_args().out.resolve()
    plot(path)
    sheet(path)
