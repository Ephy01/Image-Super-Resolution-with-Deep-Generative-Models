"""Markdown tables for the report, generated from the result files so that no number is typed by hand.

    python code/evaluation/report_tables.py --split test     # -> results/final/report_tables_test.md
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import paths

FINAL_DIR = paths.RESULTS_ROOT / "final"
ANALYSIS_DIR = paths.RESULTS_ROOT / "analysis"
LABELS = {"bicubic": "Bicubic", "realesrnet": "Real-ESRNet", "realesrgan": "Real-ESRGAN", "resshift": "ResShift", "hr": "HR reference"}
DIGITS = {"psnr": 2, "ssim": 3, "lpips": 3, "musiq": 1, "cer": 3, "arcface_sim": 3, "arcface_match": 2}
CONTENTS = ["objects", "faces", "text"]
METHODS = ["bicubic", "realesrnet", "realesrgan", "resshift"]


def number(value, metric, sign=False):
    if pd.isna(value):
        return "n/a"
    text = f"{value:+.{DIGITS[metric]}f}" if sign else f"{value:.{DIGITS[metric]}f}"
    return text.replace("+-", "-")


def span(row, metric):
    return f"{number(row[metric], metric, sign=True)} [{number(row[f'{metric}_lo'], metric)}, {number(row[f'{metric}_hi'], metric)}]"


def main_table(summary, quality):
    lines = ["| Content | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | CER ↓ / ArcFace sim ↑ | Match rate ↑ |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for content in CONTENTS:
        block = summary[(summary["content"] == content) & (summary["jpeg"] == quality) & summary["method"].isin(METHODS)].set_index("method")
        task = "cer" if content == "text" else "arcface_sim" if content == "faces" else None
        metrics = ["psnr", "ssim", "lpips", "musiq"] + ([task] if task else []) + (["arcface_match"] if content == "faces" else [])
        best = {m: (block[m].idxmin() if m in ("lpips", "cer") else block[m].idxmax()) for m in metrics}
        for method in METHODS:
            row = block.loc[method]
            cells = []
            for metric in ["psnr", "ssim", "lpips", "musiq", task, "arcface_match"]:
                if metric is None or metric not in metrics:
                    cells.append("n/a")
                    continue
                text = number(row[metric], metric)
                cells.append(f"**{text}**" if best[metric] == method else text)
            lines.append(f"| {content.capitalize()} | {LABELS[method]} | " + " | ".join(cells) + " |")
        hr = summary[(summary["content"] == content) & (summary["method"] == "hr")].iloc[0]
        lines.append(f"| {content.capitalize()} | HR reference | n/a | n/a | n/a | {number(hr['musiq'], 'musiq')} | "
                     f"{number(hr['cer'], 'cer') if content == 'text' else 'n/a'} | n/a |")
    return "\n".join(lines)


def paired_table(paired):
    lines = ["| Content | JPEG | Method (minus Real-ESRNet) | PSNR (dB) | LPIPS | MUSIQ | CER / ArcFace sim |", "|---|---|---|---|---|---|---|"]
    for quality in ["none", "10"]:
        for content in CONTENTS:
            for method in ["realesrgan", "resshift"]:
                row = paired[(paired["comparison"] == f"{method} - realesrnet") & (paired["content"] == content) & (paired["jpeg"] == quality)].iloc[0]
                task = "cer" if content == "text" else "arcface_sim" if content == "faces" else None
                lines.append(f"| {content.capitalize()} | {quality} | {LABELS[method]} | {span(row, 'psnr')} | {span(row, 'lpips')} | "
                             f"{span(row, 'musiq')} | {span(row, task) if task else 'n/a'} |")
    return "\n".join(lines)


def jpeg_table(split):
    data = json.loads((ANALYSIS_DIR / f"jpeg_hypothesis_{split}.json").read_text())
    lines = ["| Method | Content | PSNR: drop 30 to 10 minus drop 50 to 30 (dB) | SSIM | LPIPS | Verdict |", "|---|---|---|---|---|---|"]
    for entry in data["results"]:
        cells = []
        for metric in ["psnr", "ssim", "lpips"]:
            d = entry["metrics"][metric]["difference"]
            cells.append(f"{number(d['mean'], metric, sign=True)} [{number(d['lo'], metric)}, {number(d['hi'], metric)}]")
        lines.append(f"| {LABELS[entry['method']]} | {entry['content']} | " + " | ".join(cells) + f" | {entry['verdict']} |")
    return "\n".join(lines)


def runtime_table():
    rows = json.loads((ANALYSIS_DIR / "runtime.json").read_text())["rows"]
    frame = pd.DataFrame([row for row in rows if "_t4_" in row["file"]])
    lines = ["| Method | Device | Precision | Mean s/image (test) | Median s/image (test) | Peak GPU memory (MiB) | Wall time dev (s) | Wall time test (s) |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        test = frame[(frame["method"] == method) & (frame["split"] == "test")]
        dev = frame[(frame["method"] == method) & (frame["split"] == "dev")]
        if test.empty:
            continue
        t = test.iloc[0]
        memory = "n/a" if pd.isna(t["peak_memory_allocated_mib"]) else f"{t['peak_memory_allocated_mib']:.0f}"
        device = "CPU (Kaggle)" if method == "bicubic" else t["device"]
        lines.append(f"| {LABELS.get(method, method)} | {device} | {t['precision']} | {t['seconds_per_image_mean']:.3f} | {t['seconds_per_image_median']:.3f} | "
                     f"{memory} | {dev.iloc[0]['wall_seconds']:.0f} | {t['wall_seconds']:.0f} |" if not dev.empty else
                     f"| {LABELS.get(method, method)} | {device} | {t['precision']} | {t['seconds_per_image_mean']:.3f} | {t['seconds_per_image_median']:.3f} | "
                     f"{memory} | n/a | {t['wall_seconds']:.0f} |")
    splits = frame.groupby("split")["split_wall_seconds"].unique()
    lines.append("")
    lines.append("Split wall time (all methods of one notebook run, s): " + "; ".join(f"{split} {', '.join(f'{v:.0f}' for v in values)}" for split, values in splits.items()))
    return "\n".join(lines)


def seed_table(split):
    path = ANALYSIS_DIR / f"seed_variability_{split}.json"
    if not path.exists():
        return "not run"
    data = json.loads(path.read_text())
    lines = ["| Content | JPEG | Metric | Mean over seeds: min to max | Spread (max - min) | Mean per-image std |", "|---|---|---|---|---:|---:|"]
    for entry in data["results"]:
        if entry["jpeg"] not in ("none", "10"):
            continue
        for metric in ["psnr", "lpips", "musiq", "cer", "arcface_sim"]:
            if metric not in entry["metrics"]:
                continue
            m = entry["metrics"][metric]
            means = list(m["mean_by_seed"].values())
            lines.append(f"| {entry['content']} | {entry['jpeg']} | {metric} | {number(min(means), metric)} to {number(max(means), metric)} | "
                         f"{m['spread_max_minus_min']:.{DIGITS[metric] + 1}f} | {m['mean_per_image_std']:.{DIGITS[metric] + 1}f} |")
    return "\n".join(lines)


def main(split):
    summary = pd.read_csv(FINAL_DIR / f"summary_{split}.csv", dtype={"jpeg": str})
    paired = pd.read_csv(FINAL_DIR / f"paired_{split}.csv", dtype={"jpeg": str})
    sections = [
        ("Main table, no JPEG", main_table(summary, "none")),
        ("Main table, JPEG 10", main_table(summary, "10")),
        ("Paired differences", paired_table(paired)),
        ("JPEG hypothesis", jpeg_table(split)),
        ("Runtime", runtime_table()),
        ("Seeds", seed_table(split)),
    ]
    text = "\n\n".join(f"### {title}\n\n{body}" for title, body in sections)
    (FINAL_DIR / f"report_tables_{split}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    main(parser.parse_args().split)
