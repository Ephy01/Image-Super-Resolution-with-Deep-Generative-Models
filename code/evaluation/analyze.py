"""Analyses and figures for one split, from the tables of evaluate.py. Every rule is fixed in the analysis: section of the config.

    python code/evaluation/analyze.py --split test

Writes one JSON per analysis to results/analysis/ (<name>_<split>.json) and the figures to results/final/figures/.
Methods without outputs are skipped and listed as missing.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "datasets"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "models"))
import paths
from evaluate import FINAL_DIR, METRICS, bootstrap_interval, to_luma
from prepare_data import HR_DIR, LR_DIR, PROCESSED_DIR
from upscale import METHODS, OUTPUTS_DIR

CONFIG = paths.load_config()
RULES = CONFIG["analysis"]
ANALYSIS_DIR = paths.RESULTS_ROOT / "analysis"
FIGURES_DIR = FINAL_DIR / "figures"
GENERATIVE = RULES["generative"]
REFERENCE = RULES["reference"]
CONTENTS = ["objects", "faces", "text"]
QUALITIES = ["none", "50", "30", "10"]
LOWER_IS_BETTER = {"lpips", "cer"}
LABELS = {"bicubic": "Bicubic", "realesrnet": "Real-ESRNet", "realesrgan": "Real-ESRGAN", "resshift": "ResShift", "hr": "HR"}
METRIC_LABELS = {"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑", "lpips": "LPIPS ↓", "musiq": "MUSIQ ↑", "cer": "CER ↓",
                 "arcface_sim": "ArcFace similarity ↑", "arcface_match": "ArcFace match rate ↑"}


def write(name, split, data):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / f"{name}_{split}.json"
    path.write_text(json.dumps(data, indent=2, default=lambda value: bool(value) if isinstance(value, np.bool_) else float(value)))
    return path


def interval(values):
    values = pd.Series(values, dtype=float)
    lo, hi = bootstrap_interval(values)
    return {"mean": float(values.mean()), "lo": lo, "hi": hi, "n": int(values.notna().sum())}


def goodness(metric, values):
    return -values if metric in LOWER_IS_BETTER else values


# Main table ----------------------------------------------------------------------------------------------------------

def main_table(summary, split):
    rows = []
    for content in CONTENTS:
        metrics = ["psnr", "ssim", "lpips", "musiq"] + (["cer"] if content == "text" else []) + (["arcface_sim", "arcface_match"] if content == "faces" else [])
        for quality in ["none", "10"]:
            block = summary[(summary["content"] == content) & (summary["jpeg"] == quality) & summary["method"].isin(METHODS)]
            best = {}
            for metric in metrics:
                values = block.set_index("method")[metric]
                best[metric] = values.idxmin() if metric in LOWER_IS_BETTER else values.idxmax()
            for record in block.itertuples():
                row = {"content": content, "jpeg": quality, "method": record.method}
                for metric in metrics:
                    row[metric] = {"mean": getattr(record, metric), "lo": getattr(record, f"{metric}_lo"), "hi": getattr(record, f"{metric}_hi"),
                                   "best": best[metric] == record.method}
                if content == "faces":
                    row["arcface_detection_failures"] = int(record.arcface_detection_failures)
                rows.append(row)
        reference = summary[(summary["content"] == content) & (summary["method"] == "hr")].iloc[0]
        hr = {"content": content, "jpeg": "none", "method": "hr", "musiq": {"mean": reference["musiq"], "lo": reference["musiq_lo"], "hi": reference["musiq_hi"]}}
        if content == "text":
            hr["cer"] = {"mean": reference["cer"], "lo": reference["cer_lo"], "hi": reference["cer_hi"]}
        if content == "faces":
            hr["arcface_detection_failures"] = int(reference["arcface_detection_failures"])
        rows.append(hr)
    return {"split": split, "rows": rows, "note": "best = best mean among the 4 methods per content, JPEG level and column"}


# JPEG hypothesis -----------------------------------------------------------------------------------------------------

def jpeg_hypothesis(metrics, methods, split):
    rule = RULES["jpeg_hypothesis"]
    table = metrics[metrics["method"].isin(methods)].pivot_table(index=["method", "content", "image_id"], columns="jpeg", values=METRICS)
    results = []
    for method in methods:
        for content in CONTENTS:
            block = table.loc[(method, content)]
            entry = {"method": method, "content": content, "metrics": {}}
            supported = 0
            for metric in rule["metrics"]:
                good = {quality: goodness(metric, block[(metric, quality)]) for quality in ("50", "30", "10")}
                drop_high = good["50"] - good["30"]
                drop_low = good["30"] - good["10"]
                difference = interval(drop_low - drop_high)
                entry["metrics"][metric] = {"drop_50_to_30": interval(drop_high), "drop_30_to_10": interval(drop_low),
                                            "difference": difference, "accelerates": difference["lo"] > 0}
                supported += difference["lo"] > 0
            count = len(rule["metrics"])
            entry["verdict"] = "supported" if supported == count else "not supported" if supported == 0 else "mixed"
            results.append(entry)
    return {"split": split, "rule": rule, "results": results}


# Hypothesis checks ---------------------------------------------------------------------------------------------------

def paired_entry(paired, comparison, content, quality, metric, key="content"):
    row = paired[(paired["comparison"] == comparison) & (paired[key] == content) & (paired["jpeg"] == quality)]
    if row.empty:
        return None
    row = row.iloc[0]
    return {"mean": row[metric], "lo": row[f"{metric}_lo"], "hi": row[f"{metric}_hi"], "n": int(row[f"{metric}_n"])}


def hypothesis_checks(paired, paired_by_size, methods, jpeg_result, split):
    checks = {"sharper_but_lower_psnr": [], "invented_letters": [], "invented_letters_by_font_size": [], "identity_change": []}
    for method in [m for m in GENERATIVE if m in methods]:
        comparison = f"{method} - {REFERENCE}"
        for quality in ["none", "10"]:
            for content in CONTENTS:
                lpips = paired_entry(paired, comparison, content, quality, "lpips")
                musiq = paired_entry(paired, comparison, content, quality, "musiq")
                psnr = paired_entry(paired, comparison, content, quality, "psnr")
                checks["sharper_but_lower_psnr"].append({
                    "method": method, "content": content, "jpeg": quality, "lpips": lpips, "musiq": musiq, "psnr": psnr,
                    "holds": lpips["hi"] < 0 and musiq["lo"] > 0 and psnr["hi"] < 0})
            cer = paired_entry(paired, comparison, "text", quality, "cer")
            lpips = paired_entry(paired, comparison, "text", quality, "lpips")
            checks["invented_letters"].append({"method": method, "jpeg": quality, "cer": cer, "lpips": lpips,
                                               "holds": cer["lo"] > 0 and lpips["hi"] < 0})
            for size in sorted(paired_by_size["font_size"].unique()):
                cer = paired_entry(paired_by_size, comparison, size, quality, "cer", key="font_size")
                lpips = paired_entry(paired_by_size, comparison, size, quality, "lpips", key="font_size")
                checks["invented_letters_by_font_size"].append({"method": method, "jpeg": quality, "font_size": int(size), "cer": cer,
                                                                "lpips": lpips, "holds": cer["lo"] > 0 and lpips["hi"] < 0})
            similarity = paired_entry(paired, comparison, "faces", quality, "arcface_sim")
            match = paired_entry(paired, comparison, "faces", quality, "arcface_match")
            checks["identity_change"].append({"method": method, "jpeg": quality, "arcface_sim": similarity, "arcface_match": match,
                                              "holds": similarity["hi"] < 0})
    checks["jpeg_below_30"] = [{key: entry[key] for key in ("method", "content", "verdict")} for entry in jpeg_result["results"]]
    return {"split": split, "reference": REFERENCE, "rules": RULES["hypothesis_checks"], "checks": checks}


# Seeds ---------------------------------------------------------------------------------------------------------------

def seed_variability(metrics, split):
    path = FINAL_DIR / f"metrics_{split}_seeds.csv"
    if not path.exists() or "resshift" not in set(metrics["method"]):
        return None
    seeds = pd.read_csv(path, dtype={"jpeg": str})
    runs = pd.concat([metrics[metrics["method"] == "resshift"], seeds[seeds["method"].str.startswith("resshift_seed")]])
    offsets = sorted(runs["method"].unique())
    results = []
    for (content, quality), group in runs.groupby(["content", "jpeg"], sort=False):
        entry = {"content": content, "jpeg": quality, "seeds": len(offsets), "metrics": {}}
        for metric in METRICS:
            means = group.groupby("method")[metric].mean()
            if means.isna().all():
                continue
            per_image = group.groupby("image_id")[metric].std(ddof=1)
            entry["metrics"][metric] = {"mean_by_seed": {method: float(value) for method, value in means.items()},
                                        "spread_max_minus_min": float(means.max() - means.min()), "std_of_means": float(means.std(ddof=1)),
                                        "mean_per_image_std": float(per_image.mean())}
        results.append(entry)
    return {"split": split, "runs": offsets, "seed_policy": CONFIG["resshift"]["seed_policy"], "seed_offsets": CONFIG["resshift"]["seed_offsets"],
            "results": results}


# Runtime -------------------------------------------------------------------------------------------------------------

def runtime_table():
    rows = []
    for path in sorted((paths.RESULTS_ROOT / "runtime").glob("upscale_report_*.json")):
        report = json.loads(path.read_text())
        for folder, entry in report["methods"].items():
            if entry.get("cached"):
                continue
            rows.append({"file": path.name, "split": report["split"], "method": folder, "device": entry["device"],
                         "torch": report["torch"], "cuda": report.get("cuda"), "python": report.get("python"),
                         "precision": report["precision"]["bicubic" if folder == "bicubic" else "resshift" if folder.startswith("resshift") else "rrdbnet"],
                         **{key: entry.get(key) for key in ("images", "seconds_per_image_mean", "seconds_per_image_median", "wall_seconds",
                                                            "peak_memory_allocated_mib", "peak_memory_reserved_mib", "warmup_runs")},
                         "split_wall_seconds": report.get("split_wall_seconds")})
    return {"rows": rows}


# Failure cases -------------------------------------------------------------------------------------------------------

def error_crop(method_path, hr_path, size=128, window=31):
    error = np.abs(to_luma(np.asarray(Image.open(method_path).convert("RGB"))) - to_luma(np.asarray(Image.open(hr_path).convert("RGB"))))
    smooth = cv2.blur(error, (window, window))
    y, x = np.unravel_index(np.argmax(smooth), smooth.shape)
    left = int(np.clip(x - size // 2, 0, error.shape[1] - size))
    top = int(np.clip(y - size // 2, 0, error.shape[0] - size))
    return left, top, left + size, top + size


def failure_figure(case, split, methods, manifest):
    image_id, quality, method = case["image_id"], case["jpeg"], case["method"]
    hr_path = HR_DIR / f"{image_id}.png"
    box = error_crop(OUTPUTS_DIR / split / method / f"jpeg_{quality}" / f"{image_id}.png", hr_path)
    panels = [("Input", Image.open(LR_DIR / f"jpeg_{quality}" / f"{image_id}.png").resize((512, 512), Image.NEAREST))]
    panels += [(LABELS[m], Image.open(OUTPUTS_DIR / split / m / f"jpeg_{quality}" / f"{image_id}.png")) for m in methods]
    panels.append(("HR", Image.open(hr_path)))
    figure, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 2.9))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image.crop(box))
        numbers = case["values"].get(title_to_method(title), {})
        caption = "\n".join(f"{METRIC_LABELS[k].split()[0]} {v:.3g}" for k, v in numbers.items() if k in case["shown"] and pd.notna(v))
        axis.set_title(f"{title}\n{caption}", fontsize=8)
        axis.axis("off")
    figure.suptitle(f"Case {case['rule']}: {image_id}, JPEG {quality}, crop {box}", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    path = FIGURES_DIR / f"failure_{case['rule']}_{split}.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    case["crop"] = list(box)
    return str(path.relative_to(PROJECT_ROOT))


def title_to_method(title):
    return {label: method for method, label in LABELS.items()}.get(title)


def case_values(metrics, image_id, quality, methods):
    rows = metrics[(metrics["image_id"] == image_id) & (metrics["jpeg"] == quality) & metrics["method"].isin(methods)]
    values = {row.method: {metric: getattr(row, metric) for metric in METRICS} for row in rows.itertuples()}
    hr = metrics[(metrics["image_id"] == image_id) & (metrics["method"] == "hr")]
    if not hr.empty:
        values["hr"] = {metric: hr.iloc[0][metric] for metric in ("musiq", "cer")}
    return values


def failure_cases(metrics, methods, split, manifest):
    generative = [m for m in GENERATIVE if m in methods]
    complete = len(methods) == len(METHODS)
    cases, notes = [], {}
    sr = metrics[metrics["method"].isin(methods)]

    text = sr[sr["content"] == "text"].pivot_table(index=["image_id", "jpeg"], columns="method", values=["lpips", "cer"])
    candidates = []
    for (image_id, quality), row in text.iterrows():
        best_lpips, worst_cer = row["lpips"].idxmin(), row["cer"].idxmax()
        if best_lpips == worst_cer and best_lpips in generative:
            candidates.append((row["cer"][worst_cer] - row["cer"].min(), image_id, quality, worst_cer))
    if candidates:
        gap, image_id, quality, method = max(candidates)
        cases.append({"rule": "a", "image_id": image_id, "jpeg": quality, "method": method, "selection_value": gap,
                      "selection": "CER gap to the best-CER method", "eligible": len(candidates), "text_inputs": len(text),
                      "eligible_by_method": {m: sum(c[3] == m for c in candidates) for m in generative}, "shown": ["lpips", "cer"]})
    else:
        notes["a"] = "no case: no text image meets rule a"

    faces = sr[(sr["content"] == "faces") & (sr["jpeg"] == "none") & sr["method"].isin(generative)]
    failures = faces[~faces["arcface_detected"].astype(bool)][["image_id", "method"]].to_dict("records")
    worst = faces.dropna(subset=["arcface_sim"]).nsmallest(1, "arcface_sim").iloc[0]
    cases.append({"rule": "b", "image_id": worst.image_id, "jpeg": "none", "method": worst.method, "selection_value": worst.arcface_sim,
                  "selection": "lowest ArcFace similarity", "detection_failures_jpeg_none": failures, "shown": ["arcface_sim", "psnr", "lpips"]})

    jpeg10 = sr[sr["jpeg"] == "10"].pivot_table(index="image_id", columns="method", values="psnr")
    deficits = pd.concat({m: jpeg10["bicubic"] - jpeg10[m] for m in generative}, axis=1).stack()
    (image_id, method), deficit = deficits.idxmax(), deficits.max()
    cases.append({"rule": "c", "image_id": image_id, "jpeg": "10", "method": method, "selection_value": float(deficit),
                  "selection": "PSNR deficit below bicubic (dB)", "shown": ["psnr", "ssim", "lpips", "cer"]})

    if "resshift" in methods:
        small = manifest[(manifest["content"] == "text") & (manifest["font_size"] == 16)]["image_id"]
        block = sr[(sr["image_id"].isin(small)) & (sr["jpeg"] == "none")].pivot_table(index="image_id", columns="method", values="cer")
        block = block.assign(gap=block["resshift"] - block[REFERENCE]).sort_values(["resshift", "gap"], ascending=False)
        image_id = block.index[0]
        cases.append({"rule": "d", "image_id": image_id, "jpeg": "none", "method": "resshift", "selection_value": float(block.iloc[0]["resshift"]),
                      "selection": "highest ResShift CER at HR font size 16", "shown": ["cer", "psnr", "lpips"]})
    else:
        notes["d"] = "not run: ResShift outputs are missing for this split"

    from extra_metrics import Ocr
    ocr = Ocr(CONFIG["metrics"]["ocr"])
    for case in cases:
        case["values"] = case_values(metrics, case["image_id"], case["jpeg"], methods)
        case["values_by_jpeg"] = {quality: case_values(metrics, case["image_id"], quality, methods) for quality in QUALITIES}
        if case["image_id"].startswith("text"):
            info = manifest.set_index("image_id").loc[case["image_id"]]
            case["font"], case["font_size"], case["kind"] = info["font"], int(info["font_size"]), info["kind"]
        if case["image_id"].startswith("text"):
            reference = manifest.set_index("image_id").loc[case["image_id"], "text"]
            case["reference_text_first_line"] = reference.split("\n")[0]
            case["ocr_first_line"] = {m: ocr.read(OUTPUTS_DIR / split / m / f"jpeg_{case['jpeg']}" / f"{case['image_id']}.png").split("\n")[0]
                                      for m in methods}
        case["figure"] = failure_figure(case, split, methods, manifest)
    return {"split": split, "rules": RULES["failure_cases"], "methods": methods, "all_methods_present": complete, "cases": cases, "notes": notes}


# Figures -------------------------------------------------------------------------------------------------------------

def plot_jpeg_sweep(summary, methods, split):
    positions = np.arange(len(QUALITIES))
    figure, axes = plt.subplots(3, 5, figsize=(17, 9.5), sharex=True)
    for row, content in enumerate(CONTENTS):
        metrics = ["psnr", "ssim", "lpips", "musiq"] + (["cer"] if content == "text" else ["arcface_sim"] if content == "faces" else [])
        for column in range(5):
            axis = axes[row, column]
            if column >= len(metrics):
                axis.axis("off")
                continue
            metric = metrics[column]
            for method in methods:
                block = summary[(summary["content"] == content) & (summary["method"] == method)].set_index("jpeg").loc[QUALITIES]
                errors = np.vstack([block[metric] - block[f"{metric}_lo"], block[f"{metric}_hi"] - block[metric]])
                axis.errorbar(positions, block[metric], yerr=errors, marker="o", capsize=3, label=LABELS[method])
            hr = summary[(summary["content"] == content) & (summary["method"] == "hr")]
            if metric in ("musiq", "cer") and not hr.empty:
                axis.axhline(hr.iloc[0][metric], color="grey", linestyle="--", linewidth=1, label="HR")
            axis.set_xticks(positions, QUALITIES)
            axis.set_title(f"{content}: {METRIC_LABELS[metric]}", fontsize=10)
            axis.grid(alpha=0.3)
            if row == len(CONTENTS) - 1 or column == 4:
                axis.set_xlabel("JPEG quality of the input")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"{split} split: mean and 95% percentile bootstrap interval [lo, hi]")
    figure.tight_layout()
    path = FIGURES_DIR / f"jpeg_sweep_{split}.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def fidelity_perception(summary, methods, split):
    block = summary[summary["method"].isin(methods)]
    markers = {"objects": "o", "faces": "s", "text": "^"}
    colors = dict(zip(METHODS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    points = []
    for axis, metric in zip(axes, ["lpips", "musiq"]):
        for record in block.itertuples():
            x, y = record.psnr, getattr(record, metric)
            xerr = [[x - record.psnr_lo], [record.psnr_hi - x]]
            yerr = [[y - getattr(record, f"{metric}_lo")], [getattr(record, f"{metric}_hi") - y]]
            alpha = {"none": 1.0, "50": 0.75, "30": 0.5, "10": 0.3}[record.jpeg]
            axis.errorbar(x, y, xerr=xerr, yerr=yerr, marker=markers[record.content], color=colors[record.method], alpha=alpha,
                          markersize=6, capsize=2, linewidth=0.8)
            if metric == "lpips":
                points.append({"method": record.method, "content": record.content, "jpeg": record.jpeg,
                               "psnr": [record.psnr, record.psnr_lo, record.psnr_hi], "lpips": [record.lpips, record.lpips_lo, record.lpips_hi],
                               "musiq": [record.musiq, record.musiq_lo, record.musiq_hi]})
        axis.set_xlabel("PSNR (dB) ↑")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=colors[m], marker="o", linestyle="", label=LABELS[m]) for m in methods]
    handles += [plt.Line2D([], [], color="black", marker=markers[c], linestyle="", label=c) for c in CONTENTS]
    axes[1].legend(handles=handles, fontsize=8, loc="lower left")
    figure.suptitle(f"Fidelity vs perceptual quality, {split} split: one point per method x content x JPEG level (fainter = stronger JPEG); 95% intervals")
    figure.tight_layout()
    path = FIGURES_DIR / f"fidelity_perception_{split}.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path, points


# Dev vs test ---------------------------------------------------------------------------------------------------------

def dev_test_consistency():
    files = {split: FINAL_DIR / f"summary_{split}.csv" for split in ("dev", "test")}
    if not all(path.exists() for path in files.values()):
        return None
    dev, test = (pd.read_csv(files[split], dtype={"jpeg": str}) for split in ("dev", "test"))
    rows = []
    for content in CONTENTS:
        for quality in QUALITIES:
            for metric in METRICS:
                ranks = {}
                for split, table in (("dev", dev), ("test", test)):
                    block = table[(table["content"] == content) & (table["jpeg"] == quality) & table["method"].isin(METHODS)].set_index("method")[metric]
                    if block.isna().all():
                        break
                    ranks[split] = list((block.sort_values(ascending=metric in LOWER_IS_BETTER)).index)
                if len(ranks) < 2:
                    continue
                rows.append({"content": content, "jpeg": quality, "metric": metric, "dev_order": ranks["dev"], "test_order": ranks["test"],
                             "same_best": ranks["dev"][0] == ranks["test"][0], "same_order": ranks["dev"] == ranks["test"]})
    frame = pd.DataFrame(rows)
    return {"rows": rows, "same_best": int(frame["same_best"].sum()), "same_order": int(frame["same_order"].sum()), "cells": len(frame)}


def main(split="test"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    manifest = manifest[manifest["split"] == split]
    metrics = pd.read_csv(FINAL_DIR / f"metrics_{split}.csv", dtype={"jpeg": str})
    summary = pd.read_csv(FINAL_DIR / f"summary_{split}.csv", dtype={"jpeg": str})
    paired = pd.read_csv(FINAL_DIR / f"paired_{split}.csv", dtype={"jpeg": str})
    paired_by_size = pd.read_csv(FINAL_DIR / f"paired_text_by_size_{split}.csv", dtype={"jpeg": str})
    methods = [m for m in METHODS if m in set(metrics["method"])]
    missing = [m for m in METHODS if m not in methods]

    written = [write("main_table", split, dict(main_table(summary, split), missing_methods=missing))]
    jpeg_result = jpeg_hypothesis(metrics, methods, split)
    written.append(write("jpeg_hypothesis", split, jpeg_result))
    written.append(write("hypothesis_checks", split, hypothesis_checks(paired, paired_by_size, methods, jpeg_result, split)))
    written.append(write("failure_cases", split, failure_cases(metrics, methods, split, manifest)))
    seeds = seed_variability(metrics, split)
    if seeds is not None:
        written.append(write("seed_variability", split, seeds))
    sweep = plot_jpeg_sweep(summary, methods, split)
    figure, points = fidelity_perception(summary, methods, split)
    written.append(write("fidelity_perception", split, {"split": split, "figure": str(figure.relative_to(PROJECT_ROOT)), "points": points}))
    runtime = runtime_table()
    (ANALYSIS_DIR / "runtime.json").write_text(json.dumps(runtime, indent=2))
    consistency = dev_test_consistency()
    if consistency is not None:
        (ANALYSIS_DIR / "dev_test_consistency.json").write_text(json.dumps(consistency, indent=2))
    print("missing methods:", missing or "none")
    print("\n".join(str(path.relative_to(PROJECT_ROOT)) for path in written + [sweep, figure]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)
