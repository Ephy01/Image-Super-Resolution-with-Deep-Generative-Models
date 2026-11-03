import argparse
import json
import os
import sys
from pathlib import Path

import certifi
import cv2
import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "datasets"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "models"))
import paths
from download_data import read_checksums
from extra_metrics import ArcFace, Musiq, Ocr, cosine
from prepare_data import HR_DIR, JPEG_QUALITIES, PROCESSED_DIR, SCALE
from upscale import METHODS, OUTPUTS_DIR, RESULTS_DIR, pick_device

CONFIG = paths.load_config()
FINAL_DIR = RESULTS_DIR / "final"
METRICS = ["psnr", "ssim", "lpips", "musiq", "cer", "arcface_sim", "arcface_match"]
CROP_BORDER = SCALE
BOOTSTRAP_SAMPLES = CONFIG["statistics"]["bootstrap_samples"]
CONFIDENCE = CONFIG["statistics"]["confidence"]
SEED = CONFIG["statistics"]["seed"]

os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def to_luma(image):
    pixels = image.astype(np.float64) / 255.0
    return 16.0 + 65.481 * pixels[..., 0] + 128.553 * pixels[..., 1] + 24.966 * pixels[..., 2]


def crop_border(channel, border=CROP_BORDER):
    return channel[border:-border, border:-border]


def compute_psnr(output, target):
    difference = crop_border(to_luma(output)) - crop_border(to_luma(target))
    mean_squared_error = np.mean(difference ** 2)
    if mean_squared_error == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mean_squared_error))


def compute_ssim(output, target):
    first = crop_border(to_luma(output))
    second = crop_border(to_luma(target))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    def local_mean(values):
        return cv2.filter2D(values, -1, window)[5:-5, 5:-5]

    mean_first = local_mean(first)
    mean_second = local_mean(second)
    variance_first = local_mean(first ** 2) - mean_first ** 2
    variance_second = local_mean(second ** 2) - mean_second ** 2
    covariance = local_mean(first * second) - mean_first * mean_second

    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (mean_first ** 2 + mean_second ** 2 + c1) * (variance_first + variance_second + c2)
    return float(np.mean(numerator / denominator))


def to_lpips_tensor(image, device):
    pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
    return pixels.unsqueeze(0).to(device)


@torch.inference_mode()
def compute_lpips(output, target, lpips_model, device):
    distance = lpips_model(to_lpips_tensor(output, device), to_lpips_tensor(target, device))
    return float(distance.item())


def check_metrics(target, lpips_model, device):
    assert compute_psnr(target, target) == float("inf")
    assert abs(compute_ssim(target, target) - 1.0) < 1e-9
    assert compute_lpips(target, target, lpips_model, device) < 1e-6


def check_outputs(split, folder, image_ids):
    """Outputs must come from the checksummed weights and from exactly the current input files."""
    output_root = OUTPUTS_DIR / split / folder
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is missing: run inference.py for {folder} on {split} first")
    manifest = json.loads(manifest_path.read_text())
    checksums = read_checksums()
    for name, digest in manifest["key"]["weights_sha256"].items():
        if checksums.get(name) != digest:
            raise SystemExit(f"{folder}: outputs were made with {name} {digest[:12]}, scripts/SHA256SUMS has {checksums.get(name, 'none')[:12]}")
    inputs = json.loads((PROCESSED_DIR / "inputs_sha256.json").read_text())
    for quality in JPEG_QUALITIES:
        for image_id in image_ids:
            name = f"jpeg_{quality}/{image_id}.png"
            entry = manifest["files"].get(name)
            if entry is None:
                raise SystemExit(f"{folder}: {name} is not in {manifest_path}")
            if entry["input_sha256"] != inputs[f"lr/{name}"]:
                raise SystemExit(f"{folder}: {name} was made from a different input than data/processed/lr/{name}")
            if paths.sha256_file(output_root / name) != entry["output_sha256"]:
                raise SystemExit(f"{folder}: {output_root / name} changed after it was written")
    return {"device": manifest["device"], "key": manifest["key"]}


class Scorers:
    def __init__(self, device):
        metrics = CONFIG["metrics"]
        self.device = device
        self.lpips = lpips.LPIPS(net=metrics["lpips"]["net"], verbose=False).eval().to(device)
        self.musiq = Musiq(metrics["musiq"])
        self.ocr = Ocr(metrics["ocr"])
        self.arcface = ArcFace(metrics["arcface"])
        self.threshold = metrics["arcface"]["threshold"]
        if self.threshold is None or metrics["ocr"]["selected"] is None or metrics["arcface"]["det_size"] is None:
            raise SystemExit("OCR and ArcFace settings are not selected: run code/evaluation/select_settings.py and use configs/final.yaml")

    def save(self):
        self.musiq.save()
        self.ocr.save()
        self.arcface.save()

    def reference(self, image):
        path = HR_DIR / f"{image.image_id}.png"
        row = {"musiq": self.musiq(path)}
        if image.content == "text":
            row["cer"] = self.ocr.cer(path, image.text)
        if image.content == "faces":
            row["arcface_detected"] = self.arcface.embed(path) is not None
        return row

    def score(self, image, path):
        target = load_image(HR_DIR / f"{image.image_id}.png")
        output = load_image(path)
        row = {
            "psnr": compute_psnr(output, target),
            "ssim": compute_ssim(output, target),
            "lpips": compute_lpips(output, target, self.lpips, self.device),
            "musiq": self.musiq(path),
        }
        if image.content == "text":
            row["cer"] = self.ocr.cer(path, image.text)
        if image.content == "faces":
            embedding = self.arcface.embed(path)
            reference = self.arcface.embed(HR_DIR / f"{image.image_id}.png")
            similarity = cosine(embedding, reference)
            row["arcface_detected"] = embedding is not None
            row["arcface_sim"] = similarity
            # A detection failure counts as a non-match; its similarity stays missing.
            row["arcface_match"] = float(similarity > self.threshold) if embedding is not None and reference is not None else 0.0
        return row


def evaluate_split(split, manifest, scorers, methods=METHODS):
    rows = []
    images = manifest[manifest["split"] == split]
    for image in tqdm(images.itertuples(), total=len(images), desc="hr reference"):
        rows.append({"image_id": image.image_id, "content": image.content, "method": "hr", "jpeg": "none",
                     "font_size": image.font_size, **scorers.reference(image)})
    for method in methods:
        for quality in JPEG_QUALITIES:
            output_dir = OUTPUTS_DIR / split / method / f"jpeg_{quality}"
            for image in tqdm(images.itertuples(), total=len(images), desc=f"{method} jpeg_{quality}"):
                rows.append({
                    "image_id": image.image_id,
                    "content": image.content,
                    "method": method,
                    "jpeg": str(quality),
                    "font_size": image.font_size,
                    **scorers.score(image, output_dir / f"{image.image_id}.png"),
                })
        scorers.save()
    return pd.DataFrame(rows).reindex(columns=["image_id", "content", "method", "jpeg", "font_size", *METRICS, "arcface_detected"])


def bootstrap_interval(values, generator=None):
    """95% percentile bootstrap interval of the mean: (lo, hi). Missing values are dropped first."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    generator = generator or np.random.default_rng(SEED)
    indices = generator.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    lower = np.quantile(means, (1 - CONFIDENCE) / 2)
    upper = np.quantile(means, 1 - (1 - CONFIDENCE) / 2)
    return float(lower), float(upper)


def describe(group, row):
    """Mean, interval and count per metric; every interval uses its own generator seeded with SEED."""
    for metric in METRICS:
        values = group[metric].astype(float)
        row[metric] = values.mean()
        row[f"{metric}_lo"], row[f"{metric}_hi"] = bootstrap_interval(values)
        row[f"{metric}_n"] = int(values.notna().sum())
    return row


def summarize(metrics, keys=("content", "jpeg", "method")):
    rows = []
    for values, group in metrics.groupby(list(keys), sort=False):
        row = dict(zip(keys, values), images=len(group))
        if "arcface_detected" in group and group["content"].iloc[0] == "faces":
            row["arcface_detection_failures"] = int((~group["arcface_detected"].astype(bool)).sum())
        rows.append(describe(group, row))
    return pd.DataFrame(rows)


def compare_methods(metrics, baseline, method, keys=("content", "jpeg")):
    """Paired differences method - baseline on the same image and JPEG level."""
    index = ["image_id", *keys]
    first = metrics[metrics["method"] == baseline].set_index(index)
    second = metrics[metrics["method"] == method].set_index(index)
    differences = (second[METRICS] - first[METRICS]).reset_index()

    rows = []
    for values, group in differences.groupby(list(keys), sort=False):
        rows.append(describe(group, dict(zip(keys, values), comparison=f"{method} - {baseline}")))
    return pd.DataFrame(rows)


COMPARISONS = [
    ("bicubic", "realesrnet"), ("realesrnet", "realesrgan"), ("realesrnet", "resshift"),
    ("realesrgan", "resshift"), ("bicubic", "realesrgan"), ("bicubic", "resshift"),
]


def main(split="dev", methods=None, tag=""):
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    image_ids = manifest.loc[manifest["split"] == split, "image_id"].tolist()
    if methods is None:
        methods = [method for method in METHODS if (OUTPUTS_DIR / split / method / "manifest.json").exists()]
    provenance = {method: check_outputs(split, method, image_ids) for method in methods}

    device = pick_device()
    scorers = Scorers(device)
    check_metrics(load_image(HR_DIR / f"{image_ids[0]}.png"), scorers.lpips, device)

    metrics = evaluate_split(split, manifest, scorers, methods)
    summary = summarize(metrics)
    text = metrics[metrics["content"] == "text"]
    by_size = summarize(text, keys=("font_size", "jpeg", "method"))
    pairs = [(first, second) for first, second in COMPARISONS if first in methods and second in methods]
    # A run with only the ResShift seed folders has no method pairs: the paired tables stay empty.
    paired = pd.concat([compare_methods(metrics, first, second) for first, second in pairs], ignore_index=True) if pairs else pd.DataFrame()
    paired_by_size = (pd.concat([compare_methods(text, first, second, keys=("font_size", "jpeg")) for first, second in pairs], ignore_index=True)
                      if pairs else pd.DataFrame())

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(FINAL_DIR / f"metrics_{split}{tag}.csv", index=False)
    summary.to_csv(FINAL_DIR / f"summary_{split}{tag}.csv", index=False)
    by_size.to_csv(FINAL_DIR / f"summary_text_by_size_{split}{tag}.csv", index=False)
    paired.to_csv(FINAL_DIR / f"paired_{split}{tag}.csv", index=False)
    paired_by_size.to_csv(FINAL_DIR / f"paired_text_by_size_{split}{tag}.csv", index=False)

    faces = metrics[(metrics["content"] == "faces")]
    report = {
        "split": split,
        "images": len(image_ids),
        "methods": methods,
        "rows": len(metrics),
        "config": paths.config_path().name,
        "config_sha256": paths.sha256_file(paths.config_path()),
        "psnr_ssim": "Y channel (BT.601), 4-pixel border crop",
        "lpips": "AlexNet",
        "musiq": {"metric": CONFIG["metrics"]["musiq"]["metric"], "checkpoint": scorers.musiq.checkpoint, "device": CONFIG["metrics"]["musiq"]["device"]},
        "ocr": scorers.ocr.settings,
        "arcface": {"det_size": scorers.arcface.cfg["det_size"], "threshold": scorers.threshold, "models_sha256": scorers.arcface.models,
                    "detection_failures": {f"{method}/jpeg_{quality}": int((~group["arcface_detected"].astype(bool)).sum())
                                           for (method, quality), group in faces.groupby(["method", "jpeg"], sort=False)}},
        "confidence_interval": f"{CONFIDENCE:.0%} percentile bootstrap [lo, hi], {BOOTSTRAP_SAMPLES} resamples, seed {SEED} per interval",
        "outputs": provenance,
    }
    (FINAL_DIR / f"evaluation_report_{split}{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    print(summary[["content", "jpeg", "method", "psnr", "ssim", "lpips", "musiq", "cer", "arcface_sim", "arcface_match"]].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--methods", nargs="+", default=None, help="output folders; default: every method with a manifest")
    parser.add_argument("--tag", default="", help="suffix of the result files, e.g. _seeds")
    args = parser.parse_args()
    main(args.split, args.methods, args.tag)
