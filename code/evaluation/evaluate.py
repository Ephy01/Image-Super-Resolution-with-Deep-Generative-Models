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
sys.path.insert(0, str(PROJECT_ROOT / "code" / "datasets"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "models"))
from prepare_data import HR_DIR, JPEG_QUALITIES, PROCESSED_DIR, SCALE
from upscale import METHODS, OUTPUTS_DIR, RESULTS_DIR, pick_device

METRICS = ["psnr", "ssim", "lpips"]
CROP_BORDER = SCALE
BOOTSTRAP_SAMPLES = 10000
CONFIDENCE = 0.95

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


def evaluate_split(split, manifest, lpips_model, device):
    rows = []
    images = manifest[manifest["split"] == split]
    for method in METHODS:
        for quality in JPEG_QUALITIES:
            output_dir = OUTPUTS_DIR / split / method / f"jpeg_{quality}"
            for image in tqdm(images.itertuples(), total=len(images), desc=f"{method} jpeg_{quality}"):
                target = load_image(HR_DIR / f"{image.image_id}.png")
                output = load_image(output_dir / f"{image.image_id}.png")
                rows.append({
                    "image_id": image.image_id,
                    "content": image.content,
                    "method": method,
                    "jpeg": str(quality),
                    "psnr": compute_psnr(output, target),
                    "ssim": compute_ssim(output, target),
                    "lpips": compute_lpips(output, target, lpips_model, device),
                })
    return pd.DataFrame(rows)


def bootstrap_interval(values, generator):
    values = np.asarray(values)
    indices = generator.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    lower = np.quantile(means, (1 - CONFIDENCE) / 2)
    upper = np.quantile(means, 1 - (1 - CONFIDENCE) / 2)
    return float((upper - lower) / 2)


def summarize(metrics, seed):
    generator = np.random.default_rng(seed)
    rows = []
    groups = metrics.groupby(["content", "jpeg", "method"], sort=False)
    for (content, quality, method), group in groups:
        row = {"content": content, "jpeg": quality, "method": method, "images": len(group)}
        for metric in METRICS:
            row[metric] = group[metric].mean()
            row[f"{metric}_ci"] = bootstrap_interval(group[metric], generator)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_methods(metrics, baseline, method, seed):
    generator = np.random.default_rng(seed)
    keys = ["image_id", "content", "jpeg"]
    first = metrics[metrics["method"] == baseline].set_index(keys)
    second = metrics[metrics["method"] == method].set_index(keys)
    differences = (second[METRICS] - first[METRICS]).reset_index()

    rows = []
    for (content, quality), group in differences.groupby(["content", "jpeg"], sort=False):
        row = {"content": content, "jpeg": quality, "comparison": f"{method} - {baseline}"}
        for metric in METRICS:
            row[metric] = group[metric].mean()
            row[f"{metric}_ci"] = bootstrap_interval(group[metric], generator)
        rows.append(row)
    return pd.DataFrame(rows)


def main(split="dev", seed=42):
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    device = pick_device()
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)

    first_image = manifest.loc[manifest["split"] == split, "image_id"].iloc[0]
    check_metrics(load_image(HR_DIR / f"{first_image}.png"), lpips_model, device)

    metrics = evaluate_split(split, manifest, lpips_model, device)
    summary = summarize(metrics, seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / f"metrics_{split}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"summary_{split}.csv", index=False)
    paired = compare_methods(metrics, "bicubic", "realesrnet", seed)
    paired.to_csv(RESULTS_DIR / f"paired_{split}.csv", index=False)

    report = {
        "split": split,
        "images": int(manifest["split"].eq(split).sum()),
        "rows": len(metrics),
        "psnr_ssim": "Y channel (BT.601), 4-pixel border crop",
        "lpips": "AlexNet",
        "confidence_interval": f"{CONFIDENCE:.0%} bootstrap half-width, {BOOTSTRAP_SAMPLES} resamples",
        "seed": seed,
    }
    (RESULTS_DIR / f"evaluation_report_{split}.json").write_text(json.dumps(report, indent=2))
    print(summary.round(4).to_string(index=False))
    print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.split, args.seed)
