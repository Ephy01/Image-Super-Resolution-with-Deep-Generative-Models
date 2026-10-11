import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "datasets"))
from prepare_data import JPEG_QUALITIES, LR_DIR, PROCESSED_DIR, SCALE
from rrdbnet import RRDBNet

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

METHODS = ["bicubic", "realesrnet"]
MODEL_CONFIG = {
    "weights": "RealESRNet_x4plus.pth",
    "num_feat": 64,
    "num_block": 23,
    "num_grow_ch": 32,
}
WARMUP_IMAGES = 3


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return f"Apple GPU (MPS), {platform.processor()}"
    return f"CPU, {platform.processor()}"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_realesrnet(device):
    network = RRDBNet(
        num_feat=MODEL_CONFIG["num_feat"],
        num_block=MODEL_CONFIG["num_block"],
        num_grow_ch=MODEL_CONFIG["num_grow_ch"],
    )
    checkpoint = torch.load(MODELS_DIR / MODEL_CONFIG["weights"], map_location="cpu", weights_only=True)
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    network.load_state_dict(state, strict=True)
    return network.eval().to(device)


def upscale_bicubic(image):
    size = (image.width * SCALE, image.height * SCALE)
    return image.resize(size, Image.BICUBIC)


@torch.inference_mode()
def upscale_realesrnet(image, network, device):
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    batch = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
    output = network(batch).clamp(0, 1)
    synchronize(device)
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((output * 255.0).round().astype(np.uint8))


def run_method(method, image_ids, split, device):
    network = load_realesrnet(device) if method == "realesrnet" else None

    def upscale(image):
        if method == "bicubic":
            return upscale_bicubic(image)
        return upscale_realesrnet(image, network, device)

    warmup_image = Image.open(LR_DIR / "jpeg_none" / f"{image_ids[0]}.png")
    for _ in range(WARMUP_IMAGES):
        upscale(warmup_image)

    seconds = []
    for quality in JPEG_QUALITIES:
        output_dir = OUTPUTS_DIR / split / method / f"jpeg_{quality}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_id in tqdm(image_ids, desc=f"{method} jpeg_{quality}"):
            image = Image.open(LR_DIR / f"jpeg_{quality}" / f"{image_id}.png").convert("RGB")
            start = time.perf_counter()
            result = upscale(image)
            seconds.append(time.perf_counter() - start)
            result.save(output_dir / f"{image_id}.png")

    return {
        "images": len(seconds),
        "seconds_per_image_mean": float(np.mean(seconds)),
        "seconds_per_image_median": float(np.median(seconds)),
    }


def main(split="dev", methods=METHODS):
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    image_ids = manifest.loc[manifest["split"] == split, "image_id"].tolist()
    device = pick_device()

    report = {
        "split": split,
        "device": describe_device(device),
        "torch": torch.__version__,
        "precision": "float32",
        "batch_size": 1,
        "input_size": 512 // SCALE,
        "output_size": 512,
        "realesrnet_config": MODEL_CONFIG,
        "realesrnet_parameters": sum(p.numel() for p in RRDBNet().parameters()),
        "methods": {},
    }
    for method in methods:
        report["methods"][method] = run_method(method, image_ids, split, device)

    RESULTS_DIR.mkdir(exist_ok=True)
    report_json = json.dumps(report, indent=2)
    (RESULTS_DIR / f"upscale_report_{split}.json").write_text(report_json)
    print(report_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    args = parser.parse_args()
    main(args.split, args.methods)
