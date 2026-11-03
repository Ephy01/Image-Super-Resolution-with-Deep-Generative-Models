import argparse
import gc
import json
import platform
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "datasets"))
import paths
from prepare_data import JPEG_QUALITIES, LR_DIR, PROCESSED_DIR, SCALE
from rrdbnet import RRDBNet
from resshift import ResShift

CONFIG = paths.load_config()
MODELS_DIR = paths.WEIGHTS_ROOT
OUTPUTS_DIR = paths.OUTPUTS_ROOT
RESULTS_DIR = paths.RESULTS_ROOT
RUNTIME_DIR = RESULTS_DIR / "runtime"

METHODS = ["bicubic", "realesrnet", "realesrgan", "resshift"]
# Real-ESRNet and Real-ESRGAN share the RRDBNet generator and training data; only the loss differs.
WEIGHTS = dict(CONFIG["rrdbnet"]["weights"])
MODEL_CONFIG = {
    "num_feat": 64,
    "num_block": 23,
    "num_grow_ch": 32,
}
RESSHIFT_CONFIG = {
    "version": CONFIG["resshift"]["version"],
    "model_weights": CONFIG["resshift"]["model_weights"],
    "autoencoder_weights": CONFIG["resshift"]["autoencoder_weights"],
    "amp": CONFIG["resshift"]["amp"],
    "seed": 0,
}
WARMUP_IMAGES = CONFIG["runtime"]["warmup_runs"]
SEED_MASK = 0x7FFFFFFF
MODEL_CODE = {
    "bicubic": [],
    "realesrnet": ["code/models/rrdbnet.py"],
    "realesrgan": ["code/models/rrdbnet.py"],
    "resshift": ["code/models/resshift.py"],
}


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
    return f"CPU, {platform.processor() or platform.machine()}"


def device_tag(device):
    if device.type == "cuda":
        return "t4" if "T4" in torch.cuda.get_device_name(0) else "cuda"
    return device.type


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def method_folder(method, seed_offset=0):
    """Offset 0 is the main run; the other ResShift seeds go to resshift_seed<k>."""
    return method if method != "resshift" or seed_offset == 0 else f"resshift_seed{seed_offset}"


def resshift_seed(image_id, offset=0):
    return (zlib.crc32(image_id.encode("utf-8")) + offset) & SEED_MASK


def weight_files(method):
    if method in WEIGHTS:
        return [WEIGHTS[method]]
    if method == "resshift":
        return [RESSHIFT_CONFIG["model_weights"], RESSHIFT_CONFIG["autoencoder_weights"]]
    return []


def cache_key(method, seed_offset):
    """Everything that determines the output pixels apart from the input file itself."""
    key = {
        "method": method,
        "weights_sha256": {name: paths.sha256_file(MODELS_DIR / name) for name in weight_files(method)},
        "model_code_sha256": {name: paths.sha256_file(PROJECT_ROOT / name) for name in MODEL_CODE[method]},
    }
    if method in WEIGHTS:
        key["settings"] = {"model_config": MODEL_CONFIG, "precision": "float32"}
    if method == "resshift":
        key["settings"] = {
            "version": RESSHIFT_CONFIG["version"],
            "official_commit": CONFIG["resshift"]["official_commit"],
            "amp": RESSHIFT_CONFIG["amp"],
            "seed_offset": seed_offset,
            "seed_policy": CONFIG["resshift"]["seed_policy"],
        }
    return key


def read_manifest(output_root):
    path = output_root / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def cached(output_root, key, inputs):
    """True when the saved outputs were made with the same key from the same input files and are unchanged."""
    manifest = read_manifest(output_root)
    if manifest is None or manifest["key"] != key:
        return False
    for name, input_hash in inputs.items():
        entry = manifest["files"].get(name)
        if entry is None or entry["input_sha256"] != input_hash:
            return False
        if paths.sha256_file(output_root / name) != entry["output_sha256"]:
            return False
    return True


def load_rrdbnet(method, device):
    network = RRDBNet(
        num_feat=MODEL_CONFIG["num_feat"],
        num_block=MODEL_CONFIG["num_block"],
        num_grow_ch=MODEL_CONFIG["num_grow_ch"],
    )
    checkpoint = torch.load(MODELS_DIR / WEIGHTS[method], map_location="cpu", weights_only=True)
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    network.load_state_dict(state, strict=True)
    return network.eval().to(device)

def load_resshift(device):
    return ResShift(
        device=device,
        weights_dir=MODELS_DIR,
        version=RESSHIFT_CONFIG["version"],
        amp=RESSHIFT_CONFIG["amp"],
        seed=RESSHIFT_CONFIG["seed"],
    )

def upscale_bicubic(image):
    size = (image.width * SCALE, image.height * SCALE)
    return image.resize(size, Image.BICUBIC)


@torch.inference_mode()
def upscale_rrdbnet(image, network, device):
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    batch = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
    output = network(batch).clamp(0, 1)
    synchronize(device)
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((output * 255.0).round().astype(np.uint8))


def run_method(method, image_ids, split, device, seed_offset=0, inputs=None, force=False):
    if method == "bicubic":
        device = torch.device("cpu")
    folder = method_folder(method, seed_offset)
    output_root = OUTPUTS_DIR / split / folder
    inputs = inputs or {}
    wanted = {f"jpeg_{quality}/{image_id}.png": inputs[f"lr/jpeg_{quality}/{image_id}.png"]
              for quality in JPEG_QUALITIES for image_id in image_ids}
    key = cache_key(method, seed_offset)
    if not force and cached(output_root, key, wanted):
        print(f"{folder}: outputs match manifest.json (same weights, code, settings and inputs); skipped")
        return {"cached": True}
    if read_manifest(output_root) is not None:
        print(f"{folder}: manifest.json does not match the current weights, code, settings or inputs; recomputing")

    wall_start = time.perf_counter()
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if method in WEIGHTS:
        network = load_rrdbnet(method, device)
    elif method == "resshift":
        network = load_resshift(device)
    else:
        network = None

    def upscale(image, image_id=None):
        if method == "bicubic":
            return upscale_bicubic(image)
        if method == "resshift":
            output = network.upscale(np.asarray(image.convert("RGB")), seed=resshift_seed(image_id, seed_offset))
            return Image.fromarray(output)
        return upscale_rrdbnet(image, network, device)

    warmup_image = Image.open(LR_DIR / "jpeg_none" / f"{image_ids[0]}.png").convert("RGB")
    for _ in range(WARMUP_IMAGES):
        upscale(warmup_image, image_ids[0])

    seconds = []
    files = {}
    for quality in JPEG_QUALITIES:
        output_dir = output_root / f"jpeg_{quality}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_id in tqdm(image_ids, desc=f"{folder} jpeg_{quality}"):
            image = Image.open(LR_DIR / f"jpeg_{quality}" / f"{image_id}.png").convert("RGB")
            synchronize(device)
            start = time.perf_counter()
            result = upscale(image, image_id)
            synchronize(device)
            seconds.append({"image_id": image_id, "jpeg": str(quality), "seconds": time.perf_counter() - start})
            name = f"jpeg_{quality}/{image_id}.png"
            result.save(output_root / name)
            files[name] = {"input_sha256": wanted[name], "output_sha256": paths.sha256_file(output_root / name)}

    report = {
        "images": len(seconds),
        "device": describe_device(device),
        "seconds_per_image_mean": float(np.mean([row["seconds"] for row in seconds])),
        "seconds_per_image_median": float(np.median([row["seconds"] for row in seconds])),
        "wall_seconds": time.perf_counter() - wall_start,
        "warmup_runs": WARMUP_IMAGES,
    }
    if device.type == "cuda":
        report["peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    manifest = {"split": split, "folder": folder, "key": key, "device": describe_device(device), "files": files}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seconds).to_csv(RUNTIME_DIR / f"timing_{split}_{folder}_{device_tag(device)}.csv", index=False)
    del network
    return report


def main(split="dev", methods=METHODS, seed_offsets=(0,), force=False):
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    image_ids = manifest.loc[manifest["split"] == split, "image_id"].tolist()
    inputs = json.loads((PROCESSED_DIR / "inputs_sha256.json").read_text())
    device = pick_device()
    # Same global state as the stage 4 ResShift notebook; ResShift also reseeds torch per image.
    random.seed(CONFIG["data"]["seed"])
    np.random.seed(CONFIG["data"]["seed"])
    torch.manual_seed(CONFIG["data"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    report = {
        "split": split,
        "device": describe_device(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "precision": {"bicubic": "uint8 (Pillow, CPU)", "rrdbnet": "float32", "resshift": CONFIG["resshift"]["precision"]},
        "batch_size": 1,
        "input_size": 512 // SCALE,
        "output_size": 512,
        "rrdbnet_config": MODEL_CONFIG,
        "rrdbnet_weights": WEIGHTS,
        "rrdbnet_parameters": sum(p.numel() for p in RRDBNet().parameters()),
        "resshift_config": RESSHIFT_CONFIG,
        "config": paths.config_path().name,
        "config_sha256": paths.sha256_file(paths.config_path()),
        "methods": {},
    }
    split_start = time.perf_counter()
    for method in methods:
        for offset in (seed_offsets if method == "resshift" else (0,)):
            report["methods"][method_folder(method, offset)] = run_method(method, image_ids, split, device, offset, inputs, force)
    report["split_wall_seconds"] = time.perf_counter() - split_start
    if all(entry.get("cached") for entry in report["methods"].values()):
        return

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(sorted(report["methods"]))
    report_json = json.dumps(report, indent=2)
    (RUNTIME_DIR / f"upscale_report_{split}_{device_tag(device)}_{tag}.json").write_text(report_json)
    print(report_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=[CONFIG["resshift"]["seed_offset"]])
    parser.add_argument("--force", action="store_true", help="recompute even when the outputs match their manifest")
    args = parser.parse_args()
    main(args.split, args.methods, args.seed_offsets, args.force)
