"""Check the demo environment and downloaded models without running inference."""

import argparse
import hashlib
import importlib
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "ResShift"
COMMIT = "bb03b7d21614cace01787e097c8a6ab6b945227d"
CONFIG = SOURCE / "configs" / "realsr_swinunet_realesrgan256_journal.yaml"


def check_weights():
    manifest = ROOT / "provenance" / "weights_sha256.txt"
    expected_names = {
        "RealESRNet_x4plus.pth", "RealESRGAN_x4plus.pth",
        "resshift_realsrx4_s4_v3.pth", "autoencoder_vq_f4.pth",
    }
    entries = {}
    for line in manifest.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            digest, name = line.split()
            entries[name] = digest
    if set(entries) != expected_names:
        raise RuntimeError("The weight checksum manifest must list all four expected files.")
    for name, expected in entries.items():
        path = ROOT / "models" / name
        if not path.is_file():
            raise RuntimeError(f"Missing {name}. Run setup_models.py again.")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}. Run setup_models.py again.")
        print(f"Verified weight: {name}", flush=True)


def check_source():
    if not CONFIG.is_file():
        raise RuntimeError("ResShift source is missing. Run setup_models.py again.")
    result = subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip() != COMMIT:
        raise RuntimeError("ResShift source is not at the pinned commit. Run setup_models.py again.")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SOURCE))
    from omegaconf import OmegaConf
    from utils import util_common

    cfg = OmegaConf.load(CONFIG)
    if int(cfg.diffusion.params.sf) != 4 or int(cfg.diffusion.params.steps) != 4:
        raise RuntimeError("The ResShift configuration must use 4x scaling and four sampling steps.")
    for target in (cfg.model.target, cfg.autoencoder.target, cfg.diffusion.target):
        if not callable(util_common.get_obj_from_str(target)):
            raise RuntimeError(f"Invalid ResShift target: {target}")
    importlib.import_module("rrdbnet")
    importlib.import_module("diffusion_model")
    print(f"Verified ResShift source and imports: {COMMIT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    try:
        import cv2
        import numpy
        import PIL
        import torch
        import torchvision

        if args.device == "mps":
            if sys.platform != "darwin" or platform.machine() != "arm64":
                raise RuntimeError("Use native ARM64 Python on an Apple Silicon Mac.")
            if not torch.backends.mps.is_available():
                raise RuntimeError("Apple GPU (MPS) is unavailable. Check your macOS and PyTorch installation.")
        print(f"Python {platform.python_version()}, PyTorch {torch.__version__}, device {args.device}")
        print(f"Pillow {PIL.__version__}, NumPy {numpy.__version__}, OpenCV {cv2.__version__}, torchvision {torchvision.__version__}")
        check_weights()
        check_source()
    except Exception as error:
        raise SystemExit(f"Installation check failed: {error}") from None
    print("Installation check passed. No weights were loaded and no inference was run.")


if __name__ == "__main__":
    main()
