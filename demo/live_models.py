"""Run the two original RRDBNet models on the Mac's MPS device (or CUDA)."""

import hashlib
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
FILENAMES = {"realesrnet": "RealESRNet_x4plus.pth", "realesrgan": "RealESRGAN_x4plus.pth"}


class LiveModels:
    def __init__(self, enabled=True, device="auto"):
        self.models = {}
        self.weights = {}
        self.available = {}
        self.device = None
        self.torch = None
        reason = "Live learned models are disabled for this session."
        if enabled:
            try:
                import torch
                self.torch = torch
                available = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
                selected = next((name for name, ready in available.items() if ready), None) if device == "auto" else device
                if selected and available.get(selected):
                    self.device = torch.device(selected)
                reason = f"No available {device if device != 'auto' else 'MPS or CUDA'} device. Bicubic is still available."
            except ImportError:
                reason = "PyTorch is not installed in this Python environment. Bicubic is still available."
        checksums = {}
        for line in (ROOT / "provenance/weights_sha256.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                digest, name = line.split()
                checksums[name] = digest
        for method, filename in FILENAMES.items():
            path = ROOT / "models" / filename
            problem = reason if self.device is None else ""
            if self.device is not None:
                if not path.is_file():
                    problem = f"{filename} is missing. Run bash setup.command."
                elif hashlib.sha256(path.read_bytes()).hexdigest() != checksums[filename]:
                    problem = f"{filename} does not match the project's official weight checksum."
                else:
                    self.weights[method] = path
            self.available[method] = {"enabled": not bool(problem), "reason": problem, "max_side": 256}
        diffusion_files = [ROOT / "models" / name for name in
                           ["resshift_realsrx4_s4_v3.pth", "autoencoder_vq_f4.pth"]]
        problem = reason if self.device is None else ""
        if not problem:
            if not (ROOT / "third_party/ResShift/configs/realsr_swinunet_realesrgan256_journal.yaml").is_file() or importlib.util.find_spec("omegaconf") is None:
                problem = "ResShift is not set up. Run bash setup.command."
            else:
                for path in diffusion_files:
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksums[path.name]:
                        problem = "ResShift weights are missing or fail the official checksum. Run bash setup.command."
                        break
        self.available["resshift"] = {"enabled": not bool(problem), "reason": problem, "min_side": 64, "max_side": 128}
        self.label = "Apple GPU (MPS)" if self.device and self.device.type == "mps" else "CUDA GPU"
        if self.device and self.device.type == "mps":
            self.torch.mps.set_per_process_memory_fraction(0.6)

    def synchronize(self):
        if self.device.type == "mps":
            self.torch.mps.synchronize()
        else:
            self.torch.cuda.synchronize()

    def upscale(self, image, method, seed=42):
        status = self.available.get(method)
        if status is None or not status["enabled"]:
            raise ValueError(status["reason"] if status else "This method is not available for a new upload.")
        if max(image.size) > status["max_side"] or min(image.size) < status.get("min_side", 16):
            raise ValueError(f"Live {method} accepts {status.get('min_side', 16)}–{status['max_side']} pixels per side in this demo. No automatic resizing is applied.")
        if method == "resshift":
            return self.diffusion(image, seed)
        torch = self.torch
        first_run = method not in self.models
        load_seconds = 0
        if first_run:
            from rrdbnet import RRDBNet
            start = time.perf_counter()
            torch.manual_seed(42)
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            network = RRDBNet(num_feat=64, num_block=23, num_grow_ch=32)
            checkpoint = torch.load(self.weights[method], map_location="cpu", weights_only=True)
            network.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)), strict=True)
            self.models[method] = network.eval().to(self.device)
            self.synchronize()
            load_seconds = time.perf_counter() - start
        network = self.models[method]
        self.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            batch = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
            output = network(batch).clamp(0, 1)
            self.synchronize()
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = Image.fromarray((output * 255.0).round().astype(np.uint8))
        self.synchronize()
        return result, {"seconds": time.perf_counter() - start, "load_seconds": load_seconds,
                        "first_run": first_run, "device": self.label, "precision": "float32"}

    def diffusion(self, image, seed):
        load_seconds = 0
        first_run = "resshift" not in self.models
        if first_run:
            from diffusion_model import ResShift
            start = time.perf_counter()
            self.models["resshift"] = ResShift(self.device, ROOT / "models", version="v3", amp=True, seed=seed)
            self.synchronize()
            load_seconds = time.perf_counter() - start
        self.synchronize()
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="invalid value encountered in cast")
            result = self.models["resshift"].upscale(np.array(image), seed=seed)
        self.synchronize()
        return Image.fromarray(result), {"seconds": time.perf_counter() - start, "load_seconds": load_seconds,
                                        "first_run": first_run, "device": self.label, "seed": seed, "steps": 4,
                                        "precision": "float32 weights, float16 autocast"}
