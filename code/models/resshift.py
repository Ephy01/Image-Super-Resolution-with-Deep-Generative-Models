from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO = PROJECT_ROOT / "third_party" / "ResShift"
CONFIGS = {"v3": "realsr_swinunet_realesrgan256_journal.yaml"}
SCALE = 4


def _import_official_code():
    if not REPO.exists():
        raise FileNotFoundError(
            f"Official ResShift code was not found at {REPO}. "
            "Run: bash code/models/setup_resshift.sh"
        )
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf
    from utils import util_common, util_net
    return OmegaConf, util_common, util_net


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ResShift weight: {path}. Run: "
            "python code/datasets/download_data.py"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state.get("state_dict", state)


class ResShift:

    def __init__(
        self,
        device: torch.device,
        weights_dir: Path,
        version: str = "v3",
        amp: bool = True,
        seed: int = 0,
    ):
        self.device = device
        self.scale = SCALE
        OmegaConf, util_common, util_net = _import_official_code()
        if version not in CONFIGS:
            raise ValueError(f"Unsupported ResShift version {version!r}; choose from {tuple(CONFIGS)}")

        cfg = OmegaConf.load(REPO / "configs" / CONFIGS[version])
        cfg.diffusion.params.sf = self.scale
        self.diffusion = util_common.instantiate_from_config(cfg.diffusion)

        if device.type == "mps":
            for key, value in vars(self.diffusion).items():
                if isinstance(value, np.ndarray) and value.dtype == np.float64:
                    setattr(self.diffusion, key, value.astype(np.float32))

        model = util_common.instantiate_from_config(cfg.model)
        util_net.reload_model(
            model,
            _load_checkpoint(weights_dir / f"resshift_realsrx4_s4_{version}.pth"),
        )
        self.model = model.eval().to(device)

        autoencoder = util_common.get_obj_from_str(cfg.autoencoder.target)(**cfg.autoencoder.params)
        util_net.reload_model(
            autoencoder,
            _load_checkpoint(weights_dir / "autoencoder_vq_f4.pth"),
        )
        self.autoencoder = autoencoder.eval().to(device)

        for parameter in list(self.model.parameters()) + list(self.autoencoder.parameters()):
            parameter.requires_grad_(False)

        self.pad_to = int(cfg.model.params.get("lq_size", 64))
        self.seed = int(seed)
        self.amp = bool(amp and device.type in ("cuda", "mps"))

    def _autocast(self):
        if not self.amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float16)

    @torch.inference_mode()
    def upscale(self, lr: np.ndarray, seed: int | None = None) -> np.ndarray:
        if lr.ndim != 3 or lr.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB image, got shape {lr.shape}")
        torch.manual_seed(self.seed if seed is None else int(seed))
        y = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1)[None]
        y = y.float().div_(255).to(self.device).mul_(2).sub_(1)  

        height, width = y.shape[2:]
        pad_h = math.ceil(height / self.pad_to) * self.pad_to - height
        pad_w = math.ceil(width / self.pad_to) * self.pad_to - width
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")

        with self._autocast():
            output = self.diffusion.p_sample_loop(
                y=y,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=False,
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs={"lq": y},
                progress=False,
            )

        output = output[:, :, : height * self.scale, : width * self.scale]
        output = output.float().clamp(-1, 1).mul_(0.5).add_(0.5)
        pixels = output[0].permute(1, 2, 0).cpu().numpy()
        return (pixels * 255.0).round().clip(0, 255).astype(np.uint8)
