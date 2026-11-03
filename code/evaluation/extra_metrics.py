"""MUSIQ (no-reference), OCR character error rate (Tesseract) and ArcFace identity similarity (insightface).

Settings come from the metrics: section of the config. Values are cached per image file under
<SR_CACHE>/metric_values/<metric>_<settings hash>.json, keyed by the SHA-256 of the image, so a changed
setting or a changed image is always recomputed.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import paths

os.environ.setdefault("TORCH_HOME", str(paths.CACHE_ROOT / "torch"))
VALUES_DIR = paths.CACHE_ROOT / "metric_values"


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


class ValueCache:
    def __init__(self, metric, settings):
        self.path = VALUES_DIR / f"{metric}_{settings_hash(settings)}.json"
        self.values = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.settings = settings

    def get(self, image_path, compute):
        key = paths.sha256_file(image_path)
        if key not in self.values:
            self.values[key] = compute(image_path)
        return self.values[key]

    def save(self):
        """Merge with the file on disk first, so two evaluations running at the same time do not drop each other's values."""
        VALUES_DIR.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.values = {**json.loads(self.path.read_text()), **self.values}
        self.path.write_text(json.dumps(self.values))


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


# MUSIQ ---------------------------------------------------------------------------------------------------------------

class Musiq:
    def __init__(self, cfg):
        import pyiqa
        import torch

        torch.set_num_threads(max(1, os.cpu_count() // 2))
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.model = pyiqa.create_metric(cfg["metric"], device=self.device, as_loss=False)
        self.checkpoint = self.find_checkpoint()
        self.cache = ValueCache("musiq", {"metric": cfg["metric"], "device": cfg["device"], "checkpoint": self.checkpoint})

    @staticmethod
    def find_checkpoint():
        found = sorted((paths.CACHE_ROOT / "torch").rglob("musiq_koniq_ckpt-*.pth"))
        if not found:
            return None
        return {"file": found[0].name, "sha256": paths.sha256_file(found[0])}

    def compute(self, image_path):
        import torch

        pixels = torch.from_numpy(load_rgb(image_path).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        with torch.inference_mode():
            return float(self.model(pixels.to(self.device)).item())

    def __call__(self, image_path):
        return self.cache.get(image_path, self.compute)

    def save(self):
        self.cache.save()


# OCR -----------------------------------------------------------------------------------------------------------------

def tesseract_version():
    return subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0]


def tessdata_sha256(lang):
    prefix = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
    folder = re.search(r'"(.+?)"', prefix)
    path = Path(folder.group(1)) / f"{lang}.traineddata" if folder else None
    return paths.sha256_file(path) if path and path.exists() else None


def preprocess_for_ocr(image, mode):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mode == "none":
        return gray
    if mode == "upscale2_otsu":
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if mode in ("otsu", "upscale2_otsu"):
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    raise ValueError(f"unknown OCR preprocessing {mode!r}")


def normalise_text(text):
    return re.sub(r"\s+", " ", text).strip()


def character_error_rate(prediction, reference):
    from rapidfuzz.distance import Levenshtein

    reference = normalise_text(reference)
    return Levenshtein.distance(normalise_text(prediction), reference) / max(len(reference), 1)


class Ocr:
    def __init__(self, cfg, psm=None, oem=None, preprocess=None):
        import pytesseract

        self.pytesseract = pytesseract
        selected = cfg.get("selected") or {}
        self.psm = psm if psm is not None else selected["psm"]
        self.oem = oem if oem is not None else selected["oem"]
        self.preprocess = preprocess if preprocess is not None else selected["preprocess"]
        self.lang = cfg["lang"]
        self.settings = {"psm": self.psm, "oem": self.oem, "preprocess": self.preprocess, "lang": self.lang,
                         "tesseract": tesseract_version(), "tessdata_sha256": tessdata_sha256(self.lang)}
        self.cache = ValueCache("ocr", self.settings)

    def compute(self, image_path):
        image = preprocess_for_ocr(load_rgb(image_path), self.preprocess)
        return self.pytesseract.image_to_string(image, lang=self.lang, config=f"--psm {self.psm} --oem {self.oem}")

    def read(self, image_path):
        return self.cache.get(image_path, self.compute)

    def cer(self, image_path, reference):
        return character_error_rate(self.read(image_path), reference)

    def save(self):
        self.cache.save()


# ArcFace -------------------------------------------------------------------------------------------------------------

class ArcFace:
    def __init__(self, cfg, det_size=None):
        from insightface.app import FaceAnalysis

        cfg = dict(cfg, det_size=det_size or cfg["det_size"])
        self.cfg = cfg
        root = paths.CACHE_ROOT / "insightface"
        self.app = FaceAnalysis(name=cfg["model_pack"], root=str(root), allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(cfg["det_size"], cfg["det_size"]), det_thresh=cfg["det_thresh"])
        model_dir = root / "models" / cfg["model_pack"]
        self.models = {path.name: paths.sha256_file(path) for path in sorted(model_dir.glob("*.onnx"))}
        self.cache = ValueCache("arcface", {"model_pack": cfg["model_pack"], "det_size": cfg["det_size"],
                                            "det_thresh": cfg["det_thresh"], "models": self.models,
                                            "face_area": "bounding box clipped to the image"})

    def compute(self, image_path):
        """Normalised embedding of the largest detected face, or None when no face is detected.

        The area is measured inside the image: a partial face cut by the border can have a detection box that reaches
        far outside the frame, and its unclipped area exceeded the main face on 2 test images (stage 5 report, section 1).
        """
        image = load_rgb(image_path)
        faces = self.app.get(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not faces:
            return None
        height, width = image.shape[:2]

        def area(item):
            left, top, right, bottom = item.bbox
            return max(min(right, width) - max(left, 0), 0) * max(min(bottom, height) - max(top, 0), 0)

        face = max(faces, key=area)
        return [float(value) for value in face.normed_embedding]

    def embed(self, image_path):
        embedding = self.cache.get(image_path, self.compute)
        return None if embedding is None else np.asarray(embedding)

    def save(self):
        self.cache.save()


def cosine(first, second):
    if first is None or second is None:
        return float("nan")
    return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
