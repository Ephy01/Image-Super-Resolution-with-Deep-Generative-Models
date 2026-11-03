"""Choose the metric settings on the dev ground-truth images only. Never reads a test image.

    python code/evaluation/select_settings.py            # writes results/analysis/{ocr_config,arcface_threshold,musiq_setup}_dev.json

OCR: every psm x oem x preprocessing candidate is scored on the 24 dev HR text images; the lowest mean CER wins, ties
by candidate order. ArcFace: cosine similarity of every pair of different dev HR faces (impostor pairs); the threshold
is their 0.99 quantile (method "higher"), and a match is similarity > threshold, so at most 1% of impostor pairs match.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "datasets"))
import paths
from extra_metrics import ArcFace, Musiq, Ocr, character_error_rate, cosine, tesseract_version, tessdata_sha256
from prepare_data import HR_DIR, PROCESSED_DIR

ANALYSIS_DIR = paths.RESULTS_ROOT / "analysis"
SPLIT = "dev"


def dev_images(content):
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    rows = manifest[(manifest["split"] == SPLIT) & (manifest["content"] == content)]
    assert set(rows["split"]) == {SPLIT}
    return rows


def select_ocr(cfg):
    texts = dev_images("text")
    candidates = cfg["candidates"]
    grid = []
    for psm, oem, preprocess in itertools.product(candidates["psm"], candidates["oem"], candidates["preprocess"]):
        ocr = Ocr(cfg, psm=psm, oem=oem, preprocess=preprocess)
        rates = [ocr.cer(HR_DIR / f"{row.image_id}.png", row.text) for row in texts.itertuples()]
        ocr.save()
        by_size = texts.assign(cer=rates).groupby("font_size")["cer"].mean()
        grid.append({"psm": psm, "oem": oem, "preprocess": preprocess, "mean_cer": float(np.mean(rates)),
                     "cer_by_font_size": {str(int(size)): float(value) for size, value in by_size.items()}})
        print(f"psm {psm} oem {oem} {preprocess:>14}: mean CER {np.mean(rates):.4f}")
    best = min(grid, key=lambda row: row["mean_cer"])   # min keeps the first of equal values: candidate order
    report = {
        "split": SPLIT, "images": len(texts), "rule": "lowest mean CER on dev HR text images; ties by candidate order",
        "tesseract": tesseract_version(), "tessdata_sha256": tessdata_sha256(cfg["lang"]), "lang": cfg["lang"],
        "normalisation": cfg["normalisation"], "cer": cfg["cer"],
        "selected": {key: best[key] for key in ("psm", "oem", "preprocess")},
        "hr_ceiling_dev": {"mean_cer": best["mean_cer"], "cer_by_font_size": best["cer_by_font_size"]},
        "grid": grid,
    }
    (ANALYSIS_DIR / "ocr_config_dev.json").write_text(json.dumps(report, indent=2))
    return report


def embed_faces(cfg, faces, det_size):
    arcface = ArcFace(cfg, det_size=det_size)
    embeddings = {row.image_id: arcface.embed(HR_DIR / f"{row.image_id}.png") for row in faces.itertuples()}
    arcface.save()
    return arcface, embeddings


def select_arcface(cfg):
    faces = dev_images("faces")
    detection = {}
    for det_size in cfg["det_size_candidates"]:
        _, embeddings = embed_faces(cfg, faces, det_size)
        detection[det_size] = sorted(name for name, embedding in embeddings.items() if embedding is None)
        print(f"det_size {det_size}: {len(detection[det_size])} HR detection failures")
    det_size = min(cfg["det_size_candidates"], key=lambda size: len(detection[size]))
    arcface, embeddings = embed_faces(cfg, faces, det_size)
    failures = detection[det_size]
    detected = [name for name in embeddings if embeddings[name] is not None]
    impostor = np.array([cosine(embeddings[a], embeddings[b]) for a, b in itertools.combinations(detected, 2)])
    threshold = float(np.quantile(impostor, 1 - cfg["false_match_rate"], method="higher"))
    report = {
        "split": SPLIT, "images": len(faces),
        "det_size_rule": "fewest HR detection failures on dev, ties by candidate order",
        "hr_detection_failures_by_det_size": {str(size): names for size, names in detection.items()},
        "det_size": det_size, "hr_detection_failures": failures,
        "impostor_pairs": int(impostor.size), "rule": "0.99 quantile (method higher) of impostor similarities; match is similarity > threshold",
        "threshold": threshold,
        "false_match_rate_on_dev": float(np.mean(impostor > threshold)),
        "impostor_similarity": {"mean": float(impostor.mean()), "max": float(impostor.max()),
                                "quantiles": {str(q): float(np.quantile(impostor, q)) for q in (0.5, 0.9, 0.95, 0.99)}},
        "model_pack": cfg["model_pack"], "models_sha256": arcface.models, "det_thresh": cfg["det_thresh"],
    }
    (ANALYSIS_DIR / "arcface_threshold_dev.json").write_text(json.dumps(report, indent=2))
    print(f"ArcFace threshold {threshold:.4f} from {impostor.size} impostor pairs; HR detection failures: {failures}")
    return report


def setup_musiq(cfg):
    sample = dev_images("objects").iloc[0]
    musiq = Musiq(cfg)
    first = musiq.compute(HR_DIR / f"{sample.image_id}.png")
    second = musiq.compute(HR_DIR / f"{sample.image_id}.png")
    report = {
        "split": SPLIT, "metric": cfg["metric"], "device": cfg["device"], "checkpoint": Musiq.find_checkpoint(),
        "repeatability": {"image_id": sample.image_id, "first": first, "second": second, "identical": first == second},
        "input": "RGB in [0, 1], full 512x512 image", "higher_is_better": True,
    }
    (ANALYSIS_DIR / "musiq_setup_dev.json").write_text(json.dumps(report, indent=2))
    print(f"MUSIQ checkpoint {report['checkpoint']}; repeat identical: {first == second}")
    return report


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = paths.load_config()["metrics"]
    if "--skip-ocr" not in sys.argv:
        select_ocr(metrics["ocr"])
    select_arcface(metrics["arcface"])
    setup_musiq(metrics["musiq"])


if __name__ == "__main__":
    main()
