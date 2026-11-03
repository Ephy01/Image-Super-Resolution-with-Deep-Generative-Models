"""Freeze every setting for the test run: configs/default.yaml + the dev selections -> configs/final.yaml (+ .sha256).

    python code/evaluation/freeze_config.py

Refuses to overwrite an existing final.yaml whose hash file exists, so the frozen settings cannot change silently.
"""

import datetime
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import paths

FINAL = PROJECT_ROOT / "configs" / "final.yaml"
DIGEST = PROJECT_ROOT / "configs" / "final.yaml.sha256"
ANALYSIS_DIR = paths.RESULTS_ROOT / "analysis"


def main():
    if FINAL.exists() and DIGEST.exists():
        raise SystemExit(f"{FINAL.name} is already frozen ({DIGEST.read_text().split()[0][:12]}); delete both files by hand to refreeze")
    cfg = paths.load_config(PROJECT_ROOT / "configs" / "default.yaml")
    ocr = json.loads((ANALYSIS_DIR / "ocr_config_dev.json").read_text())
    arcface = json.loads((ANALYSIS_DIR / "arcface_threshold_dev.json").read_text())
    musiq = json.loads((ANALYSIS_DIR / "musiq_setup_dev.json").read_text())

    cfg["metrics"]["ocr"]["selected"] = ocr["selected"]
    cfg["metrics"]["ocr"]["tesseract"] = ocr["tesseract"]
    cfg["metrics"]["ocr"]["tessdata_sha256"] = ocr["tessdata_sha256"]
    cfg["metrics"]["arcface"]["det_size"] = arcface["det_size"]
    cfg["metrics"]["arcface"]["threshold"] = arcface["threshold"]
    cfg["metrics"]["arcface"]["models_sha256"] = arcface["models_sha256"]
    cfg["metrics"]["musiq"]["checkpoint"] = musiq["checkpoint"]
    cfg["frozen"] = {
        "date": datetime.date.today().isoformat(),
        "selected_on": "dev split only (64 images); see results/analysis/*_dev.json",
        "test_split_note": "an exploratory all-method run on test exists from cc9a873 (results/legacy_colab/); none of its numbers were used to choose any setting",
    }
    header = "# Frozen settings for the final test run. Do not edit: the SHA-256 is in final.yaml.sha256 and is checked before every test run.\n"
    FINAL.write_text(header + yaml.safe_dump(cfg, sort_keys=False, width=140))
    DIGEST.write_text(f"{paths.sha256_file(FINAL)}  final.yaml\n")
    print(DIGEST.read_text().strip())


if __name__ == "__main__":
    main()
