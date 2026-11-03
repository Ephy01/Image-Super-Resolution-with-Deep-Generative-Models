"""Pack the low-resolution inputs of one split for a GPU notebook in experiments/.

    python scripts/make_gpu_inputs.py --split test      # -> data/gpu_inputs_test.zip (+ its SHA-256 on stdout)

The archive holds data/processed/lr/jpeg_*/<image_id>.png for the split, manifest.csv and inputs_sha256.json.
The notebook checks every file against inputs_sha256.json, and inputs_sha256.json against the hash embedded in the notebook.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paths

PROCESSED = paths.DATA_ROOT / "processed"
QUALITIES = ["none", "50", "30", "10"]


def main(split):
    manifest = pd.read_csv(PROCESSED / "manifest.csv")
    image_ids = manifest.loc[manifest["split"] == split, "image_id"]
    archive_path = paths.DATA_ROOT / f"gpu_inputs_{split}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("manifest.csv", "inputs_sha256.json"):
            archive.write(PROCESSED / name, f"data/processed/{name}")
        for quality in QUALITIES:
            for image_id in image_ids:
                archive.write(PROCESSED / "lr" / f"jpeg_{quality}" / f"{image_id}.png", f"data/processed/lr/jpeg_{quality}/{image_id}.png")
    print(f"{archive_path}: {len(image_ids) * len(QUALITIES)} inputs, {archive_path.stat().st_size / 2**20:.1f} MiB, sha256 {paths.sha256_file(archive_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    main(parser.parse_args().split)
