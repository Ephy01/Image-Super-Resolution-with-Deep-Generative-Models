import argparse
import io
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

from download_data import DIV2K_DIR, FFHQ_DIR
from render_text import render_text_set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HR_DIR = PROCESSED_DIR / "hr"
LR_DIR = PROCESSED_DIR / "lr"

HR_SIZE = 512
SCALE = 4
JPEG_QUALITIES = ["none", 50, 30, 10]
JPEG_SUBSAMPLING = 2

OBJECT_DEV_COUNT = 20
FACE_DEV_COUNT = 20
TEXT_DEV_REPEATS = 1
TEXT_TEST_REPEATS = 4
TEXT_DEV_SEED = 2002
TEXT_TEST_SEED = 1001


def crop_center(image, size=HR_SIZE):
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def split_sources(paths, dev_count, seed):
    paths = sorted(paths)
    generator = random.Random(seed)
    dev_paths = set(generator.sample(paths, dev_count))
    return {path: "dev" if path in dev_paths else "test" for path in paths}


def build_objects(seed):
    rows = []
    splits = split_sources(DIV2K_DIR.glob("*.png"), OBJECT_DEV_COUNT, seed)
    for path, split in splits.items():
        image_id = f"object_{path.stem}"
        image = crop_center(Image.open(path).convert("RGB"))
        image.save(HR_DIR / f"{image_id}.png")
        rows.append({"image_id": image_id, "content": "objects", "split": split, "source": f"DIV2K/{path.name}"})
    return rows


def build_faces(seed):
    rows = []
    splits = split_sources(FFHQ_DIR.glob("*.png"), FACE_DEV_COUNT, seed)
    for path, split in splits.items():
        image_id = f"face_{path.stem}"
        image = Image.open(path).convert("RGB").resize((HR_SIZE, HR_SIZE), Image.BICUBIC)
        image.save(HR_DIR / f"{image_id}.png")
        rows.append({"image_id": image_id, "content": "faces", "split": split, "source": f"FFHQ/{path.name}"})
    return rows


def build_text():
    rows = []
    text_sets = {
        "dev": render_text_set(TEXT_DEV_REPEATS, TEXT_DEV_SEED),
        "test": render_text_set(TEXT_TEST_REPEATS, TEXT_TEST_SEED),
    }
    for split, samples in text_sets.items():
        for index, sample in enumerate(samples):
            image_id = f"text_{split}_{index:03d}"
            sample["image"].save(HR_DIR / f"{image_id}.png")
            rows.append({
                "image_id": image_id,
                "content": "text",
                "split": split,
                "source": f"rendered/{image_id}",
                "font": sample["font"],
                "font_size": sample["font_size"],
                "kind": sample["kind"],
                "text": sample["text"],
            })
    return rows


def compress_jpeg(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=JPEG_SUBSAMPLING)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def make_low_resolution(image_id):
    high_resolution = Image.open(HR_DIR / f"{image_id}.png")
    low_size = HR_SIZE // SCALE
    low_resolution = high_resolution.resize((low_size, low_size), Image.BICUBIC)
    for quality in JPEG_QUALITIES:
        degraded = low_resolution if quality == "none" else compress_jpeg(low_resolution, quality)
        degraded.save(LR_DIR / f"jpeg_{quality}" / f"{image_id}.png")


def check_leakage(manifest):
    dev = manifest[manifest["split"] == "dev"]
    test = manifest[manifest["split"] == "test"]
    shared_sources = set(dev["source"]) & set(test["source"])
    shared_texts = set(dev["text"].dropna()) & set(test["text"].dropna())
    assert not shared_sources, f"sources in both splits: {shared_sources}"
    assert not shared_texts, "identical text in dev and test"
    return {"shared_sources": len(shared_sources), "shared_texts": len(shared_texts)}


def main(seed=42):
    HR_DIR.mkdir(parents=True, exist_ok=True)
    for quality in JPEG_QUALITIES:
        (LR_DIR / f"jpeg_{quality}").mkdir(parents=True, exist_ok=True)

    rows = build_objects(seed) + build_faces(seed) + build_text()
    manifest = pd.DataFrame(rows)
    for image_id in manifest["image_id"]:
        make_low_resolution(image_id)
    manifest.to_csv(PROCESSED_DIR / "manifest.csv", index=False)

    counts = manifest.groupby(["content", "split"]).size().unstack().to_dict(orient="index")
    report = {
        "images": len(manifest),
        "images_per_content_and_split": counts,
        "hr_size": HR_SIZE,
        "lr_size": HR_SIZE // SCALE,
        "jpeg_qualities": JPEG_QUALITIES,
        "lr_images": len(manifest) * len(JPEG_QUALITIES),
        "leakage_check": check_leakage(manifest),
        "split": "by source image; dev and test text rendered with different seeds",
        "seed": seed,
    }
    report_json = json.dumps(report, indent=2)
    (PROCESSED_DIR / "data_report.json").write_text(report_json)
    print(report_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
