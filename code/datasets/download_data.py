import argparse
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DIV2K_DIR = RAW_DIR / "DIV2K_valid_HR"

FFHQ_URL = "https://huggingface.co/datasets/marcosv/ffhq-dataset/resolve/main/Part1/{:05d}.png"
FFHQ_DIR = RAW_DIR / "ffhq"
FFHQ_POOL = range(0, 1000)
FACE_COUNT = 120

WEIGHTS = {
    "RealESRNet_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


def download_file(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".part")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_bytes = int(response.headers.get("content-length", 0))
    with open(partial_path, "wb") as file, tqdm(total=total_bytes, unit="B", unit_scale=True, desc=path.name) as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            file.write(chunk)
            bar.update(len(chunk))
    partial_path.rename(path)


def download_div2k():
    if DIV2K_DIR.exists() and len(list(DIV2K_DIR.glob("*.png"))) == 100:
        return
    archive_path = RAW_DIR / "DIV2K_valid_HR.zip"
    download_file(DIV2K_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(RAW_DIR)
    archive_path.unlink()


def choose_face_ids(seed):
    generator = random.Random(seed)
    return sorted(generator.sample(list(FFHQ_POOL), FACE_COUNT))


def download_faces(seed):
    for face_id in tqdm(choose_face_ids(seed), desc="FFHQ"):
        download_file(FFHQ_URL.format(face_id), FFHQ_DIR / f"{face_id:05d}.png")


def download_weights():
    for name, url in WEIGHTS.items():
        download_file(url, MODELS_DIR / name)


def main(seed=42):
    download_div2k()
    download_faces(seed)
    download_weights()
    print(f"DIV2K images: {len(list(DIV2K_DIR.glob('*.png')))}")
    print(f"FFHQ images: {len(list(FFHQ_DIR.glob('*.png')))}")
    print(f"Weights: {sorted(path.name for path in MODELS_DIR.glob('*.pth'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
