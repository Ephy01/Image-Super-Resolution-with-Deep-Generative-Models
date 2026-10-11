import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "datasets"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "models"))
from prepare_data import HR_DIR, JPEG_QUALITIES, LR_DIR, PROCESSED_DIR
from upscale import METHODS, OUTPUTS_DIR, RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
CONTENTS = ["objects", "faces", "text"]
METRIC_LABELS = {"psnr": "PSNR (dB), higher is better", "ssim": "SSIM, higher is better", "lpips": "LPIPS, lower is better"}
METHOD_LABELS = {"bicubic": "Bicubic", "realesrnet": "Real-ESRNet"}
EXAMPLE_QUALITIES = ["none", 10]
CROP = (128, 128, 384, 384)


def plot_jpeg_sweep(summary, split):
    figure, axes = plt.subplots(len(METRIC_LABELS), len(CONTENTS), figsize=(11, 8), sharex=True)
    positions = range(len(JPEG_QUALITIES))
    for row, (metric, label) in enumerate(METRIC_LABELS.items()):
        for column, content in enumerate(CONTENTS):
            axis = axes[row, column]
            for method in METHODS:
                rows = summary[(summary["content"] == content) & (summary["method"] == method)]
                rows = rows.set_index("jpeg").loc[[str(quality) for quality in JPEG_QUALITIES]]
                axis.errorbar(positions, rows[metric], yerr=rows[f"{metric}_ci"], marker="o", capsize=3, label=METHOD_LABELS[method])
            axis.set_xticks(list(positions), [str(quality) for quality in JPEG_QUALITIES])
            axis.grid(alpha=0.3)
            if row == 0:
                axis.set_title(content)
            if column == 0:
                axis.set_ylabel(label)
            if row == len(METRIC_LABELS) - 1:
                axis.set_xlabel("JPEG quality of the input")
    axes[0, 0].legend()
    figure.suptitle(f"Baselines on the {split} split (mean and 95% confidence interval)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"jpeg_sweep_{split}.png", dpi=150)
    plt.close(figure)


def plot_examples(manifest, split):
    images = manifest[manifest["split"] == split]
    examples = [images[images["content"] == content].iloc[0] for content in CONTENTS]
    small_text = images[(images["content"] == "text") & (images["font_size"] == 24) & (images["kind"] == "words")]
    examples[2] = small_text.iloc[0]

    for quality in EXAMPLE_QUALITIES:
        columns = ["Input (128x128)"] + [METHOD_LABELS[method] for method in METHODS] + ["Ground truth"]
        figure, axes = plt.subplots(len(examples), len(columns), figsize=(12, 9.4))
        for row, example in enumerate(examples):
            low_resolution = Image.open(LR_DIR / f"jpeg_{quality}" / f"{example.image_id}.png")
            panels = [low_resolution.resize((512, 512), Image.NEAREST)]
            for method in METHODS:
                panels.append(Image.open(OUTPUTS_DIR / split / method / f"jpeg_{quality}" / f"{example.image_id}.png"))
            panels.append(Image.open(HR_DIR / f"{example.image_id}.png"))
            for column, panel in enumerate(panels):
                axes[row, column].imshow(panel.crop(CROP))
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(columns[column])
        figure.suptitle(f"Central 256x256 crops, JPEG quality of the input: {quality}")
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / f"examples_{split}_jpeg_{quality}.png", dpi=120)
        plt.close(figure)


def main(split="dev"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS_DIR / f"summary_{split}.csv", dtype={"jpeg": str})
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    plot_jpeg_sweep(summary, split)
    plot_examples(manifest, split)
    print(sorted(path.name for path in FIGURES_DIR.glob(f"*{split}*.png")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)
