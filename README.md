# Image Super-Resolution with Deep Generative Models

Computer Vision 2026 course project. Team: Iuliia Kurnaeva, Alena Petrenko, Amaliya Kharisova, Dmitrii Kuznetsov.

**Question.** How much fidelity do pretrained generative ×4 super-resolution models give up for perceptual quality compared with a non-generative baseline, and how does this depend on image content (text, faces, objects) and JPEG compression strength?

All models are pretrained. Nothing is trained in this project.

## Current state

| Stage | Status |
|---|---|
| Data loading and preprocessing | done |
| Baselines: bicubic interpolation and Real-ESRNet | done, evaluated on dev and test splits |
| GAN: Real-ESRGAN | done, evaluated on dev and test splits |
| Diffusion: ResShift | integrated and evaluated on dev and test splits |
| PSNR, SSIM and LPIPS evaluation | done |
| JPEG degradation sweep | done for quality none, 50, 30 and 10 |
| Perceptual and task metrics (MUSIQ, OCR error rate, face identity) | later |

Stage reports are in [`reports/`](reports/).

## Data

340 ground-truth images of 512×512 pixels in three content types.

| Content | Source | Test | Dev |
|---|---|---:|---:|
| Objects & scenes | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) validation images 0801–0900, central 512×512 crop | 80 | 20 |
| Faces | 120 images sampled from [FFHQ](https://github.com/NVlabs/ffhq-dataset) 00000–00999, resized 1024 → 512 | 100 | 20 |
| Text | Rendered by us: 3 fonts × 4 font sizes (16, 24, 32, 48 px) × dictionary words or random codes | 96 | 24 |

Each ground-truth image is reduced ×4 with bicubic interpolation to 128×128 and then JPEG-compressed at quality none, 50, 30 and 10. Every image gets every JPEG level, which gives 1,360 low-resolution inputs.

There is no training split: all restoration models use pretrained weights. The dev split contains 64 images and is used for debugging and preliminary comparisons. The test split contains 276 images and is used for the final evaluation. The split was generated with seed 42, with no overlap between dev and test source images.

## Pipeline

1. `code/datasets/download_data.py`: download the DIV2K validation set (449 MB), 120 FFHQ images chosen with a fixed seed, and the Real-ESRNet and Real-ESRGAN weights (64 MB each). Files that already exist are skipped.

2. `code/datasets/prepare_data.py`: build the ground-truth images, split them into dev and test by source image, create the low-resolution inputs, and check that no source image or text appears in both splits. Save `manifest.csv` (one row per image) and `data_report.json`.

3. `code/models/upscale.py`: upscale every low-resolution input of a split ×4 with bicubic interpolation, Real-ESRNet, Real-ESRGAN and ResShift diffusion. Real-ESRNet and Real-ESRGAN use the same RRDBNet network and training data and differ only in the loss (L1 vs. L1 + perceptual + GAN), so comparing them isolates the effect of generative training. Save the outputs as PNG and the timing to `results/upscale_report_<split>.json`.

   ResShift is integrated from the official implementation at a pinned commit (`bb03b7d`). The setup script is `code/models/setup_resshift.sh`. Pretrained checkpoints are downloaded by `code/datasets/download_data.py`. The diffusion sampler uses a stable per-image seed that is reused across JPEG conditions, making comparisons between compression levels more reproducible.

4. `code/evaluation/evaluate.py`: compare each output with its ground truth. PSNR and SSIM are computed on the Y channel with a 4-pixel border crop (the standard super-resolution protocol); LPIPS uses AlexNet. The script saves per-image metrics and aggregate summaries. Check the generated evaluation reports for the exact confidence-interval and paired-comparison settings.

5. `code/evaluation/make_figures.py`: plot the metrics against JPEG quality and save example comparisons.

`notebooks/data_overview.ipynb` shows the prepared data.

## Pipeline flow

1. `download_data.py`: `download_div2k` → `choose_face_ids` → `download_faces` → `download_weights`.
2. `prepare_data.py`: `build_objects` (`crop_center`) → `build_faces` → `build_text` (`render_text_set` from `render_text.py`) → `make_low_resolution` (`compress_jpeg`) → `check_leakage` → save the manifest and the report.
3. `upscale.py`: `pick_device` → load RRDBNet or ResShift → `run_method` (`upscale_bicubic`, `upscale_rrdbnet`, `ResShiftModel.upscale`) → save outputs and the report.
4. `evaluate.py`: `check_metrics` → `evaluate_split` (`compute_psnr`, `compute_ssim`, `compute_lpips`) → `summarize` → `compare_methods` → save the tables.
5. `make_figures.py`: `plot_jpeg_sweep` → `plot_examples`.

## Run

Tested with Python 3.14. The code uses an NVIDIA GPU if there is one, then an Apple GPU, then the CPU. Run the following commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run_pipeline.sh
```

`run_pipeline.sh` automates the pipeline stages configured in the script. To evaluate both dev and test splits, run the corresponding commands for each split as shown below. The pipeline includes the pinned ResShift implementation, checkpoint download, dataset preparation, upscaling, metric evaluation and figure generation. ResShift is substantially slower than the RRDBNet-based methods; runtime depends on the available accelerator. Each run may overwrite existing outputs and result tables.

The steps can also be run one at a time:

```bash
python code/datasets/download_data.py
python code/datasets/prepare_data.py

# Set up the official ResShift implementation once
bash code/models/setup_resshift.sh

# Run the dev split
python code/models/upscale.py --split dev
python code/evaluation/evaluate.py --split dev
python code/evaluation/make_figures.py --split dev

# Run the test split
python code/models/upscale.py --split test
python code/evaluation/evaluate.py --split test
python code/evaluation/make_figures.py --split test
```

## Preliminary results (dev split, 64 images; before ResShift integration)

Mean over images. Input without JPEG compression:

| Content | Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| Objects | Bicubic | 25.62 | 0.689 | 0.488 |
| Objects | Real-ESRNet | 26.31 | 0.737 | 0.376 |
| Objects | Real-ESRGAN | 24.44 | 0.682 | 0.255 |
| Faces | Bicubic | 33.40 | 0.879 | 0.289 |
| Faces | Real-ESRNet | 33.76 | 0.894 | 0.238 |
| Faces | Real-ESRGAN | 31.05 | 0.859 | 0.159 |
| Text | Bicubic | 16.46 | 0.707 | 0.436 |
| Text | Real-ESRNet | 19.31 | 0.838 | 0.186 |
| Text | Real-ESRGAN | 17.83 | 0.818 | 0.107 |

Real-ESRGAN has the lowest LPIPS for every content type but loses 1.5–2.7 dB PSNR against Real-ESRNet, the same network trained without the GAN loss. These are preliminary results from the earlier three-method experiment and do not include ResShift.

All JPEG levels: [`results/summary_dev.csv`](results/summary_dev.csv). Paired differences: [`results/paired_dev.csv`](results/paired_dev.csv).

![Metrics against JPEG quality](results/figures/jpeg_sweep_dev.png)

![Example comparison](results/figures/examples_dev_jpeg_none.png)

## Test results after ResShift integration

Mean over test images. Input without JPEG recompression:

| Content | Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| Objects | Bicubic | 26.6937 | 0.7182 | 0.4648 |
| Objects | Real-ESRNet | 27.5370 | 0.7653 | 0.3569 |
| Objects | Real-ESRGAN | 25.6481 | 0.7085 | 0.2436 |
| Objects | ResShift | 26.5597 | 0.7309 | 0.1925 |
| Faces | Bicubic | 32.0756 | 0.8688 | 0.2961 |
| Faces | Real-ESRNet | 32.6824 | 0.8864 | 0.2452 |
| Faces | Real-ESRGAN | 30.2526 | 0.8518 | 0.1657 |
| Faces | ResShift | 31.3226 | 0.8578 | 0.1126 |
| Text | Bicubic | 16.4075 | 0.7029 | 0.4383 |
| Text | Real-ESRNet | 19.2828 | 0.8363 | 0.1876 |
| Text | Real-ESRGAN | 17.8242 | 0.8169 | 0.1086 |
| Text | ResShift | 19.7180 | 0.8564 | 0.0866 |

Real-ESRNet achieves the highest PSNR and SSIM for objects and faces. ResShift achieves the highest PSNR and SSIM for text and the lowest LPIPS across all three content categories. Real-ESRGAN also achieves lower LPIPS than Real-ESRNet in all three categories, despite lower PSNR and SSIM. These results illustrate the trade-off between distortion-based and perceptual metrics; no method is the universal winner across all metrics.

These values describe only the test images without JPEG recompression. Results for JPEG qualities 50, 30 and 10 should be taken from the corresponding evaluation tables.

### Inference time

Average runtime per image on an NVIDIA Tesla T4:

| Method | Time (seconds/image) ↓ |
|---|---:|
| Bicubic | 0.00345 |
| Real-ESRNet | 0.54590 |
| Real-ESRGAN | 0.55045 |
| ResShift | 0.91256 |

Bicubic interpolation is the fastest method, while ResShift has the highest inference time in this environment. Runtime depends on the hardware and software configuration.

## Repository layout

```text
code/datasets/      download, text rendering, data preparation
code/models/        RRDBNet definition, upscaling script and ResShift setup
code/evaluation/    metrics and figures
data/processed/     manifest.csv and data_report.json (images are regenerated by the pipeline)
models/             downloaded weights (not tracked)
third_party/        pinned ResShift source (not tracked)
notebooks/          data overview
reports/            stage reports
results/            metric tables and figures
outputs/            super-resolution outputs by split, method and JPEG condition
```

## Team

| Member | Builds | Analyses and presents |
|---|---|---|
| Amaliya Kharisova | Dataset download, degradation pipeline, text generator, splits | Text |
| Dmitrii Kuznetsov | Bicubic, Real-ESRNet, Real-ESRGAN, runtime | Objects & scenes |
| Iuliia Kurnaeva | ResShift, random-seed experiment | Faces |
| Alena Petrenko | Metrics, statistics, figures, README & reproducibility | JPEG compression |

## Diffusion integration notes

ResShift is loaded from the official repository at a pinned commit (`bb03b7d`). The setup script clones that source into `third_party/ResShift/` (ignored by Git); the two pretrained checkpoints are downloaded into `models/` by `code/datasets/download_data.py`. ResShift outputs are saved under `outputs/<split>/resshift/jpeg_<quality>/`, following the same output layout as the other methods. Its stochastic sampler receives a stable per-image seed, reused across JPEG conditions.

The completed dev and test evaluations include all four methods and all four JPEG conditions. Results and generated artifacts are stored under `results/`, while the upscaled images are stored under `outputs/`. The test metrics without JPEG recompression are summarized above; full results for all degradation levels are available in the generated CSV files.

## Credits

- Real-ESRNet / Real-ESRGAN: X. Wang et al., ICCVW 2021, [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3). The RRDBNet definition follows [BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0).
- DIV2K: Agustsson & Timofte, CVPRW 2017 (academic use). FFHQ: Karras et al., CVPR 2019 (CC BY-NC-SA 4.0), downloaded from the [marcosv/ffhq-dataset](https://huggingface.co/datasets/marcosv/ffhq-dataset) mirror.
- LPIPS: Zhang et al., CVPR 2018.
