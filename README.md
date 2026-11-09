# Image Super-Resolution with Deep Generative Models

![Python](https://img.shields.io/badge/python-3.11+-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.14-red) ![GPU](https://img.shields.io/badge/GPU-T4%20(Colab%2FKaggle)-76b900) ![License](https://img.shields.io/badge/license-not%20decided-lightgrey)

Computer Vision 2026 course project. Team: Iuliia Kurnaeva, Alena Petrenko, Amaliya Kharisova, Dmitrii Kuznetsov.

**Question.** How much fidelity do pretrained generative ×4 super-resolution models give up for perceptual quality compared with a non-generative baseline, and how does this depend on image content (text, faces, objects) and JPEG compression strength?

**Hypothesis (stage 1 plan).** Generative models look sharper but score lower on PSNR, invent letters on text and alter identity on faces; all models degrade sharply below JPEG quality 30.

Four methods, all pretrained, nothing is trained: bicubic interpolation, Real-ESRNet (RRDBNet, L1 loss), Real-ESRGAN (the same RRDBNet and data, GAN loss) and ResShift v3 (diffusion, 4 sampling steps). Real-ESRNet and Real-ESRGAN differ only in the training loss, so their comparison isolates generative training.

---

## Environment Setup

1. Create the environment:

```bash
./setup.sh                      # creates .venv, installs requirements.txt, checks for Tesseract
source .venv/bin/activate
```

Tested on macOS 26 (Apple M4, MPS, Python 3.13.9, torch 2.14.1): every local step. Linux with CUDA is used through the Colab/Kaggle notebooks in `experiments/` (inference only); the stage 4 ResShift run used Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0 on a Tesla T4 (`results/resshift_dev/settings.json`). Each notebook writes its own Python, torch, CUDA and GPU versions to `results/runtime/environment_<notebook>.txt`. Windows is not supported natively: use WSL 2. The OCR metric needs the Tesseract binary (`brew install tesseract` or `sudo apt-get install tesseract-ocr`; the results use Tesseract 5.5.1 with the tessdata_fast `eng` model).

2. Download the data and the weights:

```bash
bash scripts/download_data.sh       # DIV2K validation (449 MB) + 120 FFHQ faces, then builds data/processed/
bash scripts/download_weights.sh    # 4 weight files (833 MB), verified against scripts/SHA256SUMS; pinned ResShift code
```

Paths resolve in one order everywhere: environment variable (`SR_DATA`, `SR_WEIGHTS`, `SR_OUTPUTS`, `SR_RESULTS`, `SR_THIRD_PARTY`, `SR_CACHE`), then the `paths:` block of the YAML config, then the default in [`paths.py`](paths.py).

---

## Repository Structure

```
.
├── configs/
│   ├── default.yaml          # working settings on dev, including the pre-registered analysis rules
│   ├── final.yaml            # frozen settings for the test run (written by code/evaluation/freeze_config.py)
│   └── final.yaml.sha256     # checked before every test run
├── code/
│   ├── datasets/             # download_data.py, prepare_data.py, render_text.py
│   ├── models/               # rrdbnet.py, resshift.py, upscale.py, setup_resshift.sh
│   └── evaluation/           # evaluate.py, extra_metrics.py (MUSIQ, OCR, ArcFace), select_settings.py,
│                             # freeze_config.py, reproduce.py, analyze.py, report_tables.py, make_figures.py (stages 2-4)
├── data/
│   ├── processed/            # manifest.csv, data_report.json, inputs_sha256.json (images not tracked)
│   ├── gpu_inputs_{dev,test}.zip   # low-resolution inputs for the notebooks
│   └── resshift_dev_bundle.zip     # archived stage 4 dev inputs
├── experiments/              # self-contained Colab/Kaggle T4 notebooks (01 dev, 02 test, 03 seeds)
├── notebooks/                # data overview; the executed stage 4 ResShift run
├── reports/                  # stage reports (PDF); stage5_final.md + assets/report.css
├── results/
│   ├── analysis/             # one JSON per analysis cited in the reports
│   ├── final/                # stage 5 tables (metrics, summary, paired, report_tables), figures, arcface_unclipped/
│   ├── runtime/              # timing, peak memory, environments, stage wall times
│   ├── resshift_dev/         # stage 4 ResShift provenance
│   ├── legacy_colab/         # earlier exploratory test run (see "Test split")
│   └── *_dev.csv, figures/   # stage 2-4 dev tables, unchanged
├── scripts/                  # download scripts, SHA256SUMS, make_gpu_inputs.py, build_notebooks.py, render_report.sh
├── prepare.py                # stage entry points: data,
├── inference.py              #   super-resolution,
├── evaluate.py               #   metrics,
├── analyze.py                #   analyses and figures
├── paths.py                  # path resolution (env > YAML > default)
├── setup.sh
├── run_pipeline.sh           # local dev pipeline
└── requirements.txt          # exact versions
```

The models are not trained, so there is no training script.

---

## Run Experiments

Every entry point takes `--config`; `--split test` runs only with `configs/final.yaml` whose SHA-256 matches `configs/final.yaml.sha256`. Times below come from `results/runtime/stage_times.json` and `results/runtime/upscale_report_*.json`.

| Step | Command | Hardware | Time |
|---|---|---|---|
| Data | `python prepare.py --config configs/final.yaml --only data` | Apple M4 CPU | 25 s (after the downloads) |
| Metric settings on dev | `python code/evaluation/select_settings.py` | Apple M4 CPU | not timed |
| Freeze settings | `python code/evaluation/freeze_config.py` | any | not timed |
| Inference, dev, 3 non-diffusion methods | `python inference.py --config configs/final.yaml --split dev --methods bicubic realesrnet realesrgan` | bicubic: M4 CPU; RRDBNet: M4 MPS | 6 s, 194 s, 186 s (another job ran at the same time) |
| Inference on the T4 | notebooks `experiments/01`–`03` (below) | Kaggle/Colab T4 | 409 s (dev, 4 methods), 1718 s (test, 4 methods), 4327 s (test, 4 ResShift seeds) |
| Evaluation | `python evaluate.py --config configs/final.yaml --split dev` | M4 CPU (LPIPS on MPS) | 648 s for 3 dev methods + HR with an empty metric cache, including the metric model downloads; test: about 1 s per image |
| Analyses | `python analyze.py --config configs/final.yaml --split dev` | M4 CPU | 6 s |

Each output folder `outputs/<split>/<method>/` has a `manifest.json` keyed by the weights SHA-256, the model code SHA-256, the settings and the SHA-256 of every input. `inference.py` skips a folder whose manifest matches and recomputes it otherwise; `evaluate.py` stops with a message if an output was made from other weights or other inputs. Metric values are cached per image hash under `data/cache/metric_values/`, keyed by the metric settings.

### GPU notebooks (free Colab or Kaggle T4)

No CUDA GPU is needed locally. Each notebook embeds the repository code it needs (checked by SHA-256), verifies the inputs and weights, and runs `inference.py` exactly as above.

1. `python scripts/make_gpu_inputs.py --split dev` (and `--split test`) builds `data/gpu_inputs_<split>.zip`.
2. Upload the archive and run the notebook:
   - `experiments/01_dev_all_methods_t4.ipynb`: all 4 methods on dev, T4 timing and peak memory, ResShift reproduction (input `gpu_inputs_dev.zip`);
   - `experiments/02_test_all_methods_t4.ipynb`: the single test run with the frozen settings (input `gpu_inputs_test.zip`);
   - `experiments/03_test_resshift_seeds_t4.ipynb`: ResShift seed offsets 1–4 on test (input `gpu_inputs_test.zip`).
3. Unzip each `<notebook>_results.zip` at the repository root, then:

```bash
python evaluate.py --config configs/final.yaml --split dev
python code/evaluation/reproduce.py
python evaluate.py --config configs/final.yaml --split test
python evaluate.py --config configs/final.yaml --split test --methods resshift_seed1 resshift_seed2 resshift_seed3 resshift_seed4 --tag _seeds
python analyze.py --config configs/final.yaml --split dev
python analyze.py --config configs/final.yaml --split test
python code/evaluation/report_tables.py --split test      # tables of reports/stage5_final.md
bash scripts/render_report.sh reports/stage5_final.md       # PDF via pandoc and headless Chrome
```

The stage 4 ResShift run is `notebooks/ResShift_only_dev.ipynb` with its archived inputs `data/resshift_dev_bundle.zip`; its provenance is in [`results/resshift_dev/`](results/resshift_dev/).

---

## Data

340 ground-truth images of 512×512 pixels in three content types.

| Content | Source | Test | Dev |
|---|---|---:|---:|
| Objects & scenes | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) validation images 0801–0900, central 512×512 crop | 80 | 20 |
| Faces | 120 images sampled from [FFHQ](https://github.com/NVlabs/ffhq-dataset) 00000–00999, resized 1024 → 512 | 100 | 20 |
| Text | Rendered by us: 3 fonts × 4 font sizes (16, 24, 32, 48 px) × dictionary words or random codes | 96 | 24 |

Each ground-truth image is reduced ×4 with bicubic interpolation to 128×128 and JPEG-compressed at quality none, 50, 30 and 10 (4:2:0): 1,360 inputs. There is no training split. Dev (64 images) is used for debugging and for choosing every setting; test (276 images) is for the final numbers. The split is by source image, with seed 42.

Text rendering uses Pillow's BASIC layout engine. With libraqm installed, Pillow switches to RAQM by default, which changed all 24 dev text images and their line breaks; with BASIC, all 320 dev files match the archived stage 4 inputs byte for byte and `manifest.csv` is unchanged (`results/analysis/reproduction_dev.json`, `data/processed/inputs_sha256.json`).

**Test split.** An exploratory all-method run on test exists from commit `cc9a873` (`results/legacy_colab/`). The test split is therefore not untouched. No setting was chosen with those numbers; the final test run uses `configs/final.yaml`, frozen on 2026-11-01 from dev-only selections. The test notebook ran twice because the first outputs were lost in a Kaggle restart before any metric was computed.

---

## Model Performance

Test split, 276 images, frozen settings (`configs/final.yaml`). Means over images; best per content and condition in bold. PSNR/SSIM on the Y channel with a 4-pixel border crop; LPIPS AlexNet; MUSIQ (pyiqa, KonIQ-10k checkpoint), higher is better; CER: Tesseract 5.5.1 `--psm 4 --oem 1`, chosen on dev HR text; ArcFace: buffalo_l similarity to the HR face and match rate at threshold 0.182 (1% false match rate on 190 dev impostor pairs).

**No JPEG**

| Content | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | CER ↓ / ArcFace sim ↑ | Match rate ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Objects | Bicubic | 26.69 | 0.718 | 0.465 | 33.1 | n/a | n/a |
| Objects | Real-ESRNet | **27.54** | **0.765** | 0.357 | 58.1 | n/a | n/a |
| Objects | Real-ESRGAN | 25.65 | 0.709 | 0.244 | **66.4** | n/a | n/a |
| Objects | ResShift | 26.56 | 0.731 | **0.192** | 65.7 | n/a | n/a |
| Objects | HR reference | n/a | n/a | n/a | 68.6 | n/a | n/a |
| Faces | Bicubic | 32.08 | 0.869 | 0.296 | 41.5 | **0.983** | **1.00** |
| Faces | Real-ESRNet | **32.68** | **0.886** | 0.245 | 62.3 | 0.977 | 1.00 |
| Faces | Real-ESRGAN | 30.25 | 0.852 | 0.166 | 71.5 | 0.954 | 1.00 |
| Faces | ResShift | 31.32 | 0.858 | **0.113** | **72.4** | 0.972 | 1.00 |
| Faces | HR reference | n/a | n/a | n/a | 73.6 | n/a | n/a |
| Text | Bicubic | 16.42 | 0.706 | 0.436 | 35.5 | **0.210** | n/a |
| Text | Real-ESRNet | 19.25 | 0.838 | 0.184 | 49.4 | 0.220 | n/a |
| Text | Real-ESRGAN | 17.79 | 0.819 | 0.107 | **55.0** | 0.340 | n/a |
| Text | ResShift | **19.72** | **0.859** | **0.086** | 54.0 | 0.243 | n/a |
| Text | HR reference | n/a | n/a | n/a | 57.4 | 0.019 | n/a |

**JPEG 10**

| Content | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | CER ↓ / ArcFace sim ↑ | Match rate ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Objects | Bicubic | 23.52 | 0.570 | 0.758 | 19.6 | n/a | n/a |
| Objects | Real-ESRNet | **23.85** | **0.609** | 0.595 | 40.7 | n/a | n/a |
| Objects | Real-ESRGAN | 22.67 | 0.571 | 0.440 | **60.5** | n/a | n/a |
| Objects | ResShift | 22.98 | 0.552 | **0.434** | 56.7 | n/a | n/a |
| Objects | HR reference | n/a | n/a | n/a | 68.6 | n/a | n/a |
| Faces | Bicubic | 27.38 | 0.754 | 0.642 | 20.4 | **0.689** | 0.96 |
| Faces | Real-ESRNet | **27.83** | **0.789** | 0.453 | 36.9 | 0.656 | **1.00** |
| Faces | Real-ESRGAN | 26.70 | 0.762 | 0.340 | 56.7 | 0.608 | 1.00 |
| Faces | ResShift | 27.19 | 0.755 | **0.318** | **60.5** | 0.611 | 1.00 |
| Faces | HR reference | n/a | n/a | n/a | 73.6 | n/a | n/a |
| Text | Bicubic | 15.09 | 0.579 | 0.642 | 24.2 | 0.584 | n/a |
| Text | Real-ESRNet | **15.62** | **0.689** | 0.409 | 40.2 | **0.574** | n/a |
| Text | Real-ESRGAN | 14.35 | 0.680 | 0.215 | **52.9** | 0.594 | n/a |
| Text | ResShift | 14.95 | 0.668 | **0.195** | 50.5 | 0.624 | n/a |
| Text | HR reference | n/a | n/a | n/a | 57.4 | 0.019 | n/a |

Which numbers are verified:
- All 4 methods ran on a Tesla T4 (notebooks 01 and 02); every output folder carries a manifest with input and weight hashes, checked by `evaluate.py`.
- Dev reproduction against the stage 2-4 archive (`results/analysis/reproduction_dev.json`): bicubic, Real-ESRNet and Real-ESRGAN within 0.0049 dB PSNR, 0.0002 SSIM and 0.0005 LPIPS per image. ResShift is outside the per-image tolerance (at most 0.032 dB PSNR, 0.0013 LPIPS; mean absolute PSNR difference 0.0035 dB), probably because stage 4 used CUDA 13.0 and the rerun CUDA 12.8.
- ArcFace values come after one bug fix found on test (face boxes clipped to the image); the frozen threshold and all dev values are unchanged. Pre-fix tables: `results/final/arcface_unclipped/`.

All JPEG levels with 95% percentile bootstrap intervals [lo, hi]: [`results/final/summary_test.csv`](results/final/summary_test.csv); paired differences: [`results/final/paired_test.csv`](results/final/paired_test.csv); analyses: [`results/analysis/`](results/analysis/); report: [`reports/stage5_final.pdf`](reports/stage5_final.pdf).

Runtime on a Tesla T4 (test split, batch size 1, 3 warm-ups; bicubic on the notebook CPU):

| Method | Device | Precision | Mean s/image (test) | Median s/image (test) | Peak GPU memory (MiB) | Wall time dev (s) | Wall time test (s) |
|---|---|---|---:|---:|---:|---:|---:|
| Bicubic | CPU (Kaggle) | uint8 (Pillow, CPU) | 0.003 | 0.003 | n/a | 17 | 76 |
| Real-ESRNet | Tesla T4 | float32 | 0.156 | 0.159 | 208 | 58 | 251 |
| Real-ESRGAN | Tesla T4 | float32 | 0.161 | 0.161 | 208 | 67 | 291 |
| ResShift | Tesla T4 | float32 weights, float16 autocast | 0.890 | 0.889 | 2647 | 265 | 1098 |

---

## Tech Stack

* Python 3.13, PyTorch 2.14 (local), Colab/Kaggle PyTorch on the T4
* Pillow, NumPy, pandas, OpenCV, Matplotlib
* lpips (AlexNet), pyiqa (MUSIQ), Tesseract 5 + pytesseract + RapidFuzz (CER), insightface + ONNX Runtime (ArcFace)
* omegaconf, timm, einops (ResShift)

## Credits

- Real-ESRNet / Real-ESRGAN: X. Wang et al., ICCVW 2021, [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause). The RRDBNet definition follows [BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0).
- ResShift: Z. Yue, J. Wang, C. C. Loy, NeurIPS 2023 and TPAMI 2024, [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift), commit `bb03b7d`, weights `resshift_realsrx4_s4_v3.pth` and `autoencoder_vq_f4.pth`. NTU S-Lab License 1.0: non-commercial use only. The code is cloned at setup and not redistributed here.
- DIV2K: Agustsson & Timofte, CVPRW 2017 (academic use). FFHQ: Karras et al., CVPR 2019 (CC BY-NC-SA 4.0), downloaded from the [marcosv/ffhq-dataset](https://huggingface.co/datasets/marcosv/ffhq-dataset) mirror.
- LPIPS: Zhang et al., CVPR 2018. MUSIQ: Ke et al., ICCV 2021, via [pyiqa](https://github.com/chaofengc/IQA-PyTorch). ArcFace: Deng et al., CVPR 2019, buffalo_l models from [insightface](https://github.com/deepinsight/insightface) (non-commercial research use). Tesseract OCR (Apache-2.0).

## Team

| Member | Contribution |
|---|---|
| Amaliya Kharisova | ResShift, random-seed experiments |
| Dmitrii Kuznetsov | Dataset download, degradation pipeline, splits, Bicubic and Real-ESRNet |
| Iuliia Kurnaeva | Real-ESRGAN, runtime measurements |
| Alena Petrenko | Metrics, statistics, figures, README and reproducibility |

Stage reports: [`reports/`](reports/); final report: [`reports/stage5_final.pdf`](reports/stage5_final.pdf). No license file yet: the choice is open.
