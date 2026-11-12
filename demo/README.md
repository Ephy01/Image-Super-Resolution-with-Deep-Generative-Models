# Super-resolution demo for Apple Silicon

A local interface for Bicubic, Real-ESRNet, Real-ESRGAN and ResShift. Upload your own image, optionally add its high-resolution reference, and click a method to calculate its output. The page starts empty and does not replay saved experiment results.

## Requirements

- An Apple Silicon Mac running macOS 14 or newer.
- Native ARM64 Python 3.12–3.14, excluding 3.14.1. Python 3.12 is recommended.
- Git and an internet connection for the first setup.
- Space for the Python environment and about 833 MB of model weights. Allow several GB of free disk space.

All four methods run locally. Neural models run on the Mac's Apple GPU. Bicubic runs on CPU. The setup creates an isolated `.venv` in this folder and does not install packages into your system Python or the research environment.

## First setup

```bash
git clone --branch demo https://github.com/Ephy01/Image-Super-Resolution-with-Deep-Generative-Models.git
cd Image-Super-Resolution-with-Deep-Generative-Models/demo
bash setup.command
bash start.command
```

If you already have the repository, switch to the `demo` branch and run the last two commands from its `demo/` folder.

Setup installs the dependencies, downloads all four weight files, checks their SHA-256 hashes, and downloads the official ResShift source at commit `bb03b7d21614cace01787e097c8a6ab6b945227d`. It checks the installation without running inference. Later starts work offline with those files present.

To choose a Python installation explicitly:

```bash
DEMO_PYTHON=/path/to/python3.12 bash setup.command
```

The launcher opens http://127.0.0.1:8000. Keep its Terminal window open while using the demo. Press Ctrl+C to stop it. For another port, run `bash start.command --port 8001`.

## Use the demo

1. Upload a **128 × 128** PNG, JPEG or WebP input. This size works with all four methods.
2. Optionally add its matching **512 × 512 HR reference** before running a method.
3. Click Bicubic, Real-ESRNet, Real-ESRGAN or ResShift. The first use also loads that model's weights.
4. Switch the left preview between Input and HR reference. Download the selected output with Save PNG.

The reference must show the same crop at exactly four times the input width and height. It is used only for scoring. The model receives only the low-resolution input. Changing the input or reference clears the displayed results. Clicking an already completed method displays its cached output. Refreshing the page starts with an empty interface.

For a JPEG comparison, upload an uncompressed input first, then a JPEG-compressed copy of that same input. Add the same reference again after changing the input.

There is no automatic resizing. Bicubic accepts 16–512 pixels per side, Real-ESRNet and Real-ESRGAN 16–256, and ResShift 64–128. The server listens only on localhost. Uploaded images are kept in a temporary local folder until the server stops, with a limit of 24 uploads per server session.

## Scores and reproducibility

PSNR and SSIM are calculated only with an HR reference. They use BT.601 luminance and a four-pixel border crop. Higher is better. LPIPS, MUSIQ, OCR and ArcFace are not calculated in this demo.

Inference time excludes model loading, scoring and saving. Loading time is shown separately. The first inference can be slower. These interactive measurements are separate from the research benchmark.

ResShift uses the official v3 model and four sampling steps. The seed is fixed from the input dimensions and decoded RGB pixels, so renaming a file does not change it. MPS and CUDA outputs need not be identical. These demo outputs do not replace the project's evaluation results.

## Troubleshooting

If setup cannot find Python, install native ARM64 Python from [python.org](https://www.python.org/downloads/macos/) and rerun setup. Do not run the launcher under Rosetta with an Intel Python installation.

If model downloads fail, rerun `bash setup.command`. Already verified weight files are reused. If a method remains unavailable, run:

```bash
.venv/bin/python -B scripts/check_installation.py --device mps
```

If the `.venv` came from another computer or is incompatible, move it aside and run setup again. Virtual environments are recreated locally, not shared between computers.

Pretrained model credits and research results are in the [project README](../README.md). Model weights and the downloaded ResShift checkout are not tracked in Git.
