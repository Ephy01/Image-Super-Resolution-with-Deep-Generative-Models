# ResShift dev run

The executed experiment is `notebooks/ResShift_only_dev.ipynb`. Its exact input archive is `data/resshift_dev_bundle.zip` (SHA-256 `0b9ea1766c24b96dd95f2f06521f9f1aef81b4cebd622b3d583e336549bd2eea`). The archive contains the original 64 dev HR images, 256 LR inputs, source code and baseline tables from `896fe8c`.

Only ResShift inference and its PSNR/SSIM/LPIPS evaluation were run. The three earlier methods were reused from the archive. The parent `results/` tables combine these 256 new rows with the original 768 baseline rows; CSV serialization can change the last decimal digit.

- `settings.json`: actual GPU, precision, commits, seeds and checkpoint hashes. The global/bootstrap seed is 42. Sampling uses `crc32(image_id.encode('utf-8')) & 0x7fffffff`, unchanged across JPEG levels.
- `metrics_resshift_dev.csv`, `summary_resshift_dev.csv`, `paired_resshift_dev.csv`: the exported diffusion results and paired differences.
- `timing_resshift_dev.csv`: 256 synchronized inference measurements, excluding image loading/saving, after three warm-up calls.
- `baseline_upscale_report_dev.json`: original baseline hardware and timings. These are not directly comparable with the T4 diffusion timings.
- `manifest.csv` and `bundle_manifest.json`: exact input manifest and hashes for files inside the input archive. The full manifest includes test rows, but the archive contains only dev image files.
- `environment_colab.txt`: the exported Colab environment, for reference. It contains Colab-specific paths and is not a portable installation requirements file; use the notebook's setup cell.
- `artifacts_sha256.json`: hashes of the supplied notebook, input/result ZIPs and every file in the original result ZIP, including all 256 output PNGs.

Full-resolution model outputs and the result ZIP remain outside Git, following the repository's `outputs/` convention. The notebook exports them again after a run. The committed figures and executed notebook show the visual results.

The notebook contains only code cells and retains its original outputs. Existing model and confidence-interval code is unchanged. The stored intervals are the project's bootstrap half-widths, not separately reported lower and upper percentile endpoints. This run covers one sampling seed per image; it is not a multiple-seed experiment.
