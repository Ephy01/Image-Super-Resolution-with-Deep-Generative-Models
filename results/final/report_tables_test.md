### Main table, no JPEG

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

### Main table, JPEG 10

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

### Paired differences

| Content | JPEG | Method (minus Real-ESRNet) | PSNR (dB) | LPIPS | MUSIQ | CER / ArcFace sim |
|---|---|---|---|---|---|---|
| Objects | none | Real-ESRGAN | -1.89 [-2.06, -1.73] | -0.113 [-0.129, -0.098] | +8.3 [7.0, 9.7] | n/a |
| Objects | none | ResShift | -0.98 [-1.20, -0.77] | -0.164 [-0.181, -0.148] | +7.6 [6.4, 8.9] | n/a |
| Faces | none | Real-ESRGAN | -2.43 [-2.56, -2.30] | -0.080 [-0.085, -0.074] | +9.3 [8.6, 10.0] | -0.023 [-0.025, -0.021] |
| Faces | none | ResShift | -1.36 [-1.52, -1.21] | -0.133 [-0.139, -0.126] | +10.1 [9.3, 10.9] | -0.004 [-0.006, -0.003] |
| Text | none | Real-ESRGAN | -1.47 [-1.59, -1.34] | -0.077 [-0.095, -0.061] | +5.6 [5.0, 6.3] | +0.120 [0.094, 0.146] |
| Text | none | ResShift | +0.47 [0.36, 0.58] | -0.098 [-0.119, -0.078] | +4.6 [3.8, 5.5] | +0.023 [0.013, 0.033] |
| Objects | 10 | Real-ESRGAN | -1.18 [-1.28, -1.08] | -0.155 [-0.172, -0.138] | +19.8 [18.1, 21.5] | n/a |
| Objects | 10 | ResShift | -0.86 [-1.02, -0.72] | -0.161 [-0.181, -0.141] | +16.0 [13.9, 18.1] | n/a |
| Faces | 10 | Real-ESRGAN | -1.13 [-1.20, -1.06] | -0.112 [-0.118, -0.106] | +19.8 [18.3, 21.3] | -0.048 [-0.054, -0.041] |
| Faces | 10 | ResShift | -0.63 [-0.70, -0.57] | -0.135 [-0.143, -0.127] | +23.6 [22.2, 25.0] | -0.045 [-0.053, -0.036] |
| Text | 10 | Real-ESRGAN | -1.27 [-1.33, -1.21] | -0.194 [-0.224, -0.165] | +12.7 [11.7, 13.8] | +0.021 [0.002, 0.040] |
| Text | 10 | ResShift | -0.68 [-0.78, -0.58] | -0.214 [-0.252, -0.176] | +10.3 [9.2, 11.5] | +0.051 [0.035, 0.067] |

### JPEG hypothesis

| Method | Content | PSNR: drop 30 to 10 minus drop 50 to 30 (dB) | SSIM | LPIPS | Verdict |
|---|---|---|---|---|---|
| Bicubic | objects | +0.91 [0.83, 0.99] | +0.030 [0.028, 0.032] | +0.038 [0.032, 0.044] | supported |
| Bicubic | faces | +1.37 [1.31, 1.43] | +0.028 [0.027, 0.029] | +0.059 [0.056, 0.063] | supported |
| Bicubic | text | +0.50 [0.46, 0.55] | +0.021 [0.019, 0.023] | +0.074 [0.067, 0.081] | supported |
| Real-ESRNet | objects | +0.91 [0.81, 1.02] | +0.032 [0.029, 0.036] | +0.059 [0.053, 0.065] | supported |
| Real-ESRNet | faces | +1.38 [1.29, 1.47] | +0.022 [0.021, 0.023] | +0.050 [0.047, 0.053] | supported |
| Real-ESRNet | text | +1.26 [1.11, 1.40] | +0.072 [0.068, 0.076] | +0.132 [0.119, 0.146] | supported |
| Real-ESRGAN | objects | +0.73 [0.58, 0.89] | +0.025 [0.019, 0.031] | +0.070 [0.062, 0.080] | supported |
| Real-ESRGAN | faces | +1.17 [1.05, 1.29] | +0.021 [0.019, 0.022] | +0.064 [0.059, 0.069] | supported |
| Real-ESRGAN | text | +1.27 [1.09, 1.45] | +0.068 [0.064, 0.072] | +0.064 [0.050, 0.080] | supported |
| ResShift | objects | +0.69 [0.58, 0.79] | +0.028 [0.022, 0.034] | +0.079 [0.070, 0.090] | supported |
| ResShift | faces | +1.03 [0.95, 1.10] | +0.019 [0.017, 0.021] | +0.059 [0.054, 0.065] | supported |
| ResShift | text | +1.11 [0.96, 1.28] | +0.072 [0.068, 0.076] | +0.042 [0.032, 0.051] | supported |

### Runtime

| Method | Device | Precision | Mean s/image (test) | Median s/image (test) | Peak GPU memory (MiB) | Wall time dev (s) | Wall time test (s) |
|---|---|---|---:|---:|---:|---:|---:|
| Bicubic | CPU (Kaggle) | uint8 (Pillow, CPU) | 0.003 | 0.003 | n/a | 17 | 76 |
| Real-ESRNet | Tesla T4 | float32 | 0.156 | 0.159 | 208 | 58 | 251 |
| Real-ESRGAN | Tesla T4 | float32 | 0.161 | 0.161 | 208 | 67 | 291 |
| ResShift | Tesla T4 | float32 weights, float16 autocast | 0.890 | 0.889 | 2647 | 265 | 1098 |

Split wall time (all methods of one notebook run, s): dev 409; test 1718, 4327

### Seeds

| Content | JPEG | Metric | Mean over seeds: min to max | Spread (max - min) | Mean per-image std |
|---|---|---|---|---:|---:|
| objects | none | psnr | 26.53 to 26.57 | 0.043 | 0.187 |
| objects | none | lpips | 0.189 to 0.192 | 0.0035 | 0.0082 |
| objects | none | musiq | 65.6 to 66.0 | 0.34 | 1.01 |
| faces | none | psnr | 31.28 to 31.32 | 0.047 | 0.222 |
| faces | none | lpips | 0.113 to 0.114 | 0.0014 | 0.0046 |
| faces | none | musiq | 72.4 to 72.6 | 0.23 | 0.96 |
| faces | none | arcface_sim | 0.971 to 0.972 | 0.0009 | 0.0034 |
| text | none | psnr | 19.72 to 19.74 | 0.015 | 0.113 |
| text | none | lpips | 0.086 to 0.086 | 0.0005 | 0.0030 |
| text | none | musiq | 54.0 to 54.1 | 0.13 | 0.43 |
| text | none | cer | 0.241 to 0.244 | 0.0029 | 0.0118 |
| objects | 10 | psnr | 22.94 to 22.98 | 0.041 | 0.159 |
| objects | 10 | lpips | 0.432 to 0.435 | 0.0034 | 0.0126 |
| objects | 10 | musiq | 56.1 to 57.5 | 1.45 | 2.69 |
| faces | 10 | psnr | 27.15 to 27.19 | 0.038 | 0.159 |
| faces | 10 | lpips | 0.315 to 0.318 | 0.0025 | 0.0105 |
| faces | 10 | musiq | 59.9 to 60.9 | 1.02 | 3.45 |
| faces | 10 | arcface_sim | 0.611 to 0.617 | 0.0058 | 0.0275 |
| text | 10 | psnr | 14.92 to 14.96 | 0.046 | 0.121 |
| text | 10 | lpips | 0.194 to 0.196 | 0.0020 | 0.0080 |
| text | 10 | musiq | 50.5 to 51.0 | 0.49 | 1.36 |
| text | 10 | cer | 0.621 to 0.624 | 0.0028 | 0.0251 |
