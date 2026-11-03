### Main table, no JPEG

| Content | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | CER ↓ / ArcFace sim ↑ | Match rate ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Objects | Bicubic | 25.62 | 0.689 | 0.488 | 35.7 | n/a | n/a |
| Objects | Real-ESRNet | **26.31** | **0.737** | 0.376 | 61.0 | n/a | n/a |
| Objects | Real-ESRGAN | 24.44 | 0.682 | 0.255 | **69.5** | n/a | n/a |
| Objects | ResShift | 25.47 | 0.701 | **0.212** | 68.0 | n/a | n/a |
| Objects | HR reference | n/a | n/a | n/a | 70.1 | n/a | n/a |
| Faces | Bicubic | 33.40 | 0.879 | 0.289 | 40.1 | **0.984** | **1.00** |
| Faces | Real-ESRNet | **33.76** | **0.894** | 0.238 | 62.7 | 0.978 | 1.00 |
| Faces | Real-ESRGAN | 31.05 | 0.859 | 0.159 | 70.7 | 0.953 | 1.00 |
| Faces | ResShift | 32.26 | 0.862 | **0.108** | **73.0** | 0.972 | 1.00 |
| Faces | HR reference | n/a | n/a | n/a | 74.6 | n/a | n/a |
| Text | Bicubic | 16.46 | 0.707 | 0.436 | 35.0 | **0.211** | n/a |
| Text | Real-ESRNet | 19.31 | 0.838 | 0.186 | 49.7 | 0.216 | n/a |
| Text | Real-ESRGAN | 17.84 | 0.818 | 0.107 | **54.9** | 0.338 | n/a |
| Text | ResShift | **19.76** | **0.858** | **0.085** | 54.0 | 0.241 | n/a |
| Text | HR reference | n/a | n/a | n/a | 57.2 | 0.017 | n/a |

### Main table, JPEG 10

| Content | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | CER ↓ / ArcFace sim ↑ | Match rate ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Objects | Bicubic | 22.77 | 0.543 | 0.767 | 20.5 | n/a | n/a |
| Objects | Real-ESRNet | **22.97** | **0.582** | 0.613 | 41.8 | n/a | n/a |
| Objects | Real-ESRGAN | 21.80 | 0.547 | **0.451** | **62.7** | n/a | n/a |
| Objects | ResShift | 22.26 | 0.528 | 0.456 | 57.7 | n/a | n/a |
| Objects | HR reference | n/a | n/a | n/a | 70.1 | n/a | n/a |
| Faces | Bicubic | 28.25 | 0.768 | 0.647 | 19.8 | **0.691** | 0.95 |
| Faces | Real-ESRNet | **28.72** | **0.802** | 0.456 | 38.0 | 0.638 | 0.95 |
| Faces | Real-ESRGAN | 27.65 | 0.776 | 0.342 | 56.9 | 0.597 | 0.95 |
| Faces | ResShift | 28.01 | 0.761 | **0.322** | **61.3** | 0.607 | **1.00** |
| Faces | HR reference | n/a | n/a | n/a | 74.6 | n/a | n/a |
| Text | Bicubic | 15.12 | 0.581 | 0.639 | 24.0 | 0.592 | n/a |
| Text | Real-ESRNet | **15.67** | **0.690** | 0.408 | 39.8 | **0.571** | n/a |
| Text | Real-ESRGAN | 14.38 | 0.681 | 0.212 | **53.2** | 0.604 | n/a |
| Text | ResShift | 14.96 | 0.669 | **0.195** | 50.7 | 0.636 | n/a |
| Text | HR reference | n/a | n/a | n/a | 57.2 | 0.017 | n/a |

### Paired differences

| Content | JPEG | Comparison | PSNR (dB) | LPIPS | MUSIQ | CER / ArcFace sim |
|---|---|---|---|---|---|---|
| Objects | none | Real-ESRGAN - Real-ESRNet | -1.87 [-2.25, -1.50] | -0.121 [-0.152, -0.094] | +8.5 [5.8, 11.3] | n/a |
| Objects | none | ResShift - Real-ESRNet | -0.84 [-1.11, -0.57] | -0.164 [-0.195, -0.133] | +6.9 [5.0, 9.0] | n/a |
| Faces | none | Real-ESRGAN - Real-ESRNet | -2.71 [-3.14, -2.30] | -0.079 [-0.092, -0.068] | +8.0 [7.0, 9.0] | -0.025 [-0.029, -0.021] |
| Faces | none | ResShift - Real-ESRNet | -1.51 [-1.85, -1.18] | -0.130 [-0.140, -0.119] | +10.3 [8.4, 12.4] | -0.006 [-0.010, -0.003] |
| Text | none | Real-ESRGAN - Real-ESRNet | -1.47 [-1.72, -1.23] | -0.079 [-0.119, -0.048] | +5.2 [4.1, 6.5] | +0.122 [0.075, 0.175] |
| Text | none | ResShift - Real-ESRNet | +0.45 [0.22, 0.67] | -0.101 [-0.150, -0.061] | +4.3 [2.7, 6.0] | +0.025 [0.012, 0.040] |
| Objects | 10 | Real-ESRGAN - Real-ESRNet | -1.17 [-1.39, -0.96] | -0.162 [-0.191, -0.134] | +20.9 [17.2, 24.6] | n/a |
| Objects | 10 | ResShift - Real-ESRNet | -0.71 [-0.96, -0.48] | -0.157 [-0.189, -0.126] | +15.9 [11.6, 20.2] | n/a |
| Faces | 10 | Real-ESRGAN - Real-ESRNet | -1.07 [-1.24, -0.91] | -0.114 [-0.132, -0.098] | +18.9 [15.5, 22.4] | -0.041 [-0.058, -0.022] |
| Faces | 10 | ResShift - Real-ESRNet | -0.71 [-0.88, -0.52] | -0.134 [-0.154, -0.116] | +23.3 [19.9, 26.7] | -0.025 [-0.044, -0.008] |
| Text | 10 | Real-ESRGAN - Real-ESRNet | -1.29 [-1.42, -1.17] | -0.197 [-0.258, -0.138] | +13.4 [10.9, 16.2] | +0.033 [-0.007, 0.072] |
| Text | 10 | ResShift - Real-ESRNet | -0.71 [-0.90, -0.51] | -0.214 [-0.291, -0.139] | +10.9 [8.5, 13.3] | +0.065 [0.033, 0.095] |

### JPEG hypothesis

| Method | Content | PSNR: drop 30 to 10 minus drop 50 to 30 (dB) | SSIM | LPIPS | Verdict |
|---|---|---|---|---|---|
| Bicubic | objects | +0.86 [0.73, 0.98] | +0.031 [0.028, 0.035] | +0.040 [0.031, 0.049] | supported |
| Bicubic | faces | +1.56 [1.44, 1.69] | +0.027 [0.025, 0.029] | +0.066 [0.059, 0.073] | supported |
| Bicubic | text | +0.51 [0.41, 0.61] | +0.021 [0.017, 0.025] | +0.068 [0.053, 0.083] | supported |
| Real-ESRNet | objects | +0.96 [0.72, 1.19] | +0.033 [0.028, 0.039] | +0.061 [0.051, 0.073] | supported |
| Real-ESRNet | faces | +1.51 [1.29, 1.73] | +0.021 [0.018, 0.023] | +0.054 [0.046, 0.062] | supported |
| Real-ESRNet | text | +1.27 [0.98, 1.58] | +0.072 [0.063, 0.080] | +0.128 [0.101, 0.156] | supported |
| Real-ESRGAN | objects | +0.80 [0.54, 1.06] | +0.026 [0.018, 0.033] | +0.073 [0.055, 0.095] | supported |
| Real-ESRGAN | faces | +1.22 [0.91, 1.53] | +0.019 [0.014, 0.024] | +0.070 [0.058, 0.082] | supported |
| Real-ESRGAN | text | +1.28 [0.97, 1.62] | +0.069 [0.062, 0.077] | +0.062 [0.038, 0.091] | supported |
| ResShift | objects | +0.67 [0.49, 0.85] | +0.024 [0.016, 0.033] | +0.076 [0.053, 0.106] | supported |
| ResShift | faces | +0.99 [0.88, 1.09] | +0.016 [0.011, 0.022] | +0.070 [0.058, 0.081] | supported |
| ResShift | text | +1.11 [0.82, 1.40] | +0.073 [0.064, 0.081] | +0.042 [0.024, 0.061] | supported |

### Runtime

| Method | Device | Precision | Mean s/image (test) | Median s/image (test) | Peak GPU memory (MiB) | Wall time dev (s) | Wall time test (s) |
|---|---|---|---:|---:|---:|---:|---:|
| Bicubic | CPU (Kaggle) | uint8 (Pillow, CPU) | 0.003 | 0.003 | n/a | 17 | 76 |
| Real-ESRNet | Tesla T4 | float32 | 0.156 | 0.159 | 208 | 58 | 251 |
| Real-ESRGAN | Tesla T4 | float32 | 0.161 | 0.161 | 208 | 67 | 291 |
| ResShift | Tesla T4 | float32 weights, float16 autocast | 0.890 | 0.889 | 2647 | 265 | 1098 |
| resshift_seed1 | Tesla T4 | float32 weights, float16 autocast | 0.873 | 0.875 | 2647 | n/a | 1083 |
| resshift_seed2 | Tesla T4 | float32 weights, float16 autocast | 0.881 | 0.880 | 2646 | n/a | 1078 |
| resshift_seed3 | Tesla T4 | float32 weights, float16 autocast | 0.882 | 0.881 | 2646 | n/a | 1079 |
| resshift_seed4 | Tesla T4 | float32 weights, float16 autocast | 0.883 | 0.882 | 2646 | n/a | 1080 |

Split wall time (all methods of one notebook run, s): dev 409; test 1718, 4327

### Seeds

not run
