"""The project's Y-channel PSNR/SSIM convention, without model imports."""

import math

import cv2
import numpy as np


def luma(image):
    pixels = np.asarray(image, dtype=np.float64) / 255.0
    return (16 + 65.481 * pixels[..., 0] + 128.553 * pixels[..., 1]
            + 24.966 * pixels[..., 2])[4:-4, 4:-4]


def reference_metrics(output, reference):
    if output.size != reference.size:
        raise ValueError("The reference must have exactly the output dimensions.")
    if min(output.size) < 20:
        raise ValueError("The output is too small for an 11-pixel SSIM window and border crop.")
    first, second = luma(output), luma(reference)
    mse = float(np.mean((first - second) ** 2))
    psnr = 10 * math.log10(255 ** 2 / mse) if mse else None
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    def mean(values):
        return cv2.filter2D(values, -1, window)[5:-5, 5:-5]

    a, b = mean(first), mean(second)
    va, vb = mean(first ** 2) - a ** 2, mean(second ** 2) - b ** 2
    covariance = mean(first * second) - a * b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = np.mean(((2 * a * b + c1) * (2 * covariance + c2))
                   / ((a * a + b * b + c1) * (va + vb + c2)))
    return {"psnr": psnr, "psnr_infinite": mse == 0, "ssim": float(ssim)}
