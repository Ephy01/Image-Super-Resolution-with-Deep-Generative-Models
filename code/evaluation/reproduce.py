"""Compare the per-image dev metrics of the fresh run with the archived stage 4 tables.

    python code/evaluation/reproduce.py     # -> results/analysis/reproduction_dev.json

Archive: results/metrics_dev.csv (bicubic, Real-ESRNet, Real-ESRGAN from 896fe8c; ResShift from the stage 4 T4 run).
Tolerance: 0.01 dB PSNR, 0.001 SSIM and LPIPS, per image.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import paths

TOLERANCE = {"psnr": 0.01, "ssim": 0.001, "lpips": 0.001}
KEYS = ["image_id", "content", "method", "jpeg"]


def main():
    archived = pd.read_csv(paths.RESULTS_ROOT / "metrics_dev.csv", dtype={"jpeg": str})
    fresh = pd.read_csv(paths.RESULTS_ROOT / "final" / "metrics_dev.csv", dtype={"jpeg": str})
    fresh = fresh[fresh["method"] != "hr"]
    report = {"tolerance": TOLERANCE, "archive": "results/metrics_dev.csv", "fresh": "results/final/metrics_dev.csv",
              "devices": json.loads((paths.RESULTS_ROOT / "final" / "evaluation_report_dev.json").read_text())["outputs"],
              "methods": {}}
    for method, group in fresh.groupby("method", sort=False):
        merged = group.merge(archived, on=KEYS, suffixes=("", "_archived"), validate="one_to_one")
        entry = {"images": len(merged), "archived_rows": int((archived["method"] == method).sum())}
        for metric, tolerance in TOLERANCE.items():
            difference = (merged[metric] - merged[f"{metric}_archived"]).abs()
            worst = merged.loc[difference.idxmax()]
            entry[metric] = {"max_abs_difference": float(difference.max()), "mean_abs_difference": float(difference.mean()),
                             "within_tolerance": bool((difference <= tolerance).all()), "rows_outside": int((difference > tolerance).sum()),
                             "worst": f"{worst.image_id} jpeg {worst.jpeg}"}
        entry["reproduced"] = all(entry[metric]["within_tolerance"] for metric in TOLERANCE)
        report["methods"][method] = entry
        print(method, {metric: round(entry[metric]["max_abs_difference"], 6) for metric in TOLERANCE}, "OK" if entry["reproduced"] else "OUTSIDE TOLERANCE")
    (paths.RESULTS_ROOT / "analysis" / "reproduction_dev.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
