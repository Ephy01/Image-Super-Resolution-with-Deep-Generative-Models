"""Stage 3, evaluation: PSNR, SSIM, LPIPS, MUSIQ, OCR CER (text) and ArcFace similarity (faces) for every output of a split.

    python evaluate.py --config configs/final.yaml --split dev
    python evaluate.py --config configs/final.yaml --split test
    python evaluate.py --config configs/final.yaml --split test --methods resshift_seed1 resshift_seed2 resshift_seed3 resshift_seed4 --tag _seeds

Checks each output folder against its manifest (weights in scripts/SHA256SUMS, current input hashes) and stops on a mismatch.
Tables: results/final/{metrics,summary,paired,...}_<split>.csv; intervals are 95% percentile bootstrap [lo, hi].
"""

import argparse

import paths

parser = argparse.ArgumentParser(description="Evaluation")
parser.add_argument("--config", default="configs/final.yaml")
parser.add_argument("--split", default="dev", choices=["dev", "test"])
parser.add_argument("--methods", nargs="+", default=None)
parser.add_argument("--tag", default="")
args = parser.parse_args()
paths.check_frozen(args.config, args.split)
extra = (["--methods", *args.methods] if args.methods else []) + (["--tag", args.tag] if args.tag else [])
paths.run_stage("code/evaluation/evaluate.py", ["--split", args.split, *extra], args.config)
