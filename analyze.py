"""Stage 4, analyses: main table, paired differences, JPEG-below-30 test, hypothesis checks, seed variability, runtime,
failure cases and figures, with the rules fixed in the analysis: section of the config.

    python analyze.py --config configs/final.yaml --split test       # -> results/analysis/*.json, results/final/figures/
"""

import argparse

import paths

parser = argparse.ArgumentParser(description="Analyses and figures")
parser.add_argument("--config", default="configs/final.yaml")
parser.add_argument("--split", default="test", choices=["dev", "test"])
args = parser.parse_args()
paths.check_frozen(args.config, args.split)
paths.run_stage("code/evaluation/analyze.py", ["--split", args.split], args.config)
