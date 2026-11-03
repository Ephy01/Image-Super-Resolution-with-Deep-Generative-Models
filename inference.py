"""Stage 2, inference: x4 super-resolution of every low-resolution input of a split with the configured methods.

    python inference.py --config configs/default.yaml --split dev
    python inference.py --config configs/final.yaml --split test                                   # once, frozen settings
    python inference.py --config configs/final.yaml --split test --methods resshift --seed-offsets 1 2 3 4

Outputs: outputs/<split>/<method>/jpeg_<q>/<image_id>.png with manifest.json (weights, code, settings and input hashes);
an output folder whose manifest matches is skipped. Timing and peak GPU memory: results/runtime/.
"""

import argparse

import paths

parser = argparse.ArgumentParser(description="Super-resolution inference")
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--split", default="dev", choices=["dev", "test"])
parser.add_argument("--methods", nargs="+", default=None)
parser.add_argument("--seed-offsets", nargs="+", type=int, default=None)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
paths.check_frozen(args.config, args.split)
cfg = paths.load_config(args.config)
extra = ["--methods", *(args.methods or cfg["methods"]), "--seed-offsets", *(args.seed_offsets or [cfg["resshift"]["seed_offset"]])]
paths.run_stage("code/models/upscale.py", ["--split", args.split, *extra, *(["--force"] if args.force else [])], args.config)
