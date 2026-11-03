"""Stage 1, data: download DIV2K and the FFHQ faces, build the HR images and low-resolution inputs; download and verify the weights.

    python prepare.py --config configs/final.yaml                 # data, weights and the pinned ResShift code
    python prepare.py --config configs/final.yaml --only data
"""

import argparse
import os
import subprocess
import sys

import paths

parser = argparse.ArgumentParser(description="Download and prepare the data and weights")
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--only", nargs="+", default=["data", "weights"], choices=["data", "weights"])
args = parser.parse_args()
seed = paths.load_config(args.config)["data"]["seed"]
paths.run_stage("code/datasets/download_data.py", ["--seed", seed, "--only", *args.only], args.config)
if "data" in args.only:
    paths.run_stage("code/datasets/prepare_data.py", ["--seed", seed], args.config)
if "weights" in args.only:
    environment = dict(os.environ, SR_CONFIG=os.path.abspath(args.config), PYTHON=sys.executable)
    subprocess.run(["bash", "code/models/setup_resshift.sh"], env=environment, check=True, cwd=paths.ROOT)
