"""Where the data, weights, outputs and results live.

Resolution order everywhere: environment variable > YAML value > default below.
The YAML file is the one passed with --config to a root entry point, or $SR_CONFIG, or configs/default.yaml.

SR_DATA        - data root: raw/ (downloads) and processed/ (hr/, lr/, manifest.csv)
SR_WEIGHTS     - flat folder with the downloaded weights listed in scripts/SHA256SUMS
SR_OUTPUTS     - super-resolution outputs: <split>/<method>/jpeg_<quality>/<image_id>.png plus manifest.json
SR_RESULTS     - tables, JSON reports and figures
SR_THIRD_PARTY - pinned third-party source code (ResShift)
SR_CACHE       - model caches of the metric libraries (pyiqa, insightface)
"""

import hashlib
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

DEFAULTS = {
    "data": ROOT / "data",
    "weights": ROOT / "models",
    "outputs": ROOT / "outputs",
    "results": ROOT / "results",
    "third_party": ROOT / "third_party",
    "cache": ROOT / "data" / "cache",
}
ENV_NAMES = {
    "data": "SR_DATA",
    "weights": "SR_WEIGHTS",
    "outputs": "SR_OUTPUTS",
    "results": "SR_RESULTS",
    "third_party": "SR_THIRD_PARTY",
    "cache": "SR_CACHE",
}


def config_path():
    return Path(os.environ.get("SR_CONFIG", DEFAULT_CONFIG))


def load_config(path=None):
    with open(path or config_path()) as handle:
        return yaml.safe_load(handle)


def resolve(key, cfg=None):
    """Environment variable, then the YAML value under paths:, then the default. Relative YAML paths are relative to ROOT."""
    if os.environ.get(ENV_NAMES[key]):
        return Path(os.environ[ENV_NAMES[key]])
    cfg = load_config() if cfg is None else cfg
    value = (cfg.get("paths") or {}).get(key)
    if value:
        value = Path(value)
        return value if value.is_absolute() else ROOT / value
    return DEFAULTS[key]


def use_config(path):
    """Called by the root entry points before they import code/: every module then resolves the same file."""
    os.environ["SR_CONFIG"] = str(Path(path).resolve())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_frozen(config, split):
    """The test split runs only with configs/final.yaml whose SHA-256 matches configs/final.yaml.sha256."""
    if split != "test":
        return
    config = Path(config).resolve()
    digest_file = config.with_name(config.name + ".sha256")
    if config.name != "final.yaml" or not digest_file.exists():
        raise SystemExit("the test split runs only with --config configs/final.yaml (frozen, with final.yaml.sha256)")
    expected = digest_file.read_text().split()[0]
    if sha256_file(config) != expected:
        raise SystemExit(f"{config} was changed after it was frozen: {sha256_file(config)[:12]} != {expected[:12]}")


def run_stage(script, args, config):
    """Run one script of code/ with the given config; the root entry points are thin wrappers around this."""
    import subprocess
    import sys

    import datetime
    import json
    import platform
    import time

    environment = dict(os.environ, SR_CONFIG=str(Path(config).resolve()))
    start = time.perf_counter()
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], env=environment, check=True, cwd=ROOT)
    log = resolve("results", load_config(config)) / "runtime" / "stage_times.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(log.read_text()) if log.exists() else []
    entries.append({"script": script, "args": list(map(str, args)), "config": Path(config).name, "seconds": time.perf_counter() - start,
                    "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
                    "cache": str(resolve("cache", load_config(config))), "finished": datetime.datetime.now().isoformat(timespec="seconds")})
    log.write_text(json.dumps(entries, indent=1))


_CFG = load_config()
DATA_ROOT = resolve("data", _CFG)
WEIGHTS_ROOT = resolve("weights", _CFG)
OUTPUTS_ROOT = resolve("outputs", _CFG)
RESULTS_ROOT = resolve("results", _CFG)
THIRD_PARTY_ROOT = resolve("third_party", _CFG)
CACHE_ROOT = resolve("cache", _CFG)
