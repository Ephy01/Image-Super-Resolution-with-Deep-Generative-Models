"""Write the self-contained GPU notebooks in experiments/ from the current repository files.

    python scripts/build_notebooks.py

Each notebook embeds the repository files it needs verbatim (with their SHA-256) and runs the same entry point as a
local run: python inference.py --config configs/final.yaml --split <split> ... The only upload is the input archive
from scripts/make_gpu_inputs.py, whose inputs_sha256.json hash is embedded too. Rebuild after changing any embedded file.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paths

EMBEDDED = [
    "paths.py", "inference.py", "configs/default.yaml", "configs/final.yaml", "configs/final.yaml.sha256", "scripts/SHA256SUMS",
    "code/datasets/download_data.py", "code/datasets/prepare_data.py", "code/datasets/render_text.py",
    "code/models/rrdbnet.py", "code/models/resshift.py", "code/models/upscale.py", "code/models/setup_resshift.sh",
]
PACKAGES = ["omegaconf==2.3.0", "einops==0.8.2", "timm==1.0.30", "addict==2.4.0", "future==1.0.0", "loralib==0.1.2", "PyYAML==6.0.3"]

NOTEBOOKS = [
    {
        "file": "01_dev_all_methods_t4.ipynb", "split": "dev", "methods": ["bicubic", "realesrnet", "realesrgan", "resshift"],
        "offsets": [0], "minutes": "about 10",
        "purpose": "Reproduction and runtime on dev: all 4 methods on the 256 dev inputs (bicubic on the CPU, the learned methods on the T4), "
                   "with 3 warm-ups, synchronized per-image timing, peak GPU memory and wall time. The local script "
                   "code/evaluation/reproduce.py then compares the per-image metrics with the archived stage 4 tables.",
    },
    {
        "file": "02_test_all_methods_t4.ipynb", "split": "test", "methods": ["bicubic", "realesrnet", "realesrgan", "resshift"],
        "offsets": [0], "minutes": "about 25",
        "purpose": "The single final test run: all 4 methods on the 1,104 test inputs with the frozen configs/final.yaml "
                   "(its SHA-256 is checked before anything runs). Runtime and peak memory are recorded as in notebook 01.",
    },
    {
        "file": "03_test_resshift_seeds_t4.ipynb", "split": "test", "methods": ["resshift"], "offsets": [1, 2, 3, 4], "minutes": "about 70",
        "purpose": "Seed experiment: ResShift on the 1,104 test inputs with seed offsets 1 to 4 (offset 0 is the main run of notebook 02), "
                   "so 5 seeds in total. If the GPU time runs out, set SPLIT = 'dev' in the settings cell and say so in the report.",
    },
]


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build(spec, files, hashes, inputs_hash):
    name = spec["file"].removesuffix(".ipynb")
    archive = f"gpu_inputs_{spec['split']}.zip"
    intro = f"""# {name}

{spec['purpose']}

**Runtime:** free Google Colab or Kaggle with an NVIDIA T4, {spec['minutes']} minutes. Kaggle needs *Internet on* (weights and the official ResShift code are downloaded).

**Before running:** create `data/{archive}` locally with `python scripts/make_gpu_inputs.py --split {spec['split']}`.
Colab: run the first cell and upload it when asked (or put it at `/content/{archive}`). Kaggle: add it as a dataset with any name; the notebook finds the zip, or the folder Kaggle unpacks it into, under `/kaggle/input`.

**After running:** download `{name}_results.zip` (Colab downloads it; Kaggle shows a download link under the last cell, and the zip is the only file in Output) and unzip it at the repository root.
It contains `outputs/{spec['split']}/<method>/...` (PNG outputs with `manifest.json`) and `results/runtime/...` (timing, peak memory, environment).
Then run the local commands in README.md, section *Run experiments*.

The cells below embed the repository files verbatim; each file is checked against its SHA-256 from the build. Rebuild with `python scripts/build_notebooks.py` after changing them.
"""
    settings = f"""SPLIT = {spec['split']!r}
METHODS = {spec['methods']!r}
SEED_OFFSETS = {spec['offsets']!r}
NOTEBOOK = {name!r}
INPUT_ARCHIVE = {archive!r}
INPUTS_SHA256_JSON = {inputs_hash!r}   # sha256 of data/processed/inputs_sha256.json at build time
"""
    locate = """import glob
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ON_KAGGLE = Path('/kaggle/working').exists()
WORK = Path('/tmp/sr') if ON_KAGGLE else Path('/content/sr')   # outside /kaggle/working: the results zip is the only Kaggle output
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

if ON_KAGGLE:
    # Kaggle usually unpacks an uploaded zip: accept the archive itself or its unpacked data/processed/ folder.
    found = glob.glob(f'/kaggle/input/**/{INPUT_ARCHIVE}', recursive=True)
    unpacked = glob.glob('/kaggle/input/**/data/processed/inputs_sha256.json', recursive=True)
    assert found or unpacked, f'add {INPUT_ARCHIVE} as a Kaggle dataset (the dataset name does not matter)'
    archive = Path(found[0]) if found else Path(unpacked[0]).parents[2]
else:
    archive = Path('/content') / INPUT_ARCHIVE
    if not archive.exists():
        from google.colab import files
        uploaded = files.upload()
        archive = Path('/content') / next(iter(uploaded))
if archive.is_file():
    print('input archive:', archive, hashlib.sha256(archive.read_bytes()).hexdigest())
else:
    print('unpacked inputs:', archive)

import torch
assert torch.cuda.is_available(), 'select a GPU runtime (T4)'
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
if 'T4' not in torch.cuda.get_device_name(0):
    print('WARNING: not a T4; runtime numbers will not be comparable')
"""
    embed = "FILES = " + json.dumps(files, indent=0) + "\nHASHES = " + json.dumps(hashes, indent=0) + """

for name, text in FILES.items():
    path = WORK / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == HASHES[name], f'{name} differs from the build: {actual}'
print(len(FILES), 'repository files written and verified')
"""
    install = f"""from importlib.metadata import version
constraints = WORK / 'constraints.txt'
constraints.write_text(f"torch=={{version('torch')}}\\ntorchvision=={{version('torchvision')}}\\n")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-c', str(constraints), *{PACKAGES!r}], check=True)
"""
    unpack = """import csv
import json
import shutil
if archive.is_file():
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(WORK)
else:
    shutil.copytree(archive / 'data', WORK / 'data', dirs_exist_ok=True)
hashes_file = WORK / 'data/processed/inputs_sha256.json'
assert hashlib.sha256(hashes_file.read_bytes()).hexdigest() == INPUTS_SHA256_JSON, 'inputs_sha256.json differs from the build'
expected = json.loads(hashes_file.read_text())
lr = sorted((WORK / 'data/processed/lr').rglob('*.png'))
for path in lr:
    name = str(path.relative_to(WORK / 'data/processed'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[name], f'{name} differs from inputs_sha256.json'
with open(WORK / 'data/processed/manifest.csv') as handle:
    images = sum(row['split'] == SPLIT for row in csv.DictReader(handle))
assert len(lr) == 4 * images, f'{len(lr)} inputs found, {4 * images} expected for {SPLIT}'
print(len(lr), 'inputs verified')
"""
    weights = """subprocess.run([sys.executable, 'code/datasets/download_data.py', '--only', 'weights'], check=True)   # verifies scripts/SHA256SUMS
subprocess.run(['bash', 'code/models/setup_resshift.sh'], check=True, env=dict(os.environ, PYTHON=sys.executable))
"""
    run = """command = [sys.executable, 'inference.py', '--config', 'configs/final.yaml', '--split', SPLIT, '--methods', *METHODS,
           '--seed-offsets', *map(str, SEED_OFFSETS)]
print(' '.join(command))
subprocess.run(command, check=True)   # inference.py refuses the test split unless configs/final.yaml matches final.yaml.sha256
"""
    export = """runtime = WORK / 'results/runtime'
runtime.mkdir(parents=True, exist_ok=True)
environment = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout
gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
(runtime / f'environment_{NOTEBOOK}.txt').write_text(f'python {sys.version}\\ntorch {torch.__version__} cuda {torch.version.cuda}\\n\\n{gpu}\\n{environment}')

out = (Path('/kaggle/working') if ON_KAGGLE else Path('/content')) / f'{NOTEBOOK}_results.zip'
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
    for folder in sorted((WORK / 'outputs' / SPLIT).iterdir()):
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                handle.write(path, path.relative_to(WORK))
    for path in sorted(runtime.iterdir()):
        handle.write(path, path.relative_to(WORK))
print(out, round(out.stat().st_size / 2**20, 1), 'MiB')
if ON_KAGGLE:
    from IPython.display import FileLink, display
    os.chdir('/kaggle/working')
    display(FileLink(out.name))   # click to download; after "Save Version" it is also under Output
else:
    from google.colab import files
    files.download(str(out))
"""
    cells = [markdown_cell(intro), code_cell(settings), code_cell(locate), code_cell(embed), code_cell(install),
             code_cell(unpack), code_cell(weights), code_cell(run), code_cell(export)]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                             "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "experiments" / spec["file"]
    path.write_text(json.dumps(notebook, indent=1))
    return path


def main():
    files = {name: (ROOT / name).read_text() for name in EMBEDDED}
    hashes = {name: paths.sha256_file(ROOT / name) for name in EMBEDDED}
    inputs_hash = paths.sha256_file(paths.DATA_ROOT / "processed" / "inputs_sha256.json")
    (ROOT / "experiments").mkdir(exist_ok=True)
    for spec in NOTEBOOKS:
        print(build(spec, files, hashes, inputs_hash).relative_to(ROOT))


if __name__ == "__main__":
    main()
