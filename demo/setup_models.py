"""Download verified weights and the pinned ResShift inference source."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
COMMIT = 'bb03b7d21614cace01787e097c8a6ab6b945227d'
SOURCE_URL = 'https://github.com/zsyOAOA/ResShift.git'
MIRROR = 'https://huggingface.co/Fucius/DAM-VSR/resolve/7550a74b8717fa6b2a0bf283aff53135fdaa7cc3/'
WEIGHTS = {
    'RealESRNet_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
    'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'resshift_realsrx4_s4_v3.pth': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth',
    'autoencoder_vq_f4.pth': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
}


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def setup_source():
    if shutil.which('git') is None:
        raise RuntimeError('Git is required. Install the macOS command-line tools, then rerun setup.')
    official = ROOT / 'third_party/ResShift'
    official.mkdir(parents=True, exist_ok=True)

    def git(*args, capture=False):
        return subprocess.run(['git', '-C', str(official), *args], check=True,
                              text=True, stdout=subprocess.PIPE if capture else None).stdout

    if not (official / '.git').exists():
        if any(official.iterdir()):
            raise RuntimeError(f'{official} is not a Git checkout. Move it aside and rerun setup.')
        git('init', '--quiet')
        git('remote', 'add', 'origin', SOURCE_URL)
    elif git('remote', 'get-url', 'origin', capture=True).strip() != SOURCE_URL:
        raise RuntimeError('The ResShift checkout has an unexpected origin. Move it aside and rerun setup.')
    if git('status', '--porcelain', '--untracked-files=no', capture=True).strip():
        raise RuntimeError('The ResShift source has local changes. Move it aside before rerunning setup.')
    git('config', 'core.sparseCheckout', 'true')
    git('config', 'core.sparseCheckoutCone', 'false')
    sparse = official / '.git/info/sparse-checkout'
    sparse.parent.mkdir(parents=True, exist_ok=True)
    sparse.write_text('/configs/\n/models/\n/ldm/\n/utils/\n/LICENSE\n')
    exists = subprocess.run(['git', '-C', str(official), 'cat-file', '-e', f'{COMMIT}^{{commit}}'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not exists:
        git('fetch', '--depth', '1', 'origin', COMMIT)
    git('checkout', '--detach', COMMIT)
    if git('rev-parse', 'HEAD', capture=True).strip() != COMMIT:
        raise RuntimeError('ResShift source revision verification failed.')
    print(f'ResShift source verified: {COMMIT}', flush=True)


def download(url, path, expected):
    partial = path.with_suffix('.part')
    received, reported = 0, 0
    try:
        with requests.get(url, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with partial.open('wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    received += len(chunk)
                    if received - reported >= 32 * 1024 * 1024:
                        print(f'  {path.name}: {received / (1024 * 1024):.0f} MiB', flush=True)
                        reported = received
        if digest(partial) != expected:
            raise RuntimeError(f'Checksum mismatch for {path.name}. The download was rejected.')
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mirror', action='store_true', help='Try the verified ResShift mirror first')
    args = parser.parse_args()
    expected = {}
    for line in (ROOT / 'provenance/weights_sha256.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            checksum, name = line.split()
            expected[name] = checksum
    setup_source()
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    for name, url in WEIGHTS.items():
        path = folder / name
        if path.is_file() and digest(path) == expected[name]:
            print(f'{name}: already verified', flush=True)
            continue
        urls = [url]
        if name in {'resshift_realsrx4_s4_v3.pth', 'autoencoder_vq_f4.pth'}:
            urls.append(MIRROR + name)
            if args.mirror:
                urls.reverse()
        for index, candidate in enumerate(urls):
            try:
                print(f'Downloading {name}', flush=True)
                download(candidate, path, expected[name])
                break
            except (requests.RequestException, RuntimeError) as exc:
                if index + 1 == len(urls):
                    raise RuntimeError(f'Could not prepare {name}: {exc}. Check the connection and rerun setup.') from exc
                print(f'First download failed. Trying the alternate verified source. {exc}', flush=True)
        print(f'{name}: SHA-256 verified', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, requests.RequestException) as exc:
        raise SystemExit(str(exc)) from exc
