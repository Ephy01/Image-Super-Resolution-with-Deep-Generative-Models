"""Run four super-resolution methods locally on uploaded images."""

import argparse
import base64
import binascii
import hashlib
import io
import json
import mimetypes
import tempfile
import threading
import time
import uuid
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from metrics import reference_metrics
from live_models import LiveModels

ROOT = Path(__file__).resolve().parent
MAX_BODY = 16 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 8_000_000


def pixel_hash(image):
    return hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def decode_image(value, reference=False):
    if not isinstance(value, dict) or not isinstance(value.get("data"), str):
        raise ValueError("Choose a PNG, JPEG or WebP image.")
    try:
        raw = base64.b64decode(value["data"], validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"} or getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Use a still PNG, JPEG or WebP image.")
                limit = 2048 if reference else 512
                minimum = 64 if reference else 16
                if min(source.size) < minimum or max(source.size) > limit:
                    raise ValueError(f"{'Reference' if reference else 'Input'} dimensions must be between {minimum} and {limit} pixels on each side. Images are not resized automatically.")
                source.load()
                oriented = ImageOps.exif_transpose(source).convert("RGBA")
                background = Image.new("RGBA", oriented.size, "white")
                image = Image.alpha_composite(background, oriented).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as error:
        raise ValueError("That file could not be read as a supported image.") from error
    return image


class Demo:
    def __init__(self, live=True, device="auto"):
        self.live = LiveModels(enabled=live, device=device)
        self.catalog = {"live_methods": self.live.available}
        self.temp = tempfile.TemporaryDirectory(prefix="sr-demo-")
        self.sessions = {}
        self.lock = threading.RLock()
        self.files = {"/": ROOT / "static/index.html"}
        for name in ["style.css", "app.js"]:
            self.files["/static/" + name] = ROOT / "static" / name

    def upload(self, data):
        image = decode_image(data.get("image"))
        reference = decode_image(data["reference"], reference=True) if data.get("reference") else None
        if reference and reference.size != (image.width * 4, image.height * 4):
            raise ValueError(f"The reference must be {image.width * 4} × {image.height * 4}, aligned to the input. It will not be resized to fit.")
        with self.lock:
            if len(self.sessions) >= 24:
                raise ValueError("This session has reached 24 uploads. Restart the demo to clear them.")
            key = uuid.uuid4().hex
            folder = Path(self.temp.name) / key
            folder.mkdir()
            image.save(folder / "input.png")
            input_url = f"uploads/{key}/input.png"
            self.files["/" + input_url] = folder / "input.png"
            reference_url = None
            if reference:
                reference.save(folder / "reference.png")
                reference_url = f"uploads/{key}/reference.png"
                self.files["/" + reference_url] = folder / "reference.png"
            name = str(data["image"].get("name", "Uploaded image")).replace("\\", "/").split("/")[-1][:120]
            session = {"id": key, "title": name, "input": input_url, "reference": reference_url,
                       "width": image.width, "height": image.height, "results": {},
                       "seed": int(pixel_hash(image)[:8], 16) & 0x7fffffff}
            self.sessions[key] = session
            return dict(session)

    def run(self, data):
        method = data.get("method")
        if method not in {"bicubic", "realesrnet", "realesrgan", "resshift"}:
            raise ValueError("Unknown method.")
        with self.lock:
            session = self.sessions.get(data.get("image_id"))
            if session is None:
                raise ValueError("Upload an image first, or upload it again after restarting the server.")
            if method in session["results"]:
                return {**session["results"][method], "cached": True}
            folder = Path(self.temp.name) / session["id"]
            with Image.open(folder / "input.png") as source:
                source.load()
                if method == "bicubic":
                    start = time.perf_counter()
                    output = source.resize((source.width * 4, source.height * 4), Image.Resampling.BICUBIC)
                    timing = {"seconds": time.perf_counter() - start, "device": "CPU", "load_seconds": 0, "first_run": False}
                else:
                    output, timing = self.live.upscale(source, method, seed=session["seed"])
            scores = {}
            if session["reference"]:
                with Image.open(folder / "reference.png") as reference:
                    scores = reference_metrics(output, reference)
            output.save(folder / f"{method}.png")
            url = f"uploads/{session['id']}/{method}.png"
            self.files["/" + url] = folder / f"{method}.png"
            result = {"url": url, "width": output.width, "height": output.height, "mode": "live",
                      **timing, "metrics": scores, "cached": False}
            session["results"][method] = result
            return result

    def close(self):
        self.temp.cleanup()


def make_server(demo, port=8000):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status, value):
            self.send(status, json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/catalog":
                self.json(200, demo.catalog)
                return
            with demo.lock:
                file = demo.files.get(path)
            if file and file.is_file():
                self.send(200, file.read_bytes(), mimetypes.guess_type(file)[0] or "application/octet-stream")
            else:
                self.json(404, {"error": "Not found."})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in {"/api/upload", "/api/run"}:
                self.json(404, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self.json(413, {"error": "The upload is too large. Use files totalling less than 10 MB."})
                    return
                if self.headers.get_content_type() != "application/json":
                    self.json(415, {"error": "Send JSON image data."})
                    return
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected an image request.")
                result = demo.upload(data) if path == "/api/upload" else demo.run(data)
                self.json(200, result)
            except (ValueError, TypeError) as error:
                self.json(400, {"error": str(error)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.json(500, {"error": "The operation failed. See the terminal for details."})

        def log_message(self, format, *args):
            if len(args) > 1 and str(args[1]).startswith(("4", "5")):
                super().log_message(format, *args)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the demo in your browser")
    parser.add_argument("--device", choices=["auto", "mps", "cuda"], default="auto")
    parser.add_argument("--bicubic-only", "--saved-only", dest="bicubic_only", action="store_true",
                        help="Disable learned models and run only bicubic interpolation")
    args = parser.parse_args()
    demo = Demo(live=not args.bicubic_only, device=args.device)
    try:
        server = make_server(demo, args.port)
    except OSError:
        demo.close()
        raise SystemExit(f"Port {args.port} is busy. Try: python app.py --port 8001 --open")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    live_names = [name for name, item in demo.live.available.items() if item["enabled"]]
    print(f"Super-resolution demo: {url}\nUpload an image to run bicubic or a local model. Available networks: {', '.join(live_names) or 'none'}.\nCtrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        demo.close()


if __name__ == "__main__":
    main()
