import base64
import hashlib
import io
import json
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from app import Demo, decode_image, make_server


def upload_value(image, format="PNG", name="sample.png"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return {"name": name, "data": base64.b64encode(buffer.getvalue()).decode()}


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.demo = Demo(live=False)
        rng = np.random.default_rng(42)
        self.image = Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))

    def tearDown(self):
        self.demo.close()

    def test_demo_starts_empty_and_upload_has_no_saved_results(self):
        self.assertEqual(self.demo.sessions, {})
        self.assertEqual(set(self.demo.catalog), {"live_methods"})
        self.assertFalse(any(path.startswith("/examples/") for path in self.demo.files))
        session = self.demo.upload({"image": upload_value(self.image)})
        self.assertNotIn("match", session)
        self.assertEqual(session["results"], {})
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.demo.run({"image_id": session["id"], "method": "resshift"})

    def test_live_bicubic_pixels_and_reference(self):
        expected = self.image.resize((128, 96), Image.Resampling.BICUBIC)
        session = self.demo.upload({"image": upload_value(self.image), "reference": upload_value(expected)})
        result = self.demo.run({"image_id": session["id"], "method": "bicubic"})
        with Image.open(self.demo.files["/" + result["url"]]) as actual:
            self.assertEqual(actual.size, (128, 96))
            self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertTrue(result["metrics"]["psnr_infinite"])
        self.assertIsNone(result["metrics"]["psnr"])
        self.assertAlmostEqual(result["metrics"]["ssim"], 1, places=10)
        json.dumps(result, allow_nan=False)
        cached = self.demo.run({"image_id": session["id"], "method": "bicubic"})
        self.assertTrue(cached["cached"])
        self.assertEqual(cached["seconds"], result["seconds"])

    def test_new_upload_has_no_fabricated_metrics_or_learned_outputs(self):
        session = self.demo.upload({"image": upload_value(self.image)})
        result = self.demo.run({"image_id": session["id"], "method": "bicubic"})
        self.assertEqual(result["metrics"], {})
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.demo.run({"image_id": session["id"], "method": "resshift"})
        for method in ["realesrnet", "realesrgan"]:
            with self.assertRaisesRegex(ValueError, "disabled"):
                self.demo.run({"image_id": session["id"], "method": method})

    def test_repeated_upload_has_fixed_pixel_seed_and_fresh_results(self):
        first = self.demo.upload({"image": upload_value(self.image)})
        self.demo.run({"image_id": first["id"], "method": "bicubic"})
        second = self.demo.upload({"image": upload_value(self.image, name="renamed.png")})
        digest = hashlib.sha256(b"32x24:" + self.image.tobytes()).hexdigest()
        self.assertEqual(first["seed"], int(digest[:8], 16) & 0x7fffffff)
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotIn("match", second)
        self.assertEqual(second["results"], {})
        self.assertEqual(first["seed"], second["seed"])

    def test_reference_must_fit(self):
        with self.assertRaisesRegex(ValueError, "128 × 96"):
            self.demo.upload({"image": upload_value(self.image), "reference": upload_value(Image.new("RGB", (64, 64)))})
        reference = self.image.resize((128, 96), Image.Resampling.BICUBIC)
        first = self.demo.upload({"image": upload_value(self.image),
                                  "reference": upload_value(reference)})
        self.assertIsNotNone(first["reference"])
        self.assertEqual(first["results"], {})

    def test_invalid_files_sizes_and_alpha(self):
        for invalid in [{}, {"data": "not base64"}, {"data": base64.b64encode(b"not an image").decode()},
                        upload_value(self.image, format="GIF"), upload_value(Image.new("RGB", (513, 16)))]:
            with self.assertRaises(ValueError):
                decode_image(invalid)
        decoded = decode_image(upload_value(Image.new("RGBA", (16, 16), (255, 0, 0, 0))))
        self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))

    def test_http_api_and_file_isolation(self):
        server = make_server(self.demo, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        root = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urllib.request.urlopen(root + "/api/catalog") as response:
                catalog = json.load(response)
                self.assertEqual(set(catalog), {"live_methods"})
                self.assertEqual(set(catalog["live_methods"]), {"realesrnet", "realesrgan", "resshift"})
            for path in ["/app.py", "/../app.py", "/examples/../../app.py", "/provenance/sources.json",
                         "/examples/face_00030/none/input.png"]:
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(root + path)
                self.assertEqual(raised.exception.code, 404)
                raised.exception.close()
            request = urllib.request.Request(root + "/api/upload", data=json.dumps({"image": upload_value(self.image)}).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                session = json.load(response)
            request = urllib.request.Request(root + "/api/run", data=json.dumps({"image_id": session["id"], "method": "bicubic"}).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                result = json.load(response)
            with urllib.request.urlopen(root + "/" + result["url"]) as response:
                image = Image.open(io.BytesIO(response.read()))
                self.assertEqual(image.size, (128, 96))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
