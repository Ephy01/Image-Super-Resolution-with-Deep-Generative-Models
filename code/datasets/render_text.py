import random
import string
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS = ["DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf"]
FONT_SIZES = [16, 24, 32, 48]
KINDS = ["words", "codes"]

IMAGE_SIZE = 512
MARGIN = 24
LINE_SPACING = 1.6
CODE_LENGTH = 6
CODE_ALPHABET = string.ascii_uppercase + string.digits

WORDS = [
    "about", "above", "after", "again", "always", "animal", "answer", "apple", "around", "autumn",
    "become", "before", "begin", "better", "between", "bridge", "bright", "bring", "build", "camera",
    "carry", "castle", "center", "change", "children", "circle", "city", "clean", "close", "cloud",
    "color", "common", "corner", "country", "course", "cover", "dance", "decide", "deep", "dinner",
    "doctor", "double", "dream", "drive", "early", "earth", "east", "eight", "energy", "enough",
    "evening", "every", "example", "family", "famous", "father", "field", "figure", "final", "finger",
    "first", "flower", "follow", "forest", "friend", "front", "garden", "glass", "golden", "great",
    "green", "ground", "group", "happy", "heavy", "history", "horse", "house", "hundred", "idea",
    "island", "journey", "kitchen", "known", "language", "large", "later", "learn", "letter", "light",
    "listen", "little", "machine", "market", "matter", "measure", "middle", "minute", "modern", "money",
    "morning", "mother", "mountain", "music", "nature", "never", "night", "north", "notice", "number",
    "ocean", "often", "orange", "order", "other", "paper", "people", "person", "picture", "place",
    "plant", "point", "power", "problem", "public", "question", "quick", "quiet", "reach", "ready",
    "reason", "record", "river", "round", "school", "science", "second", "seven", "shape", "short",
    "silver", "simple", "small", "sound", "south", "space", "spring", "square", "start", "story",
    "street", "strong", "summer", "system", "table", "thing", "think", "thousand", "three", "today",
    "together", "travel", "under", "until", "village", "voice", "water", "weather", "white", "window",
    "winter", "without", "wonder", "world", "write", "yellow", "young",
]


def make_token(kind, generator):
    if kind == "words":
        return generator.choice(WORDS)
    return "".join(generator.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def make_line(kind, font, max_width, generator):
    tokens = [make_token(kind, generator)]
    while True:
        candidate = tokens + [make_token(kind, generator)]
        if font.getlength(" ".join(candidate)) > max_width:
            return " ".join(tokens)
        tokens = candidate


def render_text_image(font_name, font_size, kind, generator):
    font = ImageFont.truetype(str(FONT_DIR / font_name), font_size)
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(image)

    line_height = int(font_size * LINE_SPACING)
    max_width = IMAGE_SIZE - 2 * MARGIN
    lines = []
    top = MARGIN
    while top + line_height <= IMAGE_SIZE - MARGIN:
        line = make_line(kind, font, max_width, generator)
        draw.text((MARGIN, top), line, font=font, fill="black")
        lines.append(line)
        top += line_height

    return image, "\n".join(lines)


def render_text_set(repeats, seed):
    generator = random.Random(seed)
    samples = []
    for font_name in FONTS:
        for font_size in FONT_SIZES:
            for kind in KINDS:
                for _ in range(repeats):
                    image, text = render_text_image(font_name, font_size, kind, generator)
                    samples.append({
                        "image": image,
                        "text": text,
                        "font": font_name,
                        "font_size": font_size,
                        "kind": kind,
                    })
    return samples
