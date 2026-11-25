"""Generate the demo inputs the public templates name, into example_workflows/input.

Every Load Image, Load Image Mask, Load Video and Load Audio node in a shipped
template names a file. ComfyUI ships only input/example.png, so the pack ships
the rest: flat shapes, a stick figure and a synthetic music loop, drawn here
from code so they carry no third-party rights. They exist so each template
opens and runs on a fresh install; they show the graph working, not the model
at its best. Replace them with real images, clips and music for real work.

At pack import, demo_inputs.place_demo_inputs copies any of these names that
the ComfyUI input folder lacks, and never overwrites a file already there.

Run with no arguments to rewrite every file, or pass names to rewrite some.
PIL and PyAV versions change encoded bytes, so nothing pins the files' hashes;
tests/test_demo_inputs.py checks names, sizes, modes and frame counts.
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "example_workflows" / "input"

STUDIO = (226, 222, 214)
FLOOR = (176, 166, 152)
CLAY = (176, 92, 58)
CREAM = (238, 228, 204)
COBALT = (32, 64, 150)
GOLD = (212, 168, 60)
FIGURE = (36, 40, 52)


def _backdrop(size: tuple[int, int], top=STUDIO, bottom=FLOOR, horizon: float = 0.72) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, top)
    ImageDraw.Draw(image).rectangle((0, int(height * horizon), width, height), fill=bottom)
    return image


def _vase_outline(size: int) -> list[tuple[float, float]]:
    """A vase profile mirrored about the centre line, top to bottom."""
    centre, top, bottom = size / 2, size * 0.2, size * 0.78
    right = []
    for step in range(41):
        t = step / 40
        y = top + (bottom - top) * t
        radius = size * (0.07 + 0.16 * math.sin(math.pi * (0.15 + 0.8 * t)) ** 1.5)
        right.append((centre + radius, y))
    left = [(2 * centre - x, y) for x, y in reversed(right)]
    return right + left


def _band(size: int) -> tuple[int, int, int, int]:
    return (int(size * 0.2), int(size * 0.46), int(size * 0.8), int(size * 0.56))


def _vase_scene(size: int = 1024) -> Image.Image:
    image = _backdrop((size, size))
    draw = ImageDraw.Draw(image)
    outline = _vase_outline(size)
    draw.polygon(outline, fill=CLAY)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).polygon(outline, fill=255)
    band = Image.new("RGB", (size, size), CREAM)
    band_draw = ImageDraw.Draw(band)
    left, top, right, bottom = _band(size)
    for x in range(left, right, size // 32):
        band_draw.polygon([(x, bottom), (x + size // 64, top), (x + size // 32, bottom)], fill=CLAY)
    band_mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(band_mask).rectangle(_band(size), fill=255)
    image.paste(band, (0, 0), Image.fromarray(np.minimum(np.asarray(mask), np.asarray(band_mask))))
    return image


def _band_matte(size: int = 1024) -> Image.Image:
    """White, opaque, with the vase's decoration band transparent: the region to change."""
    vase = Image.new("L", (size, size), 0)
    ImageDraw.Draw(vase).polygon(_vase_outline(size), fill=255)
    band = Image.new("L", (size, size), 0)
    ImageDraw.Draw(band).rectangle(_band(size), fill=255)
    cut = np.minimum(np.asarray(vase), np.asarray(band))
    alpha = Image.fromarray(np.where(cut > 0, 0, 255).astype(np.uint8))
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    image.putalpha(alpha)
    return image


def _tile(index: int, size: int = 1024) -> Image.Image:
    """A cobalt and gold ceramic motif on a transparent ground, one per reference slot."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    centre, radius = size / 2, size * 0.4
    draw.ellipse((centre - radius, centre - radius, centre + radius, centre + radius), fill=(*CREAM, 255))
    petals = 4 + index
    for petal in range(petals):
        angle = 2 * math.pi * petal / petals
        x, y = centre + radius * 0.55 * math.cos(angle), centre + radius * 0.55 * math.sin(angle)
        r = radius * (0.16 + 0.02 * (index % 3))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(*(COBALT if petal % 2 else GOLD), 255))
    inner = radius * 0.22
    draw.ellipse((centre - inner, centre - inner, centre + inner, centre + inner), fill=(*GOLD, 255))
    return image


def _canvas_rgba(size: int = 1024) -> Image.Image:
    return _vase_scene(size).convert("RGBA")


def _guide_rgba(size: int = 1024) -> Image.Image:
    """The vase scene with its decoration band matted out: image_10's change guide."""
    image = _vase_scene(size).convert("RGBA")
    image.putalpha(_band_matte(size).getchannel("A"))
    return image


def _pose(t: float, width: int, height: int, *, sway: float = 1.0) -> dict[str, tuple[float, float]]:
    """Joint positions of a dancing stick figure at time t seconds."""
    cx, ground, unit = width / 2 + width * 0.08 * math.sin(2 * math.pi * 0.25 * t) * sway, height * 0.9, height / 8
    bounce = unit * 0.15 * abs(math.sin(2 * math.pi * t))
    hip = (cx, ground - 3.6 * unit - bounce)
    neck = (cx, hip[1] - 2.6 * unit)
    head = (cx, neck[1] - 0.7 * unit)
    swing = math.sin(2 * math.pi * 0.5 * t)

    def limb(origin, length, angle):
        return (origin[0] + length * math.sin(angle), origin[1] + length * math.cos(angle))

    elbow_l = limb(neck, 1.3 * unit, -0.9 - 0.7 * swing)
    elbow_r = limb(neck, 1.3 * unit, 0.9 - 0.7 * swing)
    knee_l = limb(hip, 1.8 * unit, -0.25 - 0.3 * swing)
    knee_r = limb(hip, 1.8 * unit, 0.25 - 0.3 * swing)
    return {
        "head": head, "neck": neck, "hip": hip,
        "elbow_l": elbow_l, "hand_l": limb(elbow_l, 1.2 * unit, -1.6 - 0.9 * swing),
        "elbow_r": elbow_r, "hand_r": limb(elbow_r, 1.2 * unit, 1.6 - 0.9 * swing),
        "knee_l": knee_l, "foot_l": limb(knee_l, 1.7 * unit, -0.1 + 0.3 * swing),
        "knee_r": knee_r, "foot_r": limb(knee_r, 1.7 * unit, 0.1 + 0.3 * swing),
    }


BONES = (("neck", "hip"), ("neck", "elbow_l"), ("elbow_l", "hand_l"), ("neck", "elbow_r"),
         ("elbow_r", "hand_r"), ("hip", "knee_l"), ("knee_l", "foot_l"), ("hip", "knee_r"),
         ("knee_r", "foot_r"))


def _figure(size: tuple[int, int], t: float, *, ground=None, ink=FIGURE, sway: float = 1.0) -> Image.Image:
    width, height = size
    image = _backdrop(size) if ground is None else Image.new("RGB", size, ground)
    draw = ImageDraw.Draw(image)
    joints = _pose(t, width, height, sway=sway)
    thick = max(4, height // 40)
    for a, b in BONES:
        draw.line((joints[a], joints[b]), fill=ink, width=thick)
    hx, hy = joints["head"]
    r = height / 16
    draw.ellipse((hx - r, hy - r, hx + r, hy + r), fill=ink)
    return image


def _frames(size, count: int, fps: int, **figure) -> Iterator[np.ndarray]:
    for index in range(count):
        yield np.asarray(_figure(size, index / fps, **figure))


def _sliding_box(size, count: int) -> Iterator[np.ndarray]:
    """A red box crossing a studio backdrop: one object to follow through the clip."""
    width, height = size
    for index in range(count):
        image = _backdrop(size)
        x = int(width * (0.1 + 0.6 * index / max(count - 1, 1)))
        ImageDraw.Draw(image).rectangle((x, int(height * 0.4), x + width // 6, int(height * 0.7)), fill=(200, 40, 40))
        yield np.asarray(image)


def _write_video(path: Path, frames: Iterator[np.ndarray], fps: int) -> None:
    import av

    with av.open(str(path), "w", format="mp4") as container:
        stream = None
        for array in frames:
            if stream is None:
                stream = container.add_stream("libx264", rate=fps)
                stream.height, stream.width = array.shape[:2]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "32", "preset": "medium", "threads": "1"}
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        if stream is None:
            raise ValueError(f"{path.name}: no frames to write")
        for packet in stream.encode():
            container.mux(packet)


def _music(seconds: float = 25.0, rate: int = 44100) -> np.ndarray:
    """A 120 bpm loop: a decaying kick on each beat over a four-chord pad."""
    t = np.arange(int(seconds * rate)) / rate
    chords = ((261.63, 329.63, 392.00), (220.00, 261.63, 329.63), (174.61, 220.00, 261.63), (196.00, 246.94, 293.66))
    bar = (t // 2.0).astype(int) % len(chords)
    pad = np.zeros_like(t)
    for slot, notes in enumerate(chords):
        on = bar == slot
        pad[on] = sum(np.sin(2 * math.pi * f * t[on]) for f in notes) / len(notes)
    beat = t % 0.5
    kick = np.sin(2 * math.pi * (60 + 90 * np.exp(-beat * 30)) * beat) * np.exp(-beat * 9)
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.5)
    return (0.25 * pad + 0.5 * kick) * fade * 0.6


def _write_mp3(path: Path, samples: np.ndarray, rate: int = 44100) -> None:
    import av

    with av.open(str(path), "w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=rate)
        stream.layout = "mono"
        stream.bit_rate = 64000
        block = 1152
        for start in range(0, len(samples), block):
            chunk = samples[start:start + block].astype(np.float32)[None, :]
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="mono")
            frame.sample_rate = rate
            frame.pts = start
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _png(image_fn: Callable[[], Image.Image]) -> Callable[[Path], None]:
    return lambda path: image_fn().save(path, "PNG", optimize=True)


PORTRAIT, LANDSCAPE, SCAIL = (480, 832), (832, 480), (896, 512)

# Every name a public template's input node sets, beside ComfyUI's own example.png.
FILES: dict[str, Callable[[Path], None]] = {
    "qwen21_source.png": _png(_vase_scene),
    "qwen21_edit_mask_alpha.png": _png(_band_matte),
    "qwen21_canvas_rgba.png": _png(_canvas_rgba),
    "qwen21_mask_rgba.png": _png(_guide_rgba),
    **{f"qwen21_reference_{i:02d}_rgba.png": _png(lambda i=i: _tile(i)) for i in range(2, 10)},
    "reference.png": _png(_vase_scene),
    "reference_character.png": _png(lambda: _figure(SCAIL, 0.0)),
    "reference_character_mask.png": _png(lambda: _figure(SCAIL, 0.0, ground=(0, 0, 0), ink=(255, 255, 255))),
    "wan_animate2_reference.png": _png(lambda: _figure(PORTRAIT, 0.0)),
    "blue_dancer.png": _png(lambda: _figure(PORTRAIT, 0.0, ink=COBALT)),
    "driving_video.mp4": lambda path: _write_video(path, _frames(SCAIL, 81, 16), 16),
    "driving_video_mask.mp4": lambda path: _write_video(
        path, _frames(SCAIL, 81, 16, ground=(0, 0, 0), ink=(255, 255, 255)), 16),
    "wan_animate2_driving.mp4": lambda path: _write_video(path, _frames(PORTRAIT, 81, 16), 16),
    "source_video.mp4": lambda path: _write_video(path, _sliding_box(LANDSCAPE, 33), 16),
    "wan_dancer_reference_audio.mp3": lambda path: _write_mp3(path, _music()),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="file names to rewrite (default: all)")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.names) - set(FILES))
    if unknown:
        parser.error(f"unknown demo input(s): {', '.join(unknown)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.names or sorted(FILES):
        FILES[name](OUT / name)
        print(f"wrote {OUT.relative_to(REPO) / name} ({(OUT / name).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
