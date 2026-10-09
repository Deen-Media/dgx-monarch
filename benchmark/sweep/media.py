#!/usr/bin/env python3
"""Write the five synthetic inputs used by testing-only workflows.

Use the workflows' dimensions and ComfyUI input directory. Preserve existing
files unless --force is set.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FFMPEG = "/usr/bin/ffmpeg"
# WanDancer renders 480 wide by 832 high, and SCAIL-2 resizes its own reference
# pair, so one portrait serves both. FlowRVS scales its frames to 832x480 and
# cuts the first 17. 81 frames sits on the 4k+1 grid the Wan VAE needs and
# matches the SCAIL-2 row's length, which truncates pose and mask to the
# shorter of the two.
PORTRAIT = (480, 832)
VIDEO = (832, 480)
FRAMES = 81
FPS = 16
AUDIO_RATE = 24000
AUDIO_SECONDS = 5
BLOB = [PORTRAIT[0] * 0.22, PORTRAIT[1] * 0.18, PORTRAIT[0] * 0.78, PORTRAIT[1] * 0.72]  # the soft shape both portraits carry


def _portrait(path: str) -> None:
    width, height = PORTRAIT
    rows, columns = np.mgrid[0:height, 0:width]
    canvas = np.stack([columns / width * 255, rows / height * 200 + 30,
                       (1 - rows / height) * 180 + 40], axis=-1).astype(np.uint8)
    image = Image.fromarray(canvas)
    ImageDraw.Draw(image).ellipse(BLOB, fill=(240, 235, 225))
    image.filter(ImageFilter.GaussianBlur(6)).save(path)


def _portrait_mask(path: str) -> None:
    image = Image.new("L", PORTRAIT, 0)
    ImageDraw.Draw(image).ellipse(BLOB, fill=255)
    image.filter(ImageFilter.GaussianBlur(4)).convert("RGB").save(path)


def _gradient_frames():
    width, height = VIDEO
    rows, columns = np.mgrid[0:height, 0:width]
    for index in range(FRAMES):
        phase = index / FRAMES
        yield np.stack([(columns / width + phase) % 1.0 * 255,
                        rows / height * 255,
                        np.full((height, width), phase * 255)],
                       axis=-1).astype(np.uint8)


def _mask_frames():
    """One identity travelling across a white field.

    SCAIL-2 runs in replacement mode, where comfy's own mask renderer paints a
    white background and the first palette colour, blue, on the subject. A
    black field with a white box is the animation-mode convention and reads as
    a different identity.
    """
    width, height = VIDEO
    box = (width // 4, height // 2)
    for index in range(FRAMES):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        left = int(index / (FRAMES - 1) * (width - box[0]))
        top = (height - box[1]) // 2
        frame[top:top + box[1], left:left + box[0]] = (0, 0, 255)
        yield frame


def _encode(path: str, frames) -> None:
    command = [FFMPEG, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{VIDEO[0]}x{VIDEO[1]}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    with encoder.stdin as stream:
        for frame in frames:
            stream.write(frame.tobytes())
    if encoder.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {path}")


def _audio(path: str) -> None:
    steps = np.arange(AUDIO_RATE * AUDIO_SECONDS) / AUDIO_RATE
    tones = 0.5 * np.sin(2 * np.pi * 220 * steps) + 0.3 * np.sin(2 * np.pi * 440 * steps)
    with wave.open(path, "wb") as track:
        track.setnchannels(1)
        track.setsampwidth(2)
        track.setframerate(AUDIO_RATE)
        track.writeframes((tones / 0.8 * 32000).astype("<i2").tobytes())


JOBS = (
    ("dgxm_test_portrait.png", _portrait),
    ("dgxm_test_portrait_mask.png", _portrait_mask),
    ("dgxm_test_video.mp4", lambda path: _encode(path, _gradient_frames())),
    ("dgxm_test_mask.mp4", lambda path: _encode(path, _mask_frames())),
    ("dgxm_test_audio.wav", _audio),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="write the sweep's test media")
    parser.add_argument("--out-dir", default=os.path.expanduser("~/ComfyUI/input"),
                        help="directory to write into (default: ~/ComfyUI/input)")
    parser.add_argument("--force", action="store_true", help="rewrite what is there")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, write in JOBS:
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path) and not args.force:
            print(f"kept {path}")
            continue
        write(path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
