"""Generate the ComfyUI browser's JPEG previews from the shared Monarch SVG.

Run with no names to regenerate every public card, or pass template keys.
``--check`` verifies the committed exports without writing files. Install the
project's dev dependencies and DejaVu Sans to reproduce the cards.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

CARD = (768, 432)
BACKGROUND = (20, 24, 27)
TITLE_COLOUR = (244, 246, 240)
TAGLINE_COLOUR = (172, 182, 178)
ACCENT = (135, 249, 0)
LABEL = "DGX MONARCH"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
REPO = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO / "example_workflows"
LOGO = REPO / "web/brand/monarch-dark.svg"
TEXT_LEFT = 48
TEXT_RIGHT = CARD[0] - 48

# Names and descriptions are kept separate from model prompts and settings.
CARDS: dict[str, tuple[str, str]] = {
    "dgx-monarch-quickstart": ("Quickstart", "Your first image with DGX Monarch"),
    "dgx-monarch-dual-spark-split": ("Two-Spark render", "Share one image render across both Sparks"),
    "dgx-monarch-fleet": ("Fleet", "Run independent prompts on both Sparks"),
    "dgx-monarch-identity-gate": ("Identity gate", "Compare optimized loading with stock"),
    "dgx-monarch-lora-low-rss": ("LoRA stack", "Apply LoRAs with lower memory use"),
    "dgx-monarch-flux1-t2i": ("FLUX.1", "Text to image"),
    "dgx-monarch-flux2-t2i": ("FLUX.2", "Text to image"),
    "dgx-monarch-chroma-t2i": ("Chroma 1 HD", "Text to image"),
    "dgx-monarch-radiance-t2i": ("Chroma 1 Radiance", "Generate images directly in pixel space"),
    "dgx-monarch-longcat-t2i": ("LongCat-Image", "Text to image"),
    "dgx-monarch-qwen-image-t2i": ("Qwen-Image", "Text to image"),
    "dgx-monarch-qwen-image21-t2i": ("Qwen Image 2.1", "Text to image"),
    "dgx-monarch-qwen-image21-edit-rgba": ("Qwen Image 2.1 Edit", "Edit with up to ten reference images"),
    "dgx-monarch-qwen-image21-edit-mask-rgba": ("Qwen Image 2.1 Mask", "Edit with an alpha-masked reference"),
    "dgx-monarch-mage-flow-t2i": ("Mage Flow", "Text to image"),
    "dgx-monarch-mage-flow-t2i-turbo": ("Mage Flow Turbo", "Four-step image generation"),
    "dgx-monarch-mage-flow-edit": ("Mage Flow Edit", "Edit an image with supporting references"),
    "dgx-monarch-mage-flow-edit-turbo": ("Mage Flow Edit Turbo", "Four-step reference image editing"),
    "dgx-monarch-ernie-t2i": ("ERNIE-Image", "Text to image"),
    "dgx-monarch-zimage-t2i": ("Z-Image", "Text to image"),
    "dgx-monarch-zimage-dct-t2i": ("Z-Image DCT", "Image generation without a VAE"),
    "dgx-monarch-lens-t2i": ("Lens", "Text to image"),
    "dgx-monarch-omnigen2-t2i": ("OmniGen 2", "Text to image"),
    "dgx-monarch-anima-t2i": ("Anima", "Anime image generation"),
    "dgx-monarch-boogu-t2i": ("Boogu", "Text to image"),
    "dgx-monarch-pixeldit-t2i": ("PixelDiT", "Generate images directly in pixel space"),
    "dgx-monarch-pid-4k": ("PiD 4K", "Upscale an image to 4096 x 4096"),
    "dgx-monarch-krea2-t2i": ("Krea2", "Text to image"),
    "dgx-monarch-ideogram4-t2i": ("Ideogram4", "Text to image"),
    "dgx-monarch-hunyuan-image": ("HunyuanImage 2.1", "Text to image"),
    "dgx-monarch-hunyuan-video": ("HunyuanVideo 1.5", "Text to video"),
    "dgx-monarch-kandinsky5-image": ("Kandinsky 5 Image", "Text to image"),
    "dgx-monarch-kandinsky5-video-lite": ("Kandinsky 5 Video Lite", "Text to video"),
    "dgx-monarch-kandinsky5-video-pro": ("Kandinsky 5 Video Pro", "Text to video"),
    "dgx-monarch-minimax-h3-t2va": ("MiniMax H3", "Generate video and audio from text"),
    "dgx-monarch-minimax-h3-guide": ("MiniMax H3 Guide", "Guide video with a reference image"),
    "dgx-monarch-ltx-t2v": ("LTX 2.3", "Text to video"),
    "dgx-monarch-ltx-i2v-av": ("LTX 2.3 Image to Video", "Turn a still image into video and audio"),
    "dgx-monarch-ltx25-t2v": ("LTX 2.5", "Generate video and audio from text"),
    "dgx-monarch-ltx25-i2v": ("LTX 2.5 Image to Video", "Start a video from your own image"),
    "dgx-monarch-ltx25-i2v-guide": ("LTX 2.5 Guide", "Guide a video with an image"),
    "dgx-monarch-ltx25-flf2v": ("LTX 2.5 First + Last", "Set the opening and closing frames"),
    "dgx-monarch-wan-t2v": ("Wan 2.1", "Text to video"),
    "dgx-monarch-wan21-i2v": ("Wan 2.1 Image to Video", "Start a video from your own image"),
    "dgx-monarch-wan22-t2v": ("Wan 2.2", "Text to video"),
    "dgx-monarch-wan22-i2v": ("Wan 2.2 Image to Video", "Start a video from your own image"),
    "dgx-monarch-wan-animate2": ("Wan-Animate 2", "Transfer motion to a character"),
    "dgx-monarch-wan-animate2-bfloat16-warm": ("Wan-Animate 2 BF16", "Motion transfer with weights kept loaded"),
    "dgx-monarch-wan-animate2-memory-saver": ("Wan-Animate 2 Memory Saver", "Release model memory after each render"),
    "dgx-monarch-wan-animate2-base-quality": ("Wan-Animate 2 Base", "Motion transfer without a distillation LoRA"),
    "dgx-monarch-wan-animate2-distilled": ("Wan-Animate 2 Distilled", "Ten-step motion transfer"),
    "dgx-monarch-wan-flowrvs": ("Wan FlowRVS", "Create video masks from a text description"),
    "dgx-monarch-wan-bernini": ("Wan Bernini-R", "Text to video"),
    "dgx-monarch-wan-scail": ("Wan SCAIL Preview", "Animate a character from an image"),
    "dgx-monarch-wan-scail2": ("Wan SCAIL-2", "Drive a character with your own video"),
    "dgx-monarch-wandancer": ("WanDancer", "Animate a dancer to music"),
    "dgx-monarch-cogvideox-t2v": ("CogVideoX", "Text to video"),
    "dgx-monarch-cogvideox-i2v": ("CogVideoX Image to Video", "Start a video from your own image"),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def _lines(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if font.getlength(word) > width:
            raise ValueError(f"Word exceeds card width: {word}")
        if lines and font.getlength(lines[-1] + " " + word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def draw_card(title: str, tagline: str) -> Image.Image:
    card = Image.new("RGB", CARD, BACKGROUND)
    pen = ImageDraw.Draw(card)
    logo_bytes = cairosvg.svg2png(url=str(LOGO), output_width=112, output_height=112)
    logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
    card.paste(logo, (44, 32), logo)
    pen.text((177, 80), LABEL, font=_font(20, True), fill=TITLE_COLOUR, anchor="lm")
    pen.line((48, 162, 720, 162), fill=(51, 61, 55), width=1)
    title_font = _font(36, True)
    title_lines = _lines(title, title_font, TEXT_RIGHT - TEXT_LEFT)
    if len(title_lines) > 2:
        raise ValueError(f"Title exceeds two lines: {title}")
    for index, line in enumerate(title_lines):
        pen.text((TEXT_LEFT, 197 + index * 46), line,
                 font=title_font, fill=TITLE_COLOUR, anchor="lt")
    tagline_font = _font(20)
    tagline_lines = _lines(tagline, tagline_font, TEXT_RIGHT - TEXT_LEFT)
    if len(tagline_lines) > 2:
        raise ValueError(f"Description exceeds two lines: {tagline}")
    for index, line in enumerate(tagline_lines):
        pen.text((TEXT_LEFT, 305 + index * 28), line,
                 font=tagline_font, fill=TAGLINE_COLOUR, anchor="lt")
    pen.line((48, 386, 94, 386), fill=ACCENT, width=3)
    return card


def card_bytes(title: str, tagline: str) -> bytes:
    output = io.BytesIO()
    draw_card(title, tagline).save(output, "JPEG", quality=92, optimize=True)
    return output.getvalue()



def matches_preview(actual: bytes, expected: bytes) -> bool:
    """Compare decoded JPEGs, allowing tiny renderer rounding differences.

    JPEG metadata and entropy encoding can differ across library versions.
    Accept at most 8 levels per channel at any pixel and 0.25 levels averaged
    across the image. Text changes and shifted edges exceed the per-pixel cap.
    Larger font or renderer differences still require visual review and a new
    export; this is not a substitute for a pinned rendering environment.
    """
    try:
        with Image.open(io.BytesIO(actual)) as saved, Image.open(io.BytesIO(expected)) as fresh:
            if saved.format != "JPEG" or saved.size != CARD:
                return False
            difference = ImageChops.difference(saved.convert("RGB"), fresh.convert("RGB"))
            return (all(high <= 8 for _, high in difference.getextrema())
                    and max(ImageStat.Stat(difference).mean) <= 0.25)
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="template keys, no extension; default: all")
    parser.add_argument("--check", action="store_true", help="check exports without writing")
    args = parser.parse_args(argv)
    names = args.names or sorted(CARDS)
    unknown = sorted(set(names) - CARDS.keys())
    if unknown:
        parser.error(f"No card text for: {', '.join(unknown)}")
    drift = []
    for name in names:
        expected = card_bytes(*CARDS[name])
        path = TEMPLATE_DIR / f"{name}.jpg"
        if args.check:
            if not path.exists() or not matches_preview(path.read_bytes(), expected):
                drift.append(name)
        else:
            path.write_bytes(expected)
    if drift:
        print("Regenerate card previews: " + ", ".join(drift))
        return 1
    print(f"{'Checked' if args.check else 'Wrote'} {len(names)} card previews")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
