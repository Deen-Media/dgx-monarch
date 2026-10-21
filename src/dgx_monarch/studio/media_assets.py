"""Validated input images and bounded, on-demand gallery thumbnails."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import json
import warnings
from collections import OrderedDict
from typing import Any

import aiohttp
from PIL import Image, ImageOps, UnidentifiedImageError

from .comfy import ComfyClient, safe_descriptor
from .contracts import StudioError, identifier, timestamp
from .store import Store

UPLOAD_BYTES = 2 * 1024 * 1024
# The one decoded-size rule for uploads and thumbnails: Studio decodes at most
# IMAGE_PIXELS (16 megapixels, 4096 x 4096) per image, checked from the header
# before any pixel is decoded.
IMAGE_PIXELS = 16 * 1024 * 1024
# A backstop only: four bytes per RGBA pixel at the pixel limit, plus 1 MiB for
# PNG filter bytes, compression framing and embedded metadata. No 8-bit image
# within the pixel rule reaches it, so the pixel rule is what binds.
SOURCE_BYTES = IMAGE_PIXELS * 4 + 1024 * 1024
CACHE_BYTES = 32 * 1024 * 1024
_DOWNSCALE = "Downscale the image to at most 16 megapixels (4096 x 4096 pixels) and try again."


def _image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(io.BytesIO(data))
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise StudioError("image_limit", "Use a PNG, JPEG or WebP image.", 413)
            if source.width * source.height > IMAGE_PIXELS:
                raise StudioError("image_limit", _DOWNSCALE, 413)
            source.load()
            return ImageOps.exif_transpose(source).convert("RGBA")
    except StudioError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise StudioError("invalid_image", "The input is not a complete supported image.") from exc


def prepare_upload(body: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(body.get("kind"), str) or body["kind"] not in {"input", "mask"}:
        raise StudioError("asset_kind", "Choose an input image or mask.")
    if not isinstance(body.get("media_type"), str) or body["media_type"] not in {"image/png", "image/jpeg", "image/webp"}:
        raise StudioError("asset_type", "Use PNG, JPEG or WebP input.")
    encoded = body.get("data_base64")
    if not isinstance(encoded, str) or len(encoded) > ((UPLOAD_BYTES + 2) // 3) * 4:
        raise StudioError("asset_limit", "Input images must be at most 2 MiB.", 413)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise StudioError("asset_encoding", "Input image encoding is invalid.") from exc
    if len(raw) > UPLOAD_BYTES:
        raise StudioError("asset_limit", "Input images must be at most 2 MiB.", 413)
    image = _image(raw)
    output = io.BytesIO()
    # RGBA keeps mask alpha; the pixel rule, not the encoded size, bounds the result.
    image.save(output, format="PNG")
    payload = output.getvalue()
    if len(payload) > SOURCE_BYTES:
        raise StudioError("asset_limit", "The normalized image exceeds the 65 MiB backstop. " + _DOWNSCALE, 413)
    return payload, {"width": image.width, "height": image.height, "media_type": "image/png",
                     "orientation": "normalized", "sha256": hashlib.sha256(payload).hexdigest()}


def thumbnail(data: bytes) -> bytes:
    image = _image(data)
    image.thumbnail((320, 320))
    background = Image.new("RGB", image.size, "#111318")
    background.paste(image, mask=image.getchannel("A"))
    result = io.BytesIO()
    background.save(result, "JPEG", quality=78)
    return result.getvalue()


class MediaAssets:
    def __init__(self, store: Store, client: ComfyClient):
        self.store, self.client = store, client
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.cache_bytes = 0
        self.semaphore = asyncio.Semaphore(2)

    async def upload(self, body: dict[str, Any]) -> dict[str, Any]:
        async with self.semaphore:
            data, metadata = await asyncio.to_thread(prepare_upload, body)
            if self.client.session is None:
                raise StudioError("driver_offline", "Connect the driver before uploading inputs.", 503)
            name = identifier() + ".png"
            form = aiohttp.FormData()
            form.add_field("image", data, filename=name, content_type="image/png")
            form.add_field("type", "input")
            form.add_field("subfolder", "studio")
            form.add_field("overwrite", "false")
            try:
                async with self.client.session.post(self.client.origin + "/upload/image", data=form,
                                                    allow_redirects=False) as response:
                    raw = bytearray()
                    async for chunk in response.content.iter_chunked(8192):
                        raw.extend(chunk)
                        if len(raw) > 65536:
                            raise StudioError("asset_upload", "Driver upload reply exceeded the response budget.", 502)
                    if response.status != 200:
                        raise StudioError("asset_upload", "Driver did not accept the input image.", 502)
                    result = json.loads(raw)
                    if not isinstance(result, dict):
                        raise StudioError("asset_upload", "Driver returned an invalid input identity.", 502)
                    descriptor = safe_descriptor({**result, "filename": result.get("name")})
                    if descriptor["type"] != "input":
                        raise StudioError("asset_upload", "Driver did not return an input asset.", 502)
            except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                if isinstance(exc, StudioError):
                    raise
                raise StudioError("asset_upload_unknown", "Upload was interrupted; no job was submitted.", 503) from exc
            binding = "/".join(filter(None, (descriptor["subfolder"], descriptor["filename"])))
            return self.store.create("artifacts", {"id": identifier(), **descriptor, **metadata,
                                     "metadata": metadata, "kind": body["kind"], "job_id": None,
                                     "binding_value": binding, "created_at": timestamp()})

    async def preview(self, artifact_id: str) -> bytes:
        if artifact_id in self.cache:
            self.cache.move_to_end(artifact_id)
            return self.cache[artifact_id]
        async with self.semaphore:
            descriptor = self.store.get("artifacts", artifact_id)
            if descriptor.get("source") == "live_preview":
                from .live_preview import preview_frame
                return preview_frame(self.store, artifact_id)[0]
            response = await self.client.media(descriptor)
            try:
                if response.content_length and response.content_length > SOURCE_BYTES:
                    raise StudioError("thumbnail_limit", "Source exceeds the 65 MiB thumbnail backstop.", 413)
                raw = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    raw.extend(chunk)
                    if len(raw) > SOURCE_BYTES:
                        raise StudioError("thumbnail_limit", "Source exceeds the 65 MiB thumbnail backstop.", 413)
            finally:
                response.release()
            result = await asyncio.to_thread(thumbnail, bytes(raw))
            previous = self.cache.pop(artifact_id, None)
            self.cache_bytes -= len(previous) if previous else 0
            self.cache[artifact_id] = result
            self.cache_bytes += len(result)
            while self.cache_bytes > CACHE_BYTES:
                _, evicted = self.cache.popitem(last=False)
                self.cache_bytes -= len(evicted)
            return result
