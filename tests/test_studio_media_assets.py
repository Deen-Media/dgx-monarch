"""Image inputs are decoded, sanitized and bounded before crossing to ComfyUI."""
import asyncio
import base64
import hashlib
import io

import pytest
from PIL import Image

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.media_assets import SOURCE_BYTES, UPLOAD_BYTES, MediaAssets, prepare_upload, thumbnail


def image_body(mode="RGBA", size=(600, 400)):
    image = Image.new(mode, size)
    data = io.BytesIO()
    image.save(data, "PNG")
    return {"kind": "mask", "media_type": "image/png", "data_base64": base64.b64encode(data.getvalue()).decode()}


def detailed_jpeg(size=(1800, 1800)):
    """A small JPEG whose decoded pixels do not compress: its RGBA PNG passes 8 MiB."""
    image = Image.frombytes("RGB", size, hashlib.shake_256(b"studio").digest(size[0] * size[1] * 3))
    data = io.BytesIO()
    image.save(data, "JPEG", quality=35)
    return data.getvalue()


def test_mask_alpha_and_dimensions_survive_normalization_and_thumbnail_is_bounded():
    data, metadata = prepare_upload(image_body())
    with Image.open(io.BytesIO(data)) as result:
        assert result.size == (600, 400)
        assert result.getpixel((0, 0))[3] == 0
    assert metadata["orientation"] == "normalized"
    with Image.open(io.BytesIO(thumbnail(data))) as result:
        assert result.size == (320, 213)
        assert result.mode == "RGB"


@pytest.mark.parametrize(("updates", "code", "status"), [
    ({"kind": []}, "asset_kind", 400), ({"media_type": []}, "asset_type", 400),
    ({"data_base64": "!invalid"}, "asset_encoding", 400),
    ({"data_base64": base64.b64encode(b"<svg onload='alert(1)'/>").decode()}, "invalid_image", 400),
    ({"data_base64": "a" * (3 * 1024 * 1024)}, "asset_limit", 413),
    # Each bound alone: the encoded length refuses before decoding, and the decoded
    # length refuses the one byte past 2 MiB that the encoded length admits.
    ({"data_base64": "!" * (3 * 1024 * 1024)}, "asset_limit", 413),
    ({"data_base64": base64.b64encode(bytes(UPLOAD_BYTES + 1)).decode()}, "asset_limit", 413),
])
def test_untrusted_or_oversized_uploads_refuse(updates, code, status):
    with pytest.raises(StudioError) as error:
        prepare_upload({**image_body(), **updates})
    assert (error.value.code, error.value.status) == (code, status)


def test_pixel_budget_precedes_decode_and_says_to_downscale(monkeypatch):
    from dgx_monarch.studio import media_assets
    monkeypatch.setattr(media_assets, "IMAGE_PIXELS", 10)
    with pytest.raises(StudioError) as error:
        prepare_upload(image_body())
    assert (error.value.code, error.value.status) == ("image_limit", 413)
    assert "Downscale the image to at most 16 megapixels" in str(error.value)


def test_detailed_input_within_the_stated_limits_is_not_refused_for_its_png_size():
    raw = detailed_jpeg()
    assert len(raw) <= UPLOAD_BYTES
    data, metadata = prepare_upload({"kind": "input", "media_type": "image/jpeg",
                                     "data_base64": base64.b64encode(raw).decode()})
    assert len(data) > 8 * 1024 * 1024
    assert (metadata["width"], metadata["height"], metadata["media_type"]) == (1800, 1800, "image/png")
    with Image.open(io.BytesIO(data)) as result:
        assert result.mode == "RGBA"


def test_backstop_sits_above_every_png_within_the_pixel_rule():
    assert SOURCE_BYTES > 16 * 1024 * 1024 * 4


def test_normalized_size_backstop_refuses_and_says_to_downscale(monkeypatch):
    from dgx_monarch.studio import media_assets
    monkeypatch.setattr(media_assets, "SOURCE_BYTES", 64)
    with pytest.raises(StudioError) as error:
        prepare_upload(image_body())
    assert (error.value.code, error.value.status) == ("asset_limit", 413)
    assert "Downscale the image" in str(error.value)


class _Response:
    def __init__(self, data, length):
        self.data, self.content_length, self.released = data, length, False
        self.content = self

    async def iter_chunked(self, size):
        for start in range(0, len(self.data), size):
            yield self.data[start:start + size]

    def release(self):
        self.released = True


class _Client:
    def __init__(self, response):
        self.response = response

    async def media(self, descriptor):
        return self.response


class _Store:
    def get(self, kind, identity):
        return {"id": identity, "filename": "large.png", "subfolder": "", "type": "output"}


def _large_png():
    data = io.BytesIO()
    Image.open(io.BytesIO(detailed_jpeg())).save(data, "PNG")
    return data.getvalue()


@pytest.mark.parametrize("declared", [True, False])
def test_thumbnail_reads_sources_past_8_mib_within_the_pixel_rule(declared):
    data = _large_png()
    assert len(data) > 8 * 1024 * 1024
    response = _Response(data, len(data) if declared else None)
    result = asyncio.run(MediaAssets(_Store(), _Client(response)).preview("large"))
    with Image.open(io.BytesIO(result)) as image:
        assert image.size == (320, 320)
    assert response.released


@pytest.mark.parametrize("declared", [True, False])
def test_thumbnail_backstop_refuses_declared_and_streamed_sources(monkeypatch, declared):
    from dgx_monarch.studio import media_assets
    monkeypatch.setattr(media_assets, "SOURCE_BYTES", 1024)
    data = _large_png()
    response = _Response(data, len(data) if declared else None)
    with pytest.raises(StudioError) as error:
        asyncio.run(MediaAssets(_Store(), _Client(response)).preview("large"))
    assert (error.value.code, error.value.status) == ("thumbnail_limit", 413)
    assert response.released
