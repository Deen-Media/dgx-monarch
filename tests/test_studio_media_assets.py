"""Image inputs are decoded, sanitized and bounded before crossing to ComfyUI."""
import base64
import io

import pytest
from PIL import Image

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.media_assets import prepare_upload, thumbnail


def image_body(mode="RGBA", size=(600, 400)):
    image = Image.new(mode, size)
    data = io.BytesIO()
    image.save(data, "PNG")
    return {"kind": "mask", "media_type": "image/png", "data_base64": base64.b64encode(data.getvalue()).decode()}


def test_mask_alpha_and_dimensions_survive_normalization_and_thumbnail_is_bounded():
    data, metadata = prepare_upload(image_body())
    with Image.open(io.BytesIO(data)) as result:
        assert result.size == (600, 400)
        assert result.getpixel((0, 0))[3] == 0
    assert metadata["orientation"] == "normalized"
    with Image.open(io.BytesIO(thumbnail(data))) as result:
        assert result.size == (320, 213)
        assert result.mode == "RGB"


@pytest.mark.parametrize("updates", [
    {"kind": []}, {"media_type": []}, {"data_base64": "!invalid"},
    {"data_base64": base64.b64encode(b"<svg onload='alert(1)'/>").decode()},
    {"data_base64": "a" * (3 * 1024 * 1024)},
])
def test_untrusted_or_oversized_uploads_refuse(updates):
    with pytest.raises(StudioError):
        prepare_upload({**image_body(), **updates})


def test_pixel_budget_precedes_decode(monkeypatch):
    from dgx_monarch.studio import media_assets
    monkeypatch.setattr(media_assets, "IMAGE_PIXELS", 10)
    with pytest.raises(StudioError) as error:
        prepare_upload(image_body())
    assert error.value.status == 413
