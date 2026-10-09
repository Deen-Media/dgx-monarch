"""Unit coverage for narrow enum checks in the real-Comfy template canary."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "dgxm_template_widget_canary", REPO / "tests" / "canary" / "template_widget_canary.py"
)
assert _SPEC and _SPEC.loader
canary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(canary)


def test_qwen21_clip_enum_accepts_stock_value_and_rejects_family_name():
    schema = {"required": {"type": (["stable_diffusion", "qwen_image"],)}}
    assert canary.clip_loader_type_error(schema, "qwen_image") is None
    assert "qwen_image21" in canary.clip_loader_type_error(schema, "qwen_image21")


def test_create_video_static_enums_reject_shifted_values():
    schema = {"optional": {"bit_depth": (["auto", 8, 10],),
                           "color_space": (["sRGB", "HDR", "HDR PQ"],),
                           "codec": (["none", "auto"],)}}
    assert canary.create_video_enum_errors(
        schema, {"bit_depth": 8, "color_space": "sRGB", "codec": "none"}
    ) == []
    errors = canary.create_video_enum_errors(
        schema, {"bit_depth": "", "color_space": 8, "codec": "sRGB"}
    )
    assert len(errors) == 3
