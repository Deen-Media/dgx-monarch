"""Unit coverage for the narrow enum checks in the real-Comfy template canary."""
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


def test_create_video_chroma_subsampling_must_be_allowed_and_offered():
    master = {"optional": {"chroma_subsampling": ("COMBO", {"options": ["auto", "444", "422", "420"]}),
                           "codec": ("COMBO", {"options": ["none", "auto", "h264", "av1", "prores"]})}}
    assert canary.create_video_enum_errors(master, {"chroma_subsampling": "auto", "codec": "none"}) == []
    # ComfyUI 0.44.0 offers prores; 0.43 does not, so no template may hold it.
    assert "is not one of" in canary.create_video_enum_errors(master, {"codec": "prores"})[0]
    assert "is not one of" in canary.create_video_enum_errors(master, {"chroma_subsampling": "none"})[0]
    narrow = {"optional": {"codec": ("COMBO", {"options": ["none"]})}}
    assert "not offered here" in canary.create_video_enum_errors(narrow, {"codec": "h264"})[0]


def test_save_video_format_and_codec_stay_among_the_offered_options():
    def option(key, codecs):
        return {"key": key, "inputs": {"required": {"codec": ("COMFY_DYNAMICCOMBO_V3",
                                                              {"options": [{"key": c} for c in codecs]})}}}
    older = {"required": {"format": ("COMFY_DYNAMICCOMBO_V3", {"options": [
        option("auto", ["auto", "h264", "av1"]), option("mp4", ["auto", "h264", "av1"]),
        option("webm", ["auto", "av1"])]})}}
    assert canary.save_video_errors(older, {"format": "mp4", "codec": "h264"}) == []
    assert canary.save_video_errors(older, {"format": "auto", "codec": "auto"}) == []
    assert "format 'mov'" in canary.save_video_errors(older, {"format": "mov", "codec": "prores"})[0]
    assert "codec 'h264'" in canary.save_video_errors(older, {"format": "webm", "codec": "h264"})[0]
    assert canary.save_video_errors({"required": {}}, {"format": "mp4"}) == ["SaveVideo has no format options"]
