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


# Each pair is what comfyui_frontend_package 1.57.6's own processDynamicPrompt
# returned for the text under Node, with Math.random pinned to the first option.
_FRONTEND_PAIRS = (
    ("plain text, f/2.8", "plain text, f/2.8"),
    ("a{b}c", "abc"),
    ('{"k":{"n":[1,{"x":2}]}}', '"k":"n":[1,"x":2]'),
    ("x // comment\ny", "x \ny"),
    ("a /* b\nc */ d", "a  d"),
    (r"\{lit\}", "{lit}"),
    ("a|b", "a|b"),
    ("}", "}"),
    ("{unclosed", "unclosed"),
    ("end\\", "end\\undefined"),
    (r"\n stays", r"\n stays"),
    ("{x{y|z}w}", "xyw"),
    ("http://example.com", "http:"),
)


def test_the_dynamic_prompt_port_returns_what_the_frontend_parser_returned():
    for text, sent in _FRONTEND_PAIRS:
        assert canary.process_dynamic_prompt(text, lambda options: options[0]) == sent, text


def test_a_choice_block_sends_one_option_and_drops_its_braces():
    assert canary.process_dynamic_prompt("{a|b|c}", lambda options: options[-1]) == "c"


def test_dynamic_prompt_errors_name_only_flagged_widgets_holding_rewritten_text():
    schema = {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                           "clip": ("CLIP", {})},
              "optional": {"note": ("STRING", {"multiline": True})}}
    assert canary.dynamic_prompt_widgets(schema) == ["text"]
    assert canary.dynamic_prompt_errors(schema, {"text": "a photo, f/2.8", "note": "{x}"}) == []
    # A wired text arrives as a link, so the browser never rewrites it.
    assert canary.dynamic_prompt_errors(schema, {"text": ["7", 0]}) == []
    (error,) = canary.dynamic_prompt_errors(schema, {"text": '{"caption":"x"}'})
    assert error.startswith("text holds text the browser's dynamic prompt pass rewrites")
