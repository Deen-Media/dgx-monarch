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


def test_dynamic_widget_errors_hold_the_generators_rows_to_the_schema():
    """Each key the generator lists must be offered and reveal the same children in order."""
    from benchmark.sweep import convert

    def schema(children):
        return {"required": {"mode": ("COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "on", "inputs": {"required": {name: ("FLOAT", {}) for name in children}}},
            {"key": "off", "inputs": {}}]}), "label": ("STRING", {})}}

    def generated(rows):
        return [convert.DynamicName("mode", {key: [f"mode.{child}" for child in children]
                                             for key, children in rows.items()}, closed=True),
                "label"]

    master = schema(["temperature", "top_k"])
    names = convert.object_info_widget_names(master)
    rows = {"on": ["temperature", "top_k"], "off": []}
    assert canary.dynamic_widget_errors("Node", generated(rows), names, master) == []
    # A child ComfyUI moved, added or dropped shifts every later value.
    for drifted in (["top_k", "temperature"], ["temperature"], ["temperature", "top_k", "seed"]):
        (error,) = canary.dynamic_widget_errors("Node", generated({"on": drifted}), names, master)
        assert "Node.mode key 'on' generated children" in error
    (error,) = canary.dynamic_widget_errors("Node", generated({"auto": []}), names, master)
    assert "lists the key 'auto', which comfy master does not offer" in error
    # A combo that reveals nothing on this ComfyUI still offers its keys.
    bare = schema([])
    assert canary.dynamic_widget_errors(
        "Node", generated({"on": [], "off": []}), convert.object_info_widget_names(bare), bare) == []
    (error,) = canary.dynamic_widget_errors(
        "Node", generated(rows), convert.object_info_widget_names(bare), bare)
    assert "but comfy master reveals []" in error


def test_parse_chain_errors_name_the_answer_whose_caption_drifted():
    """The caption read's table, through nodes that behave and through ones that do not."""
    settings = {"key": "expanded_caption", "regex_pattern": r'"expanded_caption"\s*:\s*"(.*)"\s*\}',
                "mode": "First Group", "case_insensitive": False, "multiline": False,
                "dotall": True, "group_index": 1}
    calls = []

    def nodes(strict, loose):
        def run(kind, *args):
            calls.append(kind)
            if kind == "JsonExtractString":
                return strict
            if kind == "RegexExtract":
                assert args[1:] == (settings["regex_pattern"], "First Group", False, False, True, 1)
                return loose
            if kind == "StringTrim":
                return args[0].strip()
            return args[0] == args[1]
        return run

    # The strict read wins, and the loose read is not asked.
    assert canary.caption_from("answer", settings, nodes("A boat.", "other")) == ("A boat.", "strict")
    assert "RegexExtract" not in calls
    # An empty or blank strict read hands over to the loose one, and then to the request.
    assert canary.caption_from("answer", settings, nodes("  ", "A boat.")) == ("A boat.", "loose")
    assert canary.caption_from("answer", settings, nodes("", " \n")) == ("", "request")
    # Nodes that read nothing fail every row that expects a caption, by name.
    errors = canary.parse_chain_errors(settings, nodes("", ""))
    expecting = [what for what, _answer, _caption, read in canary.PARSE_CASES if read != "request"]
    assert len(errors) == len(expecting) == 16
    assert errors[0].startswith("caption read: a valid object: expected ('A photograph of a boat.', 'strict')")


def test_parse_settings_reads_the_graphs_own_widget_values():
    api = {"1": {"class_type": "JsonExtractString", "inputs": {"json_string": ["9", 0], "key": "k"}},
           "2": {"class_type": "RegexExtract",
                 "inputs": {"string": ["9", 0], "regex_pattern": "p", "mode": "First Group",
                            "case_insensitive": False, "multiline": False, "dotall": True,
                            "group_index": 1}}}
    assert canary.parse_settings(api) == {
        "key": "k", "regex_pattern": "p", "mode": "First Group", "case_insensitive": False,
        "multiline": False, "dotall": True, "group_index": 1}
