"""Studio reads a failed job's refusal the way dgx_monarch writes it.

studio/src/jobFailure.ts turns a job's recorded error into the class, first
sentence, remedy, troubleshooting entry and consent request that the viewer
status line and Queue show. These tests build refusals with the real builders
(``refusal()``, ``consent_descriptor.encode()`` and
``accuracy_waiver.card_tail()``), wrap one the way Monarch delivers a worker
raise, record each through the companion's own observation path, and run the
browser module under Node on the stored error. The Python parsers are the
reference, so a change on either side fails here.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dgx_monarch import accuracy_waiver, consent_descriptor
from dgx_monarch.refusal import CLASS_TITLES, PanelAction, RefusalClass, parse_refusal_tag, refusal
from dgx_monarch.studio import job_observations

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "studio/src/jobFailure.ts"
HEADLINE = ("capacity rescue available for flux/dev.safetensors on spark-a: stock residency needs 52.1 GiB "
            "of unified memory and only 40.2 GiB is available.")


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to run Studio's failure reader")
    return node


def browser(errors: list[dict], name: str = "jobFailure") -> list:
    """What the browser module makes of each recorded error."""
    program = (f"import * as failure from {json.dumps(MODULE.as_uri())};\n"
               f"const errors = {json.dumps(errors)};\n"
               f"process.stdout.write(JSON.stringify(errors.map((error) => failure.{name}(error))));")
    result = subprocess.run([_node(), "--input-type=module", "-e", program], capture_output=True,
                            text=True, timeout=60, check=True)
    return json.loads(result.stdout)


def recorded(message: str, code: str = "RuntimeError") -> dict:
    """The error the companion stores for one driver execution_error event."""
    parsed = job_observations.observation({"type": "execution_error", "data": {
        "prompt_id": "prompt", "exception_type": code, "exception_message": message}})
    assert parsed is not None
    return parsed[1]["error"]


def remote(exc: Exception, depth: int = 40) -> str:
    """The text Monarch gives ComfyUI for a worker raise: its remote traceback, then the refusal."""
    from monarch.actor import ActorError

    source = "".join(f"def frame_{index}(exc):\n    return frame_{index + 1}(exc)\n" for index in range(depth))
    source += f"def frame_{depth}(exc):\n    raise exc\n"
    frames: dict = {}
    exec(compile(source, "/srv/dgx-monarch/src/dgx_monarch/actor/worker_calls.py", "exec"), frames)  # noqa: S102
    try:
        frames["frame_0"](exc)
    except type(exc) as raised:
        return str(ActorError(raised))
    raise AssertionError("the frames did not raise")


def rescue_descriptor() -> consent_descriptor.ConsentDescriptor:
    return consent_descriptor.ConsentDescriptor(
        version=consent_descriptor.CONSENT_DESCRIPTOR_VERSION,
        kind=consent_descriptor.KIND_RESCUE_SLAB,
        consent_id=consent_descriptor.consent_id("rescue-slab", "size:35000000000", "c0ffee00c0ffee00"),
        unet_name="flux/dev.safetensors",
        file_identity="size:35000000000",
        context_fingerprint="c0ffee00c0ffee00",
        family_hint="flux",
        measured={"checkpoint_bytes": 35_000_000_000, "mem_available_bytes": 43_000_000_000, "host": "spark-a"},
        memo_context={"combo_key": "flux/dev.safetensors"},
        human_reason=("flux/dev.safetensors needs 52.1 GiB of unified memory under stock residency and only "
                      "40.2 GiB is available. Zero-copy slab residency fits it."),
        evidence=consent_descriptor.EVIDENCE_BYTE_VERIFIED,
        panel_action="Load with slab residency",
        env_fallback=consent_descriptor.ENV_SLAB_RESCUE,
    )


def rescue_refusal() -> str:
    return refusal(RefusalClass.CAPACITY, f"{HEADLINE} Zero-copy slab residency fits this checkpoint.",
                   guard="stock_load_preflight", waivable=True,
                   panel_action=PanelAction("Load with slab residency", consent_descriptor.ENV_SLAB_RESCUE),
                   troubleshooting=12) + "\n" + consent_descriptor.encode(rescue_descriptor())


def expected_consent(message: str) -> dict | None:
    parsed = consent_descriptor.parse(message)
    return None if parsed is None else {
        "kind": parsed.kind, "consentId": parsed.consent_id, "unetName": parsed.unet_name,
        "humanReason": parsed.human_reason, "panelAction": parsed.panel_action}


def expected_remedy(message: str) -> str:
    found = [match.group(0) for match in job_observations._REMEDY.finditer(message)][-1:]
    if not found or not found[0].startswith(" Open"):
        return ""
    return found[0].split(" See docs/TROUBLESHOOTING.md #")[0].strip()


def assert_agrees(error: dict, failure: dict) -> None:
    """The browser's class and consent request are the Python parsers' own."""
    tag = parse_refusal_tag(error["message"])
    assert failure["refusalClass"] == (tag.refusal_class.value if tag else None)
    assert failure["guard"] == (tag.guard if tag else None)
    assert failure["waivable"] == (tag.waivable if tag else False)
    assert failure["consent"] == expected_consent(error["message"])
    assert failure["remedy"] == expected_remedy(error["message"])


def test_a_worker_refusal_under_its_remote_traceback_reads_as_its_class_reason_and_consent():
    from dgx_monarch.consent_descriptor import SlabResidencyRescueOffer

    message = remote(SlabResidencyRescueOffer(rescue_refusal()))
    assert message.startswith("A remote actor call has failed.")
    assert message.index("[dgxm:") > job_observations.REFUSAL_HEAD_CHARS, "the tag lies past the kept head"
    error = recorded(message, code="monarch._src.actor.actor_mesh.ActorError")
    assert len(error["message"]) < len(message)
    [failure] = browser([error])
    assert_agrees(error, failure)
    assert (failure["refusalClass"], failure["guard"], failure["waivable"]) == ("C", "stock_load_preflight", True)
    assert failure["headline"] == HEADLINE
    assert failure["summary"] == f"Refused, class C ({CLASS_TITLES['C']}): {HEADLINE}"
    assert failure["remedy"].startswith('Open the DGX Monarch panel and click "Load with slab residency"')
    assert failure["remedy"].endswith("Headless: set DGXM_ALLOW_SLAB_RESCUE=1 on the driver and run again.")
    assert failure["troubleshooting"] == 12
    assert failure["consent"]["humanReason"] == rescue_descriptor().human_reason
    assert consent_descriptor.DESCRIPTOR_BEGIN not in failure["text"]
    assert failure["text"].startswith("A remote actor call has failed.")


def test_a_class_k_waiver_card_reads_with_its_kind_and_panel_button():
    accuracy_waiver.bind_model("chroma/base.safetensors", {}, [], {"ring": 2}, 2, dispatch=True)
    try:
        tail = accuracy_waiver.card_tail("ring_pad", family_hint="chroma")
    finally:
        accuracy_waiver.clear()
    assert tail, "a wired class K guard carries a card"
    text = refusal(RefusalClass.KNOWN_WRONG,
                   "this sequence length needs divisibility padding, and the exact pad exclusion is "
                   "ulysses-only. Use a uly* or single topology (docs/ADAPTERS.md).",
                   guard="ring_pad", waivable=True, panel_action=accuracy_waiver.panel_action("ring_pad"),
                   troubleshooting=21) + tail
    error = recorded(text, code="UnsupportedModelError")
    [failure] = browser([error])
    assert_agrees(error, failure)
    assert failure["refusalClass"] == "K" and failure["troubleshooting"] == 21
    assert failure["consent"]["kind"] == "waive-known-wrong:ring-pad"
    assert failure["consent"]["panelAction"] == accuracy_waiver.panel_action("ring_pad").label
    assert failure["headline"] == ("this sequence length needs divisibility padding, and the exact pad "
                                   "exclusion is ulysses-only.")


def test_unwaivable_and_plain_failures_read_without_invented_structure():
    physics = recorded(refusal(RefusalClass.PHYSICS, "this topology cannot run on one host. Use single instead.",
                               troubleshooting=16))
    plain = recorded("CUDA error: an illegal memory access was encountered", code="torch.AcceleratorError")
    long = recorded("Error(s) in loading state_dict:\n" + "\tsize mismatch for a weight.\n" * 300)
    assert "characters omitted]" in long["message"]
    failures = browser([physics, plain, long])
    for error, failure in zip((physics, plain, long), failures, strict=True):
        assert_agrees(error, failure)
    assert failures[0]["summary"] == "Refused, class P (physics): this topology cannot run on one host."
    assert (failures[0]["remedy"], failures[0]["troubleshooting"]) == ("", 16)
    assert failures[1] == {
        "code": "torch.AcceleratorError", "refusalClass": None, "guard": None, "waivable": False,
        "headline": "CUDA error: an illegal memory access was encountered",
        "summary": "torch.AcceleratorError: CUDA error: an illegal memory access was encountered",
        "text": "CUDA error: an illegal memory access was encountered", "remedy": "", "troubleshooting": None,
        "consent": None}
    # The stored text is already bounded, so the browser shows it whole.
    assert failures[2]["text"] == long["message"].rstrip()
    assert failures[2]["headline"] == "Error(s) in loading state_dict:"


def test_malformed_tags_and_descriptors_read_as_the_python_parsers_read_them():
    good = rescue_refusal()
    encoded = consent_descriptor.encode(rescue_descriptor())
    payload = json.loads(encoded[len(consent_descriptor.DESCRIPTOR_BEGIN):-len(consent_descriptor.DESCRIPTOR_END)])

    def block(body: dict) -> str:
        return consent_descriptor.DESCRIPTOR_BEGIN + json.dumps(body) + consent_descriptor.DESCRIPTOR_END

    messages = [
        good,
        "[dgxm:P] first. [dgxm:K guard=ring_pad waivable=1] second.",
        "[dgxm:C guard=stock_load_preflight] no waivable flag.",
        good + "\n" + encoded,
        good.replace(consent_descriptor.DESCRIPTOR_END, ""),
        good.replace(encoded, block({**payload, "version": 2})),
        good.replace(encoded, block({**payload, "human_reason": ""})),
        good.replace(encoded, block({**payload, "kind": "waive-everything"})),
        good.replace(encoded, block({**payload, "measured": {"nested": {"a": 1}}})),
        good.replace(encoded, block({**payload, "family_hint": 3})),
        good.replace(encoded, block(payload).replace('"checkpoint_bytes": 35000000000', '"checkpoint_bytes": 1e999')),
        good.replace(encoded, block({**payload, "human_reason": "x" * 4100})),
    ]
    errors = [recorded(message) for message in messages]
    failures = browser(errors)
    for error, failure in zip(errors, failures, strict=True):
        assert_agrees(error, failure)
    assert failures[0]["consent"] is not None
    assert all(failure["consent"] is None for failure in failures[3:])
    assert [failure["refusalClass"] for failure in failures[1:3]] == [None, None]


def test_the_browser_names_the_classes_as_the_refusal_module_does():
    program = (f"import {{ CLASS_TITLES }} from {json.dumps(MODULE.as_uri())};\n"
               "process.stdout.write(JSON.stringify(CLASS_TITLES));")
    result = subprocess.run([_node(), "--input-type=module", "-e", program], capture_output=True,
                            text=True, timeout=60, check=True)
    assert json.loads(result.stdout) == CLASS_TITLES
    source = MODULE.read_text()
    assert f"'{consent_descriptor.DESCRIPTOR_BEGIN}'" in source and f"'{consent_descriptor.DESCRIPTOR_END}'" in source
    assert f"DESCRIPTOR_BYTES = {consent_descriptor.MAX_DESCRIPTOR_BYTES};" in source
