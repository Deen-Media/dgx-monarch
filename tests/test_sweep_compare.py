"""The sweep's audio compare reads every file, rate, channel and sample it got.

An envelope correlation over a shorter prefix, a resampled copy or a channel
mix can read 1.0 against audio that differs, so the compare states each shape
and names any mismatch as an unproven measurement (CHECK) instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.sweep import compare  # noqa: E402

np = pytest.importorskip("numpy")
soundfile = pytest.importorskip("soundfile")

RATE = 24000


def _tone(seconds: float = 1.0, rate: int = RATE, rising: bool = True):
    """A tone whose loudness ramps, so its envelope is not constant."""
    steps = np.arange(int(rate * seconds)) / rate
    ramp = steps / seconds if rising else 1.0 - steps / seconds
    return 0.5 * ramp * np.sin(2 * np.pi * 220 * steps)


def _write(path: Path, samples, rate: int = RATE) -> Path:
    soundfile.write(str(path), samples, rate, subtype="PCM_16")
    return path


def test_a_shorter_prefix_of_the_reference_is_a_duration_mismatch(tmp_path):
    whole = _tone(2.0)
    reference = _write(tmp_path / "ref.wav", whole)
    prefix = _write(tmp_path / "cand.wav", whole[: whole.size // 2])
    result = compare.compare_audio([prefix], [reference])
    assert "error" in result and "samples" in result["error"]
    assert [row["frames"] for row in result["candidate"]] == [whole.size // 2]
    assert [row["frames"] for row in result["reference"]] == [whole.size]


def test_the_same_samples_at_another_rate_are_a_rate_mismatch(tmp_path):
    samples = _tone(1.0)
    reference = _write(tmp_path / "ref.wav", samples, RATE)
    resampled = _write(tmp_path / "cand.wav", samples, RATE * 2)
    result = compare.compare_audio([resampled], [reference])
    assert "error" in result and "sample rate" in result["error"]
    assert result["candidate"][0]["rate"] == RATE * 2
    assert result["reference"][0]["rate"] == RATE


def test_a_channel_count_mismatch_is_named(tmp_path):
    mono = _tone(1.0)
    reference = _write(tmp_path / "ref.wav", mono)
    stereo = _write(tmp_path / "cand.wav", np.stack((mono, mono), axis=1))
    result = compare.compare_audio([stereo], [reference])
    assert "error" in result and "channels" in result["error"]
    assert (result["candidate"][0]["channels"], result["reference"][0]["channels"]) == (2, 1)


def test_swapped_stereo_channels_never_correlate_as_one(tmp_path):
    left, right = _tone(1.0, rising=True), _tone(1.0, rising=False)
    reference = _write(tmp_path / "ref.wav", np.stack((left, right), axis=1))
    swapped = _write(tmp_path / "cand.wav", np.stack((right, left), axis=1))
    result = compare.compare_audio([swapped], [reference])
    assert "error" not in result
    assert result["envelope_correlation"] < 0.5
    assert result["identical"] is False


def test_a_file_count_mismatch_is_named_and_every_file_is_read(tmp_path):
    first = _write(tmp_path / "cand_00001_.wav", _tone(1.0))
    second = _write(tmp_path / "cand_00002_.wav", _tone(1.0, rising=False))
    reference = _write(tmp_path / "ref_00001_.wav", _tone(1.0))
    result = compare.compare_audio([first, second], [reference])
    assert "error" in result and "file count" in result["error"]
    assert (result["files"], result["reference_files"]) == (2, 1)
    # Two files on each side: the second pair differs, and the worst pair speaks.
    other = _write(tmp_path / "ref_00002_.wav", _tone(1.0))
    result = compare.compare_audio([first, second], [reference, other])
    assert "error" not in result and result["files"] == 2
    assert result["envelope_correlation"] < 0.5


def test_matching_audio_is_identical_and_reports_its_shape(tmp_path):
    samples = np.stack((_tone(1.0), _tone(1.0, rising=False)), axis=1)
    reference = _write(tmp_path / "ref.wav", samples)
    candidate = _write(tmp_path / "cand.wav", samples)
    result = compare.compare_audio([candidate], [reference])
    assert "error" not in result
    assert result["identical"] is True and result["envelope_correlation"] == 1.0
    assert result["candidate"] == [{"file": "cand.wav", "rate": RATE, "channels": 2,
                                    "frames": RATE, "seconds": 1.0}]


def test_an_unreadable_audio_file_is_an_error_not_a_crash(tmp_path):
    # soundfile raises LibsndfileError, a RuntimeError, on a file it cannot
    # decode. Uncaught, it would end the session from inside a fidelity compare.
    broken = tmp_path / "cand.flac"
    broken.write_bytes(b"not audio at all")
    reference = _write(tmp_path / "ref.wav", _tone(1.0))
    result = compare.compare_audio([broken], [reference])
    assert result["error"].startswith("LibsndfileError")


def test_identical_silence_still_reads_too_quiet(tmp_path):
    # The probe's near silence keeps its CHECK: equal constants prove nothing.
    silence = np.zeros(RATE)
    result = compare.compare_audio([_write(tmp_path / "cand.wav", silence)],
                                   [_write(tmp_path / "ref.wav", silence)])
    assert result["error"] == "audio too short or too quiet to correlate"
    assert result["identical"] is True


def test_two_empty_files_read_too_short_and_never_raise(tmp_path):
    empty = np.zeros((0, 2))
    result = compare.compare_audio([_write(tmp_path / "cand.wav", empty)],
                                   [_write(tmp_path / "ref.wav", empty)])
    assert result["error"] == "audio too short or too quiet to correlate"
    assert result["candidate"][0]["frames"] == 0


def _audio_pair(tmp_path: Path, candidate, reference) -> None:
    """One frame each side, equal, and one audio file each side."""
    from PIL import Image

    pixels = np.full((4, 4, 3), 90, dtype=np.uint8)
    for prefix in ("sweep_abc123_probe", "sweep_ref456_probe"):
        Image.fromarray(pixels).save(tmp_path / f"{prefix}_00001_.png")
    _write(tmp_path / "sweep_abc123_probe_00001_.wav", candidate)
    _write(tmp_path / "sweep_ref456_probe_00001_.wav", reference)


def _audio_cell() -> dict:
    return {"id": "abc123", "audio": True, "probe": True, "probe_reason": "",
            "waiver_guards": [], "reference": {"kind": "cell", "cell": "ref456",
                                               "bar": "step-nrms", "probe_steps": 1}}


def test_a_mismatched_audio_leg_reads_check_and_a_correlation_is_only_a_note(tmp_path):
    from benchmark.sweep import run

    whole = _tone(2.0)
    _audio_pair(tmp_path, whole[: whole.size // 2], whole)
    record = {"verdict": "PASS", "findings": [], "notes": []}
    run._fidelity(record, _audio_cell(), tmp_path, 0.10)
    assert record["findings"] == [
        "reference compare: audio envelope: sweep_abc123_probe_00001_.wav samples "
        f"{whole.size // 2} != reference sweep_ref456_probe_00001_.wav {whole.size}"]
    assert run.settle_verdict(record) == "CHECK"

    # Equal shapes, different loudness: the number is reported and scores nothing.
    for path in tmp_path.iterdir():
        path.unlink()
    _audio_pair(tmp_path, _tone(1.0, rising=False), _tone(1.0))
    record = {"verdict": "PASS", "findings": [], "notes": []}
    run._fidelity(record, _audio_cell(), tmp_path, 0.10)
    assert record["findings"] == [] and run.settle_verdict(record) == "PASS"
    assert record["fidelity"]["audio"]["envelope_correlation"] < 0
    assert len(record["notes"]) == 1 and record["notes"][0].startswith(
        "audio: lowest channel envelope correlation -")
    assert record["notes"][0].endswith("; no audio threshold applies")


def test_the_report_lists_audio_apart_from_the_pixel_columns():
    from benchmark.sweep import report

    def record(cell_id: str, audio: dict | None) -> dict:
        fidelity = {"nrms": 0.01, "max_abs": 2, "frames": 1}
        if audio is not None:
            fidelity["audio"] = audio
        return {"cell": {"id": cell_id, "label": "render", "template": "ltx",
                         "preset": "uly2", "attention": "TORCH_FLASH"},
                "verdict": "PASS", "observed": "render", "fidelity": fidelity,
                "legs": {"cold": {"outcome": "render", "wall_s": 10.0}}}

    shape = [{"file": "a.flac", "rate": 48000, "channels": 2, "frames": 48000,
              "seconds": 1.0}]
    payload = report.build([
        record("av0001", {"files": 1, "candidate": shape, "identical": False,
                          "envelope_correlation": 0.97, "channels_correlated": 2}),
        record("av0002", {"error": "audio file count 2 != reference 1"}),
        record("im0003", None)], "cluster")
    assert payload["audio"] == [
        "- `av0001` ltx uly2 TORCH_FLASH: lowest channel envelope correlation 0.97 over "
        "2 channel(s); 1 file(s), the first 2 channel(s) at 48000 Hz for 1.0 s",
        "- `av0002` ltx uly2 TORCH_FLASH: not compared: audio file count 2 != reference 1"]
    text = report.render(payload)
    section = text.split("## audio comparisons", 1)[1]
    assert "not a calibrated fidelity threshold" in section and "av0001" in section
    # The pixel table keeps only pixel numbers.
    assert "0.97" not in text.split("## findings", 1)[0]
    # A session with no audio leg still prints the section, empty.
    quiet = report.render(report.build([record("im0003", None)], "cluster"))
    assert quiet.split("## audio comparisons", 1)[1].endswith("\n- none\n")
