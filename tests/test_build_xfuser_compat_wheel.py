"""CPU-only regression coverage for the xFuser compatibility-wheel builder."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "build_xfuser_compat_wheel.py"
SPEC = importlib.util.spec_from_file_location("build_xfuser_compat_wheel", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _record(files: dict[str, bytes], record_name: str) -> bytes:
    output = io.StringIO()
    rows = [[name, builder.record_hash(data), str(len(data))] for name, data in sorted(files.items())]
    rows.append([record_name, "", ""])
    csv.writer(output, lineterminator="\r\n").writerows(rows)
    return output.getvalue().encode()


def _official_wheel(path: Path) -> Path:
    dist_info = "xfuser-0.7.0.dist-info/"
    files = {
        builder.RING_INIT: b"from .ring_flash_attn import xdit_ring_flash_attn_func\n",
        builder.VERSION_FILE: b"__version__ = version = '0.7.0'\n",
        "xfuser/LICENSE.txt": b"upstream license bytes\n",
        f"{dist_info}METADATA": b"Metadata-Version: 2.4\nName: xfuser\nVersion: 0.7.0\n",
        f"{dist_info}WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    files[f"{dist_info}RECORD"] = _record(files, f"{dist_info}RECORD")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 2, 3, 4, 6))
            info.external_attr = 0o100640 << 16
            archive.writestr(info, data)
    return path


@pytest.fixture
def official_wheel(tmp_path, monkeypatch):
    wheel = _official_wheel(tmp_path / "official.whl")
    monkeypatch.setattr(builder, "OFFICIAL_SHA256", hashlib.sha256(wheel.read_bytes()).hexdigest())
    return wheel


def _read(path: Path) -> tuple[dict[str, bytes], dict[str, zipfile.ZipInfo]]:
    with zipfile.ZipFile(path) as archive:
        return (
            {info.filename: archive.read(info) for info in archive.infolist()},
            {info.filename: info for info in archive.infolist()},
        )


def _check_record(files: dict[str, bytes], record_name: str) -> None:
    rows = list(csv.reader(io.TextIOWrapper(io.BytesIO(files[record_name]), newline="")))
    assert rows[-1] == [record_name, "", ""]
    assert {row[0] for row in rows[:-1]} == set(files) - {record_name}
    for name, digest, size in rows[:-1]:
        assert digest == builder.record_hash(files[name])
        assert size == str(len(files[name]))


def test_output_changes_only_the_allowlisted_members_and_rebuilds_record(official_wheel, tmp_path):
    output = builder.build(official_wheel, tmp_path / "output")
    source, source_infos = _read(official_wheel)
    built, built_infos = _read(output)
    old_prefix = "xfuser-0.7.0.dist-info/"
    new_prefix = f"xfuser-{builder.VERSION}.dist-info/"
    old_record = f"{old_prefix}RECORD"
    new_record = f"{new_prefix}RECORD"

    assert set(built) == {name.replace(old_prefix, new_prefix, 1) for name in source}
    assert built[builder.RING_INIT] == builder.PATCHED_RING
    assert built[builder.VERSION_FILE] == builder.PATCHED_VERSION
    assert built[f"{new_prefix}METADATA"] == source[f"{old_prefix}METADATA"].replace(
        b"\nVersion: 0.7.0\n", f"\nVersion: {builder.VERSION}\n".encode(), 1
    )
    for name, contents in source.items():
        target = name.replace(old_prefix, new_prefix, 1)
        if name not in {builder.RING_INIT, builder.VERSION_FILE, f"{old_prefix}METADATA", old_record}:
            assert built[target] == contents
        source_info = source_infos[name]
        built_info = built_infos[target]
        assert built_info.external_attr == source_info.external_attr
        assert built_info.date_time == source_info.date_time
        assert built_info.create_system == source_info.create_system
        assert built_info.create_version == source_info.create_version
        assert built_info.extract_version == source_info.extract_version
        assert built_info.extra == source_info.extra
        assert built_info.comment == source_info.comment
    _check_record(built, new_record)


def test_build_is_byte_deterministic_and_does_not_overwrite(official_wheel, tmp_path):
    first = builder.build(official_wheel, tmp_path / "one")
    second = builder.build(official_wheel, tmp_path / "two")
    assert first.read_bytes() == second.read_bytes()
    original = first.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build(official_wheel, first.parent)
    assert first.read_bytes() == original


def test_bad_hash_refuses_input_before_writing_output(official_wheel, tmp_path):
    official_wheel.write_bytes(b"not an approved wheel")
    with pytest.raises(builder.WheelValidationError, match="SHA-256 mismatch"):
        builder.build(official_wheel, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_failed_write_leaves_no_partial_output(official_wheel, tmp_path, monkeypatch):
    output_dir = tmp_path / "output"

    def fail_write(path, members):
        path.write_bytes(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(builder, "_write_wheel", fail_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        builder.build(official_wheel, output_dir)
    assert list(output_dir.iterdir()) == []


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def test_download_uses_fixed_url_validates_input_and_cleans_temporary_files(official_wheel, tmp_path):
    payload = official_wheel.read_bytes()
    seen = []

    def opener(url):
        seen.append(url)
        return _Response(payload)

    result = builder.build(None, tmp_path, opener=opener)
    assert seen == [builder.OFFICIAL_WHEEL_URL]
    assert result.exists()
    assert not list(tmp_path.glob(".xfuser-input-*"))


def test_download_rejects_bad_payload_without_leaving_a_cache(tmp_path):
    with pytest.raises(builder.WheelValidationError, match="SHA-256 mismatch"):
        builder.build(None, tmp_path, opener=lambda url: _Response(b"incorrect"))
    assert not list(tmp_path.iterdir())
