"""Execute isolated Worker attestations against installed-distribution layouts."""
from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import sys
import venv
from dataclasses import replace
from importlib.machinery import BYTECODE_SUFFIXES, EXTENSION_SUFFIXES
from pathlib import Path

import pytest

from dgx_monarch import __version__
from dgx_monarch.cli import update_worker_attestation as attestation
from dgx_monarch.cli.update_driver_pin import site_payload_digest
from dgx_monarch.config import ClusterConfig
from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256


@pytest.fixture
def installed_worker(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    environment = home / "venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    packages = environment / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (environment / "bin/monarch_worker").write_text("# installed console script fixture\n")
    package = packages / "monarch"
    package.mkdir()
    (package / "__init__.py").write_text("PIN = '0.6.0'\n")
    info = packages / "torchmonarch-0.6.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: torchmonarch\nVersion: 0.6.0\n")
    (info / "RECORD").write_text("monarch/__init__.py,,\n../../../bin/monarch_worker,,\ntorchmonarch-0.6.0.dist-info/METADATA,,\ntorchmonarch-0.6.0.dist-info/RECORD,,\n")
    source = home / "checkout/src"
    tools = home / ".local/share/dgx-monarch/releases/u-111111111111-2222222222222222/verifier"
    code = Path(attestation.__file__).parents[1]
    for destination in (source / "dgx_monarch", tools / "dgx_monarch"):
        shutil.copytree(code, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    identity = attestation.WorkerReleaseIdentity(__version__, "0.6.0", dgx_source_manifest_sha256(source / "dgx_monarch"), site_payload_digest(packages, "0.6.0"))
    config = ClusterConfig(python_bin=str(environment / "bin/python"))
    def run(pin_digest=None):
        proof = identity if pin_digest is None else replace(identity, pin_payload_digest=pin_digest)
        script = attestation.release_attestation_script(config, site=str(source), tools="$HOME/" + str(tools.relative_to(home)), identity=proof, verifier_digest=attestation.verifier_tree_digest(tools / "dgx_monarch"), captured_site=True)
        return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, env={**os.environ, "HOME": str(home)}, timeout=30)

    def check(pin_digest=None):
        """The controller's reading of the same attestation: (matched, refusal)."""
        proof = identity if pin_digest is None else replace(identity, pin_payload_digest=pin_digest)

        def host_runner(_config, _host, script, timeout):
            return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True,
                                  env={**os.environ, "HOME": str(home)}, timeout=timeout)

        return attestation.remote_release_check(
            config, config.hosts[0] if config.hosts else None, site=str(source),
            tools="$HOME/" + str(tools.relative_to(home)), identity=proof, host_runner=host_runner,
            verifier_digest=attestation.verifier_tree_digest(tools / "dgx_monarch"), captured_site=True)

    run.check = check  # type: ignore[attr-defined]
    return source, tools, packages, run


def test_installed_console_script_record_does_not_shadow_runtime(installed_worker):
    _, _, _, run = installed_worker
    result = run()
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert result.stdout.strip() == "RELEASE_MATCH"


def test_earlier_worker_source_package_still_refuses_runtime_shadow(installed_worker):
    source, _, _, run = installed_worker
    (source / "monarch").mkdir()
    (source / "monarch/__init__.py").write_text("SHADOW = True\n")
    result = run()
    assert result.returncode == 3
    assert "RELEASE_MATCH" not in result.stdout


@pytest.mark.parametrize("relative", ["__init__.py", "cli/update_driver_pin.py", "cli/update_payload_ownership.py"])
def test_verifier_ignores_unchecked_cached_bytecode(installed_worker, relative):
    _, tools, _, run = installed_worker
    module = tools / "dgx_monarch" / relative
    source = module.read_bytes()
    module.write_text("raise SystemExit(23)\n")
    cache = module.parent / "__pycache__" / f"{module.stem}.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(str(module), cfile=str(cache), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    module.write_bytes(source)
    result = run()
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert result.stdout.strip() == "RELEASE_MATCH"


def test_preloaded_verifier_dependency_is_rejected(installed_worker):
    _, _, packages, run = installed_worker
    outside = packages / "foreign_verifier.py"
    outside.write_text("# unrelated installed source\n")
    preload = "import sys,types; m=types.ModuleType('dgx_monarch.cli.update_payload_ownership'); m.__file__=" + repr(str(outside)) + "; m.foreign_namespace_files=lambda *args:set(); sys.modules[m.__name__]=m\n"
    (packages / "preload_verifier.pth").write_text(preload)
    result = run()
    assert result.returncode == 4
    assert "RELEASE_MATCH" not in result.stdout


@pytest.mark.parametrize("suffix", [".py", *EXTENSION_SUFFIXES, *BYTECODE_SUFFIXES])
def test_earlier_worker_module_file_refuses_runtime_shadow(installed_worker, suffix):
    source, _, _, run = installed_worker
    (source / ("monarch" + suffix)).write_bytes(b"# earlier import candidate\n")
    result = run()
    assert result.returncode == (4 if suffix in BYTECODE_SUFFIXES else 3)
    assert "RELEASE_MATCH" not in result.stdout


def test_pinned_module_filename_is_normalized_before_shadow_check(installed_worker):
    source, _, packages, run = installed_worker
    shutil.rmtree(packages / "monarch")
    (packages / "monarch.py").write_text("PIN = '0.6.0'\n")
    record = packages / "torchmonarch-0.6.0.dist-info/RECORD"
    record.write_text(record.read_text().replace("monarch/__init__.py", "monarch.py"))
    digest = site_payload_digest(packages, "0.6.0")
    assert run(pin_digest=digest).stdout.strip() == "RELEASE_MATCH"
    (source / "monarch").mkdir()
    (source / "monarch/__init__.py").write_text("SHADOW = True\n")
    result = run(pin_digest=digest)
    assert result.returncode == 3
    assert "RELEASE_MATCH" not in result.stdout


def _color_matcher_layout(packages, tools, monkeypatch, *, initializer=b"", false_claim=False):
    import base64
    import csv
    import hashlib
    import io

    from dgx_monarch.cli import update_payload_ownership as ownership

    def record(data):
        return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("="), str(len(data))

    pinned = packages / "tests/test_cuda.py"
    pinned.parent.mkdir(exist_ok=True)
    pinned.write_bytes(b"PINNED = True\n")
    own_record = packages / "torchmonarch-0.6.0.dist-info/RECORD"
    with own_record.open("a") as stream:
        csv.writer(stream).writerow(("tests/test_cuda.py", *record(pinned.read_bytes())))
    expected_digest = site_payload_digest(packages, "0.6.0")
    filenames = ("scotland_house.png", "scotland_pitie.png", "scotland_plain.png")
    payloads = {name: b"synthetic png " + name.encode() for name in filenames}
    known_data = {name: record(data) for name, data in payloads.items()}
    monkeypatch.setattr(ownership, "_COLOR_MATCHER_DATA", known_data, raising=False)
    verifier = tools / "dgx_monarch/cli/update_payload_ownership.py"
    with verifier.open("a") as stream:
        stream.write("\n_COLOR_MATCHER_DATA = " + repr(known_data) + "\n")
    metadata = b"Metadata-Version: 2.1\nName: color-matcher\nVersion: 0.6.0\n"
    files = {"tests/__init__.py": initializer, "tests/unit_test.py": b"FOREIGN = True\n",
             "color_matcher-0.6.0.dist-info/METADATA": metadata}
    for name, data in payloads.items():
        files["tests/data/" + name] = data
    records = []
    for name, data in files.items():
        path = packages / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records.append((name, *record(data)))
    for name, data in payloads.items():
        external = packages / "../../../tests/data" / name
        external.parent.mkdir(parents=True, exist_ok=True)
        external.write_bytes(data)
        records.append(("../../../tests/data/" + name, *record(data)))
    if false_claim:
        records.append(("tests/test_cuda.py", *record(pinned.read_bytes())))
    records.append(("color_matcher-0.6.0.dist-info/RECORD", "", ""))
    output = io.StringIO()
    csv.writer(output).writerows(records)
    (packages / "color_matcher-0.6.0.dist-info/RECORD").write_text(output.getvalue())
    return expected_digest


def test_worker_attestation_accepts_verified_color_matcher_without_changing_pin_digest(installed_worker, monkeypatch):
    _, tools, packages, run = installed_worker
    expected = _color_matcher_layout(packages, tools, monkeypatch)
    assert site_payload_digest(packages, "0.6.0") == expected
    result = run(pin_digest=expected)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert result.stdout.strip() == "RELEASE_MATCH"


@pytest.mark.parametrize("options", [{"initializer": b"EXECUTABLE = True\n"}, {"false_claim": True}])
def test_worker_attestation_still_rejects_color_matcher_shadow_or_false_claim(installed_worker, monkeypatch, options):
    _, tools, packages, run = installed_worker
    expected = _color_matcher_layout(packages, tools, monkeypatch, **options)
    result = run(pin_digest=expected)
    assert result.returncode != 0
    assert "RELEASE_MATCH" not in result.stdout


# A Worker environment built with PYTHONPYCACHEPREFIX (docs/TROUBLESHOOTING.md #155).
TAG = sys.implementation.cache_tag
PREFIX = "../../../../scratch/cache/pycache/home/user/venv/lib/python3.12/site-packages"


def _rows(*rows):
    import csv
    import io

    output = io.StringIO()
    csv.writer(output).writerows(rows)
    return output.getvalue()


def _external(packages, relative, data=b""):
    """Create the file a RECORD row outside site-packages names, as a live prefix tree has it.
    importlib.metadata drops a RECORD row whose file is missing, so the file must exist."""
    path = (packages / relative).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_the_pinned_record_s_prefix_bytecode_does_not_refuse_the_worker(installed_worker):
    _, _, packages, run = installed_worker
    row = f"{PREFIX}/monarch/__init__.{TAG}.pyc"
    _external(packages, row)
    record = packages / "torchmonarch-0.6.0.dist-info/RECORD"
    record.write_text(record.read_text() + _rows((row, "", "")))
    result = run()
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert result.stdout.strip() == "RELEASE_MATCH"


def test_any_other_row_outside_site_packages_refuses_and_names_its_cause(installed_worker):
    _, _, packages, run = installed_worker
    row = f"{PREFIX}/evil.py"
    _external(packages, row, b"EVIL = True\n")
    record = packages / "torchmonarch-0.6.0.dist-info/RECORD"
    record.write_text(record.read_text() + _rows((row, "", "")))
    result = run()
    assert result.returncode == 3 and "RELEASE_MATCH" not in result.stdout
    assert ("DGXM_ATTEST_REFUSAL=record_outside_site distribution=torchmonarch "
            "path=torchmonarch-0.6.0.dist-info/RECORD") in result.stdout
    matched, refusal = run.check()
    assert matched is False and refusal is not None
    assert (refusal.reason, refusal.distribution) == ("record_outside_site", "torchmonarch")


def _foreign_examples(packages, extra=()):
    """torchmonarch ships examples/demo.py; xfuser ships examples/xfuser_demo.py
    beside it, with pip's prefix bytecode rows; returns the new pin digest."""
    import base64
    import hashlib

    def record(path, data):
        return (path, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("="),
                str(len(data)))

    (packages / "examples").mkdir()
    (packages / "examples/demo.py").write_bytes(b"PINNED = True\n")
    own = packages / "torchmonarch-0.6.0.dist-info/RECORD"
    own.write_text(own.read_text() + _rows(record("examples/demo.py", b"PINNED = True\n")))
    expected = site_payload_digest(packages, "0.6.0")
    info = packages / "xfuser-0.8.0.dist-info"
    info.mkdir()
    metadata = b"Metadata-Version: 2.1\nName: xfuser\nVersion: 0.8.0\n"
    (info / "METADATA").write_bytes(metadata)
    (packages / "examples/xfuser_demo.py").write_bytes(b"FOREIGN = True\n")
    rows = [record("xfuser-0.8.0.dist-info/METADATA", metadata),
            record("examples/xfuser_demo.py", b"FOREIGN = True\n"),
            (f"{PREFIX}/examples/xfuser_demo.{TAG}.pyc", "", ""), *extra,
            ("xfuser-0.8.0.dist-info/RECORD", "", "")]
    (info / "RECORD").write_text(_rows(*rows))
    return expected


def test_a_foreign_record_with_prefix_bytecode_does_not_refuse_the_worker(installed_worker):
    """The issue-158 refusal on a Worker whose torchmonarch lives in its base
    environment: before this change it read as a bare prior_worker_drift."""
    _, _, packages, run = installed_worker
    expected = _foreign_examples(packages)
    assert site_payload_digest(packages, "0.6.0") == expected
    result = run(pin_digest=expected)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert result.stdout.strip() == "RELEASE_MATCH"


def test_a_foreign_record_the_check_cannot_vouch_for_names_host_reason_and_remedy(installed_worker):
    _, _, packages, run = installed_worker
    expected = _foreign_examples(packages, extra=[("../../../../scratch/x/evil.pth", "", "")])
    result = run(pin_digest=expected)
    assert result.returncode == 1 and "RELEASE_MATCH" not in result.stdout
    assert ("DGXM_ATTEST_REFUSAL=record_outside_site distribution=xfuser "
            "path=xfuser-0.8.0.dist-info/RECORD") in result.stdout
    matched, refusal = run.check(pin_digest=expected)
    assert matched is False and refusal is not None
    assert (refusal.reason, refusal.distribution) == ("record_outside_site", "xfuser")


def _shared_conftest(packages, tools, monkeypatch):
    """torchmonarch ships tests/conftest.py; the named wheel ships its own over it.
    The verifier copy on the host gets the same synthetic table, as for color-matcher."""
    import base64
    import hashlib

    from dgx_monarch.cli import update_payload_ownership as ownership

    def record(path, data):
        return (path, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("="),
                str(len(data)))

    (packages / "tests").mkdir()
    (packages / "tests/conftest.py").write_bytes(b"PINNED = True\n")
    own = packages / "torchmonarch-0.6.0.dist-info/RECORD"
    own.write_text(own.read_text() + _rows(record("tests/conftest.py", b"PINNED = True\n")))
    expected = site_payload_digest(packages, "0.6.0")
    table = {("xfuser", "0.8.0"): {"tests/conftest.py": record("tests/conftest.py", b"FOREIGN = True\n")[1:]}}
    monkeypatch.setattr(ownership, "_SHARED_PINNED", table, raising=False)
    with (tools / "dgx_monarch/cli/update_payload_ownership.py").open("a") as stream:
        stream.write("\n_SHARED_PINNED = " + repr(table) + "\n")
    info = packages / "xfuser-0.8.0.dist-info"
    info.mkdir()
    metadata = b"Metadata-Version: 2.1\nName: xfuser\nVersion: 0.8.0\n"
    (info / "METADATA").write_bytes(metadata)
    (packages / "tests/conftest.py").write_bytes(b"FOREIGN = True\n")
    (info / "RECORD").write_text(_rows(record("xfuser-0.8.0.dist-info/METADATA", metadata),
                                       record("tests/conftest.py", b"FOREIGN = True\n"),
                                       ("xfuser-0.8.0.dist-info/RECORD", "", "")))
    return expected


def test_the_named_wheel_s_copy_of_the_shared_path_attests_the_same_release(installed_worker, monkeypatch):
    _, tools, packages, run = installed_worker
    expected = _shared_conftest(packages, tools, monkeypatch)
    assert site_payload_digest(packages, "0.6.0") == expected
    result = run(pin_digest=expected)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert result.stdout.strip() == "RELEASE_MATCH"


def test_a_vanished_shared_path_refuses_on_the_worker_and_names_it(installed_worker, monkeypatch):
    _, tools, packages, run = installed_worker
    expected = _shared_conftest(packages, tools, monkeypatch)
    (packages / "tests/conftest.py").unlink()
    result = run(pin_digest=expected)
    assert result.returncode == 1 and "RELEASE_MATCH" not in result.stdout
    assert ("DGXM_ATTEST_REFUSAL=shared_payload_missing distribution=torchmonarch "
            "path=tests/conftest.py") in result.stdout
