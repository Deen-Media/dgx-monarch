"""Studio tooling: the shipped CPU scripts, their CI coverage, the contributor
checklist and the packaged-resource gate."""
from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import threading
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
STUDIO_TESTS = REPO / "studio" / "tests"
CI = REPO / ".github" / "workflows" / "ci.yml"
WHEEL_STEP = "Wheel ships exactly the hashed Studio resources"
TOKEN = "tooling-smoke-owner-token-" + "x" * 32
OBSERVER_MESSAGES = {
    "inside_repository": "Output must be outside the repository",
    "shared_directory": "mode 0700",
    "group_readable": "mode 0700",
    "world_writable_sticky": "mode 0700",
    "not_json": "Output must be a .json file",
    "existing_file": "Output already exists",
    "missing_directory": "Output directory does not exist",
}


def _ci() -> dict:
    return yaml.safe_load(CI.read_text())


def _steps() -> list[dict]:
    return _ci()["jobs"]["lint-and-unit"]["steps"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"studio_script_{name}", STUDIO_TESTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scripts(monkeypatch):
    """Fresh copies of the fixture driver and the smoke, as their CLIs load them."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    driver = _load("mock_driver")
    monkeypatch.setitem(sys.modules, "mock_driver", driver)
    return SimpleNamespace(driver=driver, smoke=_load("http_smoke"))


@pytest.fixture
def fixture_driver(scripts):
    handler = type("Handler", (scripts.driver.Handler,), {"job_seconds": 0.2})
    server = scripts.driver.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(origin=f"http://127.0.0.1:{server.server_address[1]}",
                              module=scripts.driver, handler=handler)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def _get(origin: str, path: str):
    with urlopen(origin + path, timeout=5) as response:
        return json.loads(response.read())


def _post(origin: str, path: str, body: dict):
    request = Request(origin + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read())


@pytest.mark.parametrize("path", sorted(STUDIO_TESTS.glob("*.py")), ids=lambda path: path.name)
def test_studio_scripts_refuse_explicitly_instead_of_asserting(path):
    """python -O removes assert statements, so a check written as one never runs."""
    lines = [node.lineno for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Assert)]
    assert not lines, f"{path.name} uses assert at lines {lines}"


def test_ci_and_contributing_lint_and_compile_the_studio_scripts():
    steps = {step.get("name"): step for step in _steps()}
    assert steps["ruff"]["run"].split() == [
        "ruff", "check", "__init__.py", "src", "tests", "benchmark", "tools", "studio/tests"]
    compile_line = next(line for line in steps["Surface artifact checks"]["run"].splitlines() if "compileall" in line)
    assert compile_line.split() == [
        "python", "-m", "compileall", "-q", "__init__.py", "src", "benchmark", "tests", "tools", "studio/tests"]
    contributing = (REPO / "CONTRIBUTING.md").read_text().splitlines()
    assert "ruff check __init__.py src tests benchmark tools studio/tests" in contributing
    assert "python -m compileall -q __init__.py src benchmark tests tools studio/tests" in contributing


def test_smoke_check_flags_come_from_the_companion_responses(scripts):
    workflow = scripts.driver.WORKFLOW
    good = {"first": {"id": "job"}, "replay": {"id": "job"}, "reconciled": {"id": "job"},
            "summary": {"summary": True, "submission": {"digest": "d"}},
            "detail": {"submission": {"workflow": workflow, "prompt": workflow}},
            "image": scripts.driver.image_bytes()}
    assert scripts.smoke.checks(**good) == dict.fromkeys(scripts.smoke.checks(**good), True)
    broken = {
        "duplicate_submit_same_id": {"replay": {"id": "other"}},
        "request_reconciled": {"reconciled": {"id": "other"}},
        "summary_omits_workflow": {"summary": {"summary": True, "submission": {"workflow": workflow}}},
        "immutable_workflow_preserved": {"detail": {"submission": {"workflow": {}, "prompt": workflow}}},
        "submitted_seed_preserved": {"detail": {"submission": {
            "workflow": workflow, "prompt": {"1": {"inputs": {"seed": 8}}}}}},
        "artifact_is_png": {"image": b"GIF89a"},
    }
    for flag, change in broken.items():
        flags = scripts.smoke.checks(**{**good, **change})
        assert flags.pop(flag) is False, flag
        assert all(flags.values()), (flag, flags)
    # A response that lacks the fields reports failures instead of crashing.
    empty = scripts.smoke.checks(first={}, replay={}, reconciled={}, summary={}, detail={}, image=b"")
    assert not any(empty.values())


def test_http_smoke_passes_against_the_real_companion(tmp_path, scripts, fixture_driver):
    from aiohttp import web

    from dgx_monarch.studio.comfy import ComfyClient
    from dgx_monarch.studio.server import create_app
    from dgx_monarch.studio.store import Store

    async def scenario():
        store = Store(tmp_path / "state")
        allowed: set[str] = set()
        runner = web.AppRunner(create_app(store, ComfyClient(fixture_driver.origin), TOKEN, allowed_hosts=allowed),
                               access_log=None)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            allowed.add(f"127.0.0.1:{port}")
            report = await asyncio.to_thread(scripts.smoke.run, f"http://127.0.0.1:{port}", TOKEN, timeout=30)
            return report, store.all("artifacts")
        finally:
            await runner.cleanup()
            store.close()

    report, artifacts = asyncio.run(scenario())
    flags = ("duplicate_submit_same_id", "request_reconciled", "summary_omits_workflow",
             "immutable_workflow_preserved", "submitted_seed_preserved", "artifact_is_png")
    assert {flag: report[flag] for flag in flags} == dict.fromkeys(flags, True), report
    assert report["passed"] is True and report["job_state"] == "succeeded"
    assert report["artifact_kind"] == "image"
    assert report["artifact_bytes"] == len(scripts.driver.image_bytes())
    # ComfyUI reports no output dimensions, so the companion records none.
    (artifact,) = artifacts
    assert not any(artifact["metadata"].get(key) for key in ("width", "height", "frame_rate", "duration"))


def test_fixture_driver_serves_the_comfy_shapes_studio_reads(fixture_driver):
    driver, origin = fixture_driver.module, fixture_driver.origin
    assert driver.OUTPUT == {"filename": "cpu-fixture.png", "subfolder": "", "type": "output"}
    fixture_driver.handler.job_seconds = 60
    identity = str(uuid.uuid4())
    accepted = _post(origin, "/prompt", {"prompt": driver.WORKFLOW, "prompt_id": identity})
    assert accepted == {"prompt_id": identity, "number": 0, "node_errors": {}}
    assert [row[1] for row in _get(origin, "/queue")["queue_running"]] == [identity]
    assert _get(origin, f"/history/{identity}") == {}
    driver.JOBS[identity]["created"] -= 120
    assert _get(origin, "/queue")["queue_running"] == []
    entry = _get(origin, f"/history/{identity}")[identity]
    assert entry["outputs"] == {"1": {"images": [driver.OUTPUT]}}
    assert [kind for kind, _data in entry["status"]["messages"]] == ["execution_start", "execution_success"]
    assert all(data["prompt_id"] == identity and isinstance(data["timestamp"], int)
               for _kind, data in entry["status"]["messages"])
    minted = _post(origin, "/prompt", {"prompt": driver.WORKFLOW})["prompt_id"]
    assert str(uuid.UUID(minted)) == minted and minted != identity
    driver.JOBS[minted]["created"] -= 120
    assert list(_get(origin, "/history")) == [identity, minted]
    assert list(_get(origin, "/history?max_items=1")) == [minted]
    for refused in (identity.upper(), "not-a-uuid", 7):
        with pytest.raises(HTTPError) as response:
            _post(origin, "/prompt", {"prompt": driver.WORKFLOW, "prompt_id": refused})
        assert response.value.code == 400
        assert json.loads(response.value.read())["error"]["type"] == "invalid_prompt_id"
    with pytest.raises(HTTPError) as response:
        _post(origin, "/prompt", {"workflow": driver.WORKFLOW})
    assert response.value.code == 400


def test_fixture_driver_keeps_uploads_for_the_companion_to_read_back(fixture_driver):
    import aiohttp

    payload = fixture_driver.module.image_bytes()

    async def upload():
        form = aiohttp.FormData()
        # The same fields the companion sends in media_assets.upload.
        form.add_field("image", payload, filename="upload-name.png", content_type="image/png")
        form.add_field("type", "input")
        form.add_field("subfolder", "studio")
        form.add_field("overwrite", "false")
        async with aiohttp.ClientSession() as session:
            async with session.post(fixture_driver.origin + "/upload/image", data=form) as response:
                return response.status, await response.json()

    assert asyncio.run(upload()) == (200, {"name": "upload-name.png", "subfolder": "studio", "type": "input"})
    with urlopen(fixture_driver.origin + "/view?filename=upload-name.png&subfolder=studio&type=input",
                 timeout=5) as response:
        assert response.read() == payload
    with pytest.raises(HTTPError) as missing:
        urlopen(fixture_driver.origin + "/view?filename=other.png&subfolder=studio&type=input", timeout=5)
    assert missing.value.code == 404


def _observer(output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STUDIO_TESTS / "measure_observer.py"), "--output", str(output),
         "--iterations", "100", "--repeats", "3"],
        capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def _directory(path: Path, mode: int) -> Path:
    path.mkdir()
    path.chmod(mode)
    return path


@pytest.mark.parametrize("case", sorted(OBSERVER_MESSAGES))
def test_observer_refuses_output_that_is_not_new_private_evidence(tmp_path, case):
    private = _directory(tmp_path / "private", 0o700)
    existing = private / "existing.json"
    existing.write_text("kept\n")
    output = {
        "inside_repository": REPO / f"observer-refusal-{uuid.uuid4().hex}.json",
        "shared_directory": _directory(tmp_path / "shared", 0o755) / "record.json",
        "group_readable": _directory(tmp_path / "group", 0o750) / "record.json",
        "world_writable_sticky": _directory(tmp_path / "sticky", 0o1777) / "record.json",
        "not_json": private / "record.txt",
        "existing_file": existing,
        "missing_directory": private / "missing" / "record.json",
    }[case]
    result = _observer(output)
    assert result.returncode == 2, result.stderr
    assert OBSERVER_MESSAGES[case] in result.stderr
    assert result.stdout == ""
    assert existing.read_text() == "kept\n"
    assert output == existing or not output.exists()


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="the harness hashes the git working tree")
def test_observer_writes_a_private_record_and_prints_its_hash(tmp_path):
    output = _directory(tmp_path / "evidence", 0o700) / "observer.json"
    result = _observer(output)
    assert result.returncode == 0, result.stderr
    cited = json.loads(result.stdout.strip().splitlines()[-1])
    data = output.read_bytes()
    assert cited == {"output": str(output), "sha256": hashlib.sha256(data).hexdigest()}
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    record = json.loads(data)
    assert record["assertions"] == "passed"
    assert record["harness_sha256"] == hashlib.sha256((STUDIO_TESTS / "measure_observer.py").read_bytes()).hexdigest()


def _wheel_check_script() -> str:
    steps = _steps()
    names = [step.get("name") for step in steps]
    step = steps[names.index(WHEEL_STEP)]
    assert names.index("Validate wheel installability and source completeness") < names.index(WHEEL_STEP)
    # The build step runs only on 3.12, so the check must follow the same gate.
    assert step["if"] == "matrix.python == '3.12'"
    assert step["run"].startswith("python - dist/*.whl <<'PY'\n")
    return step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def _write_wheel(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("dgx_monarch/studio/static/", b"")
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_ci_wheel_step_requires_exactly_the_hashed_studio_resources(tmp_path):
    script = _wheel_check_script()
    package = REPO / "src" / "dgx_monarch"
    served = {path.relative_to(package).as_posix(): path.read_bytes()
              for directory in ("studio/data", "studio/static")
              for path in sorted((package / directory).rglob("*")) if path.is_file()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in served.items()}
    table = f'"""Generated."""\n\nRESOURCE_SHA256 = {hashes!r}\n'.encode()
    files = {"dgx_monarch/studio/resource_hashes.py": table, "dgx_monarch/studio/__init__.py": b"",
             **{f"dgx_monarch/{name}": data for name, data in served.items()}}
    css = "dgx_monarch/" + next(name for name in served if name.endswith(".css"))
    cases = {
        "complete": (files, 0, f"ships all {len(served)} hashed Studio resources"),
        "missing_stylesheet": ({name: data for name, data in files.items() if name != css}, 1, '"wheel_missing"'),
        "extra_file": ({**files, "dgx_monarch/studio/static/assets/stray.js": b"x"}, 1, '"wheel_extra"'),
        "changed_bytes": ({**files, css: b"body{}"}, 1, '"changed"'),
    }
    for case, (contents, code, text) in cases.items():
        wheel = _write_wheel(tmp_path / f"{case}.whl", contents)
        result = subprocess.run([sys.executable, "-", str(wheel)], input=script,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == code, (case, result.stdout, result.stderr)
        assert text in result.stdout + result.stderr, case


def test_contributing_regenerates_the_studio_inventory_before_checking_it():
    text = (REPO / "CONTRIBUTING.md").read_text()
    lines = text.splitlines()
    intro = lines.index("Choose additional checks based on the files and behavior you changed:")
    after = [index for index in range(intro + 1, len(lines)) if lines[index].strip()]
    # The sentence introduces the table directly; the Studio paragraph follows it.
    assert lines[after[0]].startswith("| Change | Validation |")
    table_end = next(index for index in after if not lines[index].startswith("|"))
    regenerate = lines.index("python tools/gen_studio_inventory.py")
    assert table_end <= regenerate < lines.index("python tools/gen_studio_inventory.py --check")
    rows = {line.split("|")[1].strip(): line for line in lines[intro:table_end] if line.startswith("| ")}
    for row in ("Nodes or browser UI", "Workflow templates or model-file lists", "Studio interface (`studio/`)"):
        assert "regenerate and check the Studio inventory" in rows[row], row
    assert "[Studio development steps](docs/STUDIO.md#development)" in rows["Studio interface (`studio/`)"]
    for committed in ("src/dgx_monarch/studio/resource_hashes.py", "web/js/dgx_monarch_presentation.js"):
        assert f"`{committed}`" in text and (REPO / committed).is_file(), committed
    assert "`src/dgx_monarch/studio/data/*.json`" in text
    # One copy of the frontend build steps lives in STUDIO.md; CONTRIBUTING links it.
    assert "npm ci" not in text
    assert "python tools/gen_studio_inventory.py --check  # after regenerating it, as above" in lines
    studio = (REPO / "docs" / "STUDIO.md").read_text().splitlines()
    assert studio.count("## Development") == 1
    start = studio.index("## Development")
    assert (studio.index("python tools/gen_studio_inventory.py", start)
            < studio.index("python tools/gen_studio_inventory.py --check", start))
    assert "tools/gen_studio_inventory.py --check" in (REPO / ".github/workflows/studio.yml").read_text()


@pytest.mark.parametrize("relative", [
    "studio/tests/README.md", "docs/STUDIO_PERFORMANCE.md", "docs/STUDIO_INTERFACE_QA.md",
    "studio/tests/http_smoke.py", "studio/tests/mock_driver.py", "studio/tests/measure_observer.py",
])
def test_studio_evidence_instructions_keep_results_out_of_temporary_directories(relative):
    """Hosts clear temporary directories, so evidence written there does not last."""
    assert not re.search(r"(?<![\w.~-])/(?:tmp|var/tmp|dev/shm)\b", (REPO / relative).read_text())


def test_performance_record_cites_the_committed_harness_and_bundle():
    """The record names the harness bytes that produced its numbers and the
    bundle files whose sizes it lists, so either change shows up as stale."""
    text = (REPO / "docs" / "STUDIO_PERFORMANCE.md").read_text()
    flat = " ".join(text.split())
    harness = re.search(r"The harness SHA-256 was `([0-9a-f]{64})`", flat)
    assert harness, "cite the harness SHA-256 of the recorded run"
    assert harness.group(1) == hashlib.sha256((STUDIO_TESTS / "measure_observer.py").read_bytes()).hexdigest(), (
        "measure_observer.py changed; re-run it and update the observer section")
    assert re.search(r"The raw record of this run .* Its SHA-256 is `[0-9a-f]{64}`", flat)
    assets = REPO / "src" / "dgx_monarch" / "studio" / "static" / "assets"
    rows = dict(re.findall(r"^\| `(index-[^`]+)` \| ([\d,]+) \|", text, re.MULTILINE))
    assert set(rows) == {path.name for path in assets.iterdir()}, "update the packaged asset table"
    for name, size in rows.items():
        assert (assets / name).stat().st_size == int(size.replace(",", "")), name


def test_browser_procedure_covers_every_interface_qa_browser_row():
    qa = (REPO / "docs" / "STUDIO_INTERFACE_QA.md").read_text()
    section = qa.split("## Browser evidence\n", 1)[1].split("\n## ", 1)[0]
    rows = [line.split("|")[1].strip() for line in section.splitlines()
            if line.startswith("| ") and not line.startswith("| Check |")]
    assert len(rows) == 12
    procedure = (STUDIO_TESTS / "README.md").read_text()
    for row in rows:
        assert f"**{row}.**" in procedure, row
    flat = " ".join(procedure.split())
    for record in ("browser name and version", "RESOURCE_SHA256", "SHA-256"):
        assert record in flat, record


def test_frontend_modules_stay_below_the_documented_line_limit():
    readme = " ".join((REPO / "studio" / "README.md").read_text().split())
    assert "All TypeScript/Svelte source modules are below 500 lines." in readme
    long = {path.relative_to(REPO).as_posix(): count
            for path in sorted((REPO / "studio" / "src").rglob("*"))
            if path.suffix in {".ts", ".svelte"} and (count := len(path.read_text().splitlines())) >= 500}
    assert not long
