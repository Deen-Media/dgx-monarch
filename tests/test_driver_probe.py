"""dgxm --host probe: prefer the instance with a live mesh, and never read a
probe that got no answer as "no driver"."""
import io
import json
import sys
from types import ModuleType, SimpleNamespace

_INIT_INFO = {"DGXMonarchInit": {
    "name": "DGXMonarchInit",
    "category": "DGX Monarch",
    "output": ["DGXM_MESH"],
}}


def _driver(telemetry=None, *, init=_INIT_INFO):
    return {"init": init, "telemetry": telemetry or {"workers": []}}


def _probe_with(monkeypatch, answers, ports=(8188, 8191), calls=None):
    """Map hosts to init and telemetry answers; other hosts refuse."""
    import urllib.request

    from dgx_monarch.cli import main as cli

    monkeypatch.setattr(
        cli.doctor_mod, "_find_comfy_ports", lambda: dict.fromkeys(ports)
    )

    def fake_urlopen(url, timeout=1):
        target = getattr(url, "full_url", url)
        if calls is not None:
            calls.append((target, timeout))
        for host, answer in answers.items():
            if host in target:
                value = (answer.get("init") if "/object_info/" in target
                         else answer.get("telemetry"))
                if isinstance(value, BaseException):
                    raise value
                class _Resp(io.BytesIO):
                    def __enter__(self): return self
                    def __exit__(self, *a): return False
                return _Resp(json.dumps(value).encode())
        raise OSError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return cli._resolve_driver_host


def test_explicit_host_wins(monkeypatch):
    resolve = _probe_with(monkeypatch, {})

    def unexpected_discovery():
        raise AssertionError("an explicit host must bypass local discovery")

    from dgx_monarch.cli import main as cli

    monkeypatch.setattr(cli.doctor_mod, "_find_comfy_ports", unexpected_discovery)
    assert resolve("10.0.0.5:9000") == "10.0.0.5:9000"


def test_prefers_the_instance_with_a_mesh(monkeypatch):
    resolve = _probe_with(monkeypatch, {
        "8188": _driver({"workers": []}),
        "8191": _driver({"workers": [{"host": {"host": "a"}}]}),
    })
    assert resolve(None) == "127.0.0.1:8191"


def test_falls_back_to_first_responder_then_default(monkeypatch):
    resolve = _probe_with(monkeypatch, {"8191": _driver()})
    assert resolve(None) == "127.0.0.1:8191"
    resolve = _probe_with(monkeypatch, {}, ports=())
    assert resolve(None) == "127.0.0.1:8188"


def test_implicit_gate_top_resolution_discovers_custom_comfy_port(monkeypatch):
    resolve = _probe_with(
        monkeypatch,
        {"8195": _driver({"workers": [{"host": {"host": "driver"}}]})},
        ports=(8188, 8191, 8195),
    )
    assert resolve(None) == "127.0.0.1:8195"


def test_invalid_200_from_unrelated_main_candidate_is_not_confirmation(monkeypatch):
    resolve = _probe_with(
        monkeypatch, {"8123": _driver(init={})}, ports=(8123,))
    from dgx_monarch.cli import main as cli

    assert resolve(None) == "127.0.0.1:8188"
    assert cli._driver_probe()[0] is None


def test_driver_probe_detects_loaded_local_instance(monkeypatch):
    _probe_with(monkeypatch, {"8191": _driver()})
    from dgx_monarch.cli import main as cli

    assert cli._driver_probe()[0] == "127.0.0.1:8191"


def test_driver_probe_discovers_custom_comfy_port(monkeypatch):
    _probe_with(monkeypatch, {"8189": _driver()}, ports=(8189,))
    from dgx_monarch.cli import main as cli

    assert cli._driver_probe()[0] == "127.0.0.1:8189"


def test_slow_telemetry_does_not_erase_pack_confirmation(monkeypatch):
    resolve = _probe_with(
        monkeypatch,
        {"8189": _driver(TimeoutError("cold telemetry exceeded ranking budget"))},
        ports=(8189,),
    )
    from dgx_monarch.cli import main as cli

    assert resolve(None) == "127.0.0.1:8189"
    assert cli._driver_probe()[0] == "127.0.0.1:8189"


def test_rich_telemetry_probe_no_longer_uses_the_unsafe_one_second_budget(monkeypatch):
    calls = []
    resolve = _probe_with(
        monkeypatch, {"8195": _driver()}, ports=(8195,), calls=calls)

    assert resolve(None) == "127.0.0.1:8195"
    timeouts = {url.rsplit("/", 1)[-1]: timeout for url, timeout in calls}
    assert timeouts["DGXMonarchInit"] > 1
    assert timeouts["telemetry"] > 1


def test_telemetry_503_does_not_erase_pack_confirmation(monkeypatch):
    import urllib.error

    cold = urllib.error.HTTPError(
        "http://127.0.0.1:8195/dgxm/telemetry", 503, "cold", {},
        io.BytesIO(b'{"t": 1, "workers": [], "telemetry_error": "TimeoutError"}'),
    )
    resolve = _probe_with(monkeypatch, {"8195": _driver(cold)}, ports=(8195,))
    from dgx_monarch.cli import main as cli

    assert resolve(None) == "127.0.0.1:8195"
    assert cli._driver_probe()[0] == "127.0.0.1:8195"


def test_arbitrary_503_without_valid_init_schema_is_not_confirmation(monkeypatch):
    resolve = _probe_with(
        monkeypatch,
        {"8123": _driver(init=OSError("HTTP 503"))},
        ports=(8123,),
    )
    from dgx_monarch.cli import main as cli

    assert resolve(None) == "127.0.0.1:8188"
    assert cli._driver_probe()[0] is None


def test_update_refuses_responding_custom_port_driver(monkeypatch, capsys, tmp_path):
    calls = []
    _probe_with(
        monkeypatch,
        {"8195": _driver(TimeoutError("telemetry cold during update guard"))},
        ports=(8195,),
        calls=calls,
    )
    from dgx_monarch.cli import legacy_update
    from dgx_monarch.cli import main as cli
    from dgx_monarch.cli.update_lock import UpdateLock

    def unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("update must refuse before git or pip runs")

    monkeypatch.setattr(
        legacy_update,
        "UpdateLock",
        lambda repo: UpdateLock(repo, state_root=tmp_path / "update-locks"),
    )
    monkeypatch.setattr(legacy_update.subprocess, "run", unexpected_subprocess)
    assert cli.main(["update"]) == 2
    detail = capsys.readouterr().err
    assert "refusing to update" in detail
    assert "127.0.0.1:8195" in detail
    assert all("/dgxm/telemetry" not in url for url, _timeout in calls)


def test_legacy_update_refuses_while_verified_update_holds_checkout_lock(
    tmp_path, capsys
):
    from dgx_monarch.cli import legacy_update
    from dgx_monarch.cli.update_lock import UpdateLock

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='dgx-monarch'\n")
    lock_root = tmp_path / "update-locks"

    def lock_factory(checkout):
        return UpdateLock(checkout, state_root=lock_root)

    with UpdateLock(repo, state_root=lock_root):
        result = legacy_update.run(
            SimpleNamespace(host=None),
            load_config=lambda _args: None,
            driver_probe=lambda _host: (None, True),
            repo=repo,
            lock_factory=lock_factory,
        )

    assert result == 2
    assert "another dgxm update is already active" in capsys.readouterr().err


def test_gate_uses_discovered_custom_port(monkeypatch):
    import urllib.request

    from dgx_monarch.cli import main as cli

    monkeypatch.setattr(cli.doctor_mod, "_find_comfy_ports", lambda: {8195: 123})
    requested = []

    def urlopen(url, timeout=1):
        target = getattr(url, "full_url", url)
        requested.append(target)
        if target == "http://127.0.0.1:8195/object_info/DGXMonarchInit":
            return io.BytesIO(json.dumps(_INIT_INFO).encode())
        if target == "http://127.0.0.1:8195/dgxm/telemetry":
            return io.BytesIO(b'{"workers": []}')
        if target == "http://127.0.0.1:8195/history?max_items=1":
            return io.BytesIO(b"{}")
        raise OSError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    args = SimpleNamespace(
        host=None, list=False, report_dir="unused", timeout=10, last=True
    )
    assert cli.cmd_gate(args) == 2
    assert "http://127.0.0.1:8195/history?max_items=1" in requested


def test_top_uses_discovered_custom_port(monkeypatch):
    import urllib.request

    from dgx_monarch.cli import main as cli

    monkeypatch.setattr(cli.doctor_mod, "_find_comfy_ports", lambda: {8195: 123})
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=1: io.BytesIO(
            json.dumps(_INIT_INFO).encode()
            if "/object_info/" in getattr(url, "full_url", url)
            else b'{"workers": []}'),
    )
    called = {}
    app = ModuleType("dgx_monarch.tui.app")

    def run(**kwargs):
        called.update(kwargs)

    app.run = run
    monkeypatch.setitem(sys.modules, "dgx_monarch.tui.app", app)
    args = SimpleNamespace(
        host=None,
        interval=1.0,
        replay=None,
        record=None,
        theme="spark",
        config=None,
    )
    assert cli.cmd_top(args) == 0
    assert called["driver"] == "127.0.0.1:8195"


def _confirm_raising(monkeypatch, error):
    import urllib.request

    def fake_urlopen(_url, timeout=1):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    from dgx_monarch.cli import comfy_ports

    return comfy_ports._confirmed_driver("127.0.0.1:8188")


def test_a_refused_port_is_a_definite_answer_and_a_timeout_is_not(monkeypatch):
    """The probe answers three ways, because "no driver" gets published.

    Setup and verified update turn "no confirmed driver" into
    `render_active=False, active_leases=0`. A timed-out probe is not evidence
    for that; a refused connection is.
    """
    import urllib.error

    refused = urllib.error.URLError(ConnectionRefusedError(111, "refused"))
    assert _confirm_raising(monkeypatch, refused) is False
    assert _confirm_raising(monkeypatch, urllib.error.HTTPError(
        "http://x", 404, "Not Found", {}, None)) is False
    assert _confirm_raising(monkeypatch, ValueError("not json")) is False

    assert _confirm_raising(monkeypatch, TimeoutError("slow")) is None
    assert _confirm_raising(monkeypatch, urllib.error.URLError(
        TimeoutError("slow"))) is None
    assert _confirm_raising(monkeypatch, OSError("network is down")) is None


def test_driver_probe_reports_whether_every_candidate_answered(monkeypatch):
    from dgx_monarch.cli import comfy_ports

    verdicts = {"127.0.0.1:8188": False, "127.0.0.1:8191": None}
    monkeypatch.setattr(
        comfy_ports, "_confirmed_driver", lambda candidate: verdicts[candidate])
    assert comfy_ports.driver_probe(None, (8188, 8191)) == (None, False)

    verdicts["127.0.0.1:8191"] = False
    assert comfy_ports.driver_probe(None, (8188, 8191)) == (None, True)

    verdicts["127.0.0.1:8191"] = True
    assert comfy_ports.driver_probe(None, (8188, 8191)) == ("127.0.0.1:8191", True)


def _legacy_update_with(monkeypatch, tmp_path, probe):
    """Run the legacy update against a probe answer, with mutation forbidden."""
    from dgx_monarch.cli import legacy_update
    from dgx_monarch.cli.update_lock import UpdateLock

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='dgx-monarch'\n")

    def unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("update must refuse before git or pip runs")

    monkeypatch.setattr(legacy_update.subprocess, "run", unexpected_subprocess)
    return legacy_update.run(
        SimpleNamespace(host=None),
        load_config=lambda _args: None,
        driver_probe=probe,
        repo=repo,
        lock_factory=lambda checkout: UpdateLock(
            checkout, state_root=tmp_path / "update-locks"),
    )


def test_legacy_update_refuses_an_unobserved_driver_probe(monkeypatch, tmp_path, capsys):
    """A probe that never answered is not permission to pull, pin and restart.

    Setup and verified update refuse this case as `activity_unknown`. Read as
    "ComfyUI is not running", a 2 s `/object_info` timeout would let the
    legacy command change the release in the checkout.
    """
    from dgx_monarch.cli import probe_certainty

    exit_code = _legacy_update_with(
        monkeypatch, tmp_path, lambda _host: (None, False))

    assert exit_code == probe_certainty.UNKNOWN_EXIT
    detail = capsys.readouterr().err
    assert "activity_unknown" in detail
    assert "never answered" in detail


def test_legacy_update_proceeds_past_an_observed_miss(monkeypatch, tmp_path, capsys):
    """An observed "no driver" still updates: only the unknown answer refuses."""
    from dgx_monarch.cli import probe_certainty

    exit_code = _legacy_update_with(
        monkeypatch, tmp_path, lambda _host: (None, True))

    # The stub checkout carries no dependency table, so the run stops at the
    # pin read. Reaching that line proves it cleared the guard.
    assert exit_code not in (2, probe_certainty.UNKNOWN_EXIT)
    assert "cannot determine the exact torchmonarch pin" in capsys.readouterr().err


def test_the_argparse_equals_form_of_the_port_flag_is_a_comfy_candidate():
    from dgx_monarch.cli.comfy_ports import looks_like_comfy

    # scripts/comfy-driver.sh cds into the checkout, so the path carries no
    # "comfy" and the port flag is all that is left to recognize.
    assert looks_like_comfy(["python", "main.py", "--port=8199"])
    assert looks_like_comfy(["python", "main.py", "--port", "8199"])
    assert not looks_like_comfy(["python", "main.py", "--port="])
    assert not looks_like_comfy(["python", "main.py", "--port=0"])
    assert not looks_like_comfy(["python", "main.py", "--portable"])
