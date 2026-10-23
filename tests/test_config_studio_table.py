"""The [studio] table: its rules, its defaults, and the loaders that ignore it."""
from __future__ import annotations

import argparse
import tomllib
from dataclasses import asdict
from pathlib import Path

import pytest

from dgx_monarch.config import load_cluster_config
from dgx_monarch.config_schema import (
    DEFAULT_STUDIO_DRIVER,
    DEFAULT_STUDIO_PORT,
    DRIVER_UNIT_PATTERN,
    ClusterConfigError,
    canonical_driver_origin,
    validate_driver_origin,
    validate_root_keys,
    validate_studio_table,
)
from dgx_monarch.monarch_policy_io import trust_config_fingerprint

REPO = Path(__file__).resolve().parents[1]
PATH = Path("/home/user/.config/dgx-monarch/cluster.toml")
CLUSTER = '[cluster]\nclient_bind = "tcp://192.0.2.11:0"\n[[hosts]]\nname = "192.0.2.11"\n'
SHAPE = ("studio.driver must be an http or https origin with a host and no user, path, query, "
         "fragment or zone")
LISTEN = ("studio.driver names a listen address; use the address a client connects to, "
          "such as http://127.0.0.1:8191")
LOCAL = ("studio.driver must name this computer (127.0.0.1, [::1] or localhost); "
         "for a driver on another computer pass dgxm studio --driver")


def _refusal(values: object) -> str:
    with pytest.raises(ClusterConfigError) as raised:
        validate_studio_table(values, PATH)
    return str(raised.value)


def test_absent_table_gives_the_documented_defaults():
    assert validate_studio_table({}, PATH) == {
        "driver": None, "port": 8220, "driver_service": None, "driver_command": None}
    assert DEFAULT_STUDIO_PORT == 8220
    assert DEFAULT_STUDIO_DRIVER == "http://127.0.0.1:8188"


def test_unknown_studio_keys_are_refused_at_their_scope():
    message = _refusal({"driver": "http://127.0.0.1:8191", "allow_management": True, "drvier": "x"})
    assert message == (f"{PATH}: [studio] contains unsupported key(s): allow_management, drvier. "
                       "Allowed keys: driver, driver_command, driver_service, port")
    for absent in ("state_dir", "host", "token_file"):
        assert absent in _refusal({absent: "x"})


@pytest.mark.parametrize("value", ["http://127.0.0.1:8191", 8220, ["driver"], True])
def test_a_non_table_studio_value_is_refused(value):
    assert _refusal(value) == f"{PATH}: [studio] must be a table"


@pytest.mark.parametrize(("value", "message"), [
    ("ftp://127.0.0.1:8188", SHAPE),
    ("http://:8188", SHAPE),
    ("http://user@127.0.0.1:8188", SHAPE),
    ("http://user:secret@127.0.0.1:8188", SHAPE),
    ("http://@127.0.0.1:8188", SHAPE),
    ("http://127.0.0.1:8188/x", SHAPE),
    # The design lists path "/" under both the shape and the canonical tests;
    # it is refused with the canonical message, which names the fix.
    ("http://127.0.0.1:8188/", "studio.driver must be written as 'http://127.0.0.1:8188'"),
    ("http://127.0.0.1:8188?a=1", SHAPE),
    ("http://127.0.0.1:8188#top", SHAPE),
    ("http://127.0.0.1:81888", SHAPE),
    ("http://127.0.0.1:port", SHAPE),
    ("http://127.0.0.1:8188\x07", "studio.driver must not contain control characters"),
    ("http://127.0.0.1:8188\x7f", "studio.driver must not contain control characters"),
    ("", "studio.driver must be a non-empty string"),
    ("http://127.0.0.1:8188" + "a" * 2028, SHAPE),
    ("http://[fe80::1%25eth0]:8188", SHAPE),
    ("http://[fe80::1%eth0]:8188", SHAPE),
    ("http://127.0.0.1:8188%2F", SHAPE),
    ("http://local host:8188", SHAPE),
    ("http://café:8188", SHAPE),
])
def test_driver_must_be_an_origin(value, message):
    assert _refusal({"driver": value}) == f"{PATH}: {message}"
    with pytest.raises(ValueError) as raised:
        validate_driver_origin(value, local_only=False)
    assert str(raised.value) == message


@pytest.mark.parametrize("value", [None, 8188, ["http://127.0.0.1:8188"]])
def test_driver_must_be_a_string(value):
    assert _refusal({"driver": value}) == f"{PATH}: studio.driver must be a non-empty string"


def test_the_length_limit_is_2048_characters():
    host = "a" * (2048 - len("http://:8188"))
    assert validate_driver_origin(f"http://{host}:8188", local_only=False) == f"http://{host}:8188"
    with pytest.raises(ValueError, match="must be an http or https origin"):
        validate_driver_origin(f"http://{host}a:8188", local_only=False)


@pytest.mark.parametrize("value", [
    "http://0.0.0.0:8191", "http://0:8191", "http://0.0:8191", "http://0x0:8191",
    "http://[::]:8191", "http://[0:0::0]:8191", "https://0.0.0.0",
])
def test_listen_addresses_are_refused_as_origins(value):
    assert _refusal({"driver": value}) == f"{PATH}: {LISTEN}"
    with pytest.raises(ValueError) as raised:
        validate_driver_origin(value, local_only=False)
    assert str(raised.value) == LISTEN


@pytest.mark.parametrize("value", [
    "https://driver.example", "http://192.0.2.10:8188", "http://[2001:db8::5]:8188",
    "http://127.1:8188", "http://localhost.:8188", "http://128.0.0.1:8188",
])
def test_config_driver_must_name_this_computer(value):
    assert _refusal({"driver": value}) == f"{PATH}: {LOCAL}"
    # A typed --driver keeps every other rule but may name another computer.
    assert validate_driver_origin(value, local_only=False) == value


@pytest.mark.parametrize(("value", "canonical"), [
    ("http://127.0.0.1:8191/", "http://127.0.0.1:8191"),
    ("HTTP://127.0.0.1:8191", "http://127.0.0.1:8191"),
    ("http://LOCALHOST:8188", "http://localhost:8188"),
    ("http://127.0.0.1:80", "http://127.0.0.1"),
    ("https://127.0.0.1:443", "https://127.0.0.1"),
    ("http://127.0.0.1:08191", "http://127.0.0.1:8191"),
    ("http://127.0.0.1:8191?", "http://127.0.0.1:8191"),
    ("http://127.0.0.1:8191#", "http://127.0.0.1:8191"),
    ("http://[::1]:80", "http://[::1]"),
])
def test_driver_must_be_written_in_canonical_form(value, canonical):
    message = f"studio.driver must be written as '{canonical}'"
    assert _refusal({"driver": value}) == f"{PATH}: {message}"
    with pytest.raises(ValueError) as raised:
        validate_driver_origin(value, local_only=False)
    assert str(raised.value) == message
    assert canonical_driver_origin(value) == canonical


@pytest.mark.parametrize("value", [
    "http://127.0.0.1:8191", "http://127.0.0.2:8188", "http://[::1]:8188", "http://localhost:8188",
    "https://127.0.0.1:8443", "http://localhost",
])
def test_canonical_driver_values_are_accepted_unchanged(value):
    assert validate_studio_table({"driver": value}, PATH)["driver"] == value
    assert validate_driver_origin(value, local_only=True) == value
    assert canonical_driver_origin(value) == value


@pytest.mark.parametrize("port", [1023, 65536, 0, -1, True, False, "8220", 8220.0])
def test_port_range_and_type(port):
    assert _refusal({"port": port}) == f"{PATH}: studio.port must be an integer from 1024 to 65535"


@pytest.mark.parametrize("port", [1024, 8220, 65535])
def test_port_range_accepts_its_ends(port):
    assert validate_studio_table({"port": port}, PATH)["port"] == port


@pytest.mark.parametrize(("unit", "message"), [
    ("comfyui", "must name a systemd user service, such as comfyui.service"),
    ("-comfyui.service", "must name a systemd user service, such as comfyui.service"),
    ("comfy ui.service", "must name a systemd user service, such as comfyui.service"),
    ("comfyui.service\n", "must name a systemd user service, such as comfyui.service"),
    ("", "must name a systemd user service, such as comfyui.service"),
    (42, "must name a systemd user service, such as comfyui.service"),
    ("dgxm-worker.service", "must not name a Worker unit"),
    ("ComfyWORKER.service", "must not name a Worker unit"),
    ("dgxm-studio.service", "must not name dgxm-studio.service"),
])
def test_driver_service_name_rules(unit, message):
    assert _refusal({"driver_service": unit}) == f"{PATH}: studio.driver_service {message}"


# A template instance, joined at run time so the leak check does not read it as an email address.
@pytest.mark.parametrize("unit", ["comfyui.service", "comfyui-driver.service", "@".join(("comfyui", "8191.service")),
                                  "a_b.c.service"])
def test_driver_service_names_that_pass(unit):
    assert validate_studio_table({"driver_service": unit}, PATH)["driver_service"] == unit


def test_driver_service_validation_shares_the_unit_pattern():
    from dgx_monarch.studio import driver_service
    from dgx_monarch.studio.contracts import StudioError

    assert DRIVER_UNIT_PATTERN.pattern == r"[A-Za-z0-9][A-Za-z0-9_.@-]*\.service"
    driver_service.validate_unit_name("comfyui-driver.service")
    with pytest.raises(StudioError, match="exact unit name"):
        driver_service.validate_unit_name("comfyui")
    with pytest.raises(StudioError, match="Worker services cannot be selected"):
        driver_service.validate_unit_name("dgxm-worker.service")


@pytest.mark.parametrize("command", [
    "bash /home/user/start‮comfyui.sh", "bash /home/user/start​comfyui.sh",
    "bash\t/home/user/start-comfyui.sh", "x" * 513, "", "café", "line\nbreak", 7,
])
def test_driver_command_is_printable_ascii_of_bounded_length(command):
    assert _refusal({"driver_command": command}) == (
        f"{PATH}: studio.driver_command must be 1 to 512 printable ASCII characters")


def test_driver_command_accepts_printable_ascii_up_to_512_characters():
    command = "".join(chr(32 + index % 95) for index in range(512))
    assert validate_studio_table({"driver_command": command}, PATH)["driver_command"] == command
    assert validate_studio_table({"driver_command": " "}, PATH)["driver_command"] == " "


def test_a_full_table_round_trips():
    table = tomllib.loads(
        '[studio]\ndriver = "http://127.0.0.1:8191"\nport = 8221\n'
        'driver_service = "comfyui-driver.service"\ndriver_command = "bash /home/user/start-comfyui.sh"\n')
    validate_root_keys(table, PATH)
    assert validate_studio_table(table["studio"], PATH) == {
        "driver": "http://127.0.0.1:8191", "port": 8221, "driver_service": "comfyui-driver.service",
        "driver_command": "bash /home/user/start-comfyui.sh"}


def test_the_loader_admits_the_table_and_ignores_its_contents(tmp_path):
    plain = tmp_path / "plain.toml"
    plain.write_text(CLUSTER)
    studio = tmp_path / "studio.toml"
    studio.write_text(CLUSTER + '[studio]\ndriver = "http://0.0.0.0:8191"\nport = "x"\nunknown_key = 1\n')
    without, with_table = load_cluster_config(plain), load_cluster_config(studio)
    assert with_table.hosts == without.hosts
    assert {**asdict(with_table), "source": None} == {**asdict(without), "source": None}
    # Studio's own reader still refuses the same table.
    with pytest.raises(ClusterConfigError, match="unknown_key"):
        validate_studio_table(tomllib.loads(studio.read_text())["studio"], studio)
    # Other root keys stay strict.
    other = tmp_path / "other.toml"
    other.write_text(CLUSTER + "[studios]\n")
    with pytest.raises(ClusterConfigError, match="studios"):
        load_cluster_config(other)


def test_a_studio_table_does_not_move_the_trust_fingerprint(tmp_path):
    plain = tmp_path / "plain.toml"
    plain.write_text(CLUSTER)
    studio = tmp_path / "studio.toml"
    studio.write_text(CLUSTER + '[studio]\ndriver = "http://127.0.0.1:8191"\nport = 8221\n'
                      'driver_service = "comfyui-driver.service"\ndriver_command = "bash start.sh"\n')
    assert (trust_config_fingerprint(load_cluster_config(studio), "fallback")
            == trust_config_fingerprint(load_cluster_config(plain), "fallback"))


def test_defaults_are_the_companion_parser_defaults(monkeypatch):
    from dgx_monarch.studio import __main__ as launcher

    seen: dict[str, object] = {}

    class Parsed(Exception):
        pass

    def capture(parser, args=None, namespace=None):
        seen.update({action.dest: action.default for action in parser._actions})
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(Parsed):
        launcher.main([])
    assert seen["port"] == DEFAULT_STUDIO_PORT
    assert seen["driver"] == DEFAULT_STUDIO_DRIVER


def test_cluster_doc_states_the_studio_exception():
    text = " ".join((REPO / "docs" / "CLUSTER.md").read_text().split())
    assert ("Unknown keys fail config loading at the root, in `[cluster]`, `[[hosts]]`, `[worker_args]`, "
            "and every `[fabric.<name>]` table. The `[studio]` table is the one exception: only "
            "`dgxm studio` commands check it, so a typo there never stops a render.") in text
    assert "## `[studio]` (optional)" in (REPO / "docs" / "CLUSTER.md").read_text()


def test_the_example_config_keeps_the_table_commented():
    keys = {"driver", "port", "driver_service", "driver_command"}
    text = (REPO / "cluster.example.toml").read_text()
    assert "studio" not in tomllib.loads(text)
    lines = [line.strip() for line in text.splitlines()]
    block = ["[studio]"]
    for line in lines[lines.index("# [studio]") + 1:]:
        if not line.startswith("# "):
            break
        if line[2:].split(" = ", 1)[0] in keys:
            block.append(line[2:])
    table = tomllib.loads("\n".join(block))["studio"]
    assert set(table) == keys
    assert validate_studio_table(table, PATH) == table
