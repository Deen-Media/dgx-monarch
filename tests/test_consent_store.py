"""The persisted consent memo: keying, self-invalidation, atomic writes, modes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from dgx_monarch import consent_pending, consent_store

CONTEXT = {"combo_key": "9a3f2c81d4e6f0a1b2c3d4e5", "weight_dtype": "default"}


@pytest.fixture(autouse=True)
def _memo(tmp_path, monkeypatch):
    monkeypatch.setattr(consent_store, "MEMO_PATH",
                        str(tmp_path / "cache" / "dgx-monarch" / "consent_memo.json"))
    return consent_store.MEMO_PATH


def _checkpoint(tmp_path, name="minimax_h3_fl2va_bf16.safetensors", body=b"weights"):
    path = tmp_path / name
    path.write_bytes(body)
    return str(path)


def _record(path, key_kind="rescue-slab", **overrides):
    fields = {
        "id": "9f2c4ab17d0e4c1fa3b58e6d21c7f480",
        "kind": key_kind,
        "target_guard": "stock_load_preflight",
        "waiver_class": "C",
        "artifact": os.path.basename(path),
        "path": path,
        "file_identity": consent_store.file_identity(path),
        "memo_context": dict(CONTEXT),
        "granted_at": "2026-08-04 13:22:07",
        "granted_epoch": time.time(),
        "consent_source": "panel",
        "reason": "Stock residency cannot fit this checkpoint on this box.",
        "ledger_key": "waiver:rescue-slab:9a3f2c81d4e6f0a1b2c3d4e5",
        "combo_key": "9a3f2c81d4e6f0a1b2c3d4e5",
        "artifacts": "b" * 64,
        "artifacts_legacy": "legacy-checkpoint-digest",
        "artifacts_legacy_complete": True,
    }
    fields.update(overrides)
    return consent_store.ConsentRecord(**fields)


def test_grant_lookup_and_revoke_round_trip(tmp_path):
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None

    consent_store.grant(_record(path), key)
    found = consent_store.lookup("rescue-slab", path, CONTEXT)
    assert found is not None
    assert found.kind == "rescue-slab"
    assert found.consent_source == "panel"

    assert consent_store.revoke(key) is True
    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None
    assert consent_store.revoke(key) is False


def test_a_different_kind_or_context_is_a_different_authorization(tmp_path):
    path = _checkpoint(tmp_path)
    consent_store.grant(_record(path), consent_store.memo_key("rescue-slab", path, CONTEXT))

    other_context = {**CONTEXT, "weight_dtype": "fp8_e4m3fn"}
    assert consent_store.lookup("rescue-slab", path, other_context) is None
    assert consent_store.lookup("waive-first-load-stock", path, CONTEXT) is None


def test_an_in_place_rewrite_invalidates_the_memo_and_marks_it_stale(tmp_path):
    """A new file is a new authorization, including a same-size rewrite."""
    path = _checkpoint(tmp_path, body=b"weights")
    consent_store.grant(_record(path), consent_store.memo_key("rescue-slab", path, CONTEXT))
    assert consent_store.lookup("rescue-slab", path, CONTEXT) is not None

    with open(path, "r+b") as handle:  # same size, different bytes
        handle.write(b"WEIGHTS")
    os.utime(path, (time.time() + 5, time.time() + 5))

    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None
    rows = consent_store.list_active()
    assert len(rows) == 1
    assert rows[0]["stale"] is True
    assert "path" not in rows[0] and "file_identity" not in rows[0]


def test_a_missing_checkpoint_never_grants(tmp_path):
    path = _checkpoint(tmp_path)
    consent_store.grant(_record(path), consent_store.memo_key("rescue-slab", path, CONTEXT))
    os.unlink(path)
    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None
    assert consent_store.list_active()[0]["stale"] is True


def test_a_failed_write_leaves_the_previous_memo_intact_and_no_temp_file(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    consent_store.grant(_record(path), key)
    before = open(consent_store.MEMO_PATH).read()

    def boom(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(json, "dump", boom)
    with pytest.raises(consent_store.ConsentStoreError):
        consent_store.grant(_record(path, id="0" * 32), key)

    assert open(consent_store.MEMO_PATH).read() == before
    directory = os.path.dirname(consent_store.MEMO_PATH)
    assert [name for name in os.listdir(directory) if name.endswith(".tmp")] == []


def test_directory_fsync_failure_after_replace_is_loud(tmp_path, monkeypatch):
    """A rename that was not made durable is not reported as a durable one."""
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    consent_store.grant(_record(path), key)
    real_fsync = consent_store.os.fsync
    calls = 0

    def fail_directory_sync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory sync failed")
        return real_fsync(fd)

    monkeypatch.setattr(consent_store.os, "fsync", fail_directory_sync)
    with pytest.raises(consent_store.ConsentStoreError, match="directory sync failed"):
        consent_store.revoke(key)
    assert calls == 2


def test_directory_fsync_cancellation_is_not_recast_as_success(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    consent_store.grant(_record(path), key)
    real_fsync = consent_store.os.fsync
    calls = 0

    class Cancelled(KeyboardInterrupt):
        pass

    cancelled = Cancelled("sync interrupted")

    def cancel_directory_sync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise cancelled
        return real_fsync(fd)

    monkeypatch.setattr(consent_store.os, "fsync", cancel_directory_sync)
    with pytest.raises(Cancelled) as raised:
        consent_store.revoke(key)
    assert raised.value is cancelled


def test_a_failed_directory_sync_leaves_the_revoke_already_applied(
    tmp_path, monkeypatch,
):
    """The raise means not durable, not that the change never happened.

    ``os.replace`` has already run when the directory sync fails, so the memo
    on disk is the new one and every reader sees it. Do not report this failure
    as an unknown mutation: that points the caller at the wrong recovery. The
    consent is gone, and a retry that expects to find it writes nothing at all.
    """
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    consent_store.grant(_record(path), key)
    assert consent_store.lookup("rescue-slab", path, CONTEXT) is not None
    real_fsync = consent_store.os.fsync
    calls = 0

    def fail_directory_sync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory sync failed")
        return real_fsync(fd)

    monkeypatch.setattr(consent_store.os, "fsync", fail_directory_sync)
    with pytest.raises(consent_store.ConsentStoreError):
        consent_store.revoke(key)

    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None


def test_a_lost_consent_is_loud_not_best_effort(tmp_path, monkeypatch):
    """The family memo swallows OSError; a consent must not. A user who clicked
    and got nothing has to be told."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    unwritable = tmp_path / "locked"
    unwritable.mkdir(mode=0o500)
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(unwritable / "sub" / "consent_memo.json"))
    with pytest.raises(consent_store.ConsentStoreError):
        consent_store.grant(_record(_checkpoint(tmp_path)), "k" * 32)


def test_the_memo_file_is_private(tmp_path):
    path = _checkpoint(tmp_path)
    consent_store.grant(_record(path), consent_store.memo_key("rescue-slab", path, CONTEXT))
    assert os.stat(consent_store.MEMO_PATH).st_mode & 0o777 == 0o600
    assert os.stat(os.path.dirname(consent_store.MEMO_PATH)).st_mode & 0o077 == 0


def test_eviction_drops_the_oldest_grant(tmp_path, monkeypatch):
    monkeypatch.setattr(consent_store, "CONSENT_LIMIT", 4)
    path = _checkpoint(tmp_path)
    for index in range(6):
        consent_store.grant(_record(path, id=f"{index:032d}"), f"key-{index}")
    memo = consent_store.read()["consents"]
    assert list(memo) == ["key-2", "key-3", "key-4", "key-5"]


def test_a_foreign_schema_reads_as_empty_and_is_replaced(tmp_path):
    os.makedirs(os.path.dirname(consent_store.MEMO_PATH), exist_ok=True)
    with open(consent_store.MEMO_PATH, "w") as handle:
        json.dump({"schema": consent_store.SCHEMA + 1,
                   "consents": {"k": {"id": "x"}}, "auto_rescue": True}, handle)

    assert consent_store.read()["consents"] == {}
    assert consent_store.auto_rescue() is False

    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    consent_store.grant(_record(path), key)
    written = json.load(open(consent_store.MEMO_PATH))
    assert written["schema"] == consent_store.SCHEMA
    assert list(written["consents"]) == [key]


def test_a_malformed_row_is_treated_as_absent(tmp_path):
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    os.makedirs(os.path.dirname(consent_store.MEMO_PATH), exist_ok=True)
    with open(consent_store.MEMO_PATH, "w") as handle:
        json.dump({"schema": consent_store.SCHEMA, "auto_rescue": False,
                   "consents": {key: {"id": "x"}}}, handle)
    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None
    assert consent_store.list_active() == []


def test_schema_one_memos_are_invalidated_before_projection_or_regrant(tmp_path):
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    old = _record(path).as_json()
    old.pop("artifacts_legacy")
    old.pop("artifacts_legacy_complete")
    os.makedirs(os.path.dirname(consent_store.MEMO_PATH), exist_ok=True)
    with open(consent_store.MEMO_PATH, "w") as handle:
        json.dump({"schema": 1, "auto_rescue": True, "consents": {key: old}}, handle)

    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None
    assert consent_store.list_active() == []
    assert consent_store.auto_rescue() is False
    consent_store.grant(_record(path), key)
    assert consent_store.lookup("rescue-slab", path, CONTEXT) is not None
    assert json.load(open(consent_store.MEMO_PATH))["schema"] == consent_store.SCHEMA
    assert [row["key"] for row in consent_store.list_active()] == [key]


def test_current_schema_capacity_memo_without_legacy_identity_is_absent(tmp_path):
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)
    incomplete = _record(path).as_json()
    incomplete.pop("artifacts_legacy")
    os.makedirs(os.path.dirname(consent_store.MEMO_PATH), exist_ok=True)
    with open(consent_store.MEMO_PATH, "w") as handle:
        json.dump({"schema": consent_store.SCHEMA, "auto_rescue": False,
                   "consents": {key: incomplete}}, handle)

    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None
    assert consent_store.list_active() == []


@pytest.mark.parametrize(
    "epoch", [10 ** 1000, float("inf"), float("-inf"), float("nan"), "yesterday", None],
    ids=["overflow", "positive-inf", "negative-inf", "nan", "string", "missing"],
)
def test_malformed_or_nonfinite_grant_epoch_is_absent(tmp_path, epoch):
    raw = _record(_checkpoint(tmp_path)).as_json()
    raw["granted_epoch"] = epoch

    assert consent_store.ConsentRecord.from_json(raw) is None


def test_nonfinite_grant_epoch_is_not_persisted(tmp_path):
    path = _checkpoint(tmp_path)
    key = consent_store.memo_key("rescue-slab", path, CONTEXT)

    with pytest.raises(consent_store.ConsentStoreError, match="epoch must be finite"):
        consent_store.grant(_record(path, granted_epoch=float("inf")), key)
    assert consent_store.read()["consents"] == {}


def test_auto_rescue_toggle_round_trip(tmp_path):
    assert consent_store.auto_rescue() is False
    consent_store.set_auto_rescue(True)
    assert consent_store.auto_rescue() is True
    consent_store.set_auto_rescue(False)
    assert consent_store.auto_rescue() is False


def test_stale_entries_are_pruned_once_they_have_aged_out(tmp_path):
    path = _checkpoint(tmp_path)
    old = time.time() - consent_store.STALE_PRUNE_S - 1
    consent_store.grant(_record(path, granted_epoch=old, file_identity="gone"), "stale-key")
    fresh = _checkpoint(tmp_path, name="other.safetensors")
    consent_store.grant(_record(fresh), consent_store.memo_key("rescue-slab", fresh, CONTEXT))
    assert "stale-key" not in consent_store.read()["consents"]


def test_context_must_be_json_native_and_exact(tmp_path):
    with pytest.raises(ValueError):
        consent_pending.consent_context("rescue-slab", combo_key="abc")
    with pytest.raises(ValueError):
        consent_pending.consent_context("rescue-slab", combo_key="abc", weight_dtype="d", extra=1)
    with pytest.raises(ValueError):
        consent_pending.consent_context("no-such-kind", combo_key="abc")
    with pytest.raises(ValueError):  # NaN is not JSON compliant, same rule the ledger uses
        consent_pending.consent_context(
            "rescue-slab", combo_key="abc", weight_dtype=float("nan"))
    with pytest.raises(TypeError):
        consent_pending.consent_context("rescue-slab", combo_key="abc", weight_dtype=object())


def test_file_identity_matches_the_family_memo_definition(tmp_path):
    """The memo self-invalidates on the same rule the family memo uses. The
    function is a copy of actor/store_family.file_identity (the docstring of
    consent_store.file_identity says why), so the two must be pinned equal."""
    path = _checkpoint(tmp_path)
    donor = os.path.join(os.path.dirname(consent_store.__file__), "actor", "store_family.py")
    probe = (
        "import importlib.util, sys\n"
        "donor, checkpoint = sys.argv[1:]\n"
        "assert 'dgx_monarch.actor' not in sys.modules\n"
        "assert 'dgx_monarch.actor.worker' not in sys.modules\n"
        "spec = importlib.util.spec_from_file_location('dgxm_store_family_probe', donor)\n"
        "assert spec is not None and spec.loader is not None\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert 'dgx_monarch.transfer_utils' in sys.modules\n"
        "assert 'dgx_monarch.actor' not in sys.modules\n"
        "assert 'dgx_monarch.actor.worker' not in sys.modules\n"
        "print(module.file_identity(checkpoint))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, donor, path],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == consent_store.file_identity(path)
