"""The driver-side price for the FSDP clean-reload proof's second load."""
from __future__ import annotations

import pytest

from dgx_monarch import fsdp_reload_price as price_mod
from dgx_monarch import mesh_safety, mesh_setup
from dgx_monarch.adapters import fsdp as adapters_fsdp
from dgx_monarch.capacity_fit import ABSOLUTE_HOST_FLOOR_BYTES

GIB = 1 << 30


def _row(mem_gib: float | None, rank: int = 0, *, host: object = None) -> dict:
    if host is not None:
        return {"rank": rank, "host": host}
    if mem_gib is None:
        return {"rank": rank}
    return {"rank": rank, "host": {"mem_gib": {"MemAvailable": mem_gib}}}


class _Handle:
    def __init__(self, rows, world: int = 2):
        self.world = world
        self._rows = rows
        self.calls: list[tuple] = []

    def call_all(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if isinstance(self._rows, BaseException):
            raise self._rows
        return self._rows


def _integrated(monkeypatch, *, integrated: bool = True, size: int = 60 * GIB) -> None:
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: integrated)
    monkeypatch.setattr(price_mod, "checkpoint_bytes", lambda _name: size)
    monkeypatch.setattr(price_mod, "checkpoint_kind", lambda _name: "bf16")


def _price(handle, unet_name: str = "flux2-dev.safetensors", **options):
    return price_mod.price_fsdp_clean_reload(
        handle, {"unet_name": unet_name, "options": options})


def test_the_price_and_the_worker_guard_share_one_multiplication():
    assert price_mod.fsdp_required_bytes is adapters_fsdp.fsdp_required_bytes
    for size in (0, 1, 32 * GIB, 60 * GIB, 121 * GIB):
        # No world: the full-copy bound. A world: shards plus one block.
        assert adapters_fsdp.fsdp_required_bytes(size, checkpoint_kind="bf16") == int(
            size * adapters_fsdp.FSDP_MATERIALIZE_FACTOR) + ABSOLUTE_HOST_FLOOR_BYTES
        for world in (1, 2, 4):
            assert adapters_fsdp.fsdp_required_bytes(size, world, "bf16") == int(
                size * (1.0 / world + adapters_fsdp.FSDP_STREAM_TRANSIENT_FRACTION)
            ) + ABSOLUTE_HOST_FLOOR_BYTES


def test_supported_quantized_fsdP_uses_the_full_direct_wrap_price():
    size = 60 * GIB
    required = adapters_fsdp.fsdp_required_bytes(size, 2, "fp8")

    assert required == int(size * adapters_fsdp.FSDP_MATERIALIZE_FACTOR) + ABSOLUTE_HOST_FLOOR_BYTES
    assert required > adapters_fsdp.fsdp_required_bytes(size, 2, "bf16")
    assert adapters_fsdp.fsdp_required_bytes(size, 2) == required


def test_clean_reload_prices_quantized_direct_wrap(monkeypatch):
    size = 60 * GIB
    _integrated(monkeypatch, size=size)
    monkeypatch.setattr(price_mod, "checkpoint_kind", lambda _name: "int8")

    priced = _price(_Handle([_row(75, 0), _row(75, 1)], world=2))

    assert priced.applies and not priced.fits
    assert priced.required_bytes == adapters_fsdp.fsdp_required_bytes(size, 2, "int8")


def test_worker_guard_enforces_the_shared_fsdp_host_floor(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"x")
    size = 10 * GIB
    monkeypatch.setattr(adapters_fsdp.os.path, "getsize", lambda _path: size)
    monkeypatch.setattr(adapters_fsdp, "gpu_is_integrated", lambda: True)
    required = adapters_fsdp.fsdp_required_bytes(size, 2, "bf16")
    monkeypatch.setattr(adapters_fsdp, "mem_available_bytes", lambda: required - 1)

    with pytest.raises(mesh_safety.StockLoadCapacityError, match="host floor"):
        adapters_fsdp.fsdp_load_capacity_check(
            str(checkpoint), "checkpoint.safetensors", {}, world=2,
            checkpoint_kind="bf16")

    monkeypatch.setattr(adapters_fsdp, "mem_available_bytes", lambda: required)
    adapters_fsdp.fsdp_load_capacity_check(
        str(checkpoint), "checkpoint.safetensors", {}, world=2,
        checkpoint_kind="bf16")


@pytest.mark.parametrize(
    "size_gib, avail_gib",
    [
        pytest.param(4, 47, id="plenty"),
        pytest.param(4, 3, id="just-fits"),
        pytest.param(4, 2, id="just-short"),
        pytest.param(8, 47, id="fits-with-margin"),
        pytest.param(8, 4, id="shortfall"),
    ],
)
def test_the_price_refuses_exactly_what_the_worker_guard_refuses(
    monkeypatch, tmp_path, size_gib, avail_gib,
):
    """A re-price of the factor must move both together."""
    path = tmp_path / "checkpoint.safetensors"
    with open(path, "wb") as handle:  # sparse: no bytes are written
        handle.truncate(size_gib * GIB)
    monkeypatch.setattr(adapters_fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(
        adapters_fsdp, "mem_available_bytes", lambda: avail_gib * GIB)
    guard_refused = False
    try:
        adapters_fsdp.fsdp_load_capacity_check(
            str(path), "checkpoint.safetensors", {}, world=2,
            checkpoint_kind="bf16")
    except mesh_safety.StockLoadCapacityError:
        guard_refused = True

    _integrated(monkeypatch, size=size_gib * GIB)
    priced = _price(_Handle([_row(avail_gib, 0), _row(avail_gib, 1)], world=2))

    assert priced.applies is True
    assert (not priced.fits) is guard_refused
    assert priced.required_bytes == adapters_fsdp.fsdp_required_bytes(
        size_gib * GIB, 2, "bf16")


# The four no-claim exits

def test_a_loader_dtype_cast_makes_no_claim(monkeypatch):
    _integrated(monkeypatch)
    handle = _Handle([_row(1, 0), _row(1, 1)])

    priced = _price(handle, dtype="float8_e4m3fn")

    assert (priced.applies, priced.fits) == (False, True)
    assert handle.calls == []  # no claim means no probe either


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"weight_dtype": "bf16"}, id="driver-bf16-cast"),
        pytest.param({"weight_dtype": "fp8_e4m3fn"}, id="driver-fp8-cast"),
        pytest.param({"weight_dtype": "nonsense"}, id="unreadable-option"),
        pytest.param({"dtype": "float8_e4m3fn"}, id="worker-shaped-key"),
    ],
)
def test_a_cast_under_either_option_name_makes_no_claim(monkeypatch, options):
    """The guard skips itself on a cast, so the price must skip on one too.

    A driver request names ``weight_dtype``. Reading only the worker's
    ``dtype`` key would price every driver-side cast the worker guard waves
    through, which is the one direction a mirror may not take.
    """
    _integrated(monkeypatch)
    handle = _Handle([_row(1, 0), _row(1, 1)])

    priced = price_mod.price_fsdp_clean_reload(
        handle, {"unet_name": "flux2-dev.safetensors", "options": options})

    assert (priced.applies, priced.fits) == (False, True)
    assert priced.skipped_reason
    assert handle.calls == []


def test_the_default_dtype_is_not_a_cast(monkeypatch):
    _integrated(monkeypatch)
    handle = _Handle([_row(96.0, 0), _row(96.0, 1)])

    priced = price_mod.price_fsdp_clean_reload(
        handle, {"unet_name": "flux2-dev.safetensors",
                 "options": {"weight_dtype": "default"}})

    assert priced.applies is True


def test_a_discrete_driver_gpu_makes_no_claim(monkeypatch):
    _integrated(monkeypatch, integrated=False)
    handle = _Handle([_row(1, 0), _row(1, 1)])

    priced = _price(handle)

    assert priced.applies is False
    assert handle.calls == []


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([_row(None, 0), _row(1, 1)], id="no-host-block"),
        pytest.param([_row(1, 0), _row(None, 1, host={})], id="no-mem-block"),
        pytest.param(
            [_row(1, 0), _row(None, 1, host={"mem_gib": {}})], id="no-memavailable"),
        pytest.param(
            [_row(1, 0), _row("47.2", 1)], id="unreadable-memavailable"),
        pytest.param([_row(1, 0), _row(True, 1)], id="boolean-memavailable"),
        pytest.param([_row(1, 0), _row(-1.0, 1)], id="negative-memavailable"),
        pytest.param([_row(1, 0)], id="a-rank-did-not-answer"),
        pytest.param([], id="no-rank-answered"),
        pytest.param({"rank": 0}, id="malformed-response"),
        pytest.param(RuntimeError("status endpoint refused"), id="probe-failed"),
    ],
)
def test_an_unreadable_rank_makes_no_claim(monkeypatch, rows):
    _integrated(monkeypatch)

    priced = _price(_Handle(rows))

    assert (priced.applies, priced.fits) == (False, True)
    assert priced.skipped_reason


def test_an_unmeasurable_checkpoint_makes_no_claim(monkeypatch):
    _integrated(monkeypatch, size=0)
    handle = _Handle([_row(1, 0), _row(1, 1)])

    priced = _price(handle)

    assert priced.applies is False
    assert handle.calls == []


def test_the_price_refuses_on_the_leanest_rank(monkeypatch):
    """The load runs everywhere, so one short rank refuses the whole proof."""
    _integrated(monkeypatch, size=60 * GIB)

    # 60 GiB prices at 34.8 GiB on the pair (1/2 + 0.08) plus the 5 GiB host
    # floor; rank 1 sits under it.
    priced = _price(_Handle([_row(96.0, 0), _row(30.0, 1)]))

    assert (priced.applies, priced.fits) == (True, False)
    assert priced.rank == 1
    assert priced.mem_available_bytes == int(30.0 * GIB)
    assert priced.checkpoint_bytes == 60 * GIB
    assert priced.headroom_bytes < 0


def test_the_status_probe_is_setup_independent_and_takes_no_session(monkeypatch):
    """Reclassifying ``status`` must break here, not on a first-use render."""
    assert "status" in mesh_setup._SETUP_INDEPENDENT_ENDPOINTS
    assert "status" not in mesh_setup.SESSION_SCOPED_ENDPOINTS
    assert "status" not in mesh_setup.AMBIGUOUS_MUTATION_ENDPOINTS

    _integrated(monkeypatch)
    handle = _Handle([_row(96.0, 0), _row(96.0, 1)])
    _price(handle)

    assert [method for method, _kwargs in handle.calls] == ["status"]
    assert handle.calls[0][1] == {"timeout_s": price_mod.STATUS_TIMEOUT_S}


class _Ledger:
    def __init__(self, error: BaseException | None = None):
        self.records: list[tuple] = []
        self._error = error

    def record(self, *args):
        if self._error is not None:
            raise self._error
        self.records.append(args)


def _stop(ledger, price=None, exc=None) -> str:
    return price_mod.record_capacity_stop(
        ledger, "combo", "artifacts", "known", {"proof_scope": "fsdp_clean_reload"},
        "flux2-dev.safetensors", origin="auto_first_use", run_id="r1",
        price=price, exc=exc)


def test_a_priced_stop_records_one_inconclusive_capacity_row_with_its_numbers():
    ledger = _Ledger()
    priced = price_mod.ReloadPrice(
        True, False, 60 * GIB, 72 * GIB, int(47.2 * GIB), 1)

    sentence = _stop(ledger, price=priced)

    assert len(ledger.records) == 1
    _key, _artifacts, _commit, verdict, detail, context = ledger.records[0]
    assert verdict == "INCONCLUSIVE"
    assert detail["cross_mode"] == "CAPACITY"
    assert detail["quarantine_levers"] == []
    assert detail["measured"]["probe"] == "stock_load_preflight"
    for field in ("checkpoint_bytes", "mem_available_bytes", "required_bytes",
                  "headroom_bytes"):
        assert isinstance(detail["measured"][field], int)
    assert context == {"proof_scope": "fsdp_clean_reload"}
    assert "auto_gate=off" in sentence
    # The row carries no skip classification, so it can never grant a ceremony
    # skip on the next queue (gate_inconclusive.row_grants_skip).
    assert "inconclusive_kind" not in detail


def test_an_in_flight_stop_records_a_classification_not_the_refusal_text():
    """The row is durable and gets pasted; the refusing text is not."""
    ledger = _Ledger()
    worker_text = "/home/operator/models/flux2 on rank 1 said no room"

    sentence = _stop(ledger, exc=RuntimeError(worker_text))

    detail = ledger.records[0][4]
    assert detail["measured"] is None
    assert detail["capacity_detail"] == "RuntimeError/untagged"
    # The operator still reads the refusal, in the sentence and in the log.
    assert worker_text in sentence
    assert "capacity" in sentence


def test_a_tagged_worker_refusal_is_classified_by_its_class_and_guard():
    from dgx_monarch.refusal import RefusalClass, refusal

    tagged = refusal(RefusalClass.CAPACITY, "no room for this launch",
                     guard=price_mod.PROBE, waivable=False)
    ledger = _Ledger()

    _stop(ledger, exc=RuntimeError(tagged))

    assert ledger.records[0][4]["capacity_detail"] == (
        f"RuntimeError/class-C guard={price_mod.PROBE}")


def test_a_classification_never_carries_the_exception_body():
    body = "host box.example.com path /home/operator/ComfyUI/models"
    classification = price_mod.capacity_classification(ValueError(body))

    assert classification == "ValueError/untagged"
    assert "example.com" not in classification


class _Log:
    """The module logger does not propagate, so read it at the module seam."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _write(self, fmt, *args):
        self.lines.append(str(fmt) % args)

    info = warning = error = _write

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def test_a_price_that_applies_logs_its_numbers_on_both_branches(monkeypatch):
    """A price that only spoke on a refusal would leave a fitting ceremony
    indistinguishable from one that never priced at all, and the per-rank
    numbers that pricing the ceremony's peak needs would exist only for proofs
    that failed."""
    recorder = _Log()
    monkeypatch.setattr(price_mod, "log", recorder)
    _integrated(monkeypatch, size=10 * GIB)

    fits = _price(_Handle([_row(80.0, 0), _row(60.0, 1)]))
    assert fits.applies and fits.fits
    assert "fits" in recorder.text
    assert "rank 1" in recorder.text
    assert "60.0 GiB available" in recorder.text
    assert "10.0 GiB checkpoint" in recorder.text

    recorder.lines.clear()
    _integrated(monkeypatch, size=60 * GIB)
    short = _price(_Handle([_row(30.2, 0), _row(30.2, 1)]))
    assert short.applies and not short.fits
    assert "does not fit" in recorder.text
    assert "30.2 GiB available" in recorder.text


def test_a_no_claim_price_logs_no_numbers(monkeypatch):
    recorder = _Log()
    monkeypatch.setattr(price_mod, "log", recorder)
    _integrated(monkeypatch, integrated=False)

    assert not _price(_Handle([_row(80.0, 0)])).applies
    assert "clean-reload price" not in recorder.text


def test_the_refusing_text_reaches_the_log_even_though_the_row_drops_it(
        monkeypatch):
    recorder = _Log()
    monkeypatch.setattr(price_mod, "log", recorder)
    worker_text = "rank 1 at /home/operator/models said no room"

    _stop(_Ledger(), exc=RuntimeError(worker_text))

    assert worker_text in recorder.text
    assert "RuntimeError/untagged" in recorder.text


def test_the_probe_name_is_already_frozen_vocabulary():
    from dgx_monarch.gate_audit_vocab import MEASURED_PROBES

    assert price_mod.PROBE in MEASURED_PROBES


def test_an_unwritable_ledger_never_masks_the_refusal():
    sentence = _stop(_Ledger(OSError("read-only ledger")),
                     price=price_mod.ReloadPrice(True, False, GIB, 2 * GIB, 0, 0))

    assert sentence


def test_the_checkpoint_lookup_reads_the_driver_model_folders(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    path = tmp_path / "flux2-dev.safetensors"
    with open(path, "wb") as handle:
        handle.truncate(3 * GIB)
    monkeypatch.setitem(
        sys.modules, "folder_paths",
        SimpleNamespace(get_full_path=lambda kind, name: (
            str(path) if (kind, name) == ("diffusion_models", "flux2-dev.safetensors")
            else None)))

    assert price_mod.checkpoint_bytes("flux2-dev.safetensors") == 3 * GIB
    assert price_mod.checkpoint_bytes("absent.safetensors") == 0


def test_no_comfy_installation_measures_nothing_rather_than_raising(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "folder_paths", None)

    assert price_mod.checkpoint_bytes("flux2-dev.safetensors") == 0


def test_both_stops_carry_the_same_tag_the_worker_guard_stamps():
    """The driver stop and the worker refusal read as one guard, not two."""
    from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

    priced = price_mod.ReloadPrice(True, False, 60 * GIB, 72 * GIB, int(47.2 * GIB), 1)
    for sentence in (price_mod.operator_sentence(priced, "flux2-dev.safetensors"),
                     price_mod.operator_sentence(
                         priced, "flux2-dev.safetensors", stage="reprice"),
                     price_mod._in_flight_sentence("flux2-dev.safetensors", "no room")):
        tag = parse_leading_refusal_tag(sentence)
        assert tag is not None
        assert tag.refusal_class is RefusalClass.CAPACITY
        assert tag.guard == price_mod.PROBE == "stock_load_preflight"
        assert tag.waivable is False
        # Class C never refuses bare: a fitting strategy is always named.
        assert "fits:" in sentence
        assert "docs/TROUBLESHOOTING.md #88" in sentence


def test_the_priced_stop_names_auto_gate_off_as_the_intended_path():
    """The fresh-process proof was declined on 2026-08-26, so the refusal on a
    too-large checkpoint frames auto_gate=off as intended, not a fallback."""
    priced = price_mod.ReloadPrice(True, False, 60 * GIB, 72 * GIB, int(47.2 * GIB), 1)
    sentence = price_mod.operator_sentence(priced, "flux2-dev.safetensors")
    assert "auto_gate=off" in sentence
    assert "intended path" in sentence
    assert "2026-08-26" in sentence


def test_the_reprice_sentence_does_not_claim_nothing_happened():
    """The reprice runs after the unload and the baseline render (leg 6,
    2026-08-26), so its sentence must not carry the preflight's claim."""
    priced = price_mod.ReloadPrice(True, False, 60 * GIB, 72 * GIB, int(31.7 * GIB), 0)
    preflight = price_mod.operator_sentence(priced, "flux2-dev.safetensors")
    reprice = price_mod.operator_sentence(
        priced, "flux2-dev.safetensors", stage="reprice")
    assert "The proof was not started" in preflight
    assert "The proof was not started" not in reprice
    assert "stopped before its reload cycle" in reprice
    assert "auto_gate=off" in reprice and "2026-08-26" in reprice


def test_a_worker_tag_quoted_inside_the_in_flight_stop_stays_body_text():
    """One authoritative tag, so the class can never be read ambiguously."""
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag, refusal

    worker = refusal(RefusalClass.CAPACITY, "the FSDP launch cannot load it",
                     guard="stock_load_preflight", waivable=False)
    sentence = price_mod._in_flight_sentence("flux2-dev.safetensors", worker)

    tag = parse_refusal_tag(sentence)
    assert tag is not None
    assert tag.refusal_class is RefusalClass.CAPACITY
    assert sentence.count("[dgxm:") == 1


def test_an_in_flight_row_reaches_disk_carrying_no_worker_text(tmp_path):
    """The privacy claim, read back off the real ledger file.

    `dgxm_gate_ledger.jsonl` is what a reviewer asks an operator to paste, so
    what a fake ledger accepted proves nothing about what lands on disk.
    """
    import json
    import re

    from dgx_monarch.gate_ledger import LEDGER_NAME, GateLedger

    body = "rank 1 at /home/operator/models/flux2 on box.example.com: no room"

    price_mod.record_capacity_stop(
        GateLedger(str(tmp_path)), "combo-key", "artifact-sig", "comfy-sha",
        {"proof_scope": "fsdp_clean_reload"}, "flux2-dev.safetensors",
        origin="auto_first_use", run_id="r1", exc=RuntimeError(body))

    rows = [json.loads(line) for line in
            (tmp_path / LEDGER_NAME).read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["verdict"] == "INCONCLUSIVE"
    assert row["cross_mode"] == "CAPACITY"
    assert row["measured"] is None
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}/"
                        r"(?:untagged|class-[PCUK] guard=[\w.:-]{1,64})",
                        row["capacity_detail"])
    on_disk = (tmp_path / LEDGER_NAME).read_text()
    assert "/home/operator" not in on_disk
    assert "example.com" not in on_disk


def test_the_capacity_row_survives_a_real_ledger_round_trip(tmp_path):
    """The recorded row is the product claim, so read it back off disk.

    ``record_capacity_stop`` swallows a ledger failure so the operator still
    gets the sentence, and ``GateLedger.record`` swallows OSError of its own.
    A fake ledger therefore proves nothing about what a first-use refusal
    leaves behind for the next queue to read.
    """
    import json

    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, LEDGER_NAME, GateLedger

    ledger = GateLedger(str(tmp_path))
    priced = price_mod.ReloadPrice(
        True, False, 60 * GIB, 72 * GIB, int(47.2 * GIB), 1)

    sentence = price_mod.record_capacity_stop(
        ledger, "combo-key", "artifact-sig", "comfy-sha",
        {"proof_scope": "fsdp_clean_reload"}, "flux2-dev.safetensors",
        origin="auto_first_use", run_id="r1", price=priced)

    rows = [json.loads(line) for line in
            (tmp_path / LEDGER_NAME).read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["verdict"] == "INCONCLUSIVE"
    assert row["cross_mode"] == "CAPACITY"
    assert row["key"] == "combo-key"
    assert row["artifacts"] == "artifact-sig"
    assert row["comfy"] == "comfy-sha"
    assert row["model"] == "flux2-dev.safetensors"
    assert row["origin"] == "auto_first_use"
    assert row["gate_protocol"] == GATE_PROTOCOL_VERSION
    assert "fsdp_clean_reload" in row["capability_context"]
    assert row["measured"]["probe"] == "stock_load_preflight"
    assert row["measured"]["checkpoint_bytes"] == 60 * GIB
    assert row["measured"]["required_bytes"] == 72 * GIB
    assert row["measured"]["mem_available_bytes"] == int(47.2 * GIB)
    assert row["measured"]["headroom_bytes"] == int(47.2 * GIB) - 72 * GIB
    assert row["measured"]["weights_gib"] == 60.0
    assert row["measured"]["rank"] == 1
    assert "inconclusive_kind" not in row
    assert sentence
