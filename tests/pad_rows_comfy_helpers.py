"""Real-ComfyUI CPU harness for the per-family pad-row tests.

Each family test builds comfy's own model at toy size, runs the stock class
forward once on an unpatched model, then runs the adapter's bound forward on
deep copies, one rank thread each, through the real
``base.make_usp_attention`` dispatch and the real full-axis call. Only the
all-to-all, the sequence gather and the kernel are stubbed (the harness of
tests/test_flux_pad_exclusion.py), and the kernel is comfy's own attention on
the rows it is handed, so stock and every rank run one kernel.

The divisibility pad rows are filled with NaN instead of zeros (``nan_pads``).
Everything a sharded forward runs outside attention is row-wise, so a NaN pad
row stays in its own row until the gather trims it. If it reaches a real row's
softmax, or the shared activation scale, the output or the recorded scale is
NaN. A zero pad row would also be wrong there, but by an amount that can be
small; NaN cannot hide.

The ring cases install the same stubs with a ring degree of 2, where xfuser's
own call attends this rank's queries against every rank's keys, pads
included: what pure ring2 does with rows it cannot exclude.
"""
from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Sequence

import torch

from dgx_monarch.adapters import base
from real_comfy_helpers import COMFY_DIR, real_comfy  # noqa: F401  (the family tests import both)
from test_flux_pad_exclusion import _LOCAL, _Harness, _install_stubs, _rank

WORLD = 2


def same(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    """The same shape and the same values to the dtype's rounding.

    A rank runs its Linears on its own row count, and a CPU BLAS can pick
    another kernel for another row count (tests/test_ernie_stock_equivalence.py
    records which shapes keep their bits on this rig), so against stock the
    claim is agreement to rounding. The pad rows are NaN, so an attended pad
    fails this by any tolerance. Rank against rank the tests use torch.equal.
    """
    if candidate.shape != reference.shape:
        return False
    if torch.equal(candidate, reference):
        return True
    tolerance = 1e-9 if candidate.dtype == torch.float64 else 2e-5
    return torch.allclose(candidate, reference, rtol=tolerance, atol=tolerance)


def fill_parameters(model: torch.nn.Module, seed: int, std: float = 0.2) -> None:
    """comfy builds parameters with torch.empty; a checkpoint load fills them."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                        dtype=parameter.dtype) * std)


def nan_pads(monkeypatch) -> None:
    """Pad floating streams with NaN rows where ``shard_seq`` pads with zeros."""
    original = base.pad_seq_to_multiple

    def pad(t, multiple, dim=1):
        padded, orig = original(t, multiple, dim=dim)
        if padded.shape[dim] != orig and padded.is_floating_point():
            padded = padded.clone()
            padded.narrow(dim, orig, padded.shape[dim] - orig).fill_(float("nan"))
        return padded, orig

    monkeypatch.setattr(base, "pad_seq_to_multiple", pad)


def plain(value):
    """The tensor behind comfy's AttentionTensorContainer, or the tensor itself."""
    return value if isinstance(value, torch.Tensor) else value.take()


class StockKernelHarness(_Harness):
    """Runs one given attention kernel on the rows each call hands it.

    ``kernel(q, k, v, heads)`` takes comfy's (B, L, H*D) layout. Every call is
    recorded per rank as (entry, q, k, v) in that layout. An ``xfuser`` entry
    (an unpadded call, or a ring call) holds only this rank's queries; they are
    gathered so the kernel sees stock's full rows, and this rank's share comes
    back.
    """

    def __init__(self, world: int, kernel: Callable) -> None:
        super().__init__(world, strict=False)
        self.kernel = kernel
        self.seen: dict[int, list] = {rank: [] for rank in range(world)}
        self.lock = threading.Lock()

    def attend(self, entry, q, k, v):
        rank = _rank()
        local = q.shape[1]
        if entry == "xfuser":
            q = torch.cat(self.fabric.exchange(q), dim=1)
        batch, rows, heads, dim = q.shape
        flat = [t.reshape(t.shape[0], -1, heads * dim) for t in (q, k, v)]
        with self.lock:
            self.seen[rank].append((entry, *(t.clone() for t in flat)))
        out = self.kernel(*flat, heads).reshape(batch, rows, heads, dim)
        return out[:, rank * local:(rank + 1) * local] if entry == "xfuser" else out


def install(monkeypatch, kernel: Callable, *, ring_world: int = 1,
            world: int = WORLD) -> StockKernelHarness:
    """The stubbed fabric and kernel. Call before ``base.make_usp_attention``,
    which imports xfuser, and before any adapter binds a forward."""
    harness = StockKernelHarness(world, kernel)
    _install_stubs(monkeypatch, harness, ring_world=ring_world)
    return harness


def run_ranks(harness: StockKernelHarness, models: Sequence,
              call: Callable) -> list:
    """Run ``call(model)`` on each rank's model in its own thread.

    The models must already carry the adapter's bound forward. A rank's
    exception is re-raised once every thread has stopped.
    """
    outputs: list = [None] * len(models)
    errors: list = [None] * len(models)

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                outputs[rank] = call(models[rank])
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(len(models))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for error in errors:
        if error is not None:
            raise error
    return outputs


def replicas(model: torch.nn.Module, world: int = WORLD) -> list:
    return [copy.deepcopy(model) for _ in range(world)]


NVFP4_LAYOUT = "TensorCoreNVFP4Layout"


class MixedPrecisionLinear(torch.nn.Module):
    """Stands for comfy's mixed-precision Linear, which every NVFP4 file loads
    through: a plain Module, not a ``torch.nn.Linear``, so the full-row text
    selectors (chroma_text, hunyuan, lens and qwen_image ``_text_modules``,
    boogu_ulysses.instruct_projections) skip it. It computes what the Linear it
    replaces did, with the same weight and bias."""

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)


_MIXED: dict[type, type] = {}


def _mixed_precision_class(cls: type) -> type:
    """``MixedPrecisionLinear`` carrying comfy's own class attributes of ``cls``
    (``comfy_cast_weights`` and the like), which some forwards read."""
    if cls not in _MIXED:
        carried = {}
        for base in reversed(cls.__mro__):
            if base.__module__.startswith("torch") or base is object:
                continue
            carried.update({key: value for key, value in vars(base).items()
                            if not key.startswith("__") and key not in ("forward", "_conv_forward")})
        _MIXED[cls] = type(f"MixedPrecision{cls.__name__}", (MixedPrecisionLinear,), carried)
    return _MIXED[cls]


class ScaleRecorder:
    """Every Linear of the models quantizes as NVFP4 with no static scale.

    ``install`` marks each ``torch.nn.Linear`` with the NVFP4 layout and wraps
    it with the shared activation scale (adapters/quant_activation_scale.py),
    whose sequence-parallel reducer is the harness fabric's max (``reduce``),
    or the identity for a one-GPU run. The wrapper still calls the plain
    Linear, which ignores the scale, so the forward's output is unchanged.
    Each call records, per rank, the Linear's path and the amax the group
    reduced to: the scale every rank quantizes against.
    """

    def __init__(self, monkeypatch) -> None:
        from dgx_monarch.adapters import quant_activation_scale as qas
        from dgx_monarch.adapters import scale_rows

        self.calls: dict[int, list[tuple[str, float]]] = {}
        self._paths: dict[int, list[str]] = {}
        self._lock = threading.Lock()
        original_spans = scale_rows.spans
        original_scale = qas.shared_activation_scale

        # Keyed by thread, so a family's own rank harness works too; take()
        # numbers the threads in the order they first quantized.
        def spans(path, value):
            with self._lock:
                self._paths.setdefault(threading.get_ident(), []).append(path)
            return original_spans(path, value)

        def shared(input, plan, divisor, trace=None, spans=None):
            record: list = []
            scale = original_scale(input, plan, divisor, record, spans=spans)
            with self._lock:
                path = self._paths[threading.get_ident()][-1]
                self.calls.setdefault(threading.get_ident(), []).append(
                    (path, float(record[1])))
            return scale

        monkeypatch.setattr(scale_rows, "spans", spans)
        monkeypatch.setattr(qas, "shared_activation_scale", shared)
        monkeypatch.setenv("DGXM_SHARED_ACT_SCALE", "1")
        monkeypatch.delenv("DGXM_LOG_ACT_SCALE", raising=False)

    def take(self) -> dict[int, list[tuple[str, float]]]:
        """The calls recorded so far, one entry per thread numbered from 0, and
        a clean slate for the next run."""
        with self._lock:
            calls, self.calls, self._paths = self.calls, {}, {}
        return dict(enumerate(calls.values()))

    @staticmethod
    def group_max(world: int = WORLD):
        """A max all-reduce over ``world`` rank threads, for a family whose own
        harness has no fabric to lend: each thread's n-th call meets the other
        threads' n-th call."""
        barrier = threading.Barrier(world, timeout=60)
        lock = threading.Lock()
        rounds: dict[int, list] = {}
        local = threading.local()

        def reduce(value):
            index = getattr(local, "calls", 0)
            local.calls = index + 1
            with lock:
                rounds.setdefault(index, []).append(value.detach().clone())
            barrier.wait()
            return torch.stack(rounds[index]).amax(dim=0)

        return reduce

    def on_inject(self, monkeypatch, adapter_cls, world: int = WORLD) -> None:
        """Wrap every model ``adapter_cls.inject_usp`` binds, with one group max
        shared by the ranks, so a family's own sharded harness runs the scale."""
        reduce = self.group_max(world)
        original = adapter_cls.inject_usp

        def inject_usp(adapter, diffusion_model, ctx):
            original(adapter, diffusion_model, ctx)
            ScaleRecorder.install(diffusion_model, reduce)

        monkeypatch.setattr(adapter_cls, "inject_usp", inject_usp)

    @staticmethod
    def install(model, reduce=None) -> int:
        """Wrap every Linear of ``model``; returns how many were wrapped.

        Each Linear becomes a ``MixedPrecisionLinear``, as an NVFP4 file loads
        it, so the full-row text selectors pass it by and it takes the real
        route: its own padded rows under the family's role table.
        """
        from dgx_monarch.adapters import quant_activation_scale as qas

        for module in list(model.modules()):
            if isinstance(module, torch.nn.Linear):
                original = type(module)
                module.__class__ = _mixed_precision_class(original)
                if hasattr(original, "forward_comfy_cast_weights"):
                    # A forward an adapter or a harness bound earlier may still call
                    # comfy's own Linear.forward; its cast path runs on this class.
                    module.comfy_cast_weights = True
                module.layout_type = NVFP4_LAYOUT
        coverage = qas.install(model, {"ulysses": 2}, family="test",
                               reducers={"sp": reduce or (lambda value: value)})
        return coverage.wrapped


def fabric_max(harness):
    """The sequence-parallel max all-reduce, over the harness fabric."""
    def reduce(value):
        return torch.stack(harness.fabric.exchange(value.clone())).amax(dim=0)
    return reduce


def assert_scales_match(stock: list[tuple[str, float]], ranks: dict[int, list], world: int,
                        stock_only: frozenset[str] = frozenset()) -> None:
    """Every rank reduced every Linear call to the amax one GPU took over the
    whole tensor, call for call per Linear: no pad row (NaN here) and no lost
    real row. ``stock_only`` names Linears stock calls and the sharded forward
    provably does not need (WanDancer's single-key audio attention)."""
    assert stock, "the one-GPU run quantized nothing"

    def by_path(calls):
        grouped: dict[str, list[float]] = {}
        for path, value in calls:
            grouped.setdefault(path, []).append(value)
        return grouped

    want = by_path(stock)
    for rank in range(world):
        got = by_path(ranks.get(rank, []))
        assert set(got) == set(want) - stock_only, (rank, sorted(set(got) ^ set(want)))
        for path, values in got.items():
            assert len(values) == len(want[path]), (rank, path)
            for value, expected in zip(values, want[path], strict=True):
                assert value == value, f"rank {rank}: {path} reduced a pad row into its scale"
                assert abs(value - expected) <= 1e-5 * max(1.0, abs(expected)), (
                    rank, path, value, expected)


def grant_ring_pad_waiver(monkeypatch) -> None:
    """Answer the ring_pad guard as an accepted per-render waiver would."""
    from dgx_monarch import accuracy_waiver

    original = accuracy_waiver.waived
    monkeypatch.setattr(accuracy_waiver, "waived",
                        lambda guard: guard == base.RING_PAD_GUARD or original(guard))


def assert_ring_pad_refusal(error: BaseException) -> None:
    """The typed class K refusal of the ring pad guard, with its waiver card."""
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    assert isinstance(error, base.UnsupportedModelError)
    tag = parse_refusal_tag(str(error))
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == base.RING_PAD_GUARD and tag.waivable
