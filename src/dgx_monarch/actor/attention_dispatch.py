"""Setup-generation-bound selection of stock, sol, Wan ring and cuDNN ring USP attention."""
from __future__ import annotations

from typing import cast

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)


class _WanAttentionView:
    """Generation-bound view selecting the plain-Wan ring specialization."""

    def __init__(self, dispatch, authority_key: tuple) -> None:
        self._dispatch = dispatch
        self._authority_key = authority_key

    @property
    def sol_scope(self):
        """The dispatcher's family scope, so a bind through this view lands.

        The worker binds on whatever object the adapter hands back, and the
        Wan adapters hand back this view instead of the dispatcher.
        """
        return self._dispatch.sol_scope

    def __call__(self, *args, **kwargs):
        return self._dispatch._call_wan(self._authority_key, *args, **kwargs)


class _AttentionDispatch:
    """Mutable, setup-bound dispatcher for every USP attention route."""

    def __init__(self) -> None:
        self._impl = None
        self._kernel = ""
        self._sync = True
        self._topology: dict | None = None
        self._world: int | None = None
        self._setup_generation: int | None = None
        self._authority_key: tuple | None = None
        self._wan_impl = None
        self._wan_view: _WanAttentionView | None = None
        self._sol_scope = None
        self._effective_kernel = ""
        self._family = ""
        self._head_dim: int | None = None
        self._cudnn_ring_impl = None

    def _persistent_sol_scope(self):
        """Keep the family binding across kernel and setup reconfiguration.

        ``configure`` rebuilds implementations, but a base-model load binds the family
        only once. Store the scope on the dispatcher so rebuilt implementations remain
        bound, including when switching from another kernel to sol. ``invalidate``
        clears it when the setup's residents are unloaded.
        """
        if self._sol_scope is None:
            from ..adapters.sol_attention import SolScope

            self._sol_scope = SolScope()
        return self._sol_scope

    @property
    def kernel(self) -> str:
        """The kernel the operator selected, empty before setup ran.

        The driver stamps this render's capability context with this name; the
        implementation may run another (``effective_kernel``).
        """
        return self._kernel

    @property
    def effective_kernel(self) -> str:
        """The kernel the implementation runs.

        It differs from the selected one only where the selected kernel cannot
        carry the bound family's head dimension and another can
        (adapters/attention_capability.py).
        """
        return self._effective_kernel

    def bind_capability(self, family: str, head_dim: int | None) -> None:
        """Record the loaded family, and rebuild if that changes the kernel.

        Setup configures a kernel before any model is loaded, so the head
        dimension is unknown then and the first build takes the selected kernel
        as it stands. The family arrives at injection, which is why this is a
        second call rather than a configure argument, and why it shares the
        binding site with the sol scope.
        """
        if (family, head_dim) == (self._family, self._head_dim):
            return
        self._family, self._head_dim = family, head_dim
        if self._impl is None:
            return
        from ..adapters.attention_capability import effective_kernel

        if effective_kernel(self._kernel, family, head_dim) == self._effective_kernel:
            return
        # configure memoizes on the selected kernel, which has not changed; the
        # answer it computes from that kernel has. Clearing the memo is the one
        # way to reach the build without duplicating it here.
        selected, self._kernel = self._kernel, ""
        self.configure(selected, self._sync, topology=self._topology,
                       world=self._world, setup_generation=self._setup_generation)

    @property
    def sol_scope(self):
        """Record the loaded family independently of the selected kernel.

        A model loaded under flash or sage can render under sol without reloading.
        Recording does not validate eligibility: ``make_sol_usp_attention`` checks
        geometry at construction and ``sol_waiver_required`` checks the family on
        first use. Both raise typed refusals.
        """
        return self._persistent_sol_scope()

    @staticmethod
    def _setup_authority(
        topology, world, setup_generation
    ) -> tuple[dict | None, int | None, tuple | None]:
        if topology is None and world is None and setup_generation is None:
            # Unit tests call configure(kernel, sync) with no setup authority.
            # Such an unbound dispatcher can use only the stock path.
            return None, None, None
        if topology is None or world is None or setup_generation is None:
            raise ValueError(
                "attention setup topology, world, and generation must be supplied together"
            )
        if not isinstance(topology, dict):
            raise TypeError("attention setup topology must be a dict")
        if (
            isinstance(setup_generation, bool)
            or not isinstance(setup_generation, int)
            or setup_generation < 1
        ):
            raise ValueError("attention setup generation must be a positive integer")
        if isinstance(world, bool) or not isinstance(world, int) or world < 1:
            raise ValueError("attention setup world must be a positive integer")
        frozen = tuple(sorted(topology.items()))
        return dict(topology), world, (setup_generation, world, frozen)

    def configure(
        self,
        kernel: str,
        sync_ulysses: bool,
        *,
        topology=None,
        world=None,
        setup_generation=None,
    ) -> None:
        topology_copy, world_value, authority_key = self._setup_authority(
            topology, world, setup_generation
        )
        if (
            kernel == self._kernel
            and sync_ulysses == self._sync
            and authority_key == self._authority_key
            and self._impl is not None
        ):
            return
        if self._impl is not None and authority_key != self._authority_key:
            # Consume the old setup-bound implementation before constructing
            # anything for a replacement. A constructor failure must leave no
            # callable which can fall back to retired process groups.
            self.invalidate()
        from ..adapters import make_usp_attention
        from ..adapters.attention_capability import effective_kernel, substitution_note
        from ..adapters.base import UnsupportedModelError
        from ..adapters.sol_attention import is_sol_kernel, make_sol_usp_attention

        # A kernel too narrow for the bound family's head dimension is swapped
        # for one that carries it, here rather than at the call, so that every
        # configure entry answers alike: setup's and the per-render one.
        effective = effective_kernel(kernel, self._family, self._head_dim)
        try:
            if is_sol_kernel(kernel):
                topo = cast(dict, topology_copy)
                implementation = make_sol_usp_attention(
                    kernel,
                    sync_ulysses,
                    ulysses=int(topo.get("ulysses", 1)),
                    ring=int(topo.get("ring", 1)),
                    scope=self._persistent_sol_scope(),
                )
            else:
                implementation = make_usp_attention(effective, sync_ulysses)
        except UnsupportedModelError:
            # A typed refusal is the answer, never a reason to keep the previous
            # kernel. The driver has already stamped this render's capability
            # context with the kernel it asked for, so falling back here would
            # render one kernel's math under another kernel's vouch.
            raise
        except Exception as exc:
            # Within one setup generation, a kernel that fails to build (an
            # optional Sage install, say) keeps the current one. A replacement
            # setup never keeps an implementation that closes over the previous
            # generation's process groups.
            if self._impl is None or authority_key != self._authority_key:
                raise
            safe_call(
                log.warning,
                "attention kernel %s unavailable (%s); keeping %s",
                kernel,
                failure_summary(exc),
                self._kernel,
            )
            return

        authority_changed = authority_key != self._authority_key
        wan_implementation = None
        if (
            not authority_changed
            and self._wan_view is not None
            and kernel == "TORCH_FLASH"
            and self._wan_ring2_treatment_topology(
                topology_copy, world_value, setup_generation
            )
        ):
            # An injected Wan model already holds this view. Rebuild the
            # treatment now, before sample() can enter its first collective.
            from ..adapters import make_wan_ring_usp_attention

            wan_implementation = make_wan_ring_usp_attention(
                kernel,
                sync_ulysses,
                cast(dict, topology_copy),
                cast(int, world_value),
                cast(int, setup_generation),
            )

        self._impl = implementation
        self._cudnn_ring_impl = None
        self._kernel = kernel
        self._effective_kernel = effective
        self._sync = sync_ulysses
        self._topology = topology_copy
        self._world = world_value
        self._setup_generation = setup_generation
        self._authority_key = authority_key
        self._wan_impl = wan_implementation
        if authority_changed:
            self._wan_view = None
        note = substitution_note(kernel, self._family, self._head_dim)
        log.info("USP attention kernel: %s (sync_ulysses=%s)%s",
                 effective, sync_ulysses, f"; {note}" if note else "")

    @staticmethod
    def _wan_ring2_treatment_topology(
        topology: dict | None, world: int | None, setup_generation: int | None
    ) -> bool:
        """Select only the exact topology vouched for the Wan treatment."""

        if topology is None or type(world) is not int or world != 2:
            return False
        if type(setup_generation) is not int or setup_generation < 1:
            return False
        expected = {"dp": 1, "cfg": 1, "ulysses": 1, "ring": 2}
        if any(type(topology.get(name, 1)) is not int for name in expected):
            return False
        return (
            all(topology.get(name, 1) == degree for name, degree in expected.items())
            and topology.get("fsdp", False) is False
        )

    def for_wan(self):
        """Return this dispatcher unless the topology is the vouched Wan ring-2 setup."""

        if self._impl is None:
            raise RuntimeError("Wan USP attention requested before setup configured a kernel")
        if (
            self._topology is None
            or self._setup_generation is None
            or self._world is None
            or self._authority_key is None
        ):
            raise RuntimeError("Wan USP attention requires setup-bound topology authority")
        ring = self._topology.get("ring", 1)
        if isinstance(ring, bool) or not isinstance(ring, int) or ring < 1:
            raise RuntimeError("Wan USP attention has an invalid configured ring degree")
        if not self._wan_ring2_treatment_topology(
            self._topology, self._world, self._setup_generation
        ):
            return self
        if self._wan_view is None:
            self._wan_view = _WanAttentionView(self, self._authority_key)
        # Construct eagerly during model injection so an unavailable treatment
        # refuses the load, before the denoise loop can enter collectives.
        if self._kernel == "TORCH_FLASH":
            self._wan_implementation(self._authority_key)
        return self._wan_view

    def _wan_implementation(self, authority_key: tuple):
        if authority_key != self._authority_key or self._impl is None:
            raise RuntimeError("Wan USP attention belongs to a stale setup generation")
        if self._kernel != "TORCH_FLASH":
            self._assert_cudnn_ring_prepared()
            return self._impl
        if self._wan_impl is None:
            if (
                self._topology is None
                or self._world is None
                or self._setup_generation is None
            ):
                raise RuntimeError("Wan ring attention setup authority is unavailable")
            from ..adapters import make_wan_ring_usp_attention

            implementation = make_wan_ring_usp_attention(
                self._kernel,
                self._sync,
                self._topology,
                self._world,
                self._setup_generation,
            )
            if authority_key != self._authority_key:
                raise RuntimeError("Wan ring attention setup changed during construction")
            self._wan_impl = implementation
        return self._wan_impl

    def _call_wan(self, authority_key: tuple, *args, **kwargs):
        return self._wan_implementation(authority_key)(*args, **kwargs)

    def attestation(self) -> dict | None:
        """Bounded immutable evidence for the active Wan specialization."""

        implementation = self._wan_impl
        read_attestation = getattr(implementation, "status_attestation", None)
        if read_attestation is None:
            return None
        return read_attestation()

    def _needs_cudnn_ring(self) -> bool:
        return (
            self._effective_kernel == "TORCH_CUDNN"
            and self._topology is not None
            and self._topology.get("ring", 1) > 1
        )

    def _assert_cudnn_ring_prepared(self) -> None:
        if self._needs_cudnn_ring() and self._impl is not self._cudnn_ring_impl:
            from ..adapters.base import UnsupportedModelError
            from ..refusal import RefusalClass, refusal

            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                "Ring TORCH_CUDNN has not passed its native normalization check "
                "before the fleet readiness exchange. Use the normal sampler "
                "load path or select a supported Ulysses configuration.",
            ))

    def prepare_native_attention(self) -> None:
        """Prepare cuDNN Ring inside the sample's protected load phase.

        The family may substitute away from cuDNN during injection, and a
        warm request can change kernels without loading again. Defer the
        native probe until both paths have settled, before rank readiness.
        Valid Ring topologies exclude FSDP (Topology.validate), so a failing
        rank reaches the not-ready exchange before any Ring P2P.
        """
        if not self._needs_cudnn_ring():
            return
        if self._impl is None or self._authority_key is None:
            raise RuntimeError("cuDNN Ring preparation requires a live setup generation")
        if self._topology is None or self._topology.get("fsdp", False):
            raise RuntimeError("cuDNN Ring preparation requires a supported non-FSDP topology")
        if self._impl is self._cudnn_ring_impl:
            return
        from ..adapters.cudnn_ring_attention import make_cudnn_ring_usp_attention

        authority, previous = self._authority_key, self._impl
        implementation = make_cudnn_ring_usp_attention(self._sync)
        if (self._authority_key != authority or self._impl is not previous
                or not self._needs_cudnn_ring()):
            raise RuntimeError("cuDNN Ring setup changed during native preparation")
        self._impl = implementation
        self._cudnn_ring_impl = implementation

    def invalidate(self) -> None:
        """Drop an implementation bound to process groups being destroyed."""

        self._impl = None
        self._kernel = ""
        self._topology = None
        self._world = None
        self._setup_generation = None
        self._authority_key = None
        self._wan_impl = None
        self._wan_view = None
        self._effective_kernel = ""
        self._family = ""
        self._head_dim = None
        self._cudnn_ring_impl = None
        # A retired setup generation has already unloaded the residents whose
        # load bound this family; keeping the scope would report a bind with no
        # model behind it.
        self._sol_scope = None

    def __call__(self, *args, **kwargs):
        if self._impl is None:
            raise RuntimeError("USP attention used before setup() configured a kernel")
        self._assert_cudnn_ring_prepared()
        return self._impl(*args, **kwargs)
