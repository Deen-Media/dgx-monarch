"""Planted rank scripts for tests/test_rank_exit_scan.py. Never imported or collected:
the scan reads this file's source only. Each name says what the scan must do with it."""
import functools
import subprocess
import sys
import textwrap

import torch.multiprocessing as mp

from rank_exit_helpers import ENDING, leave

P1_NO_ENDING_SHOULD_FAIL = """
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
dist.destroy_process_group()
"""

P2_EXITS_BEFORE_THE_ENDING = """
import sys
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
dist.destroy_process_group()
sys.exit(0)
""" + ENDING


def p3_inline_script():
    subprocess.run([sys.executable, "-c", """
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
"""], check=True)


def _p4_returns_early(rank):
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=2)
    dist.destroy_process_group()
    if rank >= 0:
        return
    leave()


def _p5_target_without_leave_should_fail(rank):
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", rank=rank, world_size=2)
    dist.destroy_process_group()


P6_BACKEND_IN_A_NAME = """
import torch.distributed as dist
BACKEND = "gloo"
dist.init_process_group(BACKEND, rank=0, world_size=1)
"""


def _p8_partial(rank, extra):
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=2)
    dist.destroy_process_group()


P9_DEDENTED_WITH_ENDING_PASSES = textwrap.dedent("""
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=0, world_size=1)
""") + ENDING

P10_ENDING_FIRST_SHOULD_FAIL = ENDING + """
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
"""

P11_XFUSER_ONLY = """
from xfuser.core.distributed import init_distributed_environment
init_distributed_environment(rank=0, world_size=1, local_rank=0, backend="gloo")
"""


P13_RAISES_SYSTEMEXIT_BEFORE_THE_ENDING = """
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
dist.destroy_process_group()
raise SystemExit(0)
""" + ENDING

# Not seen, so it passes: the exit sits in a function the script defines and calls.
P14_EXIT_IN_A_HELPER_IT_CALLS = """
import sys
import torch.distributed as dist
def done():
    sys.exit(0)
dist.init_process_group("gloo", rank=0, world_size=1)
dist.destroy_process_group()
done()
""" + ENDING


P15_EXIT_IN_A_LATER_FRAGMENT = """
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
""" + "\nimport sys\nsys.exit(0)\n" + ENDING

P16_EXIT_IN_AN_EARLIER_FRAGMENT = "import sys\nsys.exit(0)\n" + """
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
""" + ENDING

PREFIX = "import os\n"
P17_FRAGMENT_IN_A_NAME = PREFIX + """
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
""" + ENDING

P18_TWO_CLEAN_FRAGMENTS_PASS = "import os\n" + """
import torch.distributed as dist
dist.init_process_group("gloo", rank=0, world_size=1)
dist.destroy_process_group()
""" + ENDING


def _p12_exits_before_leave(rank):
    import os

    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=2)
    dist.destroy_process_group()
    os._exit(0)
    leave()


def launch():
    mp.spawn(_p4_returns_early, nprocs=2)
    mp.spawn(_p12_exits_before_leave, nprocs=2)
    import multiprocessing
    multiprocessing.get_context("spawn").Process(target=_p5_target_without_leave_should_fail, args=(0,))
    mp.spawn(functools.partial(_p8_partial, extra=1), nprocs=2)
