"""Test-wide setup.

Torch defaults to one thread per core (64 here). Every tensor in this suite is tiny --
the fixture model is 3 layers of width 32 -- so multi-threading buys nothing and costs a
fork/join per op. It also makes the suite hostage to machine load: with a fit running on
the same box the suite went from ~15 s to over two minutes, purely in thread
synchronisation. One thread is both faster and reproducible.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _single_threaded_torch():
    import torch

    torch.set_num_threads(1)
