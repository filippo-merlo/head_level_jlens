"""Driver helpers. ``scripts/`` holds the only entry points that touch the model.

Each driver pins one GPU *before* importing torch, then loads the model and writes into
``results/``. Everything under ``src/`` that reads those files stays model-free.
"""

from __future__ import annotations

import os


def pin_gpu(gpu: str | int | None) -> None:
    """Restrict visibility to one GPU, numbered as ``nvidia-smi`` numbers them.

    Must run before torch is imported. A no-op when ``gpu`` is ``None``.

    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is not optional. CUDA defaults to
    ``FASTEST_FIRST``, which ranks devices by capability rather than bus order -- and
    this node is heterogeneous (5x A100 80GB + 1x H100 NVL). Under the default the H100
    is promoted to device 0 and every A100 shifts up, so ``--gpu 4`` selects an A100
    while ``nvidia-smi`` calls index 4 the H100. The failure is silent: the job lands on
    a different, often busy, GPU than the one you checked was free.
    """
    if gpu is None:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
