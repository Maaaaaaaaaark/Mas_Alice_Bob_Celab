"""Deterministic seeding (spec sec. 10.2).

Every run re-seeds Python, NumPy, and PyTorch (CPU + CUDA) from ``run_seed``.
Each Agent then derives a stable PyTorch seed from the run seed, speaker, and
per-speaker call index, providing separate reproducible sampling streams.

Reproducibility boundary (documented in the README): identical results are
expected for the same GPU/driver/library versions. Cross-version or
cross-device bitwise reproduction is not guaranteed because some CUDA
kernels are not bitwise deterministic across versions.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) global generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_torch(seed: int) -> None:
    """Re-seed only the PyTorch generators (used before each generation call)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_generation_seed(run_seed: int, speaker: str, call_index: int) -> int:
    """Derive a stable, independent seed for one agent generation call."""
    material = f"{run_seed}:{speaker}:{call_index}".encode("utf-8")
    # Stay inside PyTorch's accepted signed 63-bit seed range.
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)
