"""Analytical prefill cost model.

Prefilling n tokens that follow p already-present tokens costs
    c(n, p) = n * (1 + (p + n/2) / L0)
in "linear-token" units: the 1 is the per-token linear work (projections, MLP,
about 2 * params FLOPs), and (p + n/2) / L0 is attention over the prefix, which
grows with position. L0 is the context length at which attention FLOPs equal
the linear FLOPs per token:
    L0 = 2 * N_params / (4 * n_layers * d_model)
Llama-3-8B: 2 * 7e9 / (4 * 32 * 4096) ~= 26700 tokens.
L0 = inf gives a position-independent cost equal to the token count.
Replace L0 with a value fitted on GPU measurements once calibrated.
"""

from __future__ import annotations

import math

LLAMA3_8B_L0 = 26700.0


def chunk_cost(tokens: int, offset: int, l0: float) -> float:
    if math.isinf(l0):
        return float(tokens)
    return tokens * (1.0 + (offset + tokens / 2.0) / l0)


def prefill_cost(cached: int, new: int, l0: float) -> float:
    """Cost of a request with `cached` reused tokens and `new` recomputed ones
    (the analytical stand-in for RAGCache's profiled T(alpha, beta))."""
    return chunk_cost(new, cached, l0)
