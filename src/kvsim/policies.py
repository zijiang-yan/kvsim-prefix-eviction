"""Eviction policies.

All policies see the same events, in this order per request:
    begin_request(t, cached_tokens, new_tokens)
    on_lookahead(node, t)   for cached nodes of the next `lookahead` requests
    on_access(node, t)      for cached nodes on the request path
    on_insert(node, t)      for each newly cached node (after any evictions)
    on_evict(node, t)       after the simulator removes a victim
and victim(cache, t, pinned) picks the next node to evict, never one in `pinned`.

Each policy comes in two scopes: block (any cached node; dependency-unaware,
can leave orphans) and leaf (only cached nodes with no cached children, so
every cached node keeps its full prefix). Heaps use lazy invalidation: stale
entries are skipped when popped.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict

from .cache import Node, PrefixCache
from .costs import chunk_cost, prefill_cost


class Policy:
    name = "base"
    leaf_only = False
    lookahead = 0
    # 8-byte metadata fields kept per cached node (for the overhead estimate).
    meta_fields = 0

    def bind(self, l0: float) -> None:
        self.l0 = l0

    def begin_request(self, t: float, cached_tokens: int, new_tokens: int) -> None: ...
    def on_lookahead(self, node: Node, t: float) -> None: ...
    def on_access(self, node: Node, t: float) -> None: ...
    def on_insert(self, node: Node, t: float) -> None: ...
    def on_evict(self, node: Node, t: float) -> None: ...
    def victim(self, cache: PrefixCache, t: float, pinned: set[Node]) -> Node | None:
        raise NotImplementedError


class LRU(Policy):
    """Block-level LRU (vLLM-style hash-based prefix cache)."""

    name = "lru"
    meta_fields = 3  # timestamp + two list pointers

    def __init__(self):
        self.order: OrderedDict[Node, None] = OrderedDict()

    def on_access(self, node, t):
        node.last_access = t
        self.order.move_to_end(node)

    def on_insert(self, node, t):
        node.last_access = t
        self.order[node] = None

    def on_evict(self, node, t):
        self.order.pop(node, None)

    def victim(self, cache, t, pinned):
        for node in self.order:
            if node not in pinned:
                return node
        return None


class KeyHeap(Policy):
    """Evict the eligible node with the smallest `key(node)`.
    Eligible = cached (block scope) or cached leaf (leaf scope). Keys must not
    change with time unless the node is touched, so a heap stays exact."""

    def __init__(self, leaf_only: bool):
        self.leaf_only = leaf_only
        self.heap: list = []

    def key(self, node: Node):
        raise NotImplementedError

    def _eligible(self, node: Node) -> bool:
        return node.is_cached_leaf if self.leaf_only else node.cached

    def _push(self, node):
        if self._eligible(node):
            heapq.heappush(self.heap, (self.key(node), node.nid, node))

    def on_evict(self, node, t):
        parent = node.parent
        if self.leaf_only and parent is not None and parent.parent is not None:
            self._push(parent)  # may have just become a leaf

    def victim(self, cache, t, pinned):
        skipped, found = [], None
        while self.heap:
            entry = heapq.heappop(self.heap)
            k, _, node = entry
            if not self._eligible(node) or k != self.key(node):
                continue
            if node in pinned:
                skipped.append(entry)
                continue
            found = node
            break
        for e in skipped:
            heapq.heappush(self.heap, e)
        return found


class LeafLRU(KeyHeap):
    """LRU over cached leaves (SGLang RadixAttention / PCR eviction). With
    lookahead = w > 0, cached chunks of the next w queued requests are touched
    before each request (PCR's look-ahead priority bump)."""

    name = "leaf_lru"
    meta_fields = 1

    def __init__(self, lookahead: int = 0):
        super().__init__(leaf_only=True)
        self.lookahead = lookahead

    def key(self, node):
        return node.last_access

    def _touch(self, node, t):
        node.last_access = t
        self._push(node)

    on_access = on_insert = on_lookahead = _touch


class LFU(KeyHeap):
    """LFU with LRU tie-break; block or leaf scope."""

    meta_fields = 2

    def __init__(self, leaf_only: bool = False):
        super().__init__(leaf_only)
        self.name = "leaf_lfu" if leaf_only else "lfu"

    def key(self, node):
        return (node.freq, node.last_access)

    def on_access(self, node, t):
        node.freq += 1
        node.last_access = t
        self._push(node)

    def on_insert(self, node, t):
        node.freq = 1
        node.last_access = t
        self._push(node)


class PGDSF(KeyHeap):
    """RAGCache's Prefix-aware Greedy-Dual-Size-Frequency (Jin et al., Alg. 1):
        priority = Clock + AvgCost * Frequency
    Frequency counts every retrieval; AvgCost averages T(alpha, beta) / beta over
    the requests that had to compute the node, where alpha / beta are the
    request's cached / non-cached tokens. Clock <- max(Clock, victim priority).
    RAGCache works per document; here every chunk of a document gets the same
    updates. T is the analytical prefill model in costs.py (RAGCache profiles it)."""

    name = "pgdsf"
    meta_fields = 4  # frequency, total cost, compute count, priority

    def __init__(self):
        super().__init__(leaf_only=True)
        self.clock = 0.0
        self.req_cost_per_token = 1.0

    def key(self, node):
        return node.prio

    def begin_request(self, t, cached_tokens, new_tokens):
        if new_tokens:
            self.req_cost_per_token = prefill_cost(cached_tokens, new_tokens, self.l0) / new_tokens

    def _update(self, node, computed):
        node.freq += 1
        if computed:
            node.total_cost += self.req_cost_per_token
            node.n_computed += 1
        avg = node.total_cost / node.n_computed if node.n_computed else 0.0
        node.prio = self.clock + avg * node.freq
        self._push(node)

    def on_access(self, node, t):
        self._update(node, computed=False)

    def on_insert(self, node, t):
        node.freq = node.n_computed = 0
        node.total_cost = 0.0
        self._update(node, computed=True)

    def victim(self, cache, t, pinned):
        v = super().victim(cache, t, pinned)
        if v is not None:
            self.clock = max(self.clock, v.prio)
        return v


class ValueAware(KeyHeap):
    """Ours: evict the eligible node with the lowest time-decayed recompute-cost
    loss per byte freed,
        V(a) = c(a) / bytes(a) * S_a(t),   S_a(t) = sum_{r reached a} exp(-lambda (t - t_r)),
    with c(a) = chunk_cost(tokens, offset) (cost="position") or tokens (cost="tokens").
    S_a is one counter per node: S <- S * exp(-lambda (t - t_s)) + 1 on access.
    All nodes decay at the same rate, so ordering by V(t) equals ordering by the
    time-invariant key log S + lambda * t_s + log(c / bytes): a plain heap finds
    the exact minimum in O(log n), with no candidate sampling.
    Leaf scope is the full policy; block scope is the no-dependency ablation.
    """

    meta_fields = 2  # decayed score, score timestamp

    def __init__(self, decay: float = 1e-3, cost: str = "position", leaf_only: bool = True):
        super().__init__(leaf_only)
        if cost not in ("position", "tokens"):
            raise ValueError(f"cost must be 'position' or 'tokens', got {cost!r}")
        self.decay = decay
        self.cost = cost
        self.name = "ours" if leaf_only else "ours_nodep"

    def _decayed(self, node, t):
        if self.decay == 0.0:
            return node.score
        return node.score * math.exp(-self.decay * (t - node.score_t))

    def _cost_per_token(self, node):
        l0 = self.l0 if self.cost == "position" else math.inf
        return chunk_cost(node.tokens, node.offset, l0) / node.tokens

    def value(self, node, cache, t):
        """V(a) at time t (for analysis and tests)."""
        return self._decayed(node, t) * self._cost_per_token(node) * node.tokens / cache.nbytes(node)

    def key(self, node):
        # bytes = tokens * const, so log(c / bytes) = log(cost per token) + const.
        # Ties (e.g., lambda = 0) break by recency, as in LFU.
        v = math.log(node.score) + self.decay * node.score_t + math.log(self._cost_per_token(node))
        return (v, node.score_t)

    def on_access(self, node, t):
        node.score = self._decayed(node, t) + 1.0
        node.score_t = t
        self._push(node)

    def on_insert(self, node, t):
        node.score = 1.0
        node.score_t = t
        self._push(node)


POLICIES = {
    "lru": (LRU, (), {}),
    "lfu": (LFU, (), {"leaf_only": False}),
    "leaf_lfu": (LFU, (), {"leaf_only": True}),
    "leaf_lru": (LeafLRU, ("lookahead",), {}),
    "pgdsf": (PGDSF, (), {}),
    "ours": (ValueAware, ("decay", "cost"), {"leaf_only": True}),
    "ours_nodep": (ValueAware, ("decay", "cost"), {"leaf_only": False}),
}


def make_policy(name: str, **params) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    cls, allowed, fixed = POLICIES[name]
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(f"policy {name!r} does not take {sorted(unknown)}")
    return cls(**params, **fixed)
