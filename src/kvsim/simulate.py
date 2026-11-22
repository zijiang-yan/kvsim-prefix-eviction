"""Replay one trace against one policy under one byte budget."""

from __future__ import annotations

import time

from .cache import PrefixCache
from .costs import LLAMA3_8B_L0, chunk_cost
from .policies import Policy
from .workload import Request


def simulate(
    trace: list[Request],
    policy: Policy,
    capacity_bytes: int,
    kv_bytes_per_token: int,
    warmup_frac: float = 0.1,
    l0: float = LLAMA3_8B_L0,
) -> dict:
    cache = PrefixCache(capacity_bytes, kv_bytes_per_token)
    policy.bind(l0)
    warmup = int(len(trace) * warmup_frac)
    total_tokens = hit_tokens = evictions = decisions = 0
    total_cost = hit_cost = 0.0
    decision_ns = 0
    peak_nodes = 0

    for t, req in enumerate(trace):
        # 1. Match the longest cached prefix.
        node = cache.root
        path = []
        n_hit = 0
        req_hit = 0
        req_hit_cost = 0.0
        for content, tokens in req:
            node = cache.child(node, content, tokens)
            path.append(node)
            if n_hit == len(path) - 1 and node.cached:
                n_hit += 1
                req_hit += tokens
                req_hit_cost += chunk_cost(tokens, node.offset, l0)
        req_tokens = sum(tok for _, tok in req)
        policy.begin_request(t, req_hit, req_tokens - req_hit)

        # 2. Optional look-ahead over queued requests (PCR-style).
        for nxt in trace[t + 1 : t + 1 + policy.lookahead]:
            n = cache.root
            for content, _ in nxt:
                n = n.children.get(content)
                if n is None or not n.cached:
                    break
                policy.on_lookahead(n, t)

        # 3. Reuse hits. After the first miss a cached chunk (an orphan) still has
        #    to be recomputed, but its KV is refreshed, not re-stored.
        for node in path:
            if node.cached:
                policy.on_access(node, t)

        # 4. Store recomputed chunks, evicting as needed.
        pinned = set(path)
        stalled = False
        for node in path[n_hit:]:
            if node.cached:
                continue
            need = cache.nbytes(node)
            while cache.free < need:
                t0 = time.perf_counter_ns()
                victim = policy.victim(cache, t, pinned)
                decision_ns += time.perf_counter_ns() - t0
                decisions += 1
                if victim is None:
                    stalled = True
                    break
                cache.evict(victim)
                policy.on_evict(victim, t)
                if t >= warmup:
                    evictions += 1
            if stalled:
                break  # later chunks would not be reachable as a prefix
            cache.mark_cached(node)
            policy.on_insert(node, t)
        if stalled:
            cache.prune(path[-1])

        if t >= warmup:
            total_tokens += req_tokens
            hit_tokens += req_hit
            total_cost += chunk_cost(req_tokens, 0, l0)
            hit_cost += req_hit_cost
        peak_nodes = max(peak_nodes, cache.n_nodes)

    n_cached = sum(1 for _ in cache.iter_cached())
    return {
        "requests": len(trace) - warmup,
        "total_tokens": total_tokens,
        "hit_tokens": hit_tokens,
        "recomputed_tokens": total_tokens - hit_tokens,
        "hit_rate": hit_tokens / total_tokens if total_tokens else 0.0,
        "cost_saved_frac": hit_cost / total_cost if total_cost else 0.0,
        "evictions": evictions,
        "decision_us_mean": decision_ns / decisions / 1e3 if decisions else 0.0,
        "cached_nodes": n_cached,
        "peak_tree_nodes": peak_nodes,
        "meta_bytes_est": n_cached * policy.meta_fields * 8,
        "orphan_bytes": _orphan_bytes(cache),
    }


def _orphan_bytes(cache: PrefixCache) -> int:
    """Bytes held by cached nodes that are unreachable (an ancestor is uncached)."""
    total = 0
    stack = [(c, True) for c in cache.root.children.values()]
    while stack:
        node, reachable = stack.pop()
        if node.cached and not reachable:
            total += cache.nbytes(node)
        r = reachable and node.cached
        stack.extend((c, r) for c in node.children.values())
    return total
