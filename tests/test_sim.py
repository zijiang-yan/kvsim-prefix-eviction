import pytest

from kvsim.cache import PrefixCache
from kvsim.policies import make_policy
from kvsim.simulate import _orphan_bytes, simulate
from kvsim.workload import RagWorkload, unique_tokens

KVB = 2  # bytes per token, small for readable numbers


def test_repeat_request_hits_fully():
    trace = [[(1, 10), (2, 10)], [(1, 10), (2, 10)]]
    m = simulate(trace, make_policy("leaf_lru"), 10_000, KVB, warmup_frac=0.5)
    assert m["hit_rate"] == 1.0


def test_hit_stops_at_first_missing_prefix():
    trace = [[(1, 10), (2, 10)], [(9, 10), (2, 10)]]
    m = simulate(trace, make_policy("leaf_lru"), 10_000, KVB, warmup_frac=0.5)
    assert m["hit_tokens"] == 0  # chunk 2 under a different prefix is a different node


def test_unique_tokens_counts_prefix_tree():
    trace = [[(1, 10), (2, 10)], [(1, 10), (3, 5)]]
    assert unique_tokens(trace) == 25


@pytest.mark.parametrize("name", ["lru", "lfu", "leaf_lfu", "leaf_lru", "pgdsf", "ours", "ours_nodep"])
def test_budget_respected_and_leaf_policies_have_no_orphans(name, monkeypatch):
    trace = RagWorkload(n_requests=400, n_docs=100, shift_every=150, seed=1).generate()
    cap = int(0.1 * unique_tokens(trace) * KVB)
    policy = make_policy(name)

    seen = []
    orig = PrefixCache.mark_cached

    def checked(self, node):
        orig(self, node)
        assert self.used <= self.capacity
        seen.append(self)

    monkeypatch.setattr(PrefixCache, "mark_cached", checked)
    m = simulate(trace, policy, cap, KVB)
    assert 0.0 <= m["hit_rate"] <= 1.0
    if policy.leaf_only:
        assert _orphan_bytes(seen[-1]) == 0


@pytest.mark.parametrize("cost", ["position", "tokens"])
def test_ours_heap_evicts_exact_minimum(cost, monkeypatch):
    # The time-invariant heap key must pick the leaf with the lowest V(t).
    from kvsim.policies import ValueAware

    trace = RagWorkload(n_requests=300, n_docs=80, shift_every=100, seed=4).generate()
    cap = int(0.1 * unique_tokens(trace) * KVB)
    pol = ValueAware(decay=0.01, cost=cost)
    orig = ValueAware.victim
    checks = []

    def checked(self, cache, t, pinned):
        v = orig(self, cache, t, pinned)
        if v is not None:
            leaves = [n for n in cache.iter_cached() if n.is_cached_leaf and n not in pinned]
            best = min(self.value(n, cache, t) for n in leaves)
            assert self.value(v, cache, t) == pytest.approx(best, rel=1e-9)
            checks.append(1)
        return v

    monkeypatch.setattr(ValueAware, "victim", checked)
    simulate(trace, pol, cap, KVB, l0=1000.0)
    assert checks


def test_chunk_costs_sum_to_request_cost():
    from kvsim.costs import chunk_cost

    l0 = 1000.0
    chunks = [256, 256, 64]
    offsets = [0, 256, 512]
    total = sum(chunk_cost(n, p, l0) for n, p in zip(chunks, offsets))
    assert total == pytest.approx(chunk_cost(sum(chunks), 0, l0))


def test_position_cost_keeps_deeper_chunk():
    # Two equally hot leaves; with position cost the deeper one is worth more.
    from kvsim.cache import PrefixCache
    from kvsim.policies import ValueAware

    cache = PrefixCache(10**9, KVB)
    pol = ValueAware(decay=0.0, cost="position")
    pol.bind(100.0)
    shallow = cache.child(cache.root, 1, 10)
    a = cache.child(cache.root, 2, 10)
    deep = cache.child(a, 3, 10)
    for i, n in enumerate((shallow, a, deep)):
        cache.mark_cached(n)
        pol.on_insert(n, i)
    assert pol.victim(cache, 3, set()) is shallow


def test_pgdsf_and_lookahead_run():
    trace = RagWorkload(n_requests=400, n_docs=100, seed=2).generate()
    cap = int(0.1 * unique_tokens(trace) * KVB)
    for pol in (make_policy("pgdsf"), make_policy("leaf_lru", lookahead=4)):
        m = simulate(trace, pol, cap, KVB)
        assert 0.0 < m["hit_rate"] <= 1.0 and m["orphan_bytes"] == 0


def test_trace_workload_builds_contiguous_requests(tmp_path):
    import gzip
    import json

    from kvsim.workload import TraceWorkload

    data = {
        "paragraph_tokens": [300, 100, 50],
        "paragraph_article": [0, 0, 1],
        "questions": [
            {"a": 0, "g": 0, "t": 12, "r": [0, 1, 2]},
            {"a": 0, "g": 1, "t": 9, "r": [1, 0, 2]},
            {"a": 1, "g": 2, "t": 7, "r": [2, 0, 1]},
        ],
    }
    path = tmp_path / "t.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    for order in ("article", "shuffled"):
        trace = TraceWorkload(trace=str(path), docs_per_request=2, order=order,
                              chunk_tokens=256).generate()
        assert len(trace) == 3
        for req in trace:
            assert req[:2] == [(0, 256), (1, 256)]  # shared system prompt
        q0 = next(r for r in trace if r[-1][1] == 12)
        # paragraph 0 (300 tokens) -> 256 + 44, then paragraph 1 (100), then the question
        assert [tok for _, tok in q0[2:]] == [256, 44, 100, 12]


@pytest.mark.parametrize("name,params,stat", [
    ("leaf_lru", {}, "last_access"),
    ("leaf_lfu", {}, "freq"),
    ("ours", {"decay": 0.01, "cost": "tokens"}, "decayed"),
])
def test_prop2_scores_are_monotone_along_tree(name, params, stat, monkeypatch):
    # Proposition 2: in a prefix-closed cache, a parent's access statistics
    # dominate its child's (last access, count, and decayed count S_lambda).
    import math

    caches = []
    orig_init = PrefixCache.__init__

    def record(self, *a, **k):
        orig_init(self, *a, **k)
        caches.append(self)

    monkeypatch.setattr(PrefixCache, "__init__", record)
    trace = RagWorkload(n_requests=600, n_docs=150, shift_every=200, seed=5).generate()
    cap = int(0.1 * unique_tokens(trace) * KVB)
    simulate(trace, make_policy(name, **params), cap, KVB)
    cache, end = caches[-1], len(trace)

    def value(n):
        if stat == "decayed":
            return n.score * math.exp(-params["decay"] * (end - n.score_t))
        return getattr(n, stat)

    checked = 0
    for node in cache.iter_cached():
        parent = node.parent
        if parent is not cache.root:
            assert parent.cached
            assert value(parent) >= value(node) - 1e-12
            checked += 1
    assert checked > 0
