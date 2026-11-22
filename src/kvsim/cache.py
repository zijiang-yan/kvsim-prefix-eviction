"""Prefix tree of KV chunks with byte-accurate capacity accounting.

A node is one chunk keyed by (parent, content id), so identical content under
different prefixes is a different node (exact contiguous-prefix reuse).
A node may stay in the tree while uncached; a cached node whose ancestor is
uncached still occupies bytes but cannot produce hits (an orphan). Only
dependency-unaware policies (block LRU/LFU) can create orphans.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(eq=False)
class Node:
    nid: int
    content: int
    tokens: int
    parent: Node | None
    offset: int = 0  # tokens before this chunk in its prefix
    children: dict[int, Node] = field(default_factory=dict)
    cached: bool = False
    n_cached_children: int = 0
    # Policy-owned metadata.
    last_access: float = 0.0
    freq: int = 0
    score: float = 0.0
    score_t: float = 0.0
    prio: float = 0.0
    total_cost: float = 0.0
    n_computed: int = 0

    @property
    def is_cached_leaf(self) -> bool:
        return self.cached and self.n_cached_children == 0


class PrefixCache:
    def __init__(self, capacity_bytes: int, kv_bytes_per_token: int):
        self.capacity = capacity_bytes
        self.kv_bytes_per_token = kv_bytes_per_token
        self.root = Node(nid=0, content=-1, tokens=0, parent=None, cached=True)
        self.used = 0
        self.n_nodes = 1
        self._next_id = 1

    def nbytes(self, node: Node) -> int:
        return node.tokens * self.kv_bytes_per_token

    @property
    def free(self) -> int:
        return self.capacity - self.used

    def child(self, parent: Node, content: int, tokens: int) -> Node:
        node = parent.children.get(content)
        if node is None:
            node = Node(nid=self._next_id, content=content, tokens=tokens, parent=parent,
                        offset=parent.offset + parent.tokens)
            self._next_id += 1
            self.n_nodes += 1
            parent.children[content] = node
        return node

    def mark_cached(self, node: Node) -> None:
        assert not node.cached
        node.cached = True
        self.used += self.nbytes(node)
        node.parent.n_cached_children += 1

    def evict(self, node: Node) -> None:
        assert node.cached and node is not self.root
        node.cached = False
        self.used -= self.nbytes(node)
        node.parent.n_cached_children -= 1
        self._prune(node)

    def prune(self, node: Node) -> None:
        """Drop uncached, childless nodes so the tree only holds live structure.
        Evicted nodes are pruned too, so a re-inserted chunk starts with fresh
        policy statistics (as when RAGCache removes a node from its tree)."""
        self._prune(node)

    def _prune(self, node: Node) -> None:
        while node is not self.root and not node.cached and not node.children:
            parent = node.parent
            del parent.children[node.content]
            self.n_nodes -= 1
            node = parent

    def iter_cached(self):
        stack = list(self.root.children.values())
        while stack:
            n = stack.pop()
            if n.cached:
                yield n
            stack.extend(n.children.values())
