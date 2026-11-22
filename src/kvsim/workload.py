"""Synthetic RAG request traces.

A request is a list of (content_id, tokens) chunks:
    [system prompt chunks] + [chunks of k retrieved docs] + [query chunk]
Document popularity is Zipf(alpha) over ranks; every `shift_every` requests the
rank->document mapping is reshuffled (a hotspot shift). The query chunk is
unique per request, so it is never reused.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

Request = list[tuple[int, int]]


@dataclass
class RagWorkload:
    n_requests: int = 5000
    n_docs: int = 2000
    docs_per_request: int = 3
    zipf_alpha: float = 0.9
    doc_chunks_min: int = 2
    doc_chunks_max: int = 12
    system_chunks: int = 2
    chunk_tokens: int = 256
    query_tokens: int = 64
    shift_every: int = 0
    sort_docs: bool = False
    seed: int = 0

    def generate(self) -> list[Request]:
        rng = np.random.default_rng(self.seed)
        n_chunks = rng.integers(self.doc_chunks_min, self.doc_chunks_max + 1, self.n_docs)
        offsets = np.concatenate([[0], np.cumsum(n_chunks)])
        doc_base = self.system_chunks
        query_base = doc_base + int(offsets[-1])

        weights = 1.0 / np.arange(1, self.n_docs + 1) ** self.zipf_alpha
        weights /= weights.sum()
        rank_to_doc = rng.permutation(self.n_docs)

        system = [(i, self.chunk_tokens) for i in range(self.system_chunks)]
        trace: list[Request] = []
        for r in range(self.n_requests):
            if self.shift_every and r > 0 and r % self.shift_every == 0:
                rank_to_doc = rng.permutation(self.n_docs)
            ranks = rng.choice(self.n_docs, size=self.docs_per_request, replace=False, p=weights)
            docs = [int(rank_to_doc[x]) for x in ranks]
            if self.sort_docs:
                docs.sort()
            req = list(system)
            for d in docs:
                base = doc_base + int(offsets[d])
                req.extend((base + j, self.chunk_tokens) for j in range(int(n_chunks[d])))
            req.append((query_base + r, self.query_tokens))
            trace.append(req)
        return trace


@lru_cache(maxsize=2)
def _load_trace_file(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class TraceWorkload:
    """Requests from a retrieval trace (scripts/build_squad_trace.py).

    Each question becomes [system chunks] + [chunks of its top-k retrieved
    paragraphs] + [question chunk]. A paragraph is split into chunk_tokens
    pieces, the last one partial. order="article" keeps SQuAD's topical bursts
    (questions about one article arrive together; article order is shuffled per
    seed); order="shuffled" removes all temporal locality.
    """

    trace: str = "traces/squad_train.json.gz"
    docs_per_request: int = 3
    order: str = "article"
    sort_docs: bool = False
    system_chunks: int = 2
    chunk_tokens: int = 256
    n_requests: int | None = None
    seed: int = 0

    def generate(self) -> list[Request]:
        d = _load_trace_file(self.trace)
        ptok = d["paragraph_tokens"]
        qs = d["questions"]
        rng = np.random.default_rng(self.seed)
        if self.order == "article":
            by_article: dict[int, list[int]] = {}
            for i, q in enumerate(qs):
                by_article.setdefault(q["a"], []).append(i)
            idx = [i for a in rng.permutation(sorted(by_article)) for i in by_article[int(a)]]
        elif self.order == "shuffled":
            idx = rng.permutation(len(qs)).tolist()
        else:
            raise ValueError(f"order must be 'article' or 'shuffled', got {self.order!r}")
        if self.n_requests:
            idx = idx[: self.n_requests]

        n_pchunks = [-(-t // self.chunk_tokens) for t in ptok]
        offsets = np.concatenate([[0], np.cumsum(n_pchunks)])
        doc_base = self.system_chunks
        query_base = doc_base + int(offsets[-1])
        system = [(i, self.chunk_tokens) for i in range(self.system_chunks)]

        trace: list[Request] = []
        for r, qi in enumerate(idx):
            q = qs[qi]
            docs = q["r"][: self.docs_per_request]
            if self.sort_docs:
                docs = sorted(docs)
            req = list(system)
            for p in docs:
                base = doc_base + int(offsets[p])
                left = ptok[p]
                for j in range(n_pchunks[p]):
                    req.append((base + j, min(self.chunk_tokens, left)))
                    left -= self.chunk_tokens
            req.append((query_base + r, q["t"]))
            trace.append(req)
        return trace


def make_workload(params: dict):
    return TraceWorkload(**params) if "trace" in params else RagWorkload(**params)


def unique_tokens(trace: list[Request]) -> int:
    """Tokens in the full prefix tree of the trace (the working set)."""
    root: dict = {}
    total = 0
    for req in trace:
        node = root
        for content, tokens in req:
            if content not in node:
                node[content] = {}
                total += tokens
            node = node[content]
    return total
