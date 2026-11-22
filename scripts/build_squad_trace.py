"""Build a retrieval trace from SQuAD v1.1 (real questions over Wikipedia paragraphs).

For every question, retrieve the top-10 paragraphs with TF-IDF cosine similarity
(a lightweight stand-in for a dense retriever) and store the ranked paragraph ids.
Requests are assembled later by kvsim.workload.TraceWorkload, so docs per request,
ordering, and sorting can be swept without re-running retrieval.

Token counts are estimated as round(1.3 * words) (no tokenizer dependency).

Usage:
  curl -L -o data/raw/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  uv run --extra trace python scripts/build_squad_trace.py data/raw/train-v1.1.json traces/squad_train.json.gz
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TOP = 10


def est_tokens(text: str) -> int:
    return max(1, round(1.3 * len(text.split())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("squad_json", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    data = json.loads(args.squad_json.read_text(encoding="utf-8"))["data"]
    paragraphs, para_article, questions = [], [], []
    for a, article in enumerate(data):
        for para in article["paragraphs"]:
            pid = len(paragraphs)
            paragraphs.append(para["context"])
            para_article.append(a)
            for qa in para["qas"]:
                questions.append((a, pid, qa["question"]))
    print(f"{len(data)} articles, {len(paragraphs)} paragraphs, {len(questions)} questions")

    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    P = vec.fit_transform(paragraphs)
    Q = vec.transform([q for _, _, q in questions])

    top = np.empty((len(questions), TOP), dtype=np.int32)
    for s in range(0, len(questions), args.batch):
        sims = (Q[s : s + args.batch] @ P.T).toarray()
        idx = np.argpartition(-sims, TOP, axis=1)[:, :TOP]
        order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
        top[s : s + len(sims)] = np.take_along_axis(idx, order, axis=1)

    gold = np.array([pid for _, pid, _ in questions])
    for k in (1, 3, 5, 10):
        print(f"recall@{k}: {np.mean((top[:, :k] == gold[:, None]).any(axis=1)):.3f}")

    out = {
        "source": "SQuAD v1.1 train; TF-IDF top-10 retrieval; tokens = round(1.3 * words)",
        "paragraph_tokens": [est_tokens(p) for p in paragraphs],
        "paragraph_article": para_article,
        "questions": [
            {"a": a, "g": pid, "t": est_tokens(q), "r": top[i].tolist()}
            for i, (a, pid, q) in enumerate(questions)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
