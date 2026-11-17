"""Ranking evaluation: precision@10 and recall@20 against a labelled set."""
from __future__ import annotations

import csv
import random


def load_labels(path) -> dict[str, int]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: int(r["relevant"]) for r in csv.DictReader(f)}


def precision_at(ids, labels, k):
    return sum(labels[i] for i in ids[:k]) / k


def recall_at(ids, labels, k):
    rel = sum(labels.values())
    return sum(labels[i] for i in ids[:k]) / rel if rel else 0.0


def random_expectation(labels, k_p=10, k_r=20):
    """Expected P@k_p and R@k_r of a uniformly random ordering."""
    n, rel = len(labels), sum(labels.values())
    return rel / n, min(k_r, n) / n


def sample(papers, n=100, seed=20261117):
    rng = random.Random(seed)
    pool = sorted(papers, key=lambda p: p["id"])
    return rng.sample(pool, min(n, len(pool)))


def table(rankings: dict[str, list[str]], labels) -> str:
    rows = ["| Ranker | P@10 | R@20 |", "|---|---|---|"]
    for name, ids in rankings.items():
        ids = [i for i in ids if i in labels]
        rows.append(f"| {name} | {precision_at(ids, labels, 10):.2f} | {recall_at(ids, labels, 20):.2f} |")
    p, r = random_expectation(labels)
    rows.append(f"| random order (expected) | {p:.2f} | {r:.2f} |")
    return "\n".join(rows)
