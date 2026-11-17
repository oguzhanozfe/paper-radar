"""paper-radar CLI.

  python -m paper_radar fetch  --out data/pool.jsonl [--days 7] [--categories cs.CL,cs.AI]
  python -m paper_radar digest --pool data/pool.jsonl [--top 15] [--k 60]
  python -m paper_radar eval-sample --pool data/pool.jsonl
  python -m paper_radar eval-score
  python -m paper_radar eval-report
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import arxiv, digest, evaluate, models, rank

EVAL = Path("data/eval")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def cmd_fetch(a):
    papers = arxiv.fetch(a.categories.split(","), days=a.days, log=log)
    write_jsonl(a.out, papers)
    log(f"{len(papers)} papers -> {a.out}")


def cmd_digest(a):
    themes = rank.load_profile(Path(a.profile).read_text(encoding="utf-8"))
    if a.pool:
        papers = read_jsonl(a.pool)
    else:
        papers = arxiv.fetch(a.categories.split(","), days=a.days, log=log)
    log(f"embedding {len(papers)} papers")
    emb = rank.embed_scores(papers, themes)
    ranked = rank.llm_rank(papers, themes, emb, k=a.k, log=log)
    by_id = {p["id"]: p for p in papers}
    top, close = digest.select(ranked, by_id, top_n=a.top)
    now = datetime.now(timezone.utc)
    year, week, _ = now.isocalendar()
    name = f"{year}-W{week:02d}"
    judged = [r for r in ranked if r.get("score") is not None]
    quotes = sum(r["quote_ok"] for r in judged)
    stats = (f"Week {name}: {len(papers)} new arXiv papers in {', '.join(sorted({c for p in papers for c in p['categories'] if c in a.categories.split(',')}))}; "
             f"top {a.k} by embedding ({models.EMBED_MODEL}) re-scored by {', '.join(sorted({j['model'] for j in judged}))}; {len(top)} made the cut. "
             f"Quotes verified verbatim: {quotes}/{len(judged)}.")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    titles = [t["theme"] for t in themes]
    (out / f"{name}.md").write_text(digest.markdown(f"paper-radar {name}", top, close, titles, stats), encoding="utf-8")
    (out / f"{name}.html").write_text(digest.html(f"paper-radar {name}", top, close, titles, stats), encoding="utf-8")
    log(f"wrote {out / name}.md and .html")


def cmd_eval_sample(a):
    s = evaluate.sample(read_jsonl(a.pool), n=a.n)
    write_jsonl(EVAL / "papers.jsonl", s)
    log(f"{len(s)} papers -> {EVAL / 'papers.jsonl'}")


def cmd_eval_score(a):
    themes = rank.load_profile(Path(a.profile).read_text(encoding="utf-8"))
    papers = read_jsonl(EVAL / "papers.jsonl")
    kw = rank.keyword_scores(papers, themes)
    emb = rank.embed_scores(papers, themes)
    pipeline = rank.llm_rank(papers, themes, emb, k=a.k, log=log)
    # same batches as the pipeline for the first k papers (k is a multiple of BATCH): cache hits
    llm_all = rank.llm_rank(papers, themes, emb, k=len(papers), log=log) if a.all else []
    result = {**models.describe(), "prefilter_k": a.k, "usage": models.usage,
              "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "keyword": kw, "embedding": emb, "pipeline": pipeline, "llm_all": llm_all}
    (EVAL / "scores.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"scores -> {EVAL / 'scores.json'}")


def rankings_from(scores: dict) -> dict[str, list[str]]:
    r = {
        "keyword baseline": rank.order(scores["keyword"]),
        "embedding only": rank.order(scores["embedding"]),
        f"embedding top-{scores['prefilter_k']} → LLM (shipped)": [x["id"] for x in scores["pipeline"]],
    }
    if scores.get("llm_all"):
        r["LLM on all papers"] = [x["id"] for x in scores["llm_all"]]
    return r


def cmd_eval_report(a):
    labels = evaluate.load_labels(EVAL / "labels.csv")
    scores = json.loads((EVAL / "scores.json").read_text(encoding="utf-8"))
    print(f"{len(labels)} papers, {sum(labels.values())} labelled relevant\n")
    print(evaluate.table(rankings_from(scores), labels))
    judged = [x for x in scores["pipeline"] if x.get("score") is not None]
    print(f"\nEmbedding: {scores['embed_model']}; run {scores['run_at']}")
    for name in ("pipeline", "llm_all"):
        judged = [x for x in scores.get(name) or [] if x.get("score") is not None]
        if not judged:
            continue
        used = {}
        for x in judged:
            used[x["model"]] = used.get(x["model"], 0) + 1
        cut = [x for x in judged if x["score"] >= digest.MIN_SCORE]
        hit = sum(labels[x["id"]] for x in cut)
        print(f"{name}: {len(judged)} papers LLM-scored by {used}; "
              f"quotes verified verbatim {sum(x['quote_ok'] for x in judged)}/{len(judged)}; "
              f"digest cut (score >= {digest.MIN_SCORE}) keeps {len(cut)}, precision {hit / max(1, len(cut)):.2f}, "
              f"recall {hit / sum(labels.values()):.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="paper_radar")
    sub = p.add_subparsers(dest="cmd", required=True)
    cats = ",".join(arxiv.DEFAULT_CATEGORIES)

    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    f.add_argument("--days", type=int, default=7)
    f.add_argument("--categories", default=cats)
    f.set_defaults(fn=cmd_fetch)

    d = sub.add_parser("digest")
    d.add_argument("--pool", help="JSONL from `fetch`; fetches live when omitted")
    d.add_argument("--days", type=int, default=7)
    d.add_argument("--categories", default=cats)
    d.add_argument("--profile", default="profile.md")
    d.add_argument("--top", type=int, default=15)
    d.add_argument("--k", type=int, default=64, help="papers re-scored by the LLM after the embedding prefilter")
    d.add_argument("--out-dir", default="digests")
    d.set_defaults(fn=cmd_digest)

    s = sub.add_parser("eval-sample")
    s.add_argument("--pool", required=True)
    s.add_argument("--n", type=int, default=100)
    s.set_defaults(fn=cmd_eval_sample)

    e = sub.add_parser("eval-score")
    e.add_argument("--profile", default="profile.md")
    e.add_argument("--k", type=int, default=rank.PREFILTER_K)
    e.add_argument("--all", action="store_true", help="also LLM-score every paper (slow)")
    e.set_defaults(fn=cmd_eval_score)

    sub.add_parser("eval-report").set_defaults(fn=cmd_eval_report)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
