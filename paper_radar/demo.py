"""`python -m paper_radar demo`: rank the bundled day of arXiv papers, no key needed.

Model resolution (--mode auto):
  GEMINI_API_KEY set           -> Gemini (cloud)
  else Ollama at OLLAMA_HOST with qwen3.5:9b and qwen3-embedding:0.6b -> local
  else replay of the recorded run in demo/recorded.json
A live run that fails (quota, timeout) is redone from scratch in replay, never mixed.

The recording holds each paper's embedding similarity score (not the vectors) and the
LLM replies keyed by a hash of the prompt, so replay only covers the bundled papers
with the bundled profile.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import arxiv, digest, models, rank

DEMO = Path("demo")
PAPERS = DEMO / "papers-2026-11-13.jsonl"
RECORDED = DEMO / "recorded.json"
NAME = "2026-11-13"
K, TOP = 40, 15  # the settings of the recorded run


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def ollama_ready() -> bool:
    _, _, chat_model, embed_model = models.PROVIDERS["ollama"]
    try:
        with urllib.request.urlopen(models.OLLAMA + "/api/tags", timeout=2) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
    except (OSError, ValueError):
        return False
    return {chat_model, embed_model} <= names


def resolve(mode: str) -> str:
    if mode != "auto":
        return mode
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return "ollama" if ollama_ready() else "replay"


def _use(provider: str):
    _, _, chat_model, embed_model = models.PROVIDERS[provider]
    models.LLM = models.EMBED = provider
    models.LLM_MODELS, models.EMBED_MODEL = chat_model.split(","), embed_model


def run(mode="auto", out_dir="demo-out", record=False, profile="profile.md"):
    papers = [json.loads(line) for line in PAPERS.read_text(encoding="utf-8").splitlines() if line.strip()]
    themes = rank.load_profile(Path(profile).read_text(encoding="utf-8"))
    chosen = resolve(mode)
    if chosen != "replay":
        try:
            return _live(chosen, papers, themes, out_dir, record)
        except (Exception, SystemExit) as e:  # quota, timeout, overloaded model
            if mode != "auto":
                raise
            log(f"{chosen} run failed ({e}); redoing it in replay")
    return _replay(papers, themes, out_dir)


def _live(provider, papers, themes, out_dir, record):
    _use(provider)
    print(f"mode: {provider} (live: {','.join(models.LLM_MODELS)} + {models.EMBED_MODEL})", flush=True)
    replies = {}

    def chat(prompt):
        text, model = models.chat(prompt)
        replies[_sha(prompt)] = {"model": model, "text": text}
        return text, model
    emb = rank.embed_scores(papers, themes)
    ranked = rank.llm_rank(papers, themes, emb, k=K, chat=chat, log=log)
    digest.write(papers, themes, ranked, NAME, out_dir, arxiv.DEFAULT_CATEGORIES,
                 K, TOP, models.EMBED_MODEL, log)
    if record:
        RECORDED.write_text(json.dumps({
            "recorded_at": datetime.now(timezone.utc).date().isoformat(), "embed_model": models.EMBED_MODEL,
            "usage": models.usage, "embedding": emb, "replies": replies}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        log(f"recording -> {RECORDED}")


def _replay(papers, themes, out_dir):
    rec = json.loads(RECORDED.read_text(encoding="utf-8"))
    used = sorted({r["model"] for r in rec["replies"].values()})
    print(f"mode: replay (recorded {', '.join(used)} + {rec['embed_model']}, {rec['recorded_at']})", flush=True)

    def chat(prompt):
        hit = rec["replies"].get(_sha(prompt))
        if hit is None:
            raise SystemExit("replay only covers the bundled papers with the bundled profile.md; "
                             "set GEMINI_API_KEY or run Ollama to rank with your own profile")
        return hit["text"], hit["model"]
    ranked = rank.llm_rank(papers, themes, rec["embedding"], k=K, chat=chat, log=log)
    digest.write(papers, themes, ranked, NAME, out_dir, arxiv.DEFAULT_CATEGORIES,
                 K, TOP, rec["embed_model"], log)
