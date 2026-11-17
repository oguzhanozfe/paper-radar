"""Three rankers over the same papers: keyword baseline, embedding, embedding prefilter + LLM.

Models are reached through Ollama's HTTP API. Tests inject fakes for `embed` and `chat`.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.environ.get("PAPER_RADAR_EMBED_MODEL", "qwen3-embedding:0.6b")
LLM_MODEL = os.environ.get("PAPER_RADAR_LLM_MODEL", "qwen3.5:9b")
PREFILTER_K = 40


# ---------- profile ----------

def load_profile(text: str) -> list[dict]:
    """[{'theme', 'text', 'keywords'}] from a Markdown profile with one `##` section per theme."""
    themes = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        name, _, body = block.partition("\n")
        kw, lines = [], []
        for line in body.strip().splitlines():
            if line.lower().startswith("keywords:"):
                kw += [k.strip().lower() for k in line.split(":", 1)[1].split(",") if k.strip()]
            else:
                lines.append(line)
        themes.append({"theme": name.strip(), "text": " ".join(lines).strip(), "keywords": kw})
    if not themes:
        raise ValueError("profile has no '## theme' sections")
    return themes


def paper_text(p: dict) -> str:
    return f"{p['title']}. {p['abstract']}"


# ---------- keyword baseline ----------

def keyword_scores(papers, themes):
    """Count of distinct profile keywords present (title hits count double). Theme = most hits."""
    out = []
    for p in papers:
        title, body = p["title"].lower(), paper_text(p).lower()
        best, best_theme = 0.0, themes[0]["theme"]
        total = 0.0
        for t in themes:
            s = sum((2.0 if _has(title, k) else 1.0) for k in t["keywords"] if _has(body, k))
            total += s
            if s > best:
                best, best_theme = s, t["theme"]
        out.append({"id": p["id"], "score": total, "theme": best_theme})
    return out


def _has(text, kw):
    return re.search(r"(?<![\w-])" + re.escape(kw) + r"(?![\w-])", text) is not None


# ---------- embeddings ----------

def ollama_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 32):
        out += _post("/api/embed", {"model": EMBED_MODEL, "input": texts[i:i + 32]})["embeddings"]
    return out


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_scores(papers, themes, embed=ollama_embed):
    """Max cosine between the paper and any theme description; theme = argmax."""
    tvecs = embed([t["theme"] + ": " + t["text"] for t in themes])
    pvecs = embed([paper_text(p) for p in papers])
    out = []
    for p, v in zip(papers, pvecs):
        sims = [_cos(v, tv) for tv in tvecs]
        i = max(range(len(sims)), key=sims.__getitem__)
        out.append({"id": p["id"], "score": sims[i], "theme": themes[i]["theme"]})
    return out


# ---------- LLM ----------

PROMPT = """You rank research papers for one reader. Their interests, one theme per line:
{themes}

Paper title: {title}
Abstract: {abstract}

Rate how worth reading this paper is for this reader, 0-10:
0-2 unrelated, 3-4 adjacent topic only, 5-6 partly relevant, 7-8 clearly relevant, 9-10 must read.
Pick the closest theme. Write one sentence explaining the score, and copy one short
exact phrase (4-15 words) from the abstract that supports it."""

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "theme": {"type": "string"},
        "reason": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["score", "theme", "reason", "quote"],
}


def ollama_chat(prompt: str) -> str:
    r = _post("/api/chat", {
        "model": LLM_MODEL, "stream": False, "think": False, "format": SCHEMA,
        "options": {"temperature": 0, "seed": 7},
        "messages": [{"role": "user", "content": prompt}],
    })
    return r["message"]["content"]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    s = re.sub(r"[\"'`]", "", s)
    return " ".join(s.split()).strip(" .,;:")


def quote_verified(quote: str, abstract: str) -> bool:
    q = _norm(quote)
    return len(q.split()) >= 3 and q in _norm(abstract)


def llm_judge(paper, themes, chat=ollama_chat) -> dict:
    names = [t["theme"] for t in themes]
    prompt = PROMPT.format(
        themes="\n".join(f"- {t['theme']}: {t['text']}" for t in themes),
        title=paper["title"], abstract=paper["abstract"])
    try:
        d = json.loads(chat(prompt))
        score = max(0, min(10, int(d["score"])))
    except (ValueError, KeyError, TypeError):
        return {"id": paper["id"], "score": 0, "theme": names[0], "reason": "", "quote": "",
                "quote_ok": False, "error": "unparseable model output"}
    theme = d.get("theme", "")
    if theme not in names:  # tolerate "Tool use" for "Tool use and MCP" etc.
        t = _norm(theme)
        theme = next((n for n in names if t and (t in _norm(n) or _norm(n) in t)), names[0])
    quote = str(d.get("quote", "")).strip().strip('"')
    ok = quote_verified(quote, paper["abstract"])
    return {"id": paper["id"], "score": score, "theme": theme,
            "reason": str(d.get("reason", "")).strip(),
            "quote": quote if ok else "", "quote_ok": ok}


def llm_rank(papers, themes, embedded, k=PREFILTER_K, chat=ollama_chat, log=lambda *_: None):
    """Prefilter to the top-k by embedding score, then LLM-score those.

    Returns all papers: LLM-scored ones first (by LLM score, ties broken by embedding
    score then id), then the rest in embedding order with score None.
    """
    emb = {e["id"]: e for e in embedded}
    by_emb = sorted(papers, key=lambda p: (-emb[p["id"]]["score"], p["id"]))
    judged = []
    for i, p in enumerate(by_emb[:k]):
        j = llm_judge(p, themes, chat)
        j["embed"] = emb[p["id"]]["score"]
        judged.append(j)
        log(f"llm {i + 1}/{min(k, len(by_emb))} {j['score']} {p['title'][:60]}")
    judged.sort(key=lambda j: (-j["score"], -j["embed"], j["id"]))
    rest = [{"id": p["id"], "score": None, "theme": emb[p["id"]]["theme"], "embed": emb[p["id"]]["score"]}
            for p in by_emb[k:]]
    return judged + rest


def order(scored) -> list[str]:
    """Ids by descending score, ties broken by id (for the baselines)."""
    return [s["id"] for s in sorted(scored, key=lambda s: (-s["score"], s["id"]))]


def model_digests() -> dict:
    """{model: digest} for the embed and LLM models, so results record exactly what ran."""
    try:
        req = urllib.request.Request(OLLAMA + "/api/tags")
        with urllib.request.urlopen(req, timeout=30) as r:
            tags = json.loads(r.read())["models"]
    except OSError:
        return {}
    return {m["name"]: m["digest"] for m in tags if m["name"] in (EMBED_MODEL, LLM_MODEL)}


def _post(path, body, retries=4):
    """POST to Ollama, retrying on timeouts/errors (a shared Ollama can be busy)."""
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(10 * 2 ** attempt)
