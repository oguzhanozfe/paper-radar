"""Three rankers over the same papers: keyword baseline, embedding, embedding prefilter + LLM.

Models come from `models.py` (Gemini by default). Tests inject fakes for `embed` and `chat`.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata

from . import models

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

def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_scores(papers, themes, embed=None):
    """Max cosine between the paper and any theme description; theme = argmax."""
    embed = embed or models.embed
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

Rate each paper below by how worth reading it is for this reader, 0-10:
0-2 unrelated, 3-4 adjacent topic only, 5-6 partly relevant, 7-8 clearly relevant, 9-10 must read.
For each paper pick the closest theme, write one sentence explaining the score, and copy one
short exact phrase (4-15 words) from that paper's abstract that supports it.

{papers}

Answer with one JSON object only, no other text:
{{"results": [{{"id": "<paper id>", "score": <0-10 integer>, "theme": "<theme name>", "reason": "<one sentence>", "quote": "<exact phrase>"}}]}}
with one entry per paper, in the same order."""
BATCH = 8  # papers per LLM request (free-tier quotas count requests, not tokens)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}))
    s = re.sub(r"[\"'`]", "", s)
    return " ".join(s.split()).strip(" .,;:")


def quote_verified(quote: str, abstract: str) -> bool:
    q = _norm(quote)
    return len(q.split()) >= 3 and q in _norm(abstract)


def parse_json(text: str):
    """The JSON value in a model reply, ignoring <thought> blocks, code fences and chatter."""
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.S)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return json.loads(text)
    a = min(starts)
    b = text.rfind("}" if text[a] == "{" else "]")
    return json.loads(text[a:b + 1])


def _results(value) -> list:
    """Accept {"results": [...]}, a bare list, or a single object."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value.get("results", [value])
    return []


def _one(paper, d, names, model):
    try:
        score = max(0, min(10, int(d["score"])))
    except (ValueError, KeyError, TypeError):
        return {"id": paper["id"], "score": 0, "theme": names[0], "reason": "", "quote": "",
                "quote_ok": False, "model": model, "error": "unparseable model output"}
    theme = str(d.get("theme", ""))
    if theme not in names:  # tolerate "Tool use" for "Tool use and MCP" etc.
        t = _norm(theme)
        theme = next((n for n in names if t and (t in _norm(n) or _norm(n) in t)), names[0])
    quote = str(d.get("quote", "")).strip().strip('"')
    ok = quote_verified(quote, paper["abstract"])
    return {"id": paper["id"], "score": score, "theme": theme, "reason": str(d.get("reason", "")).strip(),
            "quote": quote if ok else "", "quote_ok": ok, "model": model}


def llm_judge_batch(papers, themes, chat=None) -> list[dict]:
    """Score several papers in one request. `chat(prompt)` returns text or (text, model)."""
    chat = chat or models.chat
    names = [t["theme"] for t in themes]
    prompt = PROMPT.format(
        themes="\n".join(f"- {t['theme']}: {t['text']}" for t in themes),
        papers="\n\n".join(f"Paper id: {p['id']}\nTitle: {p['title']}\nAbstract: {p['abstract']}" for p in papers))
    reply = chat(prompt)
    text, model = reply if isinstance(reply, tuple) else (reply, "")
    try:
        results = _results(parse_json(text))
        by_id = {str(r.get("id", "")).strip(): r for r in results if isinstance(r, dict)}
    except (ValueError, KeyError, TypeError, AttributeError):
        by_id = {}
    return [_one(p, by_id.get(p["id"], {}), names, model) for p in papers]


def llm_rank(papers, themes, embedded, k=PREFILTER_K, chat=None, log=lambda *_: None):
    """Prefilter to the top-k by embedding score, then LLM-score those in batches.

    Returns all papers: LLM-scored ones first (by LLM score, ties broken by embedding
    score then id), then the rest in embedding order with score None.
    """
    emb = {e["id"]: e for e in embedded}
    by_emb = sorted(papers, key=lambda p: (-emb[p["id"]]["score"], p["id"]))
    judged = []
    for i in range(0, min(k, len(by_emb)), BATCH):
        batch = by_emb[i:min(i + BATCH, k)]
        for j in llm_judge_batch(batch, themes, chat):
            j["embed"] = emb[j["id"]]["score"]
            judged.append(j)
        log(f"llm {len(judged)}/{min(k, len(by_emb))} scored")
    judged.sort(key=lambda j: (-j["score"], -j["embed"], j["id"]))
    rest = [{"id": p["id"], "score": None, "theme": emb[p["id"]]["theme"], "embed": emb[p["id"]]["score"]}
            for p in by_emb[k:]]
    return judged + rest


def order(scored) -> list[str]:
    """Ids by descending score, ties broken by id (for the baselines)."""
    return [s["id"] for s in sorted(scored, key=lambda s: (-s["score"], s["id"]))]
