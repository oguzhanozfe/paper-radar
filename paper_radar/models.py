"""Model backends. Gemini (free tier) by default; DeepSeek or a local Ollama optional.

Choose with env vars (keys are read from the environment only, never logged):
  PAPER_RADAR_LLM    gemini | deepseek | ollama   (default gemini)
  PAPER_RADAR_EMBED  gemini | ollama              (default gemini; DeepSeek has no embeddings)
  PAPER_RADAR_LLM_MODEL / PAPER_RADAR_EMBED_MODEL to override the model names.
PAPER_RADAR_LLM_MODEL may list several models, comma-separated: when one is out of quota or
overloaded the next is tried, and every result records the model that produced it.
Gemini and DeepSeek are called through their OpenAI-compatible endpoints. LLM replies are cached
on disk (.cache/llm, keyed by a hash of the prompt) so reruns cost no requests.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
PROVIDERS = {
    # name: (OpenAI-compatible base URL, key env var, default chat model, default embed model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
               "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemma-4-31b-it", "gemini-embedding-001"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat", None),
    "ollama": (OLLAMA + "/v1", None, "qwen3.5:9b", "qwen3-embedding:0.6b"),
}
LLM = os.environ.get("PAPER_RADAR_LLM", "gemini")
EMBED = os.environ.get("PAPER_RADAR_EMBED", "gemini")
LLM_MODELS = (os.environ.get("PAPER_RADAR_LLM_MODEL") or PROVIDERS[LLM][2]).split(",")
LLM_MODEL = LLM_MODELS[0]
CACHE = os.environ.get("PAPER_RADAR_CACHE", ".cache/llm")
EMBED_MODEL = os.environ.get("PAPER_RADAR_EMBED_MODEL") or PROVIDERS[EMBED][3]
if not EMBED_MODEL:
    raise SystemExit(f"provider {EMBED!r} has no embedding model; set PAPER_RADAR_EMBED=gemini or ollama")

usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "embed_calls": 0, "embedded_texts": 0}


def describe() -> dict:
    return {"llm_models": f"{LLM}:{','.join(LLM_MODELS)}", "embed_model": f"{EMBED}:{EMBED_MODEL}"}


def embed(texts: list[str], batch: int = 100) -> list[list[float]]:
    """Embeddings, cached on disk per model and text hash."""
    path = os.path.join(CACHE, "..", f"embed-{EMBED}-{EMBED_MODEL.replace('/', '_')}.json")
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        r = _post(EMBED, "/embeddings", {"model": EMBED_MODEL, "input": [t for _, t in chunk]})
        vecs = [d["embedding"] for d in sorted(r["data"], key=lambda d: d.get("index", 0))]  # stable sort keeps order
        cache.update({k: v for (k, _), v in zip(chunk, vecs)})
        usage["embed_calls"] += 1
        usage["embedded_texts"] += len(chunk)
    if todo:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return [cache[k] for k in keys]


class Unavailable(Exception):
    """Model out of quota for the day, or overloaded: try the next model."""


def chat(prompt: str) -> tuple[str, str]:
    """(reply text, model that produced it). Cached on disk by prompt hash."""
    path = os.path.join(CACHE, hashlib.sha256(prompt.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            hit = json.load(f)
        usage["cache_hits"] = usage.get("cache_hits", 0) + 1
        return hit["text"], hit["model"]
    errors = []
    for model in LLM_MODELS:
        body = {"model": model, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}}
        if LLM == "ollama":
            body["reasoning_effort"] = "none"  # qwen3.5 thinking off
        try:
            r = _post(LLM, "/chat/completions", body)
        except Unavailable as e:
            errors.append(f"{model}: {e}")
            continue
        usage["llm_calls"] += 1
        u = r.get("usage") or {}
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
        text = r["choices"][0]["message"]["content"] or ""
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": f"{LLM}:{model}", "text": text}, f)
        return text, f"{LLM}:{model}"
    raise RuntimeError("no model available: " + "; ".join(errors))


def _post(provider: str, path: str, body: dict, retries: int = 4) -> dict:
    base, key_env, _, _ = PROVIDERS[provider]
    headers = {"Content-Type": "application/json"}
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise SystemExit(f"{key_env} is not set (or choose another backend, see paper_radar/models.py)")
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            # a daily quota will not recover by waiting; an overloaded model may not either
            if e.code == 429 and "PerDay" in msg or e.code == 503 and attempt >= 1:
                raise Unavailable(f"HTTP {e.code}") from None
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise (Unavailable if e.code in (429, 503) else RuntimeError)(f"{provider} {path}: HTTP {e.code}") from None
            # per-minute limits say how long to wait ("Please retry in 33.4s"); otherwise back off
            m = re.search(r"retry in ([\d.]+)s", msg)
            wait = float(m.group(1)) + 1 if m else min(60, 5 * 2 ** attempt)
        except OSError as e:  # timeouts, dropped connections: give it one more try, then move on
            if attempt >= 1:
                raise Unavailable(f"{type(e).__name__}") from None
            wait = 10
        time.sleep(wait)
    raise RuntimeError("unreachable")
