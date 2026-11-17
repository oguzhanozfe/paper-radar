"""Fetch papers from the official arXiv API (https://info.arxiv.org/help/api/).

Rules followed: one request at a time, >= 3 s between requests, pages of at most
1000 results, a generic User-Agent. Thank you to arXiv for use of its open access
interoperability.
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

API = "https://export.arxiv.org/api/query"
USER_AGENT = "paper-radar/0.1 (+https://github.com/oguzhanozfe/paper-radar)"
DEFAULT_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "cs.MA", "cs.HC"]
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}
SPACING_S = 3.1
PAGE = 500

GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+/[\w.-]+", re.I)
# "code available", "code is released", "our code" ... without a link
CODE_HINT_RE = re.compile(r"\b(?:code|implementation)s?\b[^.]{0,40}\b(?:available|released|open[- ]sourced?|public)\b", re.I)


def _text(el, path):
    found = el.find(path, NS)
    return " ".join(found.text.split()) if found is not None and found.text else ""


def code_links(*texts: str) -> list[str]:
    """GitHub repo URLs found in the given texts, normalised and deduplicated."""
    out = []
    for t in texts:
        for m in GITHUB_RE.findall(t or ""):
            url = "https://" + re.sub(r"^(?:https?://)?(?:www\.)?", "", m).rstrip(".")
            if url.lower() not in (u.lower() for u in out):
                out.append(url)
    return out


def parse_feed(xml_text: str) -> tuple[int, list[dict]]:
    root = ET.fromstring(xml_text)
    total = int(_text(root, "os:totalResults") or 0)
    papers = []
    for e in root.findall("a:entry", NS):
        raw_id = _text(e, "a:id")
        if "/abs/" not in raw_id:  # error entries have no /abs/ id
            continue
        pid = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[1])
        abstract = _text(e, "a:summary")
        comment = _text(e, "arxiv:comment")
        prim = e.find("arxiv:primary_category", NS)
        papers.append({
            "id": pid,
            "title": _text(e, "a:title"),
            "abstract": abstract,
            "comment": comment,
            "authors": [_text(a, "a:name") for a in e.findall("a:author", NS)],
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
            "primary": prim.get("term") if prim is not None else "",
            "published": _text(e, "a:published"),
            "url": f"https://arxiv.org/abs/{pid}",
            "code": code_links(abstract, comment),
            "code_hint": bool(CODE_HINT_RE.search(abstract + " " + comment)),
        })
    return total, papers


def _get(params: dict, retries: int = 5) -> str:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8")
        except OSError as e:  # HTTPError (429/503) and network errors
            if attempt == retries - 1:
                raise
            wait = 60 * 2 ** attempt
            retry_after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = max(wait, int(retry_after))
            print(f"arXiv request failed ({e}); waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch(categories=None, days: int = 7, end: datetime | None = None,
          get=_get, sleep=time.sleep, log=print) -> list[dict]:
    """All papers submitted in the `days` before `end` in any of `categories`."""
    categories = categories or DEFAULT_CATEGORIES
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    fmt = "%Y%m%d%H%M"
    cats = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cats}) AND submittedDate:[{start.strftime(fmt)} TO {end.strftime(fmt)}]"
    seen: dict[str, dict] = {}
    offset, total = 0, None
    empty_retries = 0
    while total is None or offset < total:
        if offset or total is not None:
            sleep(SPACING_S)
        total_now, page = parse_feed(get({
            "search_query": query, "start": offset, "max_results": PAGE,
            "sortBy": "submittedDate", "sortOrder": "descending",
        }))
        total = total_now if total is None else total
        if not page:
            # the API sometimes returns an empty page mid-way; retry a couple of times
            empty_retries += 1
            if empty_retries > 3:
                log(f"giving up at offset {offset}/{total}: empty pages")
                break
            continue
        empty_retries = 0
        for p in page:
            seen.setdefault(p["id"], p)
        offset += len(page)
        log(f"fetched {offset}/{total}")
    return list(seen.values())
