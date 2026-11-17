"""Render a ranked week as Markdown and as email-ready HTML (inline styles, no external assets)."""
from __future__ import annotations

from html import escape

MIN_SCORE = 7      # LLM score needed to make the top list
CLOSE_SCORE = 5    # scores 5-6 (or 7+ beyond top N) land in "skipped but close"


def select(ranked: list[dict], papers: dict, top_n: int = 15, close_n: int = 10):
    """Split LLM-ranked results into (top, close). Only LLM-scored papers are eligible."""
    scored = [r for r in ranked if r.get("score") is not None]
    top = [r for r in scored if r["score"] >= MIN_SCORE][:top_n]
    taken = {r["id"] for r in top}
    close = [r for r in scored if r["id"] not in taken and r["score"] >= CLOSE_SCORE][:close_n]
    return [dict(papers[r["id"]], **r) for r in top], [dict(papers[r["id"]], **r) for r in close]


def _grouped(items, themes):
    groups = {t: [] for t in themes}
    for it in items:
        groups.setdefault(it["theme"], []).append(it)
    return [(t, g) for t, g in groups.items() if g]


def _md_escape(s: str) -> str:
    return s.replace("[", "\\[").replace("]", "\\]").replace("<", "&lt;")


def markdown(title, top, close, themes, stats) -> str:
    out = [f"# {title}", "", stats, ""]
    for theme, items in _grouped(top, themes):
        out += [f"## {theme}", ""]
        for p in items:
            code = " · code: " + ", ".join(f"<{u}>" for u in p["code"]) if p["code"] else ""
            out.append(f"- **[{_md_escape(p['title'])}]({p['url']})** — score {p['score']}/10{code}")
            out.append(f"  - {_md_escape(p['reason'])}")
            if p.get("quote"):
                out.append(f"  - > “{_md_escape(p['quote'])}”")
        out.append("")
    if close:
        out += ["## Skipped but close", ""]
        for p in close:
            out.append(f"- [{_md_escape(p['title'])}]({p['url']}) — {p['score']}/10, {p['theme']}: {_md_escape(p['reason'])}")
        out.append("")
    out += ["---", "Thank you to arXiv for use of its open access interoperability. "
            "Reasons are written by the LLM named above; quotes are shown only when found verbatim in the abstract.", ""]
    return "\n".join(out)


def html(title, top, close, themes, stats) -> str:
    e = escape
    a = 'style="color:#1a4fa3;text-decoration:none"'
    body = [f'<h1 style="font-size:22px;margin:0 0 4px">{e(title)}</h1>',
            f'<p style="color:#666;font-size:13px;margin:0 0 20px">{e(stats)}</p>']
    for theme, items in _grouped(top, themes):
        body.append(f'<h2 style="font-size:17px;border-bottom:1px solid #ddd;padding-bottom:4px;margin:24px 0 8px">{e(theme)}</h2>')
        for p in items:
            code = "".join(f' <a {a} href="{e(u)}">[code]</a>' for u in p["code"])
            quote = (f'<div style="color:#555;font-style:italic;border-left:3px solid #ccc;padding-left:8px;margin-top:4px">'
                     f'“{e(p["quote"])}”</div>') if p.get("quote") else ""
            body.append(
                f'<div style="margin:0 0 14px"><a {a} href="{e(p["url"])}"><b>{e(p["title"])}</b></a>'
                f' <span style="color:#888;font-size:12px">{p["score"]}/10</span>{code}'
                f'<div style="margin-top:3px">{e(p["reason"])}</div>{quote}</div>')
    if close:
        body.append('<h2 style="font-size:17px;border-bottom:1px solid #ddd;padding-bottom:4px;margin:24px 0 8px">Skipped but close</h2><ul style="padding-left:18px">')
        for p in close:
            body.append(f'<li style="margin-bottom:6px"><a {a} href="{e(p["url"])}">{e(p["title"])}</a>'
                        f' <span style="color:#888;font-size:12px">{p["score"]}/10 · {e(p["theme"])}</span><br>{e(p["reason"])}</li>')
        body.append("</ul>")
    body.append('<p style="color:#888;font-size:12px;margin-top:28px">Thank you to arXiv for use of its open access interoperability. '
                'Reasons are written by the LLM named above; quotes are shown only when found verbatim in the abstract.</p>')
    return ('<!doctype html><html><head><meta charset="utf-8"><title>' + e(title) + '</title></head>'
            '<body style="margin:0;background:#f6f6f6"><div style="max-width:680px;margin:0 auto;background:#fff;'
            'padding:24px;font-family:Helvetica,Arial,sans-serif;font-size:14px;line-height:1.45;color:#222">'
            + "\n".join(body) + "</div></body></html>\n")
