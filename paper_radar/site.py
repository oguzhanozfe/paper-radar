"""Build the GitHub Pages site in docs/: every digest's HTML plus an index page."""
from __future__ import annotations

import shutil
from html import escape
from pathlib import Path

REPO = "https://github.com/oguzhanozfe/paper-radar"


def build(digests="digests", out="docs"):
    src, dst = Path(digests), Path(out)
    dst.mkdir(exist_ok=True)
    (dst / ".nojekyll").write_text("", encoding="utf-8")
    items = []
    for h in sorted(src.glob("*.html"), reverse=True):  # by name, newest first; mtime is not stable in git
        shutil.copyfile(h, dst / h.name)
        md = h.with_suffix(".md")
        stats = md.read_text(encoding="utf-8").splitlines()[2] if md.exists() else ""
        items.append(f'<li style="margin-bottom:14px"><a style="color:#1a4fa3;font-weight:bold" href="{escape(h.name)}">'
                     f'{escape(h.stem)}</a><div style="color:#555;font-size:13px">{escape(stats)}</div></li>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>paper-radar digests</title></head>
<body style="margin:0;background:#f6f6f6"><div style="max-width:680px;margin:0 auto;background:#fff;padding:24px;font-family:Helvetica,Arial,sans-serif;font-size:15px;line-height:1.5;color:#222">
<h1 style="font-size:24px;margin:0 0 6px">paper-radar</h1>
<p style="margin:0 0 16px">New arXiv papers ranked against a written interest profile: embedding prefilter, then an LLM
score with a one-sentence reason and a quote checked against the abstract. Each digest below is a real run.</p>
<p style="margin:0 0 20px;font-size:14px"><a style="color:#1a4fa3" href="{REPO}">Code and measured ranking results</a> ·
<a style="color:#1a4fa3" href="{REPO}/blob/main/profile.md">the interest profile</a> ·
run it yourself with no key: <code>python -m paper_radar demo</code></p>
<h2 style="font-size:17px;border-bottom:1px solid #ddd;padding-bottom:4px">Digests</h2>
<ul style="padding-left:18px">
{chr(10).join(items)}
</ul>
<p style="color:#888;font-size:12px;margin-top:28px">Thank you to arXiv for use of its open access interoperability.</p>
</div></body></html>
"""
    (dst / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} digest(s) -> {dst}/index.html")
