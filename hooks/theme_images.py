"""MkDocs hook: make README ``<picture>`` elements follow the Material theme toggle.

The README uses ``<picture>`` with ``prefers-color-scheme`` sources, which
GitHub and PyPI render correctly but which follow the OS setting rather than
the site's light/dark toggle. At build time, each such ``<picture>`` is
replaced by two ``<img>`` tags marked ``#only-light`` and ``#only-dark``,
which Material for MkDocs shows or hides based on the active color scheme.
"""

import re

PICTURE = re.compile(r"<picture>(.*?)</picture>", re.DOTALL)
SOURCE = re.compile(
    r'<source\s+media="\(prefers-color-scheme:\s*(light|dark)\)"\s+srcset="([^"]+)"\s*/?>'
)
IMG = re.compile(r"<img\b[^>]*>")
SRC = re.compile(r'src="[^"]*"')


def _replace(match: re.Match) -> str:
    inner = match.group(1)
    sources = dict(SOURCE.findall(inner))
    img = IMG.search(inner)
    if img is None or not {"light", "dark"} <= sources.keys():
        return match.group(0)
    tag = img.group(0)
    return "".join(
        SRC.sub(lambda _: f'src="{sources[scheme]}#only-{scheme}"', tag, count=1)
        for scheme in ("light", "dark")
    )


def on_page_content(html, page, config, files):
    return PICTURE.sub(_replace, html)
