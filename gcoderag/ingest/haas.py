"""Best-effort Haas mill G/M-code page ingestion (SPEC 3.2); local only.

Haas content is copyrighted: raw pages and parsed docs stay under the gitignored `data/` directory.
Any failure (robots disallow, HTTP error, JS-rendered page) results in a documented skip, never an exception.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import urllib.robotparser
from pathlib import Path
from typing import Any

from gcoderag.config import get_paths
from gcoderag.ingest import HTTP_RETRIES, HTTP_TIMEOUT_S, USER_AGENT, Doc, slugify
from gcoderag.textnorm import extract_codes, normalize_code

logger = logging.getLogger(__name__)

ROBOTS_URL = "https://www.haascnc.com/robots.txt"
BASE = "https://www.haascnc.com/service/online-operator-s-manuals/mill-operator-s-manual/"
PAGES = {
    "mill-gcodes": BASE + "mill---g-codes.html",
    "mill-mcodes": BASE + "mill---m-codes.html",
}
SANITY_MARKERS = ("G01", "Linear Interpolation")
MIN_TEXT_CHARS = 40
REQUEST_INTERVAL_S = 1.0


def _get(url: str) -> tuple[int, str]:
    """GET `url` with retries; return (status_code, text). Status 0 means a network failure."""
    import requests

    last_status = 0
    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = requests.get(url, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
            last_status = response.status_code
            if response.ok:
                response.encoding = response.encoding or "utf-8"
                return response.status_code, response.text
            if 400 <= response.status_code < 500:
                return response.status_code, ""
        except Exception as exc:  # noqa: BLE001 - any network error is retried, then reported
            logger.warning("haas fetch failed (%s): %s", attempt + 1, exc)
        time.sleep(REQUEST_INTERVAL_S * (attempt + 1))
    return last_status, ""


def robots_allows(urls: list[str]) -> tuple[bool, str]:
    """Check robots.txt for every url; return (allowed, reason)."""
    status, text = _get(ROBOTS_URL)
    if status != 200:
        return False, f"robots.txt not readable (HTTP {status})"
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(text.splitlines())
    blocked = [u for u in urls if not parser.can_fetch(USER_AGENT, u)]
    return (not blocked, "allowed" if not blocked else f"robots.txt disallows {blocked}")


def parse_page(html: str, page_key: str, url: str) -> list[Doc]:
    """Parse a Haas manual page: one doc per h2/h3/h4 heading, plus one per reference-table row (pure)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    used: dict[str, int] = {}
    docs = [*_heading_docs(soup, page_key, url, used), *_table_row_docs(soup, page_key, url, used)]
    return docs


def _heading_docs(soup: Any, page_key: str, url: str, used: dict[str, int]) -> list[Doc]:
    """One document per heading: heading text plus following sibling text until the next heading."""
    docs: list[Doc] = []
    for heading in soup.find_all(["h2", "h3", "h4"]):
        title = re.sub(r"\s+", " ", heading.get_text(" ", strip=True))
        if not title:
            continue
        parts: list[str] = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ("h2", "h3", "h4"):
                break
            text = sibling.get_text("\n", strip=True)
            if text:
                parts.append(text)
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip()
        if len(body) < MIN_TEXT_CHARS:
            continue
        docs.append(_make_doc(page_key, url, title, body, used))
    return docs


def _table_row_docs(soup: Any, page_key: str, url: str, used: dict[str, int]) -> list[Doc]:
    """One document per reference-table row whose first cell is a G/M code: `<code> <name> (Group <group>)`."""
    docs: list[Doc] = []
    for row in soup.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in row.find_all(["td", "th"])]
        if len(cells) < 2 or not re.fullmatch(r"[GM]\s?\d+(\.\d+)?", cells[0]):
            continue
        code, name = normalize_code(cells[0]), cells[1]
        group = cells[2] if len(cells) > 2 and cells[2] else "n/a"
        title = f"{code} {name}"
        text = f"{code} {name} (Group {group}). See {url}."
        docs.append(_make_doc(page_key, url, title, text, used, body_is_text=True))
    return docs


def _make_doc(page_key: str, url: str, title: str, body: str, used: dict[str, int], body_is_text: bool = False) -> Doc:
    """Build a Haas `Doc` with a de-duplicated slug id."""
    slug = slugify(title)
    used[slug] = used.get(slug, 0) + 1
    if used[slug] > 1:
        slug = f"{slug}-{used[slug]}"
    page_name = "G-codes" if page_key.endswith("gcodes") else "M-codes"
    return Doc(
        doc_id=f"haas:{page_key}:{slug}",
        source="haas",
        title=title,
        codes=extract_codes(title),
        url=url,
        section_path=f"haas mill {page_name} > {title}",
        text=body if body_is_text else f"{title}\n\n{body}",
    )


def ingest(raw_dir: Path) -> tuple[list[Doc], dict[str, Any]]:
    """Run the SPEC 3.2 procedure; write STATUS.json; return (docs, status). Never raises on network problems."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"status": "skipped", "reason": "", "pages": {}, "docs": 0}
    docs: list[Doc] = []
    try:
        allowed, reason = robots_allows(list(PAGES.values()))
        if not allowed:
            status["reason"] = reason
        else:
            docs = _download_and_parse(raw_dir, status)
    except Exception as exc:  # noqa: BLE001 - best effort by design
        logger.warning("haas ingest failed: %s", exc)
        status.update(status="skipped", reason=f"unexpected error: {exc}")
        docs = []
    status["docs"] = len(docs)
    (raw_dir / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return docs, status


def _download_and_parse(raw_dir: Path, status: dict[str, Any]) -> list[Doc]:
    """Download each page (1 req/s), sanity-check the G-code page, and parse; updates `status` in place."""
    pages_html: dict[str, str] = {}
    for key, url in PAGES.items():
        cached = raw_dir / f"{key}.html"
        if cached.exists():
            html, code = cached.read_text(encoding="utf-8"), 200
        else:
            time.sleep(REQUEST_INTERVAL_S)
            code, html = _get(url)
            if code == 200:
                cached.write_text(html, encoding="utf-8")
        status["pages"][key] = {"url": url, "http_status": code}
        if code != 200:
            status["reason"] = f"HTTP {code} for {url}"
            return []
        pages_html[key] = html
    gcode_html = pages_html["mill-gcodes"]
    if not all(marker in gcode_html for marker in SANITY_MARKERS):
        status["reason"] = "G-code page lacks 'G01'/'Linear Interpolation' in raw HTML (JavaScript-rendered)"
        return []
    docs = [doc for key, html in pages_html.items() for doc in parse_page(html, key, PAGES[key])]
    if not docs:
        status["reason"] = "pages downloaded but no sections parsed"
        return []
    status.update(status="ok", reason="downloaded and parsed")
    return docs


def main() -> int:
    """Run the Haas ingest and print the status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    docs, status = ingest(get_paths().raw / "haas")
    print(f"haas status: {status['status']} ({status['reason']}); docs: {len(docs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
