"""Download and parse the LinuxCNC G-code AsciiDoc sources into documents (SPEC 3.1, 3.3).

Run `python -m gcoderag.ingest.linuxcnc` to download (if needed), parse and print `docs: N`.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

from gcoderag.config import get_paths
from gcoderag.ingest import HTTP_RETRIES, HTTP_TIMEOUT_S, USER_AGENT, Doc, slugify
from gcoderag.textnorm import extract_codes

logger = logging.getLogger(__name__)

RAW_BASE_URL = "https://raw.githubusercontent.com/LinuxCNC/linuxcnc/master/docs/src/gcode/"
PUBLIC_BASE_URL = "https://linuxcnc.org/docs/stable/html/gcode/"
REQUIRED_FILES = ("g-code.adoc", "m-code.adoc", "overview.adoc")
OPTIONAL_FILES = (
    "o-code.adoc", "other-code.adoc", "coordinates.adoc", "tool-compensation.adoc",
    "machining-center.adoc", "rs274ngc.adoc",
)
MIN_TEXT_CHARS = 40
ANCHOR_LOOKBACK = 3

_HEADING_RE = re.compile(r"^(=+)\s+(.*\S)\s*$")
_ANCHOR_RE = re.compile(r"^\[\[([^\]]+)\]\]\s*$")
_INDEX_TERM_RE = re.compile(r"\(\(\(.*?\)\)\)")
_XREF_RE = re.compile(r"<<[^,>]+,\s*([^>]+)>>|<<([^>]+)>>")
_INLINE_ANCHOR_RE = re.compile(r"\[\[[^\]]*\]\]")
_HEADING_BASE_CODE_RE = re.compile(r"\b(G\s?0*10)\s+L\d+", re.IGNORECASE)


def download_sources(raw_dir: Path) -> list[Path]:
    """Download the LinuxCNC .adoc files into `raw_dir` (skipping existing files); return the files present.

    Required files raise RuntimeError when they cannot be fetched; optional files are skipped on 404.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    present: list[Path] = []
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        target = raw_dir / name
        if not target.exists():
            text = fetch_text(RAW_BASE_URL + name)
            if text is None:
                if name in REQUIRED_FILES:
                    raise RuntimeError(f"required LinuxCNC source could not be downloaded: {name}")
                logger.info("skipping optional source %s", name)
                continue
            target.write_text(text, encoding="utf-8")
        present.append(target)
    return present


def fetch_text(url: str) -> str | None:
    """GET `url` with retries; return the body text, or None on 404 / repeated failure."""
    import requests

    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = requests.get(url, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            response.encoding = "utf-8"
            return response.text
        except Exception as exc:  # noqa: BLE001 - network errors of any kind trigger a retry
            logger.warning("fetch failed (%s/%s) %s: %s", attempt + 1, HTTP_RETRIES + 1, url, exc)
            time.sleep(1.0 * (attempt + 1))
    return None


def clean_title(heading: str) -> str:
    """Strip `(((index terms)))` and inline anchors from a heading and collapse whitespace."""
    title = _INLINE_ANCHOR_RE.sub("", _INDEX_TERM_RE.sub("", heading))
    title = _XREF_RE.sub(lambda m: (m.group(1) or m.group(2)).strip(), title)
    return re.sub(r"\s+", " ", title).strip()


def heading_codes(title: str) -> list[str]:
    """Return normalized codes named in a heading, plus the base `G10` for 'G10 L2' style headings."""
    codes = extract_codes(title)
    if _HEADING_BASE_CODE_RE.search(title) and "G10" not in codes:
        codes.append("G10")
    return codes


def _clean_table_row(line: str) -> str:
    """Turn an AsciiDoc table row (`|a |b`) into `a | b`."""
    cells = [cell.strip() for cell in line.split("|")]
    return " | ".join(cell for cell in cells if cell)


def clean_body(lines: list[str]) -> str:
    """Apply the SPEC 3.3 cleaning rules to the raw lines of a section body."""
    out: list[str] = []
    in_comment = False
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("////"):
            in_comment = not in_comment
            continue
        if in_comment or _is_dropped_line(stripped):
            continue
        if stripped.startswith("|"):
            line = _clean_table_row(stripped)
        elif re.match(r"^\.[A-Za-z0-9]", stripped):
            line = stripped[1:]  # block title such as ".G4 Example Line"
        line = _INDEX_TERM_RE.sub("", line)
        line = _INLINE_ANCHOR_RE.sub("", line)
        line = _XREF_RE.sub(lambda m: (m.group(1) or m.group(2)).strip(), line)
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_dropped_line(stripped: str) -> bool:
    """True for markup-only lines: comments, images, attributes, includes, fences and block attributes."""
    if stripped.startswith("//") or stripped.startswith(("image::", "include::")):
        return True
    if re.match(r"^:[\w-]+:", stripped):
        return True
    if re.fullmatch(r"(-{4,}|\.{4,}|={4,}|\*{4,}|_{4,}|\+{4,}|\|===|,===|!===)", stripped):
        return True
    if re.fullmatch(r"\[[^\[\]]*\]", stripped):  # block attributes: [source,ngc], [NOTE], [width=...]
        return True
    return bool(_ANCHOR_RE.match(stripped))


def _find_anchor(lines: list[str], heading_index: int) -> str | None:
    """Return the nearest `[[anchor]]` within ANCHOR_LOOKBACK lines above a heading, not crossing another heading."""
    for offset in range(1, ANCHOR_LOOKBACK + 1):
        idx = heading_index - offset
        if idx < 0 or _HEADING_RE.match(lines[idx]):
            return None
        match = _ANCHOR_RE.match(lines[idx].strip())
        if match:
            return match.group(1).split(",")[0].strip()
    return None


def _section_bounds(lines: list[str]) -> list[tuple[int, int]]:
    """Return (heading_index, end_index) for every level-2 heading; a section ends at the next heading of level <= 2."""
    starts = [i for i, ln in enumerate(lines) if (m := _HEADING_RE.match(ln)) and len(m.group(1)) == 2]
    bounds: list[tuple[int, int]] = []
    for start in starts:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            m = _HEADING_RE.match(lines[j])
            if m and len(m.group(1)) <= 2:
                end = j
                break
        bounds.append((start, end))
    return bounds


def _trim_trailing_anchors(body: list[str]) -> list[str]:
    """Drop anchor/blank lines at the end of a body (they belong to the next heading)."""
    end = len(body)
    while end > 0 and (not body[end - 1].strip() or _ANCHOR_RE.match(body[end - 1].strip())):
        end -= 1
    return body[:end]


def parse_adoc(text: str, file_stem: str) -> list[Doc]:
    """Parse one LinuxCNC AsciiDoc file into documents, one per level-2 section (pure function)."""
    lines = _strip_comment_blocks(text.splitlines())
    docs: list[Doc] = []
    used_ids: dict[str, int] = {}
    for start, end in _section_bounds(lines):
        title = clean_title(_HEADING_RE.match(lines[start]).group(2))  # type: ignore[union-attr]
        body = clean_body(_trim_trailing_anchors(lines[start + 1:end]))
        if len(body) < MIN_TEXT_CHARS:
            continue
        anchor = _find_anchor(lines, start) or f"sec-{slugify(title)}"
        doc_id = _unique_id(f"linuxcnc:{anchor}", used_ids)
        docs.append(Doc(
            doc_id=doc_id,
            source="linuxcnc",
            title=title,
            codes=heading_codes(title),
            url=f"{PUBLIC_BASE_URL}{file_stem}.html#{anchor}",
            section_path=f"{file_stem} > {title}",
            text=f"{title}\n\n{body}",
        ))
    return docs


def _strip_comment_blocks(lines: list[str]) -> list[str]:
    """Remove `////` comment blocks before section detection so commented-out headings are ignored."""
    out: list[str] = []
    in_comment = False
    for line in lines:
        if line.strip().startswith("////"):
            in_comment = not in_comment
            continue
        if not in_comment:
            out.append(line)
    return out


def _unique_id(doc_id: str, used: dict[str, int]) -> str:
    """Return `doc_id`, or `doc_id-2`, `-3`... when it was already used (records the use in `used`)."""
    count = used.get(doc_id, 0) + 1
    used[doc_id] = count
    return doc_id if count == 1 else f"{doc_id}-{count}"


def load_all(raw_dir: Path) -> list[Doc]:
    """Parse every downloaded .adoc file in `raw_dir` (required files first, then optional, in fixed order)."""
    docs: list[Doc] = []
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        path = raw_dir / name
        if path.exists():
            docs.extend(parse_adoc(path.read_text(encoding="utf-8"), path.stem))
    return _dedupe_ids(docs)


def _dedupe_ids(docs: list[Doc]) -> list[Doc]:
    """Make doc ids unique across files by suffixing repeats with -2, -3 (SPEC 3.3)."""
    used: dict[str, int] = {}
    out: list[Doc] = []
    for doc in docs:
        new_id = _unique_id(doc.doc_id, used)
        out.append(doc if new_id == doc.doc_id else Doc(**{**doc.__dict__, "doc_id": new_id}))
    return out


def main() -> int:
    """Download (if needed) and parse the LinuxCNC sources; print the document count."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_dir = get_paths().raw / "linuxcnc"
    download_sources(raw_dir)
    docs = load_all(raw_dir)
    print(f"docs: {len(docs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
