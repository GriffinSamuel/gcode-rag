"""Split documents into overlapping word-bounded chunks (SPEC 3.4)."""
from __future__ import annotations

import re

from gcoderag.ingest import Chunk, Doc

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TITLE_ONLY_WORDS = 20  # a chunk this small (typically just the title) always absorbs the next unit


def _words(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


def _window_split(text: str, size: int) -> list[str]:
    """Split text into consecutive, non-overlapping windows of `size` words (overlap is added when packing)."""
    words = text.split()
    size = max(1, size)
    return [" ".join(words[start:start + size]) for start in range(0, len(words), size)]


def split_long_paragraph(paragraph: str, target_words: int, max_words: int, overlap_words: int) -> list[str]:
    """Split a paragraph longer than `max_words` on sentence boundaries (then words), with overlap between pieces."""
    units: list[str] = []
    for sentence in _SENTENCE_RE.split(paragraph):
        if _words(sentence) > max_words:
            units.extend(_window_split(sentence, target_words - overlap_words))
        elif sentence.strip():
            units.append(sentence.strip())
    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and _words(" ".join(current + [unit])) > target_words:
            pieces.append(" ".join(current))
            tail = " ".join(pieces[-1].split()[-overlap_words:]) if overlap_words else ""
            current = [tail] if tail else []
        current.append(unit)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_doc(doc: Doc, target_words: int = 250, max_words: int = 400, overlap_words: int = 40) -> list[Chunk]:
    """Chunk one document: pack paragraphs up to `target_words`; split paragraphs over `max_words` with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if _words(paragraph) > max_words:
            units.extend(split_long_paragraph(paragraph, target_words, max_words, overlap_words))
        else:
            units.append(paragraph)
    texts: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and _words("\n\n".join(current + [unit])) > target_words:
            texts.append("\n\n".join(current))
            current = []
        current.append(unit)
    if current:
        texts.append("\n\n".join(current))
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{index}",
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            codes=list(doc.codes),
            url=doc.url,
            chunk_index=index,
            text=text,
        )
        for index, text in enumerate(texts)
    ]


def chunk_docs(docs: list[Doc], **kwargs: int) -> list[Chunk]:
    """Chunk every document in order."""
    return [chunk for doc in docs for chunk in chunk_doc(doc, **kwargs)]
