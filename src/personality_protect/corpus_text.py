"""Normalize raw corpus text (LinkedIn/CSV/article paste-ups) without rewriting voice.

Lives here rather than in ``sft.py`` because every consumer of the corpus needs
it, not just training: the retrieval index embeds and hands back the same text,
so an uncleaned exemplar puts CSS declarations from an article export straight
into the writing prompt.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_QUOTE_ARTIFACT_RE = re.compile(r'(^|\n)\s*"+\s*(\n|$)')
_TRAILING_ESCAPED_QUOTE_RE = re.compile(r'\\"+')
# LinkedIn article exports often paste Medium/Ghost CSS ahead of the body.
_CSS_RULE_RE = re.compile(
    r"(?ms)^[ \t]*[a-zA-Z_*#.][^{;\n]{0,160}\{[^{}]*\}[ \t]*\n?"
)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


class _HTMLTextExtractor(HTMLParser):
    """Pull text from HTML while keeping block-level paragraph breaks."""

    _BLOCK = frozenset(
        {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "blockquote",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        if tag == "br":
            self._chunks.append("\n")
        elif tag in self._BLOCK:
            self._chunks.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK:
            self._chunks.append("\n\n")

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _strip_html_css(text: str) -> str:
    """Remove embedded CSS rules and HTML tags from corpus/article paste-ups."""
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", "\n", text)
    prev = None
    while prev != text:
        prev = text
        text = _CSS_RULE_RE.sub("\n", text)
    if _HTML_TAG_RE.search(text):
        parser = _HTMLTextExtractor()
        try:
            parser.feed(text)
            extracted = parser.text()
            if extracted.strip():
                text = extracted
        except Exception:
            text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def normalize_corpus_text(text: str) -> str:
    """Fix LinkedIn/CSV quote wrapping and strip HTML/CSS without rewriting voice."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_html_css(text)
    text = _TRAILING_ESCAPED_QUOTE_RE.sub('"', text)
    lines_out: list[str] = []
    blank_pending = False
    for raw_line in text.split("\n"):
        s = raw_line.strip()
        if not s or set(s) <= {'"'}:
            blank_pending = True
            continue
        # Unwrap CSV paragraph quotes: "paragraph." / paragraph."
        if s.startswith('"'):
            s = s[1:]
        if s.endswith('"') and not s.endswith('\\"'):
            s = s[:-1]
        s = s.strip()
        if not s:
            blank_pending = True
            continue
        if blank_pending and lines_out:
            lines_out.append("")
        blank_pending = False
        lines_out.append(s)
    text = "\n".join(lines_out)
    text = _QUOTE_ARTIFACT_RE.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
