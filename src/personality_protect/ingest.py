"""Ingest LinkedIn exports and local writing paths into the profile index."""

from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from personality_protect.config import ProfileConfig, ProfilePaths, load_config, save_config
from personality_protect.models import Piece, append_index

_DATE_RE = re.compile(
    r"(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})"
    r"|(?P<m2>\d{1,2})[/-](?P<d2>\d{1,2})[/-](?P<y2>\d{4})"
)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._chunks)


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def _parse_date(raw: str | None) -> tuple[str | None, int | None]:
    if not raw:
        return None, None
    raw = raw.strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S UTC",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M",
        "%d %b %Y",
        "%b %d, %Y",
        "%Y/%m/%d",
    ):
        try:
            dt = datetime.strptime(raw[: len(fmt) + 8].strip(), fmt)
            return dt.strftime("%Y-%m-%d"), dt.year
        except ValueError:
            continue
    m = _DATE_RE.search(raw)
    if m:
        if m.group("y"):
            y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
        else:
            y, mo, d = int(m.group("y2")), int(m.group("m2")), int(m.group("d2"))
        try:
            dt = datetime(y, mo, d)
            return dt.strftime("%Y-%m-%d"), dt.year
        except ValueError:
            return None, None
    # Year-only fallback
    ym = re.search(r"\b(19|20)\d{2}\b", raw)
    if ym:
        year = int(ym.group(0))
        return f"{year}-01-01", year
    return None, None


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return _clean_text(re.sub(r"<[^>]+>", " ", html))
    return _clean_text(parser.text())


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    # LinkedIn CSVs are often UTF-8 with BOM
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as fh:
                sample = fh.read(4096)
                fh.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.DictReader(fh, dialect=dialect)
                return [{k: (v or "") for k, v in row.items() if k is not None} for row in reader]
        except UnicodeDecodeError:
            continue
    return []


def _pick(row: dict[str, str], *names: str) -> str:
    lower = { (k or "").strip().lower(): v for k, v in row.items() }
    for name in names:
        if name.lower() in lower and lower[name.lower()].strip():
            return lower[name.lower()].strip()
    # fuzzy contains
    for key, val in lower.items():
        for name in names:
            if name.lower() in key and val.strip():
                return val.strip()
    return ""


def _maybe_unpack_zip(path: Path, cache_dir: Path) -> Path:
    """If path is a zip, unpack once into profile cache and return extract root."""
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        return path
    dest = cache_dir / f"linkedin_{_stable_id(str(path.resolve()), str(path.stat().st_mtime))}"
    marker = dest / ".unpacked"
    if marker.is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest)
    marker.write_text("ok\n", encoding="utf-8")
    return dest


def _find_linkedin_files(root: Path) -> dict[str, list[Path]]:
    """Locate Shares*/Comments*/Articles* under an export tree."""
    found: dict[str, list[Path]] = {"shares": [], "comments": [], "articles": []}
    if not root.exists():
        return found
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        stem = p.stem.lower()
        if name.endswith(".csv") and ("share" in stem or stem.startswith("share")):
            found["shares"].append(p)
        elif name.endswith(".csv") and "comment" in stem:
            found["comments"].append(p)
        elif "article" in stem and name.endswith((".csv", ".html", ".htm")):
            found["articles"].append(p)
        elif p.parent.name.lower().startswith("article") and name.endswith((".html", ".htm", ".txt", ".md")):
            found["articles"].append(p)
    return found


def ingest_linkedin(export_path: Path, paths: ProfilePaths) -> list[Piece]:
    """Parse LinkedIn export directory or zip into pieces (read in place / cache unpack)."""
    export_path = export_path.expanduser().resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"LinkedIn export not found: {export_path}")

    root = _maybe_unpack_zip(export_path, paths.cache_dir)
    files = _find_linkedin_files(root)
    pieces: list[Piece] = []

    for csv_path in files["shares"]:
        for row in _read_csv_rows(csv_path):
            text = _clean_text(
                _pick(row, "ShareCommentary", "Commentary", "Share Commentary", "text", "Body", "Content")
            )
            if not text:
                continue
            date_raw = _pick(row, "Date", "SharedDate", "Created Date", "Time", "Timestamp")
            date, year = _parse_date(date_raw)
            pid = _stable_id("linkedin_post", text[:200], date or "", str(csv_path))
            pieces.append(
                Piece(
                    id=pid,
                    source="linkedin_post",
                    text=text,
                    path=str(csv_path),
                    date=date,
                    year=year,
                    title=_pick(row, "ShareLink", "Url", "URL")[:120],
                    meta={"file": csv_path.name},
                )
            )

    for csv_path in files["comments"]:
        for row in _read_csv_rows(csv_path):
            text = _clean_text(_pick(row, "Message", "Comments", "Comment", "text", "Body", "Content"))
            if not text:
                continue
            date_raw = _pick(row, "Date", "Created Date", "Time", "Timestamp")
            date, year = _parse_date(date_raw)
            pid = _stable_id("linkedin_comment", text[:200], date or "", str(csv_path))
            pieces.append(
                Piece(
                    id=pid,
                    source="linkedin_comment",
                    text=text,
                    path=str(csv_path),
                    date=date,
                    year=year,
                    meta={"file": csv_path.name},
                )
            )

    for art_path in files["articles"]:
        if art_path.suffix.lower() == ".csv":
            for row in _read_csv_rows(art_path):
                text = _clean_text(
                    _pick(row, "Content", "Article Content", "Body", "text", "ShareCommentary")
                )
                if art_path.suffix.lower() in {".html", ".htm"} or "<" in text[:50]:
                    text = _html_to_text(text)
                if not text:
                    continue
                date_raw = _pick(row, "Date", "Published At", "Created Date")
                date, year = _parse_date(date_raw)
                title = _pick(row, "Title", "Headline", "Article Title")
                pid = _stable_id("linkedin_article", text[:200], date or "", title)
                pieces.append(
                    Piece(
                        id=pid,
                        source="linkedin_article",
                        text=text,
                        path=str(art_path),
                        date=date,
                        year=year,
                        title=title,
                        meta={"file": art_path.name},
                    )
                )
        else:
            raw = art_path.read_text(encoding="utf-8", errors="replace")
            text = _html_to_text(raw) if art_path.suffix.lower() in {".html", ".htm"} else _clean_text(raw)
            if not text:
                continue
            # Try date from filename or mtime
            date, year = _parse_date(art_path.stem)
            if date is None:
                mtime = datetime.fromtimestamp(art_path.stat().st_mtime)
                date, year = mtime.strftime("%Y-%m-%d"), mtime.year
            pid = _stable_id("linkedin_article", text[:200], str(art_path))
            pieces.append(
                Piece(
                    id=pid,
                    source="linkedin_article",
                    text=text,
                    path=str(art_path),
                    date=date,
                    year=year,
                    title=art_path.stem,
                    meta={"file": art_path.name},
                )
            )

    return pieces


_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".eml",
    ".html",
    ".htm",
    ".org",
    ".text",
}


def _guess_source(path: Path, hint: str | None) -> str:
    if hint:
        return hint
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "email" in name or "email" in parent or path.suffix.lower() == ".eml":
        return "email"
    if "note" in name or "note" in parent or "obsidian" in parent:
        return "note"
    return "doc"


def ingest_local_paths(
    targets: Iterable[Path],
    *,
    source_hint: str | None = None,
    recursive: bool = True,
) -> list[Piece]:
    """Read local emails/docs/notes in place (no copy)."""
    pieces: list[Piece] = []
    files: list[Path] = []
    for target in targets:
        target = target.expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        if target.is_file():
            files.append(target)
        elif recursive:
            for p in target.rglob("*"):
                if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES:
                    files.append(p)
        else:
            for p in target.iterdir():
                if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES:
                    files.append(p)

    for path in sorted(set(files)):
        raw = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() in {".html", ".htm"}:
            text = _html_to_text(raw)
        elif path.suffix.lower() == ".eml":
            # Strip crude headers
            if "\n\n" in raw:
                text = _clean_text(raw.split("\n\n", 1)[1])
            else:
                text = _clean_text(raw)
        else:
            text = _clean_text(raw)
        if not text:
            continue
        date, year = _parse_date(path.stem)
        if date is None:
            # Look for a date line near the top
            for line in text.splitlines()[:5]:
                date, year = _parse_date(line)
                if date:
                    break
        if date is None:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            date, year = mtime.strftime("%Y-%m-%d"), mtime.year
        source = _guess_source(path, source_hint)
        pid = _stable_id(source, str(path), text[:120])
        pieces.append(
            Piece(
                id=pid,
                source=source,
                text=text,
                path=str(path),
                date=date,
                year=year,
                title=path.stem,
                meta={"suffix": path.suffix.lower()},
            )
        )
    return pieces


def register_source(config: ProfileConfig, kind: str, path: str) -> ProfileConfig:
    entry = {"kind": kind, "path": path}
    # de-dupe
    for existing in config.sources:
        if existing.get("kind") == kind and existing.get("path") == path:
            return config
    config.sources.append(entry)
    return config


def run_ingest(
    paths: ProfilePaths,
    *,
    linkedin: Path | None = None,
    local: list[Path] | None = None,
    source_hint: str | None = None,
) -> tuple[int, list[Piece]]:
    """Ingest sources into the profile index. Returns (added_count, all_new_pieces)."""
    config = load_config(paths)
    new_pieces: list[Piece] = []

    if linkedin is not None:
        lp = linkedin.expanduser().resolve()
        batch = ingest_linkedin(lp, paths)
        new_pieces.extend(batch)
        config = register_source(config, "linkedin", str(lp))

    if local:
        batch = ingest_local_paths(local, source_hint=source_hint)
        new_pieces.extend(batch)
        for p in local:
            config = register_source(config, source_hint or "local", str(p.expanduser().resolve()))

    added = append_index(paths.index_path, new_pieces)
    save_config(paths, config)
    return added, new_pieces
