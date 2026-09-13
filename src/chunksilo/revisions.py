#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Revision grouping for messy document version control.

Documents in the indexed directories carry ad-hoc revision markers - a
revision-named directory (".../PB3/Spec.docx") or a token in the filename
("Spec PB3.docx", "Spec_v2.pdf") - with no consistent convention, plus copies
made to collect review comments. This module recognises files that are
revisions of one document, groups them, and picks the newest member of each
group by file modification time so the indexer can skip the rest.

Revision labels are never ordered: tokens are only stripped from names to form
the group key. Which member is "latest" is decided by mtime alone.

Pure string/path logic: no I/O, no config reads at import time.
"""
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, NamedTuple

from .cfgload import DEFAULT_REVIEW_COPY_PATTERNS

# Revision-token alternatives shared by filenames and directory names. Case
# rules are encoded per alternative, so nothing here compiles with IGNORECASE.
#
# Filename suffixes (stripped from the end of the stem, separator required):
_FILENAME_REV_ALTS = [
    # PA1, pb3, PC12 - preliminary marker. The digit is REQUIRED in filenames
    # so initials like "Spec PM.docx" are not stripped.
    r"[Pp][A-Za-z]\d{1,2}",
    # A, B, C2 - single UPPERCASE letter only, so "Spec a.docx" and
    # "Spec USA.docx" stay untouched.
    r"[A-Z]\d{0,2}",
    # v2, V1.0, v1_2
    r"[vV]\d+(?:[._]\d+){0,2}",
    # rev2, Revision 3, ver_1, utg 2, utgåva B
    r"(?:[Rr][Ee][Vv](?:ision)?|[Vv][Ee][Rr](?:sion)?|[Uu][Tt][Gg](?:[åa]va)?)"
    r"[\s._\-]*(?:\d{1,3}|[A-Za-z]\d{0,2})",
]

# Whole directory components. Anchored: "B" and "PB3" are revision dirs,
# "Backup", "B2B" and "2024" are not. Single letters are accepted in either
# case here (a dir named "b" is a revision dir; a file "Spec b.docx" is not).
_DIR_REV_ALTS = [
    # 1, 2, 3 - but not \d{4}: years are never revision dirs
    r"\d{1,2}",
    r"[A-Za-z]",
    # PA, PB1 - the digit is optional for directories
    r"[Pp][A-Za-z]\d{0,2}",
    r"[vV]\d+(?:[._]\d+){0,2}",
    r"(?:[Rr][Ee][Vv](?:ision)?|[Vv][Ee][Rr](?:sion)?|[Uu][Tt][Gg](?:[åa]va)?)"
    r"[\s._\-]*(?:\d{1,3}|[A-Za-z]\d{0,2})?",
]

# A trailing single letter after one of these words names a sibling document
# ("Appendix B"), not a revision, so the bare-letter alternative must not
# strip it.
_SINGLE_LETTER_GUARD = re.compile(
    r"(?i:\b(?:appendix|annex|attachment|exhibit|schedule|part|section"
    r"|chapter|figure|table|bilaga|del|kapitel|avsnitt|tabell|figur))[\s._\-]*$"
)

_BARE_LETTER_TOKEN = re.compile(r"[A-Z]\d{0,2}$")


def _nfc(text: str) -> str:
    # SMB/macOS mounts can deliver "å" decomposed (NFD), which would break
    # every pattern containing it.
    return unicodedata.normalize("NFC", text)


def _compile_alts(alts: list[str], extra: tuple[str, ...], template: str) -> re.Pattern:
    parts = list(alts) + [f"(?i:{frag})" for frag in extra]
    return re.compile(template.format(alts="|".join(parts)))


def _compile_user_patterns(patterns: Iterable[str], where: str) -> tuple[re.Pattern, ...]:
    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error as e:
            raise ValueError(f"invalid regex in {where}: {pat!r} ({e})") from e
    return tuple(compiled)


class Member(NamedTuple):
    path: str
    mtime: float


@dataclass
class GroupingResult:
    """Outcome of partitioning a scanned file list."""

    keep: list[str] = field(default_factory=list)  # input order preserved
    superseded: set[str] = field(default_factory=set)
    review_copies: dict[str, str] = field(default_factory=dict)  # path -> pattern
    group_of: dict[str, str] = field(default_factory=dict)
    doc_id_of: dict[str, str] = field(default_factory=dict)
    latest_of_group: dict[str, str] = field(default_factory=dict)  # key -> path


@dataclass(frozen=True)
class RevisionPolicy:
    enabled: bool = True
    index_superseded: bool = False
    doc_id_patterns: tuple[re.Pattern, ...] = ()
    review_copy_patterns: tuple[re.Pattern, ...] = ()
    _filename_suffix: re.Pattern = None  # type: ignore[assignment]
    _dir_component: re.Pattern = None  # type: ignore[assignment]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RevisionPolicy":
        cfg = (config.get("indexing") or {}).get("versioning") or {}
        extra = tuple(cfg.get("extra_revision_tokens") or ())
        # Validate the extra fragments individually before folding them into
        # the combined grammar, so a bad one is named in the error.
        _compile_user_patterns(extra, "indexing.versioning.extra_revision_tokens")
        review = cfg.get("review_copy_patterns")
        if review is None:
            review = DEFAULT_REVIEW_COPY_PATTERNS
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            index_superseded=bool(cfg.get("index_superseded", False)),
            doc_id_patterns=_compile_user_patterns(
                cfg.get("doc_id_patterns") or (), "indexing.versioning.doc_id_patterns"
            ),
            review_copy_patterns=_compile_user_patterns(
                review, "indexing.versioning.review_copy_patterns"
            ),
            _filename_suffix=_compile_alts(
                _FILENAME_REV_ALTS, extra, r"[\s._\-]+\(?({alts})\)?$"
            ),
            _dir_component=_compile_alts(_DIR_REV_ALTS, extra, r"^\(?(?:{alts})\)?$"),
        )

    def review_copy_match(self, path: str) -> str | None:
        """Pattern that marks this file as a review copy, or None."""
        stem = _nfc(PurePath(path).stem)
        for pattern in self.review_copy_patterns:
            if pattern.search(stem):
                return pattern.pattern
        return None

    def extract_doc_id(self, path: str) -> str | None:
        """Document ID from the full path, tried patterns in order."""
        normalized = _nfc(path)
        for pattern in self.doc_id_patterns:
            m = pattern.search(normalized)
            if m:
                return (m.group(1) if pattern.groups else m.group(0)).upper()
        return None

    def is_revision_dir(self, component: str) -> bool:
        return bool(self._dir_component.match(_nfc(component)))

    def strip_revision_suffixes(self, stem: str) -> str:
        stem = _nfc(stem)
        while True:
            m = self._filename_suffix.search(stem)
            if not m:
                break
            if _BARE_LETTER_TOKEN.fullmatch(m.group(1)) and _SINGLE_LETTER_GUARD.search(
                stem[: m.start()]
            ):
                break
            stem = stem[: m.start()]
        # A stem that IS a revision token ("PB3.docx" inside a per-document
        # folder) contributes nothing to identity.
        if self._dir_component.match(stem):
            return ""
        return stem.rstrip(" ._-")

    def group_key(self, path: str) -> str:
        doc_id = self.extract_doc_id(path)
        if doc_id:
            return f"id:{doc_id}"
        pure = PurePath(path)
        parent = "/".join(
            part for part in pure.parent.parts if not self.is_revision_dir(part)
        )
        stem = self.strip_revision_suffixes(pure.stem)
        ext = pure.suffix.casefold()
        if ext == ".doc":  # the indexer converts .doc to .docx anyway
            ext = ".docx"
        return f"path:{_nfc(parent).casefold()}/{_nfc(stem).casefold()}{ext}"


def partition(files: Iterable[tuple[str, float]], policy: RevisionPolicy) -> GroupingResult:
    """Split a scanned (path, mtime) list into keepers and skipped files.

    The latest member of each group is the one with the newest mtime (path
    string breaks ties, purely for run-to-run determinism). With
    ``policy.index_superseded`` the older members stay in ``keep`` but are
    still listed in ``superseded`` so the indexer can stamp them.
    """
    result = GroupingResult()
    if not policy.enabled:
        result.keep = [path for path, _ in files]
        return result

    groups: dict[str, list[Member]] = {}
    ordered: list[str] = []
    for path, mtime in files:
        match = policy.review_copy_match(path)
        if match is not None:
            result.review_copies[path] = match
            continue
        ordered.append(path)
        key = policy.group_key(path)
        result.group_of[path] = key
        doc_id = policy.extract_doc_id(path)
        if doc_id:
            result.doc_id_of[path] = doc_id
        groups.setdefault(key, []).append(Member(path, mtime))

    for key, members in groups.items():
        latest = max(members, key=lambda m: (m.mtime, m.path))
        result.latest_of_group[key] = latest.path
        for member in members:
            if member.path != latest.path:
                result.superseded.add(member.path)

    if policy.index_superseded:
        result.keep = ordered
    else:
        result.keep = [p for p in ordered if p not in result.superseded]
    return result
