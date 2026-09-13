# SPDX-License-Identifier: Apache-2.0
"""Present indexed paths at the network-share location users actually know.

Search results carry ``file://`` URIs built from the path the *indexer*
walked. When that path is a mount point on the machine running ChunkSilo, it
means nothing to a client on another machine. The ``shares`` configuration
states where each mounted tree really lives (``prefix`` -> ``unc``), so
results for covered files can name the share itself: an ``smb://`` URI plus
the Windows UNC form of the same location.

Everything here is pure string work over configuration: like the rest of the
result-formatting path it must never touch the filesystem (AGENTS.md, Search
Invariants). The mapping is matched component-wise after ``normpath``, so
``..`` cannot carry one share's mapping across into another's.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShareMapping:
    """One local-mount -> share mapping from the ``shares`` configuration."""

    prefix: str
    prefix_parts: tuple[str, ...]
    host: str
    share_parts: tuple[str, ...]


def _path_parts(path: str) -> list[str]:
    """Path components after normalisation, with empties and ``.`` dropped.

    normpath collapses ``.`` and ``..`` first, so a path cannot climb out of
    its share and still match. Any ``..`` that survives (a relative path
    climbing above its own root) is kept as a component precisely so it cannot
    match a real prefix.
    """
    return [
        part for part in os.path.normpath(path).split(os.sep) if part and part != "."
    ]


def build_share_mappings(config: dict[str, Any]) -> tuple[ShareMapping, ...]:
    """Mappings from the ``shares`` list, longest prefix first.

    Invalid entries are skipped with a warning rather than raised: a typo must
    degrade that entry's URIs to the ``file://`` fallback, never break search.
    """
    mappings: list[ShareMapping] = []
    for rule in config.get("shares") or []:
        if not isinstance(rule, dict):
            continue
        prefix = str(rule.get("prefix") or "").strip()
        unc = str(rule.get("unc") or "").strip().replace("\\", "/")
        prefix_parts = tuple(_path_parts(prefix)) if prefix else ()
        unc_parts = [part for part in unc.split("/") if part]
        if (
            not os.path.isabs(prefix)
            or not prefix_parts  # a "/" prefix would map the whole filesystem
            or not unc.startswith("//")
            or len(unc_parts) < 2  # need at least //server/share
        ):
            logger.warning(
                "Ignoring shares entry: prefix must be an absolute path and unc "
                "must look like //server/share (prefix=%r, unc=%r)",
                rule.get("prefix"), rule.get("unc"),
            )
            continue
        mappings.append(
            ShareMapping(
                prefix=os.path.normpath(prefix),
                prefix_parts=prefix_parts,
                host=unc_parts[0],
                share_parts=tuple(unc_parts[1:]),
            )
        )
    mappings.sort(key=lambda mapping: len(mapping.prefix_parts), reverse=True)
    return tuple(mappings)


def to_share_uris(
    normalized_path: str, mappings: tuple[ShareMapping, ...]
) -> tuple[str, str] | None:
    """``(smb:// URI, UNC path)`` for a covered path, or None.

    The ``smb://`` form percent-encodes each segment (spaces, ``#`` and ``?``
    would silently truncate the URI); the UNC form is raw, because Windows
    opens ``\\\\server\\share\\...`` verbatim and does not understand percent
    escapes.
    """
    if not mappings:
        return None
    parts = _path_parts(normalized_path)
    for mapping in mappings:
        if parts[: len(mapping.prefix_parts)] == list(mapping.prefix_parts):
            segments = list(mapping.share_parts) + parts[len(mapping.prefix_parts):]
            smb = "smb://" + mapping.host + "/" + "/".join(
                quote(segment, safe="") for segment in segments
            )
            unc = "\\\\" + mapping.host + "\\" + "\\".join(segments)
            return smb, unc
    return None
