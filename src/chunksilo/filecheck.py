# SPDX-License-Identifier: Apache-2.0
"""Dry-run of the indexing scan: same filters, no index, no writes.

Runs the exact phase-1 scan the indexer uses (directory availability,
timeout-guarded walk, directory pruning, glob matching, stat, cross-source
dedup, revision partitioning) and reports a per-file verdict with the
deciding reason. Never loads models, hashes files, or writes anywhere -
the state database is only ever opened read-only.
"""
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from . import revisions

# Verdict statuses, in the order they appear in reports.
STATUSES = (
    "index",
    "superseded",
    "review_copy",
    "excluded",
    "not_included",
    "unreadable",
    "duplicate",
)

_STATUS_LABELS = {
    "index": "would index",
    "superseded": "superseded",
    "review_copy": "review copy",
    "excluded": "excluded",
    "not_included": "not included",
    "unreadable": "unreadable",
    "duplicate": "duplicate",
}


@dataclass
class FileVerdict:
    path: str
    status: str  # one of STATUSES
    reason: str
    pattern: str | None = None  # glob or review-copy regex that decided
    group: str | None = None  # revision group key
    latest_in_group: str | None = None
    indexed: bool | None = None  # in the state DB (None = DB absent or n/a)


@dataclass
class PrunedDir:
    path: str
    pattern: str


@dataclass
class DirectoryReport:
    path: str  # absolute
    enabled: bool
    available: bool
    unavailable_reason: str | None = None
    recursive: bool = True
    scan_complete: bool = True
    scan_incomplete_reason: str | None = None
    pruned_dirs: list[PrunedDir] = field(default_factory=list)
    unreadable_dirs: list[str] = field(default_factory=list)  # absolute
    files: list[FileVerdict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class CheckReport:
    directories: list[DirectoryReport] = field(default_factory=list)
    totals: dict[str, int] = field(default_factory=dict)
    state_db: str = ""
    state_db_present: bool = False
    target: FileVerdict | None = None  # single-path mode
    error: str | None = None  # config-level failure


def run_check(config: dict[str, Any], target: str | None = None) -> CheckReport:
    """Scan all configured directories with the indexer's filters, no writes.

    The caller must have activated ``config`` process-wide (load_config plus
    index._config) so cfgload.get() inside the walk answers from the same
    file; the CLI handler does this.
    """
    from . import index as index_module

    state_db = Path(config["storage"]["storage_dir"]) / "ingestion_state.db"
    report = CheckReport(state_db=str(state_db))

    try:
        index_config = index_module._parse_index_config(config.get("indexing") or {})
    except ValueError as exc:
        report.error = str(exc)
        return report

    reports_by_entry: dict[int, DirectoryReport] = {}
    for dc in index_config.directories:
        dir_report = DirectoryReport(
            path=str(dc.path.absolute()),
            enabled=dc.enabled,
            available=False,
            recursive=dc.recursive,
        )
        if not dc.enabled:
            dir_report.unavailable_reason = "disabled in config (enabled: false)"
        reports_by_entry[id(dc)] = dir_report
        report.directories.append(dir_report)

    data_source = index_module.MultiDirectoryDataSource(index_config)

    for dc in data_source.unavailable_dirs:
        dir_report = reports_by_entry[id(dc)]
        dir_report.unavailable_reason = data_source.unscanned_roots().get(
            str(dc.path.absolute()), "directory unavailable"
        )

    # Phase 1 scan, source by source, with one shared seen-set replicating
    # MultiDirectoryDataSource.iter_candidates dedup (first source wins).
    seen: dict[str, str] = {}  # abs path -> source dir that saw it first
    candidates: list[Any] = []  # FileCandidate, in scan order
    verdict_report: dict[str, DirectoryReport] = {}  # candidate path -> report
    source_of: dict[str, Any] = {}  # candidate path -> LocalFileSystemSource

    for source in data_source.sources:
        dir_report = reports_by_entry[id(source.config)]
        dir_report.available = True

        events: list[Any] = []
        for candidate in source.iter_candidates(on_event=events.append):
            if candidate.path in seen:
                dir_report.files.append(FileVerdict(
                    path=candidate.path,
                    status="duplicate",
                    reason=f"already scanned via {seen[candidate.path]}",
                ))
                continue
            seen[candidate.path] = candidate.source_dir
            candidates.append(candidate)
            verdict_report[candidate.path] = dir_report
            source_of[candidate.path] = source

        dir_report.scan_complete = source.scan_status.complete
        if not source.scan_status.complete:
            dir_report.scan_incomplete_reason = source.scan_status.reason

        for event in events:
            if event.kind == "pruned_dir":
                dir_report.pruned_dirs.append(PrunedDir(event.path, event.pattern))
            elif event.kind == "excluded_file":
                if event.pattern is not None:
                    dir_report.files.append(FileVerdict(
                        path=event.path,
                        status="excluded",
                        reason=f"excluded by pattern {event.pattern!r}",
                        pattern=event.pattern,
                    ))
                else:
                    dir_report.files.append(FileVerdict(
                        path=event.path,
                        status="not_included",
                        reason="no include pattern matched",
                    ))
            elif event.kind == "unreadable_file":
                dir_report.files.append(FileVerdict(
                    path=event.path,
                    status="unreadable",
                    reason="could not stat (error or timeout); "
                           "would be skipped and its deletion withheld",
                ))
            elif event.kind == "unreadable_dir":
                dir_report.unreadable_dirs.append(event.path)

    # Revision partitioning across ALL sources at once, exactly like
    # build_index - doc-ID groups can span directories.
    policy = revisions.RevisionPolicy.from_config(config)
    grouping = revisions.partition(
        [(c.path, c.mtime) for c in candidates], policy
    )

    tracked, report.state_db_present = _read_state_db(state_db)

    for candidate in candidates:
        path = candidate.path
        dir_report = verdict_report[path]
        group = grouping.group_of.get(path)
        latest = grouping.latest_of_group.get(group) if group else None

        if path in grouping.review_copies:
            pattern = grouping.review_copies[path]
            dir_report.files.append(FileVerdict(
                path=path,
                status="review_copy",
                reason=f"review copy (filename matched {pattern!r}); never indexed",
                pattern=pattern,
            ))
            continue

        if path in grouping.superseded and not policy.index_superseded:
            dir_report.files.append(FileVerdict(
                path=path,
                status="superseded",
                reason=f"older revision; latest in group is {latest}",
                group=group,
                latest_in_group=latest,
            ))
            continue

        # Kept: re-derive the include pattern (pure string work, no I/O).
        _, pattern = source_of[path]._match_decision(Path(path))
        if path in grouping.superseded:
            reason = (
                "older revision, indexed and stamped superseded "
                f"(index_superseded: true); latest in group is {latest}"
            )
        elif pattern is not None:
            reason = f"matched include pattern {pattern!r}"
        else:
            reason = "no include patterns configured; everything passes"
        dir_report.files.append(FileVerdict(
            path=path,
            status="index",
            reason=reason,
            pattern=pattern,
            group=group,
            latest_in_group=latest if latest != path else None,
            indexed=(path in tracked) if report.state_db_present else None,
        ))

    for dir_report in report.directories:
        dir_report.files.sort(key=lambda v: v.path)
        dir_report.pruned_dirs.sort(key=lambda p: p.path)
        dir_report.unreadable_dirs.sort()
        dir_report.counts = {status: 0 for status in STATUSES}
        for verdict in dir_report.files:
            dir_report.counts[verdict.status] += 1
        dir_report.counts["pruned_dirs"] = len(dir_report.pruned_dirs)
        dir_report.counts["unreadable_dirs"] = len(dir_report.unreadable_dirs)

    report.totals = {status: 0 for status in STATUSES}
    report.totals["pruned_dirs"] = 0
    report.totals["unreadable_dirs"] = 0
    for dir_report in report.directories:
        for key, count in dir_report.counts.items():
            report.totals[key] += count

    if target is not None:
        report.target = _resolve_target(target, report, data_source)

    return report


def _read_state_db(state_db: Path) -> tuple[set[str], bool]:
    """Indexed paths from the state DB, strictly read-only; never creates it."""
    if not state_db.exists():
        return set(), False
    try:
        with sqlite3.connect(f"file:{state_db}?mode=ro", uri=True) as conn:
            return {row[0] for row in conn.execute("SELECT path FROM files")}, True
    except sqlite3.OperationalError:
        return set(), True


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _resolve_target(target, report, data_source):
    """Verdict for one path, using the full scan that already ran."""
    path = str(Path(target).expanduser().absolute())

    for dir_report in report.directories:
        for verdict in dir_report.files:
            if verdict.path == path:
                return verdict
        for pruned in dir_report.pruned_dirs:
            if _under(path, pruned.path):
                return FileVerdict(
                    path=path,
                    status="excluded",
                    reason=f"ancestor directory {pruned.path} pruned "
                           f"by pattern {pruned.pattern!r}",
                    pattern=pruned.pattern,
                )
        for unreadable_dir in dir_report.unreadable_dirs:
            if _under(path, unreadable_dir):
                return FileVerdict(
                    path=path,
                    status="unreadable",
                    reason=f"ancestor directory {unreadable_dir} could not be "
                           "listed; files under it are kept, not re-indexed",
                )

    for dir_report in report.directories:
        if not _under(path, dir_report.path):
            continue
        if not dir_report.enabled:
            return FileVerdict(
                path=path,
                status="excluded",
                reason=f"directory {dir_report.path} is disabled (enabled: false)",
            )
        if not dir_report.available:
            return FileVerdict(
                path=path,
                status="unreadable",
                reason=f"directory {dir_report.path} is unavailable "
                       f"({dir_report.unavailable_reason})",
            )
        if not dir_report.recursive and str(Path(path).parent) != dir_report.path:
            return FileVerdict(
                path=path,
                status="excluded",
                reason=f"directory {dir_report.path} is configured "
                       "recursive: false; only top-level files are scanned",
            )
        # Under an available directory but the scan produced no verdict:
        # explain from the matcher directly.
        for source in data_source.sources:
            if str(source.base_dir.absolute()) != dir_report.path:
                continue
            matched, pattern = source._match_decision(Path(path))
            exists = Path(path).is_file()
            if not matched and pattern is not None:
                reason = f"excluded by pattern {pattern!r}"
                status = "excluded"
            elif not matched:
                reason = "no include pattern matched"
                status = "not_included"
            else:
                status = "excluded"
                reason = "matches the patterns but the scan did not see it"
                if not dir_report.scan_complete:
                    reason += f" (scan incomplete: {dir_report.scan_incomplete_reason})"
            if not exists:
                reason += "; file does not exist"
            return FileVerdict(path=path, status=status, reason=reason,
                               pattern=pattern if not matched else None)

    return FileVerdict(
        path=path,
        status="excluded",
        reason="not under any configured indexing directory",
    )


def exit_code(report: CheckReport) -> int:
    """0 = clean (or target would be indexed); 1 = problem or target skipped."""
    if report.error:
        return 1
    if report.target is not None:
        return 0 if report.target.status == "index" else 1
    for dir_report in report.directories:
        if dir_report.enabled and (
            not dir_report.available or not dir_report.scan_complete
        ):
            return 1
    return 0


def to_json(report: CheckReport) -> dict[str, Any]:
    from dataclasses import asdict

    data = asdict(report)
    # asdict turns PrunedDir into {"path", "pattern"} dicts already.
    return data


def render_text(report: CheckReport, out: TextIO | None = None) -> None:
    if out is None:
        out = sys.stdout
    if report.error:
        print(f"Configuration error: {report.error}", file=out)
        return

    if report.target is not None:
        _render_target(report, out)
        return

    db_note = "present" if report.state_db_present else "not created yet"
    print(f"State DB: {report.state_db} ({db_note})", file=out)

    for dir_report in report.directories:
        print(file=out)
        print(f"{dir_report.path}", file=out)
        if not dir_report.enabled:
            print("  disabled in config (enabled: false) - not scanned", file=out)
            continue
        if not dir_report.available:
            print(f"  UNAVAILABLE: {dir_report.unavailable_reason}", file=out)
            continue
        mode = "recursive" if dir_report.recursive else "top-level files only"
        print(f"  scanned ({mode})", file=out)
        if not dir_report.scan_complete:
            print(f"  WARNING: scan incomplete - {dir_report.scan_incomplete_reason}; "
                  "files under unvisited subtrees are missing from this report",
                  file=out)
        for pruned in dir_report.pruned_dirs:
            print(f"  pruned dir    {pruned.path}  [by {pruned.pattern!r}]", file=out)
        for unreadable_dir in dir_report.unreadable_dirs:
            print(f"  WARNING: could not list {unreadable_dir}; files under it are "
                  "missing from this report and their deletion is withheld",
                  file=out)
        for verdict in dir_report.files:
            label = _STATUS_LABELS[verdict.status].ljust(13)
            suffix = ""
            if verdict.status == "index":
                if verdict.indexed is True:
                    suffix = "  (indexed)"
                elif verdict.indexed is False:
                    suffix = "  (not indexed yet)"
            print(f"  {label} {verdict.path}  [{verdict.reason}]{suffix}", file=out)

    print(file=out)
    print(f"Summary: {_summary_line(report.totals)}", file=out)


def _render_target(report: CheckReport, out: TextIO) -> None:
    verdict = report.target
    print(f"Path: {verdict.path}", file=out)
    label = _STATUS_LABELS[verdict.status]
    print(f"Verdict: {label} - {verdict.reason}", file=out)
    if verdict.status == "index":
        if verdict.indexed is True:
            print("Index: already present in the state DB", file=out)
        elif verdict.indexed is False:
            print("Index: not indexed yet - run chunksilo --build-index", file=out)
    if verdict.group:
        members = [
            other
            for dir_report in report.directories
            for other in dir_report.files
            if other.group == verdict.group and other.path != verdict.path
        ]
        if members:
            print(f"Revision group: {verdict.group}", file=out)
            for member in members:
                print(f"  {_STATUS_LABELS[member.status].ljust(13)} {member.path}",
                      file=out)


def _summary_line(totals: dict[str, int]) -> str:
    parts = []
    names = {
        "index": "would index",
        "superseded": "superseded",
        "review_copy": "review copies",
        "excluded": "excluded",
        "not_included": "not matching any include",
        "unreadable": "unreadable",
        "duplicate": "duplicates",
        "pruned_dirs": "directories pruned",
        "unreadable_dirs": "directories unlistable",
    }
    for key, name in names.items():
        if totals.get(key):
            parts.append(f"{totals[key]} {name}")
    return ", ".join(parts) if parts else "nothing found"
