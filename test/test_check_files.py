#!/usr/bin/env python3
"""Tests for the --check-files dry-run diagnostics.

Covers the reason-returning matchers, the ScanEvent callback on
iter_candidates, filecheck.run_check verdicts, and the CLI wiring.
The check must be strictly read-only: it must never create the state DB
or write anywhere.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chunksilo import filecheck
from chunksilo.index import (
    DirectoryConfig,
    FileInfo,
    IngestionState,
    LocalFileSystemSource,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(tmp_path, **overrides):
    """Create a LocalFileSystemSource pointing at tmp_path."""
    cfg = DirectoryConfig(path=tmp_path, **overrides)
    return LocalFileSystemSource(cfg)


def _config_for(tmp_path, dirs, **indexing_extra):
    """Minimal config dict for run_check."""
    indexing = {"directories": dirs}
    indexing.update(indexing_extra)
    return {
        "storage": {"storage_dir": str(tmp_path / "storage")},
        "indexing": indexing,
    }


def _verdict_by_name(report, name):
    """The FileVerdict whose path ends with name, across all directories."""
    for dir_report in report.directories:
        for verdict in dir_report.files:
            if Path(verdict.path).name == name:
                return verdict
    raise AssertionError(f"no verdict for {name!r} in report")


# ===========================================================================
# TestMatchDecision - direct coverage of the reason-returning matchers
# ===========================================================================


class TestMatchDecision:
    def test_dir_component_exclude_returns_pattern(self, tmp_path):
        (tmp_path / ".git").mkdir()
        source = _make_source(tmp_path, exclude=["**/.git/**"], include=[])
        matched, pattern = source._match_decision(tmp_path / ".git" / "x.pdf")
        assert matched is False
        assert pattern == "**/.git/**"

    def test_right_anchored_exclude_returns_pattern(self, tmp_path):
        source = _make_source(tmp_path, exclude=["*.tmp"], include=[])
        matched, pattern = source._match_decision(tmp_path / "a.tmp")
        assert (matched, pattern) == (False, "*.tmp")

    def test_exact_name_exclude_returns_pattern(self, tmp_path):
        source = _make_source(tmp_path, exclude=[".DS_Store"], include=[])
        matched, pattern = source._match_decision(tmp_path / ".DS_Store")
        assert (matched, pattern) == (False, ".DS_Store")

    def test_include_hit_returns_include_pattern(self, tmp_path):
        source = _make_source(tmp_path, exclude=[], include=["**/*.pdf"])
        matched, pattern = source._match_decision(tmp_path / "doc.pdf")
        assert (matched, pattern) == (True, "**/*.pdf")

    def test_include_miss_returns_none(self, tmp_path):
        source = _make_source(tmp_path, exclude=[], include=["**/*.pdf"])
        matched, pattern = source._match_decision(tmp_path / "doc.xlsx")
        assert (matched, pattern) == (False, None)

    def test_empty_include_passes_everything(self, tmp_path):
        source = _make_source(tmp_path, exclude=[], include=[])
        matched, pattern = source._match_decision(tmp_path / "anything.bin")
        assert (matched, pattern) == (True, None)

    def test_exclude_wins_over_include(self, tmp_path):
        source = _make_source(tmp_path, exclude=["*.pdf"], include=["**/*.pdf"])
        matched, pattern = source._match_decision(tmp_path / "doc.pdf")
        assert (matched, pattern) == (False, "*.pdf")

    def test_case_insensitive_by_default(self, tmp_path):
        source = _make_source(tmp_path, exclude=[], include=["**/*.PDF"])
        matched, pattern = source._match_decision(tmp_path / "doc.pdf")
        assert matched is True
        # Pattern is echoed as written in config, not lowercased
        assert pattern == "**/*.PDF"

    def test_case_sensitive_when_configured(self, tmp_path):
        source = _make_source(
            tmp_path, exclude=[], include=["**/*.PDF"], case_sensitive=True
        )
        matched, pattern = source._match_decision(tmp_path / "doc.pdf")
        assert (matched, pattern) == (False, None)

    def test_exclude_pattern_returned_in_original_case(self, tmp_path):
        source = _make_source(tmp_path, exclude=["*.TMP"], include=[])
        matched, pattern = source._match_decision(tmp_path / "x.tmp")
        assert (matched, pattern) == (False, "*.TMP")

    def test_directory_skip_pattern(self, tmp_path):
        source = _make_source(tmp_path, exclude=["**/node_modules/**", "*.tmp"])
        assert source._directory_skip_pattern("node_modules") == "**/node_modules/**"
        assert source._directory_skip_pattern("src") is None
        # Non-dir-form excludes never prune traversal
        assert source._directory_skip_pattern("a.tmp") is None

    def test_directory_skip_is_always_case_sensitive(self, tmp_path):
        """Pins existing behavior: dir pruning ignores case_sensitive=False."""
        source = _make_source(tmp_path, exclude=["**/node_modules/**"])
        assert source.config.case_sensitive is False
        assert source._directory_skip_pattern("NODE_MODULES") is None

    @pytest.mark.parametrize("name", ["doc.pdf", "doc.xlsx", "a.tmp", ".DS_Store"])
    def test_wrapper_parity(self, tmp_path, name):
        """Bool wrappers must agree with the reason-returning variants."""
        source = _make_source(
            tmp_path, exclude=["*.tmp", ".DS_Store"], include=["**/*.pdf"]
        )
        path = tmp_path / name
        assert source._matches_patterns(path) == source._match_decision(path)[0]

    @pytest.mark.parametrize("dir_name", ["node_modules", ".git", "src"])
    def test_dir_wrapper_parity(self, tmp_path, dir_name):
        source = _make_source(tmp_path, exclude=["**/node_modules/**", "**/.git/**"])
        assert source._should_skip_directory(dir_name) == (
            source._directory_skip_pattern(dir_name) is not None
        )


# ===========================================================================
# TestScanEvents - the on_event callback on iter_candidates
# ===========================================================================


def _make_tree(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "c.pdf").write_text("x")
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.xlsx").write_text("x")
    return _make_source(
        tmp_path, include=["**/*.pdf"], exclude=["**/.git/**"]
    )


class TestScanEvents:
    def test_candidates_identical_with_and_without_events(self, tmp_path):
        """Drift guard: the callback must not change what gets scanned."""
        source = _make_tree(tmp_path)
        plain = [c.path for c in source.iter_candidates()]
        events = []
        observed = [c.path for c in source.iter_candidates(on_event=events.append)]
        assert plain == observed

    def test_pruned_dir_and_excluded_file_events(self, tmp_path):
        source = _make_tree(tmp_path)
        events = []
        candidates = list(source.iter_candidates(on_event=events.append))

        assert [Path(c.path).name for c in candidates] == ["a.pdf"]
        by_kind = {e.kind: e for e in events}
        assert by_kind["pruned_dir"].path == str((tmp_path / ".git").absolute())
        assert by_kind["pruned_dir"].pattern == "**/.git/**"
        assert by_kind["excluded_file"].path == str((tmp_path / "b.xlsx").absolute())
        assert by_kind["excluded_file"].pattern is None  # no include matched

    def test_exclude_pattern_carried_on_event(self, tmp_path):
        (tmp_path / "x.tmp").write_text("x")
        source = _make_source(tmp_path, include=[], exclude=["*.tmp"])
        events = []
        list(source.iter_candidates(on_event=events.append))
        assert events[0].kind == "excluded_file"
        assert events[0].pattern == "*.tmp"

    def test_unreadable_file_event(self, tmp_path):
        (tmp_path / "a.pdf").write_text("x")
        source = _make_source(tmp_path, include=["**/*.pdf"], exclude=[])
        events = []
        with patch.object(LocalFileSystemSource, "_stat_candidate", return_value=None):
            candidates = list(source.iter_candidates(on_event=events.append))
        assert candidates == []
        assert [e.kind for e in events] == ["unreadable_file"]
        assert events[0].path == str((tmp_path / "a.pdf").absolute())

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can list a directory without read permission")
    def test_unreadable_dir_event(self, tmp_path):
        sub = tmp_path / "locked"
        sub.mkdir()
        (sub / "a.pdf").write_text("x")
        source = _make_source(tmp_path, include=["**/*.pdf"], exclude=[])
        events = []
        sub.chmod(0)
        try:
            candidates = list(source.iter_candidates(on_event=events.append))
        finally:
            sub.chmod(0o755)
        assert candidates == []
        assert [e.kind for e in events] == ["unreadable_dir"]
        assert events[0].path == str(sub.absolute())
        # The rest of the tree was walked: the scan is complete, only this
        # subtree is withheld from deletion.
        assert source.scan_status.complete
        assert list(source.unscanned_roots()) == [str(sub.absolute())]

    def test_non_recursive_events(self, tmp_path):
        (tmp_path / "a.pdf").write_text("x")
        (tmp_path / "b.xlsx").write_text("x")
        source = _make_source(
            tmp_path, include=["**/*.pdf"], exclude=[], recursive=False
        )
        events = []
        candidates = list(source.iter_candidates(on_event=events.append))
        assert [Path(c.path).name for c in candidates] == ["a.pdf"]
        assert [(e.kind, Path(e.path).name) for e in events] == [
            ("excluded_file", "b.xlsx")
        ]


# ===========================================================================
# TestRunCheck - the filecheck orchestration
# ===========================================================================


class TestRunCheck:
    def test_basic_verdicts_and_counts(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / ".git").mkdir()
        (docs / ".git" / "hidden.pdf").write_text("x")
        (docs / "a.pdf").write_text("x")
        (docs / "b.xlsx").write_text("x")
        config = _config_for(tmp_path, [
            {"path": str(docs), "include": ["**/*.pdf"], "exclude": ["**/.git/**"]}
        ])

        report = filecheck.run_check(config)

        assert report.error is None
        assert _verdict_by_name(report, "a.pdf").status == "index"
        assert "**/*.pdf" in _verdict_by_name(report, "a.pdf").reason
        assert _verdict_by_name(report, "b.xlsx").status == "not_included"
        dir_report = report.directories[0]
        assert [p.pattern for p in dir_report.pruned_dirs] == ["**/.git/**"]
        assert dir_report.counts["index"] == 1
        assert dir_report.counts["not_included"] == 1
        assert dir_report.counts["pruned_dirs"] == 1
        assert report.totals["index"] == 1
        assert filecheck.exit_code(report) == 0

    def test_review_copy_and_superseded(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        old = docs / "Spec A.docx"
        new = docs / "Spec B.docx"
        review = docs / "Spec review.docx"
        for f in (old, new, review):
            f.write_text("x")
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        config = _config_for(tmp_path, [str(docs)])

        report = filecheck.run_check(config)

        superseded = _verdict_by_name(report, "Spec A.docx")
        assert superseded.status == "superseded"
        assert superseded.latest_in_group == str(new.absolute())
        kept = _verdict_by_name(report, "Spec B.docx")
        assert kept.status == "index"
        assert kept.group == superseded.group
        copy = _verdict_by_name(report, "Spec review.docx")
        assert copy.status == "review_copy"
        assert copy.pattern is not None

    def test_index_superseded_keeps_older_revisions(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        old = docs / "Spec A.docx"
        new = docs / "Spec B.docx"
        for f in (old, new):
            f.write_text("x")
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        config = _config_for(
            tmp_path, [str(docs)], versioning={"index_superseded": True}
        )

        report = filecheck.run_check(config)

        older = _verdict_by_name(report, "Spec A.docx")
        assert older.status == "index"
        assert "stamped superseded" in older.reason

    def test_disabled_directory_not_scanned(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.pdf").write_text("x")
        config = _config_for(tmp_path, [{"path": str(docs), "enabled": False}])

        report = filecheck.run_check(config)

        dir_report = report.directories[0]
        assert dir_report.enabled is False
        assert dir_report.files == []
        assert filecheck.exit_code(report) == 0  # disabled is not an error

    def test_unavailable_directory_fails_exit_code(self, tmp_path):
        config = _config_for(tmp_path, [str(tmp_path / "missing")])

        report = filecheck.run_check(config)

        dir_report = report.directories[0]
        assert dir_report.available is False
        assert dir_report.unavailable_reason
        assert filecheck.exit_code(report) == 1

    def test_zero_directories_is_config_error(self, tmp_path):
        config = {
            "storage": {"storage_dir": str(tmp_path / "storage")},
            "indexing": {},
        }

        report = filecheck.run_check(config)

        assert report.error is not None
        assert filecheck.exit_code(report) == 1

    def test_duplicate_directory_reported(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.pdf").write_text("x")
        config = _config_for(tmp_path, [str(docs), str(docs)])

        report = filecheck.run_check(config)

        first, second = report.directories
        assert [v.status for v in first.files] == ["index"]
        assert [v.status for v in second.files] == ["duplicate"]
        assert str(docs.absolute()) in second.files[0].reason

    def test_non_recursive_skips_subdirectories(self, tmp_path):
        docs = tmp_path / "docs"
        (docs / "sub").mkdir(parents=True)
        (docs / "top.md").write_text("x")
        (docs / "sub" / "deep.md").write_text("x")
        config = _config_for(tmp_path, [{"path": str(docs), "recursive": False}])

        report = filecheck.run_check(config)

        names = [Path(v.path).name for v in report.directories[0].files]
        assert "top.md" in names
        assert "deep.md" not in names
        assert report.directories[0].recursive is False

    def test_state_db_absent_and_never_created(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.pdf").write_text("x")
        config = _config_for(tmp_path, [str(docs)])
        state_db = tmp_path / "storage" / "ingestion_state.db"

        report = filecheck.run_check(config)

        assert report.state_db_present is False
        assert _verdict_by_name(report, "a.pdf").indexed is None
        # Read-only invariant: the check must not create storage or the DB
        assert not state_db.exists()
        assert not (tmp_path / "storage").exists()

    def test_state_db_presence_annotation(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        indexed_file = docs / "old.pdf"
        new_file = docs / "new.pdf"
        indexed_file.write_text("x")
        new_file.write_text("x")
        config = _config_for(tmp_path, [str(docs)])

        (tmp_path / "storage").mkdir()
        state = IngestionState(tmp_path / "storage" / "ingestion_state.db")
        state.update_file_state(
            FileInfo(path=str(indexed_file.absolute()), hash="h", last_modified=0.0),
            ["doc1"],
        )

        report = filecheck.run_check(config)

        assert report.state_db_present is True
        assert _verdict_by_name(report, "old.pdf").indexed is True
        assert _verdict_by_name(report, "new.pdf").indexed is False

    def test_target_kept_file(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.pdf").write_text("x")
        config = _config_for(tmp_path, [str(docs)])

        report = filecheck.run_check(config, target=str(docs / "a.pdf"))

        assert report.target is not None
        assert report.target.status == "index"
        assert filecheck.exit_code(report) == 0

    def test_target_superseded_file(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        old = docs / "Spec A.docx"
        new = docs / "Spec B.docx"
        for f in (old, new):
            f.write_text("x")
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        config = _config_for(tmp_path, [str(docs)])

        report = filecheck.run_check(config, target=str(old))

        assert report.target.status == "superseded"
        assert report.target.latest_in_group == str(new.absolute())
        assert filecheck.exit_code(report) == 1

    def test_target_under_pruned_directory(self, tmp_path):
        docs = tmp_path / "docs"
        (docs / ".git").mkdir(parents=True)
        (docs / ".git" / "x.pdf").write_text("x")
        config = _config_for(tmp_path, [str(docs)])

        report = filecheck.run_check(config, target=str(docs / ".git" / "x.pdf"))

        assert report.target.status == "excluded"
        assert "pruned" in report.target.reason
        assert report.target.pattern == "**/.git/**"
        assert filecheck.exit_code(report) == 1

    def test_target_in_subdir_of_non_recursive_directory(self, tmp_path):
        docs = tmp_path / "docs"
        (docs / "sub").mkdir(parents=True)
        (docs / "sub" / "deep.md").write_text("x")
        config = _config_for(tmp_path, [{"path": str(docs), "recursive": False}])

        report = filecheck.run_check(config, target=str(docs / "sub" / "deep.md"))

        assert report.target.status == "excluded"
        assert "recursive: false" in report.target.reason

    def test_target_outside_configured_directories(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        config = _config_for(tmp_path, [str(docs)])

        report = filecheck.run_check(config, target=str(tmp_path / "elsewhere.pdf"))

        assert report.target.status == "excluded"
        assert "not under any configured" in report.target.reason
        assert filecheck.exit_code(report) == 1

    def test_target_nonexistent_file_under_directory(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        config = _config_for(tmp_path, [str(docs)])

        report = filecheck.run_check(config, target=str(docs / "ghost.pdf"))

        assert "does not exist" in report.target.reason
        assert filecheck.exit_code(report) == 1


# ===========================================================================
# TestCheckFilesCli - CLI wiring
# ===========================================================================


def _write_config(tmp_path, docs):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"storage:\n  storage_dir: \"{tmp_path / 'storage'}\"\n"
        f"indexing:\n  directories:\n    - \"{docs}\"\n"
    )
    return config_file


class TestCheckFilesCli:
    def _run_main(self, args):
        from chunksilo.cli import main

        with patch("sys.argv", ["chunksilo"] + args):
            with pytest.raises(SystemExit) as exc_info:
                main()
        return exc_info.value.code

    def test_full_scan_exit_zero(self, tmp_path, capsys):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("x")
        config_file = _write_config(tmp_path, docs)

        code = self._run_main(["--check-files", "--config", str(config_file)])

        assert code == 0
        out = capsys.readouterr().out
        assert "would index" in out
        assert "a.md" in out

    def test_json_output(self, tmp_path, capsys):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("x")
        config_file = _write_config(tmp_path, docs)

        code = self._run_main(
            ["--check-files", "--config", str(config_file), "--json"]
        )

        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["totals"]["index"] == 1
        assert data["directories"][0]["files"][0]["status"] == "index"

    def test_skipped_target_exits_one(self, tmp_path, capsys):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "notes.xlsx").write_text("x")
        config_file = _write_config(tmp_path, docs)

        code = self._run_main([
            "--check-files", str(docs / "notes.xlsx"), "--config", str(config_file)
        ])

        assert code == 1
        assert "not included" in capsys.readouterr().out

    def test_unavailable_directory_exits_one(self, tmp_path):
        config_file = _write_config(tmp_path, tmp_path / "missing")

        code = self._run_main(["--check-files", "--config", str(config_file)])

        assert code == 1

    def test_query_conflict_is_an_error(self, tmp_path):
        config_file = _write_config(tmp_path, tmp_path)

        code = self._run_main(
            ["some query", "--check-files", "--config", str(config_file)]
        )

        assert code == 2  # argparse error
