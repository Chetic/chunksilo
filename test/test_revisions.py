#!/usr/bin/env python3
"""Unit tests for revision grouping (revisions.py).

Pure string/path logic: no filesystem, no config files.
"""

import unicodedata

import pytest

from chunksilo.revisions import RevisionPolicy, partition


@pytest.fixture
def policy():
    return RevisionPolicy.from_config({})


# =============================================================================
# Filename suffix stripping
# =============================================================================


class TestStripRevisionSuffixes:
    @pytest.mark.parametrize(
        "stem,expected",
        [
            ("Spec PB3", "Spec"),
            ("Spec pb3", "Spec"),
            ("Spec_v2", "Spec"),
            ("Spec V1.0", "Spec"),
            ("Spec A", "Spec"),
            ("Spec B2", "Spec"),
            ("Report rev 3", "Report"),
            ("Report Revision 3", "Report"),
            ("Plan utgåva 2", "Plan"),
            ("Plan utg 2", "Plan"),
            ("Spec (PB3)", "Spec"),
            ("Spec v2 PB1", "Spec"),  # iterative
            ("Spec-B", "Spec"),
        ],
    )
    def test_stripped(self, policy, stem, expected):
        assert policy.strip_revision_suffixes(stem) == expected

    @pytest.mark.parametrize(
        "stem",
        [
            "Vitamin",  # no separator, nothing to strip
            "Spec USA",  # multi-letter uppercase is not a revision
            "Report 2024",  # bare digits never strip from filenames
            "Chapter 3",
            "Spec final",
            "Spec a",  # lowercase single letter is a word, not a revision
            "Spec PM",  # P-form without digit = initials
            "CV",  # no separator
        ],
    )
    def test_untouched(self, policy, stem):
        assert policy.strip_revision_suffixes(stem) == stem

    def test_nfd_input_normalized(self, policy):
        decomposed = unicodedata.normalize("NFD", "Plan utgåva 2")
        assert policy.strip_revision_suffixes(decomposed) == "Plan"

    @pytest.mark.parametrize(
        "stem",
        ["Appendix B", "Annex A", "Bilaga C", "Part B", "Chapter A", "Tabell B"],
    )
    def test_single_letter_guard(self, policy, stem):
        """A letter naming a sibling document is not a revision."""
        assert policy.strip_revision_suffixes(stem) == stem

    def test_whole_stem_revision_token(self, policy):
        """A stem that IS a revision token carries no identity."""
        assert policy.strip_revision_suffixes("PB3") == ""
        assert policy.strip_revision_suffixes("v2") == ""
        assert policy.strip_revision_suffixes("B") == ""


# =============================================================================
# Directory components
# =============================================================================


class TestIsRevisionDir:
    @pytest.mark.parametrize(
        "component", ["PB3", "PA", "B", "b", "2", "12", "v1.0", "V2", "Rev", "Revision 2", "utgåva 3", "(PB3)"]
    )
    def test_revision_dirs(self, policy, component):
        assert policy.is_revision_dir(component)

    @pytest.mark.parametrize(
        "component", ["Backup", "B2B", "2024", "123", "Specs", "PBX123", "v2-final"]
    )
    def test_normal_dirs(self, policy, component):
        assert not policy.is_revision_dir(component)


# =============================================================================
# Review-copy detection
# =============================================================================


class TestReviewCopyDetection:
    @pytest.mark.parametrize(
        "name",
        [
            "/x/Spec B (review comments JD).docx",
            "/x/Copy of Spec.docx",
            "/x/Spec - Copy (2).docx",
            "/x/Kopia av Spec.docx",
            "/x/Spec - kopia.docx",
            "/x/Spec B granskning.docx",
            "/x/Spec kommentarer.docx",
        ],
    )
    def test_detected(self, policy, name):
        assert policy.review_copy_match(name) is not None

    def test_word_boundary(self, policy):
        assert policy.review_copy_match("/x/copyright.docx") is None
        assert policy.review_copy_match("/x/Spec.docx") is None

    def test_known_over_match_trade_off(self, policy):
        # Documented default behavior: a document legitimately named after the
        # review activity matches too. The skip is logged per file and the
        # patterns are configurable.
        assert policy.review_copy_match("/x/Literature Review.docx") is not None

    def test_directory_names_ignored(self, policy):
        assert policy.review_copy_match("/reviews/Spec.docx") is None


# =============================================================================
# Doc-ID extraction and group keys
# =============================================================================


class TestGroupKey:
    def test_doc_id_pattern_wins_over_path(self):
        policy = RevisionPolicy.from_config(
            {"indexing": {"versioning": {"doc_id_patterns": [r"\b(DOC-\d{5})\b"]}}}
        )
        a = policy.group_key("/share/proj1/PB1/DOC-12345 Spec.docx")
        b = policy.group_key("/share/archive/DOC-12345 Spec B.docx")
        assert a == b == "id:DOC-12345"

    def test_doc_id_whole_match_when_no_group(self):
        policy = RevisionPolicy.from_config(
            {"indexing": {"versioning": {"doc_id_patterns": [r"DOC\d{5}"]}}}
        )
        assert policy.extract_doc_id("/x/notes.pdf") is None
        assert policy.extract_doc_id("/x/DOC12345 spec.pdf") == "DOC12345"
        # Matching is case-insensitive and the ID is upper-cased, consistent
        # with the case-folded path keys.
        assert policy.extract_doc_id("/x/doc12345 spec.pdf") == "DOC12345"

    def test_bad_doc_id_pattern_raises(self):
        with pytest.raises(ValueError, match=r"doc_id_patterns.*\("):
            RevisionPolicy.from_config(
                {"indexing": {"versioning": {"doc_id_patterns": ["("]}}}
            )

    def test_revision_dirs_stripped_from_parent(self, policy):
        a = policy.group_key("/share/proj/Spec/PB1/Spec.docx")
        b = policy.group_key("/share/proj/Spec/PB2/Spec.docx")
        c = policy.group_key("/share/proj/Spec/B/Spec.docx")
        assert a == b == c

    def test_filename_revisions_group(self, policy):
        assert policy.group_key("/share/Spec PB3.docx") == policy.group_key(
            "/share/Spec B.docx"
        )

    def test_different_directories_do_not_group(self, policy):
        assert policy.group_key("/share/proj1/Spec.docx") != policy.group_key(
            "/share/proj2/Spec.docx"
        )

    def test_extension_is_part_of_key(self, policy):
        assert policy.group_key("/share/Spec B.pdf") != policy.group_key(
            "/share/Spec B.docx"
        )

    def test_doc_and_docx_group(self, policy):
        assert policy.group_key("/share/Spec A.doc") == policy.group_key(
            "/share/Spec B.docx"
        )

    def test_case_insensitive_keys(self, policy):
        assert policy.group_key("/share/SPEC B.DOCX") == policy.group_key(
            "/share/spec.docx"
        )

    def test_whole_stem_revision_groups_by_folder(self, policy):
        a = policy.group_key("/share/proj/Spec/PB1.docx")
        b = policy.group_key("/share/proj/Spec/B.docx")
        assert a == b

    def test_extra_revision_tokens(self):
        policy = RevisionPolicy.from_config(
            {"indexing": {"versioning": {"extra_revision_tokens": [r"draft\d+"]}}}
        )
        assert policy.strip_revision_suffixes("Spec draft3") == "Spec"
        assert policy.is_revision_dir("Draft2")


# =============================================================================
# Partitioning
# =============================================================================


class TestPartition:
    def test_latest_by_mtime_wins(self, policy):
        files = [
            ("/s/Spec PB1.docx", 100.0),
            ("/s/Spec PB2.docx", 200.0),
            ("/s/Spec B.docx", 300.0),
        ]
        result = partition(files, policy)
        assert result.keep == ["/s/Spec B.docx"]
        assert result.superseded == {"/s/Spec PB1.docx", "/s/Spec PB2.docx"}
        key = result.group_of["/s/Spec B.docx"]
        assert result.latest_of_group[key] == "/s/Spec B.docx"

    def test_mtime_beats_label(self, policy):
        # The decided rule: mtime alone picks the latest, labels never order.
        files = [("/s/Spec B.docx", 100.0), ("/s/Spec PB1.docx", 200.0)]
        result = partition(files, policy)
        assert result.keep == ["/s/Spec PB1.docx"]

    def test_tie_break_deterministic(self, policy):
        files = [("/s/Spec A.docx", 100.0), ("/s/Spec B.docx", 100.0)]
        assert partition(files, policy).keep == partition(list(reversed(files)), policy).keep

    def test_ungrouped_files_all_kept_in_order(self, policy):
        files = [("/s/One.docx", 1.0), ("/t/Two.pdf", 2.0), ("/u/Three.md", 3.0)]
        result = partition(files, policy)
        assert result.keep == [p for p, _ in files]
        assert not result.superseded

    def test_review_copies_always_skipped(self, policy):
        files = [
            ("/s/Spec B.docx", 100.0),
            ("/s/Spec B (review comments JD).docx", 999.0),
        ]
        result = partition(files, policy)
        assert result.keep == ["/s/Spec B.docx"]
        assert "/s/Spec B (review comments JD).docx" in result.review_copies

    def test_disabled_keeps_everything(self):
        policy = RevisionPolicy.from_config(
            {"indexing": {"versioning": {"enabled": False}}}
        )
        files = [("/s/Spec PB1.docx", 1.0), ("/s/Spec B.docx", 2.0), ("/s/Copy of X.docx", 3.0)]
        result = partition(files, policy)
        assert result.keep == [p for p, _ in files]
        assert not result.superseded
        assert not result.review_copies

    def test_index_superseded_keeps_all_but_marks(self):
        policy = RevisionPolicy.from_config(
            {"indexing": {"versioning": {"index_superseded": True}}}
        )
        files = [("/s/Spec PB1.docx", 1.0), ("/s/Spec B.docx", 2.0)]
        result = partition(files, policy)
        assert result.keep == ["/s/Spec PB1.docx", "/s/Spec B.docx"]
        assert result.superseded == {"/s/Spec PB1.docx"}

    def test_doc_id_of_populated(self):
        policy = RevisionPolicy.from_config(
            {"indexing": {"versioning": {"doc_id_patterns": [r"(\d{4}-\d{2})"]}}}
        )
        files = [("/s/1234-01 Spec.docx", 1.0)]
        result = partition(files, policy)
        assert result.doc_id_of["/s/1234-01 Spec.docx"] == "1234-01"
