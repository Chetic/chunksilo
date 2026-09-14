"""Tests for share-location presentation of search results (shareuri.py).

Results for files indexed from a mounted network share must name the location
the user actually knows (smb:// plus the Windows UNC form), derived purely
from the ``shares`` config - never from the mount table, and never by touching
the filesystem.
"""
import logging

from llama_index.core.schema import NodeWithScore, TextNode

from chunksilo import search
from chunksilo.search import _format_bm25_matches, _resolve_result_uris
from chunksilo.shareuri import build_share_mappings, to_share_uris

DOCS = {"prefix": "/mnt/docs", "unc": "//nas/docs"}


def _mappings(*rules):
    return build_share_mappings({"shares": list(rules)})


class TestBuildShareMappings:
    def test_valid_entry(self):
        (mapping,) = _mappings(DOCS)
        assert mapping.host == "nas"
        assert mapping.share_parts == ("docs",)
        assert mapping.prefix_parts == ("mnt", "docs")

    def test_nested_share_path(self):
        (mapping,) = _mappings({"prefix": "/mnt/eng", "unc": "//nas/dept/engineering"})
        assert mapping.host == "nas"
        assert mapping.share_parts == ("dept", "engineering")

    def test_windows_style_unc_is_normalized(self):
        (mapping,) = _mappings({"prefix": "/mnt/x", "unc": "\\\\nas\\docs"})
        assert mapping.host == "nas"
        assert mapping.share_parts == ("docs",)

    def test_invalid_entries_are_skipped_with_a_warning(self, caplog):
        """A typo must degrade that entry's URIs, never break search."""
        with caplog.at_level(logging.WARNING, logger="chunksilo.shareuri"):
            mappings = _mappings(
                {"prefix": "relative/path", "unc": "//h/s"},
                {"prefix": "/ok", "unc": "not-a-unc"},
                {"prefix": "/", "unc": "//h/s"},  # would map the whole filesystem
                {"prefix": "/mnt/x", "unc": "//hostonly"},  # no share component
                DOCS,
            )
        assert len(mappings) == 1
        assert mappings[0].prefix == "/mnt/docs"
        assert caplog.text.count("Ignoring shares entry") == 4

    def test_sorted_longest_prefix_first(self):
        mappings = _mappings(
            {"prefix": "/mnt", "unc": "//h/top"},
            {"prefix": "/mnt/docs", "unc": "//h/deep"},
        )
        assert [m.share_parts for m in mappings] == [("deep",), ("top",)]


class TestToShareUris:
    def test_covered_path(self):
        smb, unc = to_share_uris("/mnt/docs/spec.pdf", _mappings(DOCS))
        assert smb == "smb://nas/docs/spec.pdf"
        assert unc == "\\\\nas\\docs\\spec.pdf"

    def test_smb_is_encoded_and_unc_is_raw(self):
        """Windows opens UNC verbatim; percent escapes belong in the URI only."""
        smb, unc = to_share_uris("/mnt/docs/10 - Design Notes/a#b.docx", _mappings(DOCS))
        assert smb == "smb://nas/docs/10%20-%20Design%20Notes/a%23b.docx"
        assert unc == "\\\\nas\\docs\\10 - Design Notes\\a#b.docx"

    def test_longest_prefix_wins(self):
        mappings = _mappings(
            {"prefix": "/mnt", "unc": "//h/broad"},
            {"prefix": "/mnt/docs", "unc": "//h/narrow"},
        )
        smb, _ = to_share_uris("/mnt/docs/f.txt", mappings)
        assert smb == "smb://h/narrow/f.txt"
        smb, _ = to_share_uris("/mnt/other/f.txt", mappings)
        assert smb == "smb://h/broad/other/f.txt"

    def test_dotdot_cannot_cross_into_a_share(self):
        """normpath moves the path out of the covered tree before matching."""
        assert to_share_uris("/mnt/docs/../secrets/f.txt", _mappings(DOCS)) is None

    def test_prefix_matching_is_component_anchored(self):
        assert to_share_uris("/mnt/docs-archive/f.txt", _mappings(DOCS)) is None

    def test_uncovered_path(self):
        assert to_share_uris("/opt/docs/f.txt", _mappings(DOCS)) is None
        assert to_share_uris("/opt/docs/f.txt", ()) is None


class TestSearchIntegration:
    CONFIG = {"indexing": {"directories": []}, "shares": [DOCS]}

    def test_share_covered_result(self):
        assert _resolve_result_uris("/mnt/docs/spec.pdf", self.CONFIG) == (
            "smb://nas/docs/spec.pdf",
            "\\\\nas\\docs\\spec.pdf",
        )

    def test_uncovered_result_keeps_the_exact_file_uri(self):
        config = {"indexing": {"directories": []}, "shares": []}
        assert _resolve_result_uris("/opt/docs/read me.md", config) == (
            "file:///opt/docs/read%20me.md",
            None,
        )

    def test_relative_path_joins_the_first_directory_then_maps(self, monkeypatch):
        monkeypatch.setattr(search, "_configured_directories_cache", None)
        config = {"indexing": {"directories": ["/mnt/docs"]}, "shares": [DOCS]}
        assert _resolve_result_uris("guides/spec.pdf", config) == (
            "smb://nas/docs/guides/spec.pdf",
            "\\\\nas\\docs\\guides\\spec.pdf",
        )

    def test_bm25_matches_carry_uri_and_unc(self):
        node = TextNode(
            text="x",
            metadata={"file_path": "/mnt/docs/a.pdf", "file_name": "a.pdf"},
        )
        (entry,) = _format_bm25_matches(
            [NodeWithScore(node=node, score=1.2)], self.CONFIG
        )
        assert entry == {
            "uri": "smb://nas/docs/a.pdf",
            "unc": "\\\\nas\\docs\\a.pdf",
            "score": 1.2,
        }
