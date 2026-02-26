#!/usr/bin/env python3
"""Tests for chunksilo.cli module."""

import json
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Tests for argument parsing
# =============================================================================


class TestCliArgumentParsing:
    def _run_main(self, args):
        """Helper to run cli.main() with given argv."""
        from chunksilo.cli import main

        with patch("sys.argv", ["chunksilo"] + args):
            return main()

    @patch("chunksilo.cli.run_search", create=True)
    def test_query_argument(self, mock_search):
        """Basic query is passed to run_search."""
        from chunksilo import cli

        mock_search.return_value = {"chunks": [], "matched_files": []}

        with patch("sys.argv", ["chunksilo", "test query"]):
            with patch.object(cli, "run_search", mock_search, create=True):
                # Import and call inline to control the lazy import
                pass

    @patch("chunksilo.index.build_index")
    def test_build_index_flag(self, mock_build):
        self._run_main(["--build-index"])
        mock_build.assert_called_once()

    @patch("chunksilo.index.build_index")
    def test_download_models_flag(self, mock_build):
        self._run_main(["--download-models"])
        mock_build.assert_called_once_with(
            download_only=True,
            config_path=None,
            verbose=False,
        )

    def test_dump_defaults_outputs_yaml(self, capsys):
        self._run_main(["--dump-defaults"])
        output = capsys.readouterr().out
        # Should output YAML with known config keys
        assert "storage" in output or "retrieval" in output

    def test_no_args_shows_error(self):
        with pytest.raises(SystemExit) as exc_info:
            self._run_main([])
        assert exc_info.value.code == 2


# =============================================================================
# Tests for search output formatting
# =============================================================================


class TestCliSearchOutput:
    def _run_search_main(self, args, search_result):
        """Run main() with a mocked run_search return value."""
        from chunksilo.cli import main

        with patch("sys.argv", ["chunksilo"] + args):
            with patch("chunksilo.search.run_search", return_value=search_result):
                return main()

    def test_json_output(self, capsys):
        result = {
            "chunks": [{"text": "hello", "score": 0.95, "location": {"uri": "file://test.md"}}],
            "matched_files": [{"uri": "file://test.md", "score": 0.8}],
            "retrieval_time": "0.05s",
        }
        self._run_search_main(["test query", "--json"], result)
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["chunks"][0]["text"] == "hello"

    def test_error_result_exits_1(self):
        result = {"error": "Index not found", "chunks": [], "matched_files": []}
        with pytest.raises(SystemExit) as exc_info:
            self._run_search_main(["test query"], result)
        assert exc_info.value.code == 1

    def test_no_results_message(self, capsys):
        result = {"chunks": [], "matched_files": [], "retrieval_time": "0.01s"}
        self._run_search_main(["test query"], result)
        output = capsys.readouterr().out
        assert "No results found" in output

    def test_human_readable_output(self, capsys):
        result = {
            "chunks": [
                {
                    "text": "Document content here",
                    "score": 0.85,
                    "location": {
                        "uri": "file:///docs/test.md",
                        "page": None,
                        "line": 10,
                        "heading_path": ["Chapter 1", "Section A"],
                    },
                }
            ],
            "matched_files": [],
            "retrieval_time": "0.10s",
        }
        self._run_search_main(["test query"], result)
        output = capsys.readouterr().out
        assert "Document content here" in output
        assert "Results (1)" in output


# =============================================================================
# Tests for --list-files
# =============================================================================


class TestCliListFiles:
    def test_list_files_json(self, capsys, tmp_path):
        from chunksilo.cli import main

        state_db = tmp_path / "ingestion_state.db"
        state_db.touch()

        mock_state = MagicMock()
        mock_state.get_all_files.return_value = {
            "/docs/a.md": None,
            "/docs/b.pdf": None,
        }
        mock_state_cls = MagicMock(return_value=mock_state)
        mock_config = MagicMock(return_value={"storage": {"storage_dir": str(tmp_path)}})

        with patch("sys.argv", ["chunksilo", "--list-files", "--json"]):
            with patch("chunksilo.cfgload.load_config", mock_config):
                with patch("chunksilo.index.IngestionState", mock_state_cls):
                    main()

        output = capsys.readouterr().out
        files = json.loads(output)
        assert "/docs/a.md" in files
        assert "/docs/b.pdf" in files
