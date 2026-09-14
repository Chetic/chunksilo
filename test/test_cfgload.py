"""Tests for the configuration loader (cfgload.py).

An explicit-path load becomes the process-wide active configuration, so a
--config argument governs every later cfgload.get() call in the process.
"""
import logging

import yaml

from chunksilo import cfgload


def _write(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestActiveConfig:
    def test_explicit_path_becomes_active(self, tmp_path):
        cfg_path = _write(tmp_path / "a.yaml", {"indexing": {"chunk_size": 999}})

        loaded = cfgload.load_config(cfg_path)

        assert loaded["indexing"]["chunk_size"] == 999
        assert cfgload.get("indexing.chunk_size") == 999
        assert cfgload.load_config() is loaded
        assert cfgload._active_path == cfg_path

    def test_string_path_is_accepted(self, tmp_path):
        cfg_path = _write(tmp_path / "a.yaml", {"indexing": {"chunk_size": 321}})

        cfgload.load_config(str(cfg_path))

        assert cfgload.get("indexing.chunk_size") == 321

    def test_another_path_replaces_the_active_config(self, tmp_path):
        first = _write(tmp_path / "a.yaml", {"indexing": {"chunk_size": 1}})
        second = _write(tmp_path / "b.yaml", {"indexing": {"chunk_size": 2}})

        cfgload.load_config(first)
        cfgload.load_config(second)

        assert cfgload.get("indexing.chunk_size") == 2
        assert cfgload._active_path == second

    def test_same_path_reuses_the_cache(self, tmp_path):
        cfg_path = _write(tmp_path / "a.yaml", {"indexing": {"chunk_size": 7}})

        first = cfgload.load_config(cfg_path)
        second = cfgload.load_config(cfg_path)

        assert first is second

    def test_missing_file_uses_defaults_and_is_still_active(self, tmp_path):
        missing = tmp_path / "missing.yaml"

        loaded = cfgload.load_config(missing)

        assert loaded["indexing"]["chunk_size"] == cfgload._DEFAULTS["indexing"]["chunk_size"]
        assert cfgload._active_path == missing
        assert cfgload.load_config() is loaded

    def test_reload_clears_the_active_config(self, tmp_path):
        cfg_path = _write(tmp_path / "a.yaml", {"indexing": {"chunk_size": 5}})
        cfgload.load_config(cfg_path)

        cfgload.reload_config()

        assert cfgload._active_path == cfgload.CONFIG_PATH

    def test_returned_config_never_aliases_the_defaults(self, tmp_path):
        loaded = cfgload.load_config(tmp_path / "missing.yaml")

        loaded["storage"]["storage_dir"] = "/somewhere/else"
        loaded["indexing"]["defaults"]["include"].append("**/*.rst")

        assert cfgload._DEFAULTS["storage"]["storage_dir"] == "./storage"
        assert "**/*.rst" not in cfgload.DEFAULT_INCLUDE_PATTERNS


class TestMerging:
    def test_empty_value_keeps_the_default(self, tmp_path):
        cfg_path = tmp_path / "a.yaml"
        cfg_path.write_text("indexing:\n  chunk_size:\n  chunk_overlap: 9\n", encoding="utf-8")

        loaded = cfgload.load_config(cfg_path)

        assert loaded["indexing"]["chunk_size"] == 512
        assert loaded["indexing"]["chunk_overlap"] == 9

    def test_commented_out_section_keeps_the_defaults(self, tmp_path):
        cfg_path = tmp_path / "a.yaml"
        cfg_path.write_text("retrieval:\nstorage:\n  storage_dir: ./s\n", encoding="utf-8")

        loaded = cfgload.load_config(cfg_path)

        assert loaded["retrieval"]["embed_top_k"] == 20
        assert loaded["storage"]["storage_dir"] == "./s"

    def test_removed_options_are_reported(self, tmp_path, caplog):
        cfg_path = _write(
            tmp_path / "old.yaml",
            {
                "indexing": {
                    "timeout": {"per_file_seconds": 10},
                    "enable_adaptive_batching": False,
                },
                "retrieval": {"bm25_similarity_top_k": 3},
            },
        )

        with caplog.at_level(logging.WARNING, logger="chunksilo.cfgload"):
            loaded = cfgload.load_config(cfg_path)

        warned = [r.getMessage() for r in caplog.records if "no longer used" in r.getMessage()]
        assert len(warned) == 3
        assert any("indexing.timeout" in m and "indexing.per_file_seconds" in m for m in warned)
        assert any("indexing.enable_adaptive_batching" in m for m in warned)
        assert any("retrieval.bm25_similarity_top_k" in m for m in warned)
        # The current schema still answers with its defaults.
        assert loaded["indexing"]["per_file_seconds"] == 300

    def test_current_options_do_not_warn(self, tmp_path, caplog):
        cfg_path = _write(tmp_path / "new.yaml", {"indexing": {"per_file_seconds": 10}})

        with caplog.at_level(logging.WARNING, logger="chunksilo.cfgload"):
            cfgload.load_config(cfg_path)

        assert not [r for r in caplog.records if "no longer used" in r.getMessage()]
