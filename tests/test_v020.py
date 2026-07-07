"""Tests for v0.2.0 features: LLM, config, multi-language, historical, hooks."""

import os
import sys
import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from apprentice.config import Config, load_config, save_config, _parse_toml
from apprentice.llm.client import LLMClient, get_client
from apprentice.llm.ask import ask, build_context
from apprentice.llm.fix import propose_fix
from apprentice.llm.summarize import summarize_function, summarize_codebase
from apprentice.model.store import init_store
from apprentice.model.entities import Plan, Observation
from apprentice.indexer.python_parser import index_repo, discover_all_files
from apprentice.indexer.registry import get_parser_for_file, supported_extensions, supported_languages
from apprentice.indexer.embedder import Embedder
from apprentice.analyzer.proactive import run_all_analyzers
from apprentice.analyzer.historical import analyze_complexity_trends
from apprentice.hooks import install_hook, uninstall_hook, is_hook_installed


@pytest.fixture
def tmp_repo():
    d = tempfile.mkdtemp(prefix="apprentice_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def write_file(root, rel_path, content):
    abs_path = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w") as f:
        f.write(textwrap.dedent(content))


# =============================================================================
# Config tests
# =============================================================================

class TestConfig:

    def test_defaults(self):
        config = Config.defaults()
        assert config.complexity_warn == 15
        assert config.complexity_error == 30
        assert config.embedding_backend == "tfidf"
        assert ".py" in config.file_extensions

    def test_toml_roundtrip(self, tmp_repo):
        config = Config.defaults()
        config.complexity_warn = 20
        config.llm_backend = "openai"
        save_config(config, tmp_repo)

        loaded = load_config(tmp_repo)
        assert loaded.complexity_warn == 20
        assert loaded.llm_backend == "openai"

    def test_toml_parser(self):
        toml_text = """
[llm]
backend = "anthropic"
model = "claude-sonnet-4-20250514"

[analyzer]
complexity_warn = 25
"""
        data = _parse_toml(toml_text)
        assert data["llm"]["backend"] == "anthropic"
        assert data["analyzer"]["complexity_warn"] == 25

    def test_env_override(self, tmp_repo, monkeypatch):
        monkeypatch.setenv("APPRENTICE_LLM_BACKEND", "zai")
        config = load_config(tmp_repo)
        assert config.llm_backend == "zai"


# =============================================================================
# LLM tests (mock mode)
# =============================================================================

class TestLLM:

    def test_mock_client(self):
        client = LLMClient(backend="mock")
        assert not client.is_real()
        resp = client.complete("system", "hello")
        assert resp.backend == "mock"
        assert len(resp.text) > 0

    def test_auto_backend_falls_back_to_mock(self, monkeypatch):
        # Clear all API keys AND ensure z-ai CLI is not found
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ZAI_API_KEY"]:
            monkeypatch.delenv(key, raising=False)
        # Mock shutil.which to return None for z-ai
        import shutil
        original_which = shutil.which
        monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "z-ai" else original_which(cmd))
        client = LLMClient()
        assert client.backend == "mock"

    def test_ask_with_mock(self, tmp_repo):
        write_file(tmp_repo, "mod.py", """
            def foo(x):
                return x + 1
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)

        client = LLMClient(backend="mock")
        embedder = Embedder(backend="tfidf")
        answer = ask(store, tmp_repo, "what does foo do?", client=client, embedder=embedder)
        assert "Mock" in answer or "mock" in answer  # mock mode indicator

    def test_fix_with_mock(self, tmp_repo):
        write_file(tmp_repo, "mod.py", """
            def foo(x):
                # TODO: implement this
                return None
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)

        # Add a TODO observation
        from apprentice.analyzer.proactive import analyze_todos_without_plan
        obs_list = analyze_todos_without_plan(store, tmp_repo, ["mod.py"])
        assert len(obs_list) >= 1
        for o in obs_list:
            store.add_observation(o)

        client = LLMClient(backend="mock")
        result = propose_fix(store, tmp_repo, obs_list[0].id, client=client)
        assert result["observation"] is not None
        assert len(result["diff"]) > 0

    def test_summarize_with_mock(self, tmp_repo):
        write_file(tmp_repo, "mod.py", """
            def calculate(x, y):
                '''Add two numbers.'''
                return x + y
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)

        client = LLMClient(backend="mock")
        summary = summarize_function(store, tmp_repo, "mod.calculate", client=client)
        assert len(summary) > 0


# =============================================================================
# Multi-language tests
# =============================================================================

class TestMultiLanguage:

    def test_registry_has_python_and_js(self):
        langs = supported_languages()
        assert "python" in langs
        assert "javascript" in langs

    def test_parser_for_python(self):
        parser = get_parser_for_file("test.py")
        assert parser is not None
        assert parser.language_name == "python"

    def test_parser_for_javascript(self):
        parser = get_parser_for_file("test.js")
        assert parser is not None
        assert parser.language_name == "javascript"

    def test_parser_for_typescript(self):
        parser = get_parser_for_file("test.tsx")
        assert parser is not None

    def test_parser_for_unknown(self):
        parser = get_parser_for_file("test.go")
        assert parser is None

    def test_index_javascript(self, tmp_repo):
        write_file(tmp_repo, "app.js", """
            function add(a, b) {
                return a + b;
            }

            const multiply = (a, b) => {
                return a * b;
            };

            class Calculator {
                constructor() {
                    this.result = 0;
                }
                add(x) {
                    this.result += x;
                    return this.result;
                }
            }
        """)
        store = init_store(tmp_repo)
        stats = index_repo(tmp_repo, store, verbose=False)
        assert stats["files"] == 1
        assert stats["functions"] >= 2  # add + multiply at minimum
        assert stats["classes"] >= 1

        fns = store.all_functions()
        fn_names = [f.name for f in fns]
        assert "add" in fn_names
        assert "multiply" in fn_names

    def test_index_mixed_languages(self, tmp_repo):
        write_file(tmp_repo, "mod.py", "def py_fn(): return 1")
        write_file(tmp_repo, "app.js", "function js_fn() { return 1; }")
        store = init_store(tmp_repo)
        stats = index_repo(tmp_repo, store, verbose=False)
        assert stats["files"] == 2

        files = store.all_files()
        langs = {f.language for f in files}
        assert "python" in langs
        assert "javascript" in langs


# =============================================================================
# Historical tracking tests
# =============================================================================

class TestHistorical:

    def test_history_recorded_on_index(self, tmp_repo):
        write_file(tmp_repo, "mod.py", """
            def foo(x):
                if x > 0:
                    return x
                return -1
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)

        history = store.function_history("mod.foo")
        assert len(history) >= 1
        assert history[0]["complexity"] >= 2

    def test_complexity_trends_detected(self, tmp_repo):
        # v1: simple function
        write_file(tmp_repo, "mod.py", """
            def foo(x):
                return x + 1
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)

        # v2: more complex
        write_file(tmp_repo, "mod.py", """
            def foo(x):
                if x > 0:
                    if x > 10:
                        for i in range(x):
                            if i % 2 == 0:
                                continue
                    return x
                elif x < -10:
                    return -1
                return 0
        """)
        index_repo(tmp_repo, store, verbose=False)

        trends = store.complexity_trends(min_changes=2)
        foo_trend = [t for t in trends if t["qualified_name"] == "mod.foo"]
        assert len(foo_trend) >= 1
        assert foo_trend[0]["max_c"] > foo_trend[0]["min_c"]


# =============================================================================
# Git hooks tests
# =============================================================================

class TestHooks:

    def test_install_and_uninstall(self, tmp_repo):
        # Create a fake .git directory
        os.makedirs(os.path.join(tmp_repo, ".git", "hooks"), exist_ok=True)

        assert not is_hook_installed(tmp_repo)

        path = install_hook(tmp_repo)
        assert os.path.exists(path)
        assert is_hook_installed(tmp_repo)

        # Verify it's executable
        assert os.access(path, os.X_OK)

        removed = uninstall_hook(tmp_repo)
        assert removed
        assert not is_hook_installed(tmp_repo)

    def test_hook_script_content(self, tmp_repo):
        os.makedirs(os.path.join(tmp_repo, ".git", "hooks"), exist_ok=True)
        install_hook(tmp_repo)

        hook_path = os.path.join(tmp_repo, ".git", "hooks", "pre-commit")
        with open(hook_path, "r") as f:
            content = f.read()
        assert "Apprentice pre-commit hook" in content
        assert "apprentice watch" in content


# =============================================================================
# Daemon tests
# =============================================================================

class TestDaemon:

    def test_daemon_detects_changes(self, tmp_repo):
        from apprentice.daemon import Daemon
        from apprentice.config import Config

        write_file(tmp_repo, "mod.py", "def foo(): return 1")
        config = Config.defaults()
        daemon = Daemon(tmp_repo, config)

        # Initial state — no changes (just indexed)
        changed = daemon.get_changed_files()
        # The file exists but hasn't been indexed yet
        assert "mod.py" in changed

        # After indexing, no changes
        daemon.store = init_store(tmp_repo)
        index_repo(tmp_repo, daemon.store, verbose=False, config=config)
        daemon._init_hashes()

        changed = daemon.get_changed_files()
        assert len(changed) == 0

        # Modify the file
        write_file(tmp_repo, "mod.py", "def foo(): return 2")
        changed = daemon.get_changed_files()
        assert "mod.py" in changed

    def test_daemon_run_once(self, tmp_repo):
        from apprentice.daemon import Daemon
        from apprentice.config import Config

        write_file(tmp_repo, "mod.py", """
            def foo():
                # TODO: implement
                return None
        """)
        config = Config.defaults()
        daemon = Daemon(tmp_repo, config)
        n_obs = daemon.run_once()
        assert n_obs >= 1  # should find the TODO
