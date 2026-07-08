"""
Tests for the Apprentice — also serves as a demonstration.

Run: pytest tests/ -v
"""

import os
import sys
import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest

# Make the apprentice package importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from apprentice.model.store import Store, init_store
from apprentice.model.entities import File, Function, Plan, Observation, hash_content
from apprentice.indexer.python_parser import index_repo, index_python_file
from apprentice.indexer.embedder import Embedder, cosine
from apprentice.analyzer.proactive import (
    run_all_analyzers,
    analyze_plan_drift,
    analyze_duplication,
    analyze_dead_code,
    analyze_complexity,
    analyze_todos_without_plan,
    analyze_new_pattern,
)


@pytest.fixture
def tmp_repo():
    """A temporary directory that we'll treat as a repo root."""
    d = tempfile.mkdtemp(prefix="apprentice_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def write_file(root, rel_path, content):
    abs_path = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w") as f:
        f.write(textwrap.dedent(content))


# =============================================================================
# Indexer tests
# =============================================================================

class TestIndexer:

    def test_indexes_simple_function(self, tmp_repo):
        write_file(tmp_repo, "mymod.py", """
            def greet(name):
                return f"hello, {name}"
        """)
        store = init_store(tmp_repo)
        stats = index_repo(tmp_repo, store, verbose=False)
        assert stats["files"] == 1
        assert stats["functions"] == 1
        fns = store.all_functions()
        assert len(fns) == 1
        assert fns[0].name == "greet"
        # The summary should mention the return
        assert "return" in fns[0].ast_summary or "returns" in fns[0].ast_summary

    def test_indexes_class_with_methods(self, tmp_repo):
        write_file(tmp_repo, "shapes.py", """
            class Square:
                def __init__(self, side):
                    self.side = side
                def area(self):
                    return self.side ** 2
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        classes = store.all_classes()
        assert len(classes) == 1
        assert classes[0].name == "Square"
        assert "area" in classes[0].method_names

    def test_complexity_counter(self, tmp_repo):
        write_file(tmp_repo, "complex.py", """
            def branchy(x, y, z):
                if x > 0:
                    if y > 0:
                        return 1
                    else:
                        return 2
                elif z > 0:
                    for i in range(10):
                        if i % 2 == 0:
                            continue
                return 0
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        fns = store.all_functions()
        assert len(fns) == 1
        # 1 base + 3 ifs + 1 for-else + 1 inner if = 6
        assert fns[0].complexity >= 5

    def test_dead_code_detection(self, tmp_repo):
        write_file(tmp_repo, "dead.py", """
            def used_fn():
                return 1
            def unused_fn():
                return 2
            def caller():
                return used_fn()
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        fns = {f.name: f for f in store.all_functions()}
        # used_fn is called by caller() — NOT dead
        assert fns["used_fn"].is_dead is False
        # unused_fn has no callers — dead
        assert fns["unused_fn"].is_dead is True
        # caller has no callers — also dead (correctly; nothing calls it).
        # The tool can't know if 'caller' is an entry point unless it's
        # named 'main' or starts with 'test_'.
        assert fns["caller"].is_dead is True

    def test_duplication_detection(self, tmp_repo):
        write_file(tmp_repo, "dup.py", """
            def helper_a(x):
                result = x * 2
                return result + 1

            def helper_b(x):
                result = x * 2
                return result + 1
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        cliches = store.all_cliches(min_instances=2)
        assert len(cliches) == 1
        assert len(cliches[0].instances) == 2

    def test_call_graph_direct_calls(self, tmp_repo):
        """Verify the call graph correctly resolves direct function calls."""
        write_file(tmp_repo, "calls.py", """
            def foo():
                return bar() + baz()
            def bar():
                return 1
            def baz():
                return 2
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        fns = {f.name: f for f in store.all_functions()}
        # foo calls bar and baz, so bar and baz should have foo as a caller
        foo_qname = fns["foo"].qualified_name
        assert foo_qname in fns["bar"].callers, (
            f"bar.callers = {fns['bar'].callers}, expected to include {foo_qname}"
        )
        assert foo_qname in fns["baz"].callers, (
            f"baz.callers = {fns['baz'].callers}, expected to include {foo_qname}"
        )
        # bar and baz should NOT be dead (foo calls them)
        assert fns["bar"].is_dead is False
        assert fns["baz"].is_dead is False

    def test_call_graph_self_method_calls(self, tmp_repo):
        """Verify that self.method() calls within a class are resolved."""
        write_file(tmp_repo, "cls.py", """
            class Calculator:
                def add(self, x, y):
                    return self.validate(x) + self.validate(y)
                def validate(self, x):
                    if x < 0:
                        raise ValueError
                    return x
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        fns = {f.name: f for f in store.all_functions()}
        # Calculator.add calls self.validate, so validate should have add as a caller
        add_qname = fns["add"].qualified_name
        assert add_qname in fns["validate"].callers, (
            f"validate.callers = {fns['validate'].callers}, "
            f"expected to include {add_qname}"
        )
        # validate should NOT be dead
        assert not fns["validate"].is_dead, (
            "validate is flagged as dead — self.method() call resolution is broken"
        )

    def test_decorated_functions_are_live(self, tmp_repo):
        write_file(tmp_repo, "app.py", """
            ROUTES = {}

            def route(path):
                def decorator(fn):
                    ROUTES[path] = fn
                    return fn
                return decorator

            @route("/health")
            def healthcheck(request):
                return {"ok": True}
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        fn = store.get_function("app.healthcheck")
        assert fn is not None
        assert fn.is_dead is False
        assert "<decorator>" in fn.callers

    def test_module_registry_keeps_handler_live(self, tmp_repo):
        write_file(tmp_repo, "plugins.py", """
            def process_payment(event):
                return event["amount"] * 100

            HANDLERS = {"payment.created": process_payment}
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        fn = store.get_function("plugins.process_payment")
        assert fn is not None
        assert fn.is_dead is False
        assert "<module>" in fn.callers

    def test_import_alias_module_registry_keeps_target_live(self, tmp_repo):
        write_file(tmp_repo, "jobs.py", """
            def nightly_cleanup(db, cutoff):
                return db.delete_older_than(cutoff)
        """)
        write_file(tmp_repo, "app.py", """
            from jobs import nightly_cleanup as cleanup_job

            SCHEDULED_JOBS = [cleanup_job]
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        fn = store.get_function("jobs.nightly_cleanup")
        assert fn is not None
        assert fn.is_dead is False
        assert "<module>" in fn.callers

    def test_module_registered_class_methods_are_live(self, tmp_repo):
        write_file(tmp_repo, "registry.py", """
            PARSERS = []

            class PythonParser:
                def language_name(self):
                    return "python"
                def parse_file(self, path):
                    return path

            def register_parser(parser):
                PARSERS.append(parser)

            register_parser(PythonParser())
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        parse_file = store.get_function("registry.PythonParser.parse_file")
        language_name = store.get_function("registry.PythonParser.language_name")
        assert parse_file is not None
        assert language_name is not None
        assert parse_file.is_dead is False
        assert language_name.is_dead is False
        assert "<class:PythonParser>" in parse_file.callers


# =============================================================================
# Embedder tests
# =============================================================================

class TestEmbedder:

    def test_tfidf_vectors_same_dim(self, tmp_repo):
        write_file(tmp_repo, "mod.py", """
            def fn_a(x):
                return x + 1
            def fn_b(y):
                return y * 2
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        emb = Embedder(backend="tfidf")
        n = emb.index_all(store, tmp_repo)
        assert n == 2
        fns = store.all_functions()
        v1, b1 = store.get_embedding(fns[0].qualified_name)
        v2, b2 = store.get_embedding(fns[1].qualified_name)
        assert b1 == "tfidf" and b2 == "tfidf"
        assert len(v1) == len(v2)
        # Same function should be perfectly similar to itself
        sim = cosine(v1, v1)
        assert 0.99 <= sim <= 1.01

    def test_asthash_works_offline(self):
        emb = Embedder(backend="asthash")
        v = emb.vectorize_function(None, "def f(x): return x + 1")
        assert len(v) == 256
        assert any(abs(x) > 0 for x in v)


# =============================================================================
# Proactive analyzer tests
# =============================================================================

class TestProactive:

    def test_drift_detection(self, tmp_repo):
        # Set up: plan about "auth", introduce "ui" code
        write_file(tmp_repo, "auth.py", """
            def login(user):
                return user.authenticate()
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)

        # State an auth plan
        plan = Plan(
            id="test1",
            description="refactor authentication to use JWT",
            keywords=["auth"],
        )
        store.upsert_plan(plan)

        # Now introduce a UI file (drift)
        write_file(tmp_repo, "ui.py", """
            def render_button(label):
                return f"<button>{label}</button>"
            def render_form(fields):
                return "<form>" + "".join(render_button(f) for f in fields) + "</form>"
        """)
        index_repo(tmp_repo, store, verbose=False)

        obs = analyze_plan_drift(store, tmp_repo, ["ui.py"])
        assert len(obs) >= 1
        assert obs[0].kind == "drift"
        assert "ui" in obs[0].message.lower() or "drift" in obs[0].message.lower()

    def test_duplication_observation(self, tmp_repo):
        write_file(tmp_repo, "dup.py", """
            def helper_a(x):
                result = x * 2
                return result + 1

            def helper_b(x):
                result = x * 2
                return result + 1
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        obs = analyze_duplication(store, tmp_repo, ["dup.py"])
        # Should find the duplication
        assert any(o.kind == "duplication" for o in obs)

    def test_complexity_observation(self, tmp_repo):
        # Write a function with complexity > 30 (threshold for 'error')
        # Each 'if x == N: return N' adds 1 to complexity. Need 30+ branches.
        # Write directly to avoid textwrap.dedent issues with f-string interpolation
        branches = "\n".join(
            f"    if x == {i}: return {i}" for i in range(40)
        )
        code = f"def heavy_branches(x):\n{branches}\n    return -1\n"
        abs_path = os.path.join(tmp_repo, "complex.py")
        with open(abs_path, "w") as f:
            f.write(code)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        fns = store.all_functions()
        assert len(fns) == 1
        # Should have complexity >= 40 (1 base + 40 ifs)
        assert fns[0].complexity >= 30
        obs = analyze_complexity(store, tmp_repo, ["complex.py"])
        assert len(obs) >= 1
        assert obs[0].kind == "complexity_creep"
        assert obs[0].severity == "error"

    def test_todo_without_plan(self, tmp_repo):
        write_file(tmp_repo, "todo.py", """
            def process(data):
                # TODO: add validation
                return data
            def other(x):
                # FIXME: this is wrong
                return x * 2
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        obs = analyze_todos_without_plan(store, tmp_repo, ["todo.py"])
        assert len(obs) == 2
        kinds = {o.kind for o in obs}
        assert "todo_without_plan" in kinds
        # FIXME should be warning
        fixme_obs = [o for o in obs if "FIXME" in o.message]
        assert fixme_obs[0].severity == "warning"

    def test_todo_with_matching_plan_no_observation(self, tmp_repo):
        write_file(tmp_repo, "todo.py", """
            def process(data):
                # TODO: add validation for auth
                return data
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        # Plan mentions auth
        plan = Plan(
            id="test2",
            description="improve auth validation",
            keywords=["auth"],
        )
        store.upsert_plan(plan)
        obs = analyze_todos_without_plan(store, tmp_repo, ["todo.py"])
        assert len(obs) == 0  # plan matches

    def test_todo_scanner_skips_test_files(self, tmp_repo):
        write_file(tmp_repo, "tests/test_markers.py", """
            def test_fixture_marker():
                # TODO: fixture marker used by another test
                assert True
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        obs = analyze_todos_without_plan(store, tmp_repo, ["tests/test_markers.py"])
        assert obs == []

    def test_python_todo_scanner_ignores_string_literals(self, tmp_repo):
        write_file(tmp_repo, "strings.py", '''
            def fixture():
                return """
                # TODO: this is test fixture text, not a real comment
                """

            def real_marker():
                # TODO: wire this to the queue
                return None
        ''')
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        obs = analyze_todos_without_plan(store, tmp_repo, ["strings.py"])
        assert len(obs) == 1
        assert "wire this to the queue" in obs[0].message

    def test_dead_code_skips_abstract_interface_methods(self, tmp_repo):
        write_file(tmp_repo, "interfaces.py", """
            from abc import ABC, abstractmethod

            class Parser(ABC):
                @abstractmethod
                def parse_file(self, path):
                    raise NotImplementedError

                def should_ignore(self, path):
                    return False
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        obs = analyze_dead_code(store, tmp_repo, ["interfaces.py"])
        qnames = {o.function_qualified_name for o in obs}
        assert "interfaces.Parser.parse_file" not in qnames
        assert "interfaces.Parser.should_ignore" not in qnames

    def test_new_pattern_skips_cross_class_interface_methods(self, tmp_repo):
        write_file(tmp_repo, "parsers.py", """
            class BaseParser:
                def parse_file(self, path):
                    raise NotImplementedError

            class PythonParser:
                def parse_file(self, path):
                    return path

            class JavaScriptParser:
                def parse_file(self, path):
                    return path
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        obs = analyze_new_pattern(store, tmp_repo, ["parsers.py"])
        assert not [o for o in obs if o.kind == "new_pattern"]

    def test_duplication_skips_test_helpers(self, tmp_repo):
        write_file(tmp_repo, "tests/test_a.py", """
            def make_user():
                name = "ross"
                email = "ross@example.com"
                return {"name": name, "email": email}
        """)
        write_file(tmp_repo, "tests/test_b.py", """
            def make_user_again():
                name = "ross"
                email = "ross@example.com"
                return {"name": name, "email": email}
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        obs = analyze_duplication(store, tmp_repo, ["tests/test_a.py", "tests/test_b.py"])
        assert obs == []

    def test_duplication_skips_cross_class_interface_methods(self, tmp_repo):
        write_file(tmp_repo, "parsers.py", """
            class PythonParser:
                def should_ignore(self, path):
                    return False

            class JavaScriptParser:
                def should_ignore(self, path):
                    return False
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        obs = analyze_duplication(store, tmp_repo, ["parsers.py"])
        assert obs == []

    def test_plan_drift_skips_tests_and_low_signal_refactor_config(self, tmp_repo):
        write_file(tmp_repo, "config.py", """
            def load_config(path):
                return {"path": path}
        """)
        write_file(tmp_repo, "tests/test_api.py", """
            def test_endpoint(client):
                return client.get("/api")
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        store.upsert_plan(Plan(
            id="refactor",
            description="refactor indexing internals",
            keywords=["refactor", "index"],
        ))
        obs = analyze_plan_drift(store, tmp_repo, ["config.py", "tests/test_api.py"])
        assert obs == []

    def test_plan_drift_matches_snake_case_keywords(self, tmp_repo):
        write_file(tmp_repo, "service.py", """
            def handle_api_request(payload):
                return payload
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)
        store.upsert_plan(Plan(
            id="refactor",
            description="refactor indexing internals",
            keywords=["refactor", "index"],
        ))
        obs = analyze_plan_drift(store, tmp_repo, ["service.py"])
        assert [o.kind for o in obs] == ["drift"]
        assert "api" in obs[0].message

    def test_full_watch_pipeline(self, tmp_repo):
        """End-to-end: index, plan, change, watch, get observations."""
        # Initial state
        write_file(tmp_repo, "main.py", """
            def main():
                print("hello")
        """)
        store = init_store(tmp_repo)
        index_repo(tmp_repo, store, verbose=False)

        # State a plan
        plan = Plan(
            id="e2e",
            description="add user authentication",
            keywords=["auth"],
        )
        store.upsert_plan(plan)

        # Make a change that introduces drift, duplication, complexity, todo
        branches = "\n".join(
            f"    if x == {i}: return {i}" for i in range(40)
        )
        ui_code = f"""
def render_a(x):
    result = x * 2
    return result + 1
def render_b(x):
    result = x * 2
    return result + 1
def huge_branches(x):
{branches}
    # TODO: refactor this
    return -1
"""
        abs_path = os.path.join(tmp_repo, "ui.py")
        with open(abs_path, "w") as f:
            f.write(ui_code)
        # Re-index
        index_repo(tmp_repo, store, verbose=False)

        # Run all analyzers
        all_obs = run_all_analyzers(store, tmp_repo, ["ui.py"])
        # Should have at least: drift, duplication, complexity, todo
        kinds = {o.kind for o in all_obs}
        assert "drift" in kinds
        assert "duplication" in kinds
        assert "complexity_creep" in kinds
        assert "todo_without_plan" in kinds


# =============================================================================
# Persistence tests
# =============================================================================

class TestPersistence:

    def test_survives_restart(self, tmp_repo):
        """The persistence test: open store, write data, close, reopen."""
        write_file(tmp_repo, "mod.py", "def foo(): return 1")
        store1 = init_store(tmp_repo)
        index_repo(tmp_repo, store1, verbose=False)
        assert store1.function_count() == 1

        # Simulate restart by creating a new Store instance pointing at same db
        from apprentice.model.store import default_db_path
        store2 = Store(default_db_path(tmp_repo))
        store2.init_schema()
        assert store2.function_count() == 1
        fns = store2.all_functions()
        assert fns[0].name == "foo"

    def test_plan_persists(self, tmp_repo):
        store1 = init_store(tmp_repo)
        plan = Plan(id="persist-test", description="test plan", keywords=["test"])
        store1.upsert_plan(plan)

        from apprentice.model.store import default_db_path
        store2 = Store(default_db_path(tmp_repo))
        store2.init_schema()
        plans = store2.active_plans()
        assert len(plans) == 1
        assert plans[0].id == "persist-test"
