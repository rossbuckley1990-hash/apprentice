"""
Self-hosting tests — the Apprentice indexes its own codebase and verifies
its own analyzers don't produce false positives about its own code.

This is the test the reviewer requested: if the call graph and dead-code
analyzer are correct, run_all_analyzers should NOT be flagged as dead,
because it's called from the CLI.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from apprentice.model.store import init_store
from apprentice.indexer.python_parser import index_repo
from apprentice.analyzer.proactive import run_all_analyzers


REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def self_indexed():
    """Index the Apprentice's own codebase into a temp store."""
    d = tempfile.mkdtemp(prefix="apprentice_self_")
    store = init_store(d)
    # Index the real repo
    index_repo(str(REPO_ROOT), store, verbose=False)
    yield store
    shutil.rmtree(d, ignore_errors=True)


class TestSelfHosting:

    def test_run_all_analyzers_not_dead(self, self_indexed):
        """The critical test: run_all_analyzers is called from the CLI,
        so it must NOT be flagged as dead code."""
        store = self_indexed
        fn = store.get_function("apprentice.analyzer.proactive.run_all_analyzers")
        if fn is None:
            pytest.skip("run_all_analyzers not found in index")
        assert not fn.is_dead, (
            f"run_all_analyzers was flagged as dead code! "
            f"Callers: {fn.callers}. "
            f"This means the call graph is broken — the function IS called "
            f"from apprentice.interface.cli.cmd_watch."
        )

    def test_store_methods_not_dead(self, self_indexed):
        """Store methods like upsert_function are called from the indexer
        and CLI, so they must NOT be flagged as dead."""
        store = self_indexed
        fn = store.get_function("apprentice.model.store.Store.upsert_function")
        if fn is None:
            pytest.skip("Store.upsert_function not found")
        assert not fn.is_dead, (
            f"Store.upsert_function was flagged as dead! "
            f"Callers: {fn.callers}. "
            f"This is called from index_repo and should have callers."
        )

    def test_collect_calls_not_dead(self, self_indexed):
        """_collect_calls is called from index_python_file, so it must
        NOT be flagged as dead."""
        store = self_indexed
        fn = store.get_function("apprentice.indexer.python_parser._collect_calls")
        if fn is None:
            pytest.skip("_collect_calls not found")
        assert not fn.is_dead, (
            f"_collect_calls was flagged as dead! "
            f"Callers: {fn.callers}. "
            f"This is called from _handle_func in the indexer."
        )

    def test_ast_summary_not_corrupted(self, self_indexed):
        """Verify that _ast_summary is computed on the UNCORRUPTED AST
        (the old bug mutated the AST before computing the summary)."""
        store = self_indexed
        fn = store.get_function("apprentice.indexer.python_parser._body_hash")
        if fn is None:
            pytest.skip("_body_hash not found")
        # _body_hash calls _norm_ast, which used to mutate the AST.
        # If the summary includes 'calls _norm_ast', the AST wasn't corrupted.
        # If it says 'calls _local', the AST was mutated.
        assert "_local" not in fn.ast_summary, (
            f"_body_hash ast_summary contains '_local' — the AST mutation bug "
            f"is still present. Summary: {fn.ast_summary}"
        )

    def test_no_phantom_functions_after_rename(self, self_indexed):
        """Rename a function and re-index; the old name should NOT persist."""
        store = self_indexed
        # Find a function to "rename"
        fns = store.all_functions()
        if not fns:
            pytest.skip("No functions in index")
        # Pick a real function from the codebase
        test_fn = fns[0]
        old_qname = test_fn.qualified_name
        assert store.get_function(old_qname) is not None

        # Read its file, rename the function, re-index
        root = str(REPO_ROOT)
        abs_path = os.path.join(root, test_fn.file_path)
        with open(abs_path, "r") as f:
            content = f.read()
        # Rename: replace 'def test_fn.name(' with 'def renamed_test_fn('
        new_content = content.replace(
            f"def {test_fn.name}(",
            f"def renamed_{test_fn.name}(",
        )
        if new_content == content:
            pytest.skip("Could not rename function (pattern not found)")
        with open(abs_path, "w") as f:
            f.write(new_content)
        try:
            index_repo(root, store, verbose=False)
            # Old qualified name should be gone
            assert store.get_function(old_qname) is None, (
                f"Phantom function {old_qname} still exists after rename. "
                f"Stale rows are not being cleaned up."
            )
        finally:
            # Restore the original file
            with open(abs_path, "w") as f:
                f.write(content)

    def test_dead_code_count_is_reasonable(self, self_indexed):
        """After the call-graph fix (module-prefixed methods + value-references
        as liveness), the dead-code false positive rate should be low.
        Was 63% before fixes, 22% after first fix. Target: <15%."""
        store = self_indexed
        all_fns = store.all_functions()
        dead_fns = [f for f in all_fns if f.is_dead]
        dead_ratio = len(dead_fns) / max(len(all_fns), 1)
        # After the v0.4.0 fixes (module prefix + value references),
        # this should be well under 15%.
        assert dead_ratio < 0.15, (
            f"{len(dead_fns)}/{len(all_fns)} ({dead_ratio:.0%}) functions are "
            f"flagged as dead. The call graph still has false positives — "
            f"target is <15% after module-prefix + value-reference fixes."
        )

    def test_dedup_on_repeated_watch(self, self_indexed):
        """Running watch twice on the same files should NOT produce
        duplicate observations (deterministic IDs + dedup)."""
        import tempfile
        from apprentice.model.entities import hash_content

        store = self_indexed
        root = str(REPO_ROOT)

        # Run watch on all files
        all_files = [f.path for f in store.all_files()]
        obs1 = run_all_analyzers(store, root, all_files)
        for o in obs1:
            store.add_observation(o)

        unacked1 = store.unacknowledged_observations(limit=500)
        count1 = len(unacked1)

        # Run again — should produce no NEW observations
        obs2 = run_all_analyzers(store, root, all_files)
        for o in obs2:
            store.add_observation(o)

        unacked2 = store.unacknowledged_observations(limit=500)
        count2 = len(unacked2)

        assert count2 == count1, (
            f"Running watch twice produced different unacked counts: "
            f"{count1} -> {count2}. Dedup is not working."
        )
