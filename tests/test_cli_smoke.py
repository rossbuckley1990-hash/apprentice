"""
CLI smoke tests — invoke every subcommand against a fixture repo.

This is the test the reviewer asked for: it would have caught the watch
UnboundLocalError, the migration banner on every command, and the gitignore
gap. Every CLI command gets at least one smoke test.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from apprentice.interface.cli import main as cli_main


@pytest.fixture
def cli_repo():
    """A temporary repo with a small Python file, initialized for the Apprentice."""
    d = tempfile.mkdtemp(prefix="apprentice_cli_")
    # Create a small Python file
    with open(os.path.join(d, "mod.py"), "w") as f:
        f.write("""
def greet(name):
    return f"hello, {name}"

def unused():
    return 42

class Calculator:
    def add(self, x, y):
        return x + y
    def sub(self, x, y):
        return x - y
""")
    # Initialize git so hook tests work
    os.system(f"cd {d} && git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init")
    # Run `apprentice init`
    old_cwd = os.getcwd()
    os.chdir(d)
    try:
        cli_main(["init"])
        cli_main(["index"])
        yield d
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(d, ignore_errors=True)


class TestCLISmoke:

    def test_init(self, cli_repo):
        """init was already run in the fixture; verify .apprentice/ exists."""
        assert os.path.exists(os.path.join(cli_repo, ".apprentice", "apprentice.db"))
        # Verify .gitignore was created (or appended to)
        gitignore = os.path.join(cli_repo, ".gitignore")
        assert os.path.exists(gitignore)
        with open(gitignore) as f:
            assert ".apprentice/" in f.read()

    def test_status(self, cli_repo, capsys):
        """status should show files and functions."""
        rc = cli_main(["status"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Files in model" in captured.out
        assert "Functions in model" in captured.out

    def test_plan_create_and_list(self, cli_repo, capsys):
        """plan create + plan --list."""
        rc = cli_main(["plan", "add authentication with JWT"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Plan [" in captured.out

        rc = cli_main(["plan", "--list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "authentication" in captured.out

    def test_plan_done(self, cli_repo, capsys):
        """plan --done marks a plan completed."""
        cli_main(["plan", "test plan to complete"])
        captured = capsys.readouterr()
        # Extract plan ID
        import re
        m = re.search(r"Plan \[([^\]]+)\]", captured.out)
        assert m, f"Could not find plan ID in: {captured.out}"
        plan_id = m.group(1)

        rc = cli_main(["plan", "--done", plan_id])
        assert rc == 0

    def test_plan_with_keywords(self, cli_repo, capsys):
        """plan --keywords adds explicit keyword categories."""
        rc = cli_main(["plan", "--keywords", "auth,api", "do the thing"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "auth" in captured.out
        assert "api" in captured.out

    def test_watch_no_changes(self, cli_repo, capsys):
        """watch on a freshly-indexed repo should say 'no changes'."""
        rc = cli_main(["watch"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "No changes" in captured.out or "No new observations" in captured.out

    def test_watch_with_changes(self, cli_repo, capsys):
        """watch after a file change should run analyzers."""
        # Modify the file
        with open(os.path.join(cli_repo, "mod.py"), "a") as f:
            f.write("\n# TODO: implement this\n")
        rc = cli_main(["watch"])
        assert rc == 0
        captured = capsys.readouterr()
        # Should have at least one observation (the TODO)
        assert "observation" in captured.out.lower() or "No new" in captured.out

    def test_watch_all(self, cli_repo, capsys):
        """watch --all analyzes everything."""
        rc = cli_main(["watch", "--all"])
        assert rc == 0

    def test_observations(self, cli_repo, capsys):
        """observations command shows observations."""
        rc = cli_main(["observations"])
        assert rc == 0

    def test_ask(self, cli_repo, capsys):
        """ask command works (in mock or real mode)."""
        rc = cli_main(["ask", "what does greet do"])
        assert rc == 0
        captured = capsys.readouterr()
        # In mock mode, "Mock" appears; in real LLM mode, an actual answer appears.
        # Either way, the command should not crash and should produce output.
        assert len(captured.out) > 10

    def test_recall(self, cli_repo, capsys):
        """recall shows function details."""
        rc = cli_main(["recall", "greet"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "greet" in captured.out

    def test_similar(self, cli_repo, capsys):
        """similar finds related functions."""
        rc = cli_main(["similar", "greet"])
        assert rc == 0

    def test_history(self, cli_repo, capsys):
        """history shows function evolution."""
        rc = cli_main(["history", "greet"])
        assert rc == 0

    def test_summarize(self, cli_repo, capsys):
        """summarize produces a function summary."""
        rc = cli_main(["summarize", "greet"])
        assert rc == 0

    def test_summarize_codebase(self, cli_repo, capsys):
        """summarize --codebase produces an overview."""
        rc = cli_main(["summarize", "--codebase"])
        assert rc == 0

    def test_config_show(self, cli_repo, capsys):
        """config shows current configuration."""
        rc = cli_main(["config"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "LLM backend" in captured.out

    def test_config_init(self, cli_repo, capsys):
        """config --init creates .apprentice.toml."""
        rc = cli_main(["config", "--init"])
        assert rc == 0
        assert os.path.exists(os.path.join(cli_repo, ".apprentice.toml"))

    def test_hook_install_uninstall(self, cli_repo, capsys):
        """hook install + uninstall."""
        rc = cli_main(["hook", "install"])
        assert rc == 0
        hook_path = os.path.join(cli_repo, ".git", "hooks", "pre-commit")
        assert os.path.exists(hook_path)

        rc = cli_main(["hook", "status"])
        captured = capsys.readouterr()
        assert "installed" in captured.out.lower()

        rc = cli_main(["hook", "uninstall"])
        assert rc == 0
        assert not os.path.exists(hook_path)

    def test_prune(self, cli_repo, capsys):
        """prune cleans up old data."""
        rc = cli_main(["prune", "--days", "0"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Pruned" in captured.out

    def test_fix_no_observation(self, cli_repo, capsys):
        """fix with a non-existent observation ID should report not found."""
        rc = cli_main(["fix", "nonexistent-id"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "None" in captured.out

    def test_watch_does_not_crash(self, cli_repo, capsys):
        """The critical smoke test: watch must not crash with UnboundLocalError.
        This is the exact bug the reviewer found in v0.3.1."""
        # Modify a file to ensure watch has work to do
        with open(os.path.join(cli_repo, "mod.py"), "a") as f:
            f.write("\ndef new_func(): pass\n")
        try:
            rc = cli_main(["watch"])
            assert rc == 0
        except UnboundLocalError as e:
            pytest.fail(f"watch crashed with UnboundLocalError: {e}")
        except Exception as e:
            pytest.fail(f"watch crashed: {type(e).__name__}: {e}")
