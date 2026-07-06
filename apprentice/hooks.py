"""
Git hook integration.

`apprentice hook install` creates a pre-commit hook that runs `apprentice watch`
on staged files. If any 'error' severity observations are found, the commit
is blocked (configurable).
"""

from __future__ import annotations
import os
import sys
import stat
from pathlib import Path
from typing import List, Optional

from .config import Config


HOOK_SCRIPT = """#!/bin/bash
# Apprentice pre-commit hook
# Runs proactive analysis on staged files and blocks commit on errors.
#
# Exit code contract:
#   0 = no blocking observations (commit allowed)
#   1 = blocking observations found (commit blocked)
#   2 = analyzer crashed (fail-closed: block to be safe)
#
# This hook is fail-closed: if the Apprentice is not installed or crashes,
# the commit is blocked. Override with SKIP_APPRENTICE=1 git commit ...

set -e

# Allow bypass via env var
if [ "$SKIP_APPRENTICE" = "1" ]; then
    echo "  [apprentice] SKIP_APPRENTICE=1, skipping pre-commit check"
    exit 0
fi

# Check if apprentice is installed
if ! command -v apprentice &>/dev/null; then
    echo "  [apprentice] not found — install with 'pip install apprentice'"
    echo "  [apprentice] To bypass: SKIP_APPRENTICE=1 git commit ..."
    exit 2
fi

# Get staged files (Python + JS/TS)
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\\.(py|js|ts|jsx|tsx)$' || true)

if [ -z "$STAGED" ]; then
    exit 0
fi

echo "  [apprentice] analyzing staged files..."

# Run watch on staged files only
# The --staged flag produces exit code:
#   0 = no blocking observations
#   1 = error-severity observations found
#   2 = analyzer crash
set +e
apprentice watch --staged
EXIT_CODE=$?
set -e

case $EXIT_CODE in
    0)
        echo "  [apprentice] OK — no blocking observations"
        exit 0
        ;;
    1)
        echo ""
        echo "  [apprentice] BLOCKING COMMIT — error-severity observations found"
        echo "  To bypass: SKIP_APPRENTICE=1 git commit ..."
        echo "  To fix: address the observations above, then re-commit"
        exit 1
        ;;
    *)
        echo ""
        echo "  [apprentice] FAIL-CLOSED — analyzer exited with code $EXIT_CODE"
        echo "  This may indicate a crash. To bypass: SKIP_APPRENTICE=1 git commit ..."
        exit 2
        ;;
esac
"""


def install_hook(repo_root: str) -> str:
    """Install the pre-commit hook. Returns the hook path."""
    hook_dir = Path(repo_root) / ".git" / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hook_dir / "pre-commit"

    with open(hook_path, "w") as f:
        f.write(HOOK_SCRIPT)

    # Make executable
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return str(hook_path)


def uninstall_hook(repo_root: str) -> bool:
    """Remove the Apprentice pre-commit hook. Returns True if removed."""
    hook_path = Path(repo_root) / ".git" / "hooks" / "pre-commit"
    if hook_path.exists():
        # Only remove if it's our hook
        with open(hook_path, "r") as f:
            content = f.read()
        if "Apprentice pre-commit hook" in content:
            hook_path.unlink()
            return True
    return False


def is_hook_installed(repo_root: str) -> bool:
    """Check if the Apprentice hook is installed."""
    hook_path = Path(repo_root) / ".git" / "hooks" / "pre-commit"
    if not hook_path.exists():
        return False
    try:
        with open(hook_path, "r") as f:
            content = f.read()
        return "Apprentice pre-commit hook" in content
    except OSError:
        return False
