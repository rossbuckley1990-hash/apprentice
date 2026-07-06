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

set -e

# Check if apprentice is installed
if ! command -v apprentice &>/dev/null; then
    echo "  [apprentice] not found — skipping pre-commit check"
    exit 0
fi

# Get staged Python/JS files
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\\.(py|js|ts|jsx|tsx)$' || true)

if [ -z "$STAGED" ]; then
    exit 0
fi

echo "  [apprentice] analyzing staged files..."

# Run watch on staged files only
OUTPUT=$(apprentice watch --staged 2>&1) || true

# Check for error-severity observations
if echo "$OUTPUT" | grep -q "severity: error"; then
    echo ""
    echo "  [apprentice] BLOCKING COMMIT — error-severity observations found:"
    echo "$OUTPUT" | grep -A2 "severity: error"
    echo ""
    echo "  To bypass: git commit --no-verify"
    echo "  To fix: address the observations above, then re-commit"
    exit 1
fi

# Show warnings but don't block
if echo "$OUTPUT" | grep -q "severity: warning"; then
    echo "  [apprentice] warnings found (commit allowed):"
    echo "$OUTPUT" | grep -A1 "severity: warning" | head -10
fi

echo "  [apprentice] OK — no blocking observations"
exit 0
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
