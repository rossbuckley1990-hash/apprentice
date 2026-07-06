"""
LLM-powered fix synthesis.

Given an Observation, the Apprentice can propose a fix — the 1988 spec's
"synthesis engine" for routine code changes.

The fix is always presented as a diff for the user to review and approve.
The Apprentice NEVER auto-applies fixes without confirmation.
"""

from __future__ import annotations
import os
import textwrap
from typing import Optional, List, Dict, Any

from .client import LLMClient, get_client
from ..model.store import Store
from ..model.entities import Observation


SYSTEM_PROMPT = """You are the Programmer's Apprentice, generating a fix for a code observation.

Given:
- The observation (what was found)
- The relevant source code
- The codebase context

Generate a minimal patch that addresses the observation. Output the fix as a unified diff.

Rules:
1. Output ONLY the diff, no explanations before or after.
2. Use proper unified diff format with ---/+++ headers and @@ hunks.
3. Keep changes minimal — fix the specific issue, don't refactor surrounding code.
4. If the observation is informational (not a bug), output "NO_FIX_NEEDED".
5. If you don't have enough context, output "INSUFFICIENT_CONTEXT: <what's missing>".

Example output:
```diff
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,7 +10,9 @@
 def login(username, password):
-    if validate(password):
+    if validate(password) and username:
         return create_session(username)
     return None
```"""


def _read_function_source(root: str, file_path: str, start_line: int, end_line: int) -> str:
    """Read the source code of a function from the file."""
    abs_path = os.path.join(root, file_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        start = max(0, start_line - 1)
        end = min(len(lines), end_line)
        return "".join(lines[start:end])
    except (OSError, IndexError):
        return ""


def _read_file_context(root: str, file_path: str, around_line: Optional[int] = None, radius: int = 20) -> str:
    """Read a file, optionally focused around a specific line."""
    abs_path = os.path.join(root, file_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if around_line is None:
            return "".join(lines)
        start = max(0, around_line - 1 - radius)
        end = min(len(lines), around_line + radius)
        # Add line numbers
        numbered = []
        for i, line in enumerate(lines[start:end], start=start + 1):
            marker = ">>> " if i == around_line else "    "
            numbered.append(f"{marker}{i:4d}  {line}")
        return "".join(numbered)
    except OSError:
        return f"[unable to read {file_path}]"


def propose_fix(
    store: Store,
    root: str,
    observation_id: str,
    client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Propose a fix for an observation.

    Returns:
        {
            "observation": Observation,
            "source_context": str,
            "diff": str,        # the proposed patch
            "explanation": str, # brief explanation of the fix
        }
    """
    if client is None:
        client = get_client()

    # Find the observation
    obs = None
    for o in store.unacknowledged_observations(limit=200):
        if o.id == observation_id:
            obs = o
            break
    if obs is None:
        for o in store.all_observations(limit=500):
            if o.id == observation_id:
                obs = o
                break
    if obs is None:
        return {
            "observation": None,
            "source_context": "",
            "diff": "",
            "explanation": f"Observation {observation_id} not found.",
        }

    # Gather source context
    source_context = ""
    if obs.file_path:
        source_context = _read_file_context(
            root, obs.file_path, around_line=obs.line, radius=30
        )
    elif obs.function_qualified_name:
        fn = store.get_function(obs.function_qualified_name)
        if fn:
            source_context = _read_function_source(
                root, fn.file_path, fn.start_line, fn.end_line
            )

    # Build the prompt
    obs_desc = f"""
OBSERVATION:
  kind: {obs.kind}
  severity: {obs.severity}
  message: {obs.message}
  file: {obs.file_path or '(unknown)'}
  line: {obs.line or '(unknown)'}
  function: {obs.function_qualified_name or '(unknown)'}
"""

    # Add function info if available
    fn_info = ""
    if obs.function_qualified_name:
        fn = store.get_function(obs.function_qualified_name)
        if fn:
            fn_info = f"""
FUNCTION DETAILS:
  name: {fn.qualified_name}
  args: {fn.arg_names}
  complexity: {fn.complexity}
  callers: {len(fn.callers)}
  summary: {fn.ast_summary}
"""

    user_msg = f"""{obs_desc}{fn_info}

SOURCE CONTEXT:
{source_context if source_context else '(no source available)'}

Generate a minimal diff to address this observation."""

    response = client.complete(SYSTEM_PROMPT, user_msg, max_tokens=1500)
    diff = response.text.strip()

    # Generate a brief explanation
    explanation = _explain_fix(obs, diff)

    return {
        "observation": obs,
        "source_context": source_context,
        "diff": diff,
        "explanation": explanation,
    }


def _explain_fix(obs: Observation, diff: str) -> str:
    """Generate a one-line explanation of the fix."""
    if diff == "NO_FIX_NEEDED":
        return f"This observation ({obs.kind}) is informational — no code fix needed."
    if diff.startswith("INSUFFICIENT_CONTEXT"):
        return diff
    return f"Proposed fix for {obs.kind}: {obs.message[:80]}..."
