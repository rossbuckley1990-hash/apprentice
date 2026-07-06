"""
LLM-powered summarization — the 1988 spec's "explain in terms of intent."

Generates human-readable summaries of functions, files, and the overall
codebase using the persistent model + LLM.
"""

from __future__ import annotations
import os
from typing import Optional, List

from .client import LLMClient, get_client
from ..model.store import Store
from ..model.entities import Function, File


SUMMARY_SYSTEM = """You are the Programmer's Apprentice. Summarize code concisely.
Focus on: what the code does, why it exists, and any notable patterns or issues.
Keep summaries to 2-3 sentences. Use technical language appropriate for a developer."""


def summarize_function(
    store: Store,
    root: str,
    qualified_name: str,
    client: Optional[LLMClient] = None,
) -> str:
    """Generate a natural-language summary of a function."""
    if client is None:
        client = get_client()

    fn = store.get_function(qualified_name)
    if fn is None:
        # Try suffix match
        for f in store.all_functions():
            if f.qualified_name.endswith(qualified_name) or f.name == qualified_name:
                fn = f
                break
    if fn is None:
        return f"Function '{qualified_name}' not found in the codebase model."

    # Read source
    abs_path = os.path.join(root, fn.file_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        source = "".join(lines[fn.start_line - 1 : fn.end_line])
    except OSError:
        source = "(unable to read source)"

    context = f"""FUNCTION: {fn.qualified_name}
FILE: {fn.file_path}:{fn.start_line}-{fn.end_line}
ARGS: {fn.arg_names}
COMPLEXITY: {fn.complexity}
CALLERS: {len(fn.callers)} ({', '.join(fn.callers[:3])})
AST SUMMARY: {fn.ast_summary}
DOCSTRING: {fn.docstring or '(none)'}

SOURCE:
{source}

Summarize this function in 2-3 sentences."""

    response = client.complete(SUMMARY_SYSTEM, context, max_tokens=300)
    return response.text


def summarize_codebase(
    store: Store,
    root: str,
    client: Optional[LLMClient] = None,
) -> str:
    """Generate a high-level summary of the entire codebase."""
    if client is None:
        client = get_client()

    files = store.all_files()
    functions = store.all_functions()
    classes = store.all_classes()
    cliches = store.all_cliches(min_instances=2)
    plans = store.active_plans()
    obs = store.unacknowledged_observations()

    # Find the most complex functions
    by_complexity = sorted(functions, key=lambda f: -f.complexity)[:5]
    # Find the most-called functions (hubs)
    by_callers = sorted(functions, key=lambda f: -len(f.callers))[:5]
    # Dead functions
    dead = [f for f in functions if f.is_dead]

    context = f"""CODEBASE SUMMARY REQUEST

REPO STATISTICS:
  Files: {len(files)}
  Functions: {len(functions)}
  Classes: {len(classes)}
  Cliché groups: {len(cliches)}
  Dead functions: {len(dead)}
  Active plans: {len(plans)}
  Unacknowledged observations: {len(obs)}

TOP 5 MOST COMPLEX FUNCTIONS:
{chr(10).join(f"  [{f.complexity}] {f.qualified_name} ({f.file_path}:{f.start_line})" for f in by_complexity)}

TOP 5 MOST-CALLED FUNCTIONS (hubs):
{chr(10).join(f"  [{len(f.callers)} callers] {f.qualified_name}" for f in by_callers)}

ACTIVE PLANS:
{chr(10).join(f"  [{p.id}] {p.description}" for p in plans) if plans else "  (none)"}

Provide a 4-5 sentence summary of this codebase's health and notable characteristics."""

    response = client.complete(SUMMARY_SYSTEM, context, max_tokens=500)
    return response.text
