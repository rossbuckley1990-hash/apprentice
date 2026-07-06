"""
LLM-powered natural-language Q&A about the codebase.

This is the 1988 spec's "explain in terms of intent" — the Apprentice
answers questions about your code using its persistent model + an LLM.
"""

from __future__ import annotations
import os
from typing import List, Dict, Any, Optional

from .client import LLMClient, get_client
from ..model.store import Store
from ..model.entities import Function, Plan
from ..indexer.embedder import Embedder, cosine


SYSTEM_PROMPT = """You are the Programmer's Apprentice, a persistent coding agent that maintains a living model of the user's codebase.

You have access to:
- A structured codebase model (files, functions, classes, call graph, clichés)
- The user's active plans (stated intents)
- Recent proactive observations

Answer the user's question using this context. Be specific: cite function names, file paths, and line numbers from the model. If the answer requires looking at code you don't have in context, say so.

Keep answers concise. Use code references like `module.function` or `path:line`."""


def build_context(store: Store, query: str, embedder: Optional[Embedder] = None) -> str:
    """Build a context string for the LLM from the codebase model.

    Combines:
    - Relevant functions (found by keyword + embedding similarity)
    - Active plans
    - Recent unacknowledged observations
    """
    parts: List[str] = []

    # 1. Repo overview
    parts.append(f"REPO OVERVIEW:")
    parts.append(f"  Files: {store.file_count()}")
    parts.append(f"  Functions: {store.function_count()}")
    cliches = store.all_cliches(min_instances=2)
    parts.append(f"  Cliché groups: {len(cliches)}")
    parts.append("")

    # 2. Active plans
    plans = store.active_plans()
    if plans:
        parts.append("ACTIVE PLANS:")
        for p in plans:
            parts.append(f"  [{p.id}] {p.description}")
            if p.keywords:
                parts.append(f"       keywords: {', '.join(p.keywords)}")
        parts.append("")

    # 3. Recent observations
    obs = store.unacknowledged_observations(limit=5)
    if obs:
        parts.append("RECENT OBSERVATIONS:")
        for o in obs:
            parts.append(f"  [{o.kind}] {o.message[:120]}")
        parts.append("")

    # 4. Find relevant functions by keyword
    query_terms = query.lower().split()
    matches: List[tuple[Function, int]] = []
    for fn in store.all_functions():
        score = 0
        text = (fn.name + " " + fn.qualified_name + " " +
                (fn.ast_summary or "") + " " +
                (fn.docstring or "") + " " +
                " ".join(fn.arg_names)).lower()
        for term in query_terms:
            if term in fn.name.lower():
                score += 5
            if term in text:
                score += 1
        if score > 0:
            matches.append((fn, score))

    matches.sort(key=lambda x: -x[1])

    if matches:
        parts.append(f"RELEVANT FUNCTIONS (top {min(10, len(matches))}):")
        for fn, score in matches[:10]:
            parts.append(f"  {fn.qualified_name}")
            parts.append(f"    file: {fn.file_path}:{fn.start_line}")
            parts.append(f"    summary: {fn.ast_summary}")
            if fn.docstring:
                parts.append(f"    doc: {fn.docstring[:100]}")
            parts.append(f"    callers: {len(fn.callers)}, complexity: {fn.complexity}")
        parts.append("")

    # 5. If we have embeddings, try semantic search
    if embedder and matches:
        try:
            top_match = matches[0][0]
            similar = embedder.find_similar(store, top_match.qualified_name, top_k=5)
            if similar:
                parts.append("SEMANTICALLY SIMILAR FUNCTIONS:")
                for qn, sim in similar[:5]:
                    parts.append(f"  [{sim:.2f}] {qn}")
                parts.append("")
        except Exception:
            pass  # embeddings might not be ready

    return "\n".join(parts)


def ask(
    store: Store,
    root: str,
    question: str,
    client: Optional[LLMClient] = None,
    embedder: Optional[Embedder] = None,
) -> str:
    """Answer a natural-language question about the codebase."""
    if client is None:
        client = get_client()
    if embedder is None:
        embedder = Embedder()

    context = build_context(store, question, embedder)
    user_msg = f"CODEBASE CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    response = client.complete(SYSTEM_PROMPT, user_msg, max_tokens=1500)
    return response.text
