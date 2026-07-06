"""
The Apprentice CLI.

Commands:
  apprentice init              Initialize the Apprentice in this repo
  apprentice index [--rebuild] Index (or re-index) the codebase
  apprentice status            Show what the Apprentice knows
  apprentice plan <text>       State an intent (the Apprentice tracks it)
  apprentice plan --list       List active plans
  apprentice plan --done ID    Mark a plan completed
  apprentice watch [--all]     Run proactive analyzers (on changed files, or all)
  apprentice observations      Show unacknowledged observations
  apprentice ack <ID>          Acknowledge an observation
  apprentice ask <question>    Ask about the codebase (uses persistent model)
  apprentice recall <name>     Show what the Apprentice knows about a function
  apprentice similar <name>    Find functions similar to this one
"""

from __future__ import annotations
import os
import sys
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Optional

from .. import __version__
from ..model.entities import Plan, Observation
from ..model.store import Store, init_store, default_db_path
from ..indexer.python_parser import index_repo, discover_python_files
from ..indexer.embedder import Embedder
from ..analyzer.proactive import run_all_analyzers


# =============================================================================
# Helpers
# =============================================================================

def find_repo_root(start: Optional[str] = None) -> str:
    """Find the repo root by looking for .apprentice or .git."""
    p = Path(start or os.getcwd()).resolve()
    while True:
        if (p / ".apprentice").is_dir():
            return str(p)
        if (p / ".git").is_dir():
            return str(p)
        if p.parent == p:
            # Reached filesystem root — use cwd
            return os.getcwd()
        p = p.parent


def get_store() -> tuple[str, Store]:
    root = find_repo_root()
    store = init_store(root)
    return root, store


def changed_files_since_last_index(store: Store, root: str) -> List[str]:
    """Return files whose content_hash differs from what's stored, plus new files."""
    from ..model.entities import hash_content
    changed = []
    for rel_path in discover_python_files(root):
        abs_path = os.path.join(root, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        new_hash = hash_content(content)
        existing = store.get_file(rel_path)
        if not existing or existing.content_hash != new_hash:
            changed.append(rel_path)
    return changed


# =============================================================================
# Commands
# =============================================================================

def cmd_init(args):
    root = os.getcwd()
    store = init_store(root)
    gitignore = Path(root) / ".gitignore"
    if gitignore.exists():
        with open(gitignore, "r") as f:
            content = f.read()
        if ".apprentice/" not in content:
            with open(gitignore, "a") as f:
                if not content.endswith("\n"):
                    f.write("\n")
                f.write("# Apprentice local state\n.apprentice/\n")
            print("  Added .apprentice/ to .gitignore")
    print(f"  Apprentice initialized at {root}")
    print(f"  State: {default_db_path(root)}")
    print(f"  Next: run `apprentice index` to build the codebase model.")


def cmd_index(args):
    root, store = get_store()
    if args.rebuild:
        # Clear existing model
        Path(default_db_path(root)).unlink(missing_ok=True)
        store = init_store(root)
    print(f"  Indexing {root} ...")
    stats = index_repo(root, store, verbose=True)
    print()
    print(f"  Files indexed:     {stats['files']}")
    print(f"  Files changed:     {stats['changed']}")
    print(f"  Functions found:   {stats['functions']}")
    print(f"  Classes found:     {stats['classes']}")
    print(f"  Dead functions:    {stats['dead_functions']}")
    print(f"  Cliché groups:     {stats['cliches']}")

    # Embeddings
    print()
    print("  Computing embeddings (offline TF-IDF by default)...")
    embedder = Embedder()
    n_emb = embedder.index_all(store, root, force=args.rebuild)
    print(f"  Embedded {n_emb} functions using backend: {embedder.backend}")


def cmd_status(args):
    root, store = get_store()
    last_snap = store.last_snapshot()
    plans = store.active_plans()
    unacked = store.unacknowledged_observations()

    print(f"  Apprentice v{__version__}")
    print(f"  Repo: {root}")
    print(f"  Files in model:    {store.file_count()}")
    print(f"  Functions in model: {store.function_count()}")
    print(f"  Active plans:      {len(plans)}")
    print(f"  Unacked observations: {len(unacked)}")
    if last_snap:
        print(f"  Last watch run:    {last_snap['run_at']}")
        print(f"    (checked {last_snap['files_checked']} files, emitted {last_snap['observations_emitted']} observations)")
    if plans:
        print()
        print("  Active plans:")
        for p in plans:
            print(f"    [{p.id}] {p.description[:80]}")


def cmd_plan(args):
    root, store = get_store()
    if args.list:
        plans = store.all_plans()
        if not plans:
            print("  No plans.")
            return
        for p in plans:
            status_mark = "✓" if p.status == "completed" else ("✗" if p.status == "abandoned" else "●")
            print(f"  {status_mark} [{p.id}] {p.description[:80]}")
            if p.status == "active":
                print(f"     keywords: {', '.join(p.keywords[:5])}")
                print(f"     created:  {p.created}")
        return

    if args.done:
        plan = store.get_plan(args.done)
        if plan is None:
            print(f"  No plan with id {args.done}")
            return
        plan.status = "completed"
        store.upsert_plan(plan)
        print(f"  Marked plan {args.done} as completed.")
        return

    if not args.text:
        print("  Usage: apprentice plan <description of what you're doing>")
        print("         apprentice plan --list")
        print("         apprentice plan --done <id>")
        return

    plan_id = uuid.uuid4().hex[:8]
    description = " ".join(args.text)
    # Extract keywords (use the same drift-keyword extractor)
    from ..analyzer.proactive import _extract_drift_keywords
    keywords = sorted(_extract_drift_keywords(description))
    plan = Plan(
        id=plan_id,
        description=description,
        keywords=keywords,
    )
    store.upsert_plan(plan)
    print(f"  Plan [{plan_id}] recorded: {description}")
    print(f"  Detected keywords: {', '.join(keywords) if keywords else '(none — generic)'}")
    print(f"  The Apprentice will check new code against this plan on `apprentice watch`.")


def cmd_watch(args):
    root, store = get_store()
    if args.all:
        changed = discover_python_files(root)
    else:
        changed = changed_files_since_last_index(store, root)
    if not changed:
        print("  No changes since last index. Use --all to analyze everything.")
        return

    print(f"  Analyzing {len(changed)} changed files...")
    # Re-index to update the model with the latest changes
    print("  (re-indexing to refresh the model...)")
    index_repo(root, store, verbose=False)

    observations = run_all_analyzers(store, root, changed)

    # Persist observations
    for obs in observations:
        store.add_observation(obs)

    store.log_snapshot(
        files_checked=len(changed),
        observations_emitted=len(observations),
        notes=f"watch on {len(changed)} files",
    )

    if not observations:
        print("  No new observations. The codebase looks consistent with active plans.")
        return

    print()
    print(f"  {len(observations)} observations:")
    print()
    _print_observations(observations)


def cmd_observations(args):
    root, store = get_store()
    if args.all:
        obs = store.all_observations(limit=200)
    else:
        obs = store.unacknowledged_observations(limit=100)
    if not obs:
        print("  No observations.")
        return
    _print_observations(obs)


def cmd_ack(args):
    root, store = get_store()
    for obs_id in args.ids:
        store.acknowledge_observation(obs_id)
        print(f"  Acknowledged: {obs_id}")


def cmd_ask(args):
    """Heuristic 'ask' — searches functions by keyword in name/summary.
    A real LLM-backed version would use the embeddings + an LLM call."""
    root, store = get_store()
    query = " ".join(args.query).lower()
    query_terms = query.split()

    matches = []
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
    if not matches:
        print(f"  No functions matching '{query}'.")
        return
    print(f"  Top matches for '{query}':")
    for fn, score in matches[:10]:
        print(f"    [{score:>2}] {fn.qualified_name}")
        print(f"          {fn.file_path}:{fn.start_line}")
        print(f"          {fn.ast_summary}")
        if fn.docstring:
            print(f"          \"{fn.docstring[:80]}\"")
        if fn.is_dead:
            print(f"          (no callers — dead code)")


def cmd_recall(args):
    root, store = get_store()
    qname = args.name
    fn = store.get_function(qname)
    if fn is None:
        # Try to find by suffix match
        for f in store.all_functions():
            if f.qualified_name.endswith(qname) or f.name == qname:
                fn = f
                break
    if fn is None:
        print(f"  No function found matching '{qname}'.")
        return
    print(f"  {fn.qualified_name}")
    print(f"  file:    {fn.file_path}:{fn.start_line}-{fn.end_line}")
    print(f"  args:    {fn.arg_names}")
    print(f"  summary: {fn.ast_summary}")
    print(f"  complexity: {fn.complexity}")
    if fn.docstring:
        print(f"  doc:     {fn.docstring}")
    print(f"  callers: {len(fn.callers)}")
    for c in fn.callers[:5]:
        print(f"    - {c}")
    if len(fn.callers) > 5:
        print(f"    ... and {len(fn.callers) - 5} more")
    if fn.is_dead:
        print("  ⚠ no callers (dead code)")


def cmd_similar(args):
    root, store = get_store()
    qname = args.name
    # Resolve
    fn = store.get_function(qname)
    if fn is None:
        for f in store.all_functions():
            if f.qualified_name.endswith(qname) or f.name == qname:
                fn = f
                qname = fn.qualified_name
                break
    if fn is None:
        print(f"  No function found matching '{qname}'.")
        return
    embedder = Embedder()
    # Make sure embeddings exist
    existing = store.get_embedding(qname)
    if existing is None:
        print(f"  Computing embeddings (one-time)...")
        embedder.index_all(store, root)
    similar = embedder.find_similar(store, qname, top_k=10)
    if not similar:
        print(f"  No similar functions found (or embeddings missing).")
        return
    print(f"  Functions similar to {qname}:")
    for qn, sim in similar:
        print(f"    [{sim:.3f}] {qn}")


# =============================================================================
# Output formatting
# =============================================================================

SEVERITY_SYMBOL = {
    "error": "✗",
    "warning": "⚠",
    "info": "●",
}


def _print_observations(obs_list: List[Observation]):
    for obs in obs_list:
        sym = SEVERITY_SYMBOL.get(obs.severity, "?")
        ack = "" if not obs.acknowledged else " (acknowledged)"
        print(f"  {sym} [{obs.kind}] {obs.id}{ack}")
        print(f"     {obs.message}")
        loc_parts = []
        if obs.file_path:
            loc_parts.append(obs.file_path)
        if obs.line:
            loc_parts.append(f"line {obs.line}")
        if obs.function_qualified_name:
            loc_parts.append(f"fn {obs.function_qualified_name}")
        if loc_parts:
            print(f"     location: {' '.join(loc_parts)}")
        print()


# =============================================================================
# Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apprentice",
        description="The Programmer's Apprentice — a persistent, proactive coding agent.",
    )
    p.add_argument("--version", action="version", version=f"apprentice {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize the Apprentice in this repo")
    p_init.set_defaults(func=cmd_init)

    p_index = sub.add_parser("index", help="Index the codebase")
    p_index.add_argument("--rebuild", action="store_true", help="Rebuild the model from scratch")
    p_index.set_defaults(func=cmd_index)

    p_status = sub.add_parser("status", help="Show what the Apprentice knows")
    p_status.set_defaults(func=cmd_status)

    p_plan = sub.add_parser("plan", help="State an intent (the Apprentice tracks it)")
    p_plan.add_argument("text", nargs="*", help="Plan description")
    p_plan.add_argument("--list", action="store_true", help="List all plans")
    p_plan.add_argument("--done", metavar="ID", help="Mark a plan as completed")
    p_plan.set_defaults(func=cmd_plan)

    p_watch = sub.add_parser("watch", help="Run proactive analyzers on changed files")
    p_watch.add_argument("--all", action="store_true", help="Analyze all files, not just changed")
    p_watch.set_defaults(func=cmd_watch)

    p_obs = sub.add_parser("observations", help="Show observations")
    p_obs.add_argument("--all", action="store_true", help="Show acknowledged too")
    p_obs.set_defaults(func=cmd_observations)

    p_ack = sub.add_parser("ack", help="Acknowledge observations")
    p_ack.add_argument("ids", nargs="+", help="Observation IDs to acknowledge")
    p_ack.set_defaults(func=cmd_ack)

    p_ask = sub.add_parser("ask", help="Ask about the codebase (keyword search)")
    p_ask.add_argument("query", nargs="+", help="Search terms")
    p_ask.set_defaults(func=cmd_ask)

    p_recall = sub.add_parser("recall", help="Show what the Apprentice knows about a function")
    p_recall.add_argument("name", help="Function name or qualified name")
    p_recall.set_defaults(func=cmd_recall)

    p_similar = sub.add_parser("similar", help="Find functions similar to this one")
    p_similar.add_argument("name", help="Function name or qualified name")
    p_similar.set_defaults(func=cmd_similar)

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        return 130
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
