"""
The Apprentice CLI — v0.2.0 with LLM, daemon, hooks, history.

Commands:
  apprentice init              Initialize
  apprentice index [--rebuild] Index the codebase
  apprentice status            Show what the Apprentice knows

  apprentice plan <text>       State an intent
  apprentice plan --list       List plans
  apprentice plan --done ID    Mark a plan completed

  apprentice watch [--all] [--staged]  Run proactive analyzers
  apprentice observations      Show observations
  apprentice ack <ID>          Acknowledge observations

  apprentice ask <question>    Natural-language Q&A (LLM-powered)
  apprentice fix <obs-id>      Propose a fix for an observation (LLM-powered)
  apprentice summarize <name>  Summarize a function (LLM-powered)
  apprentice summarize --codebase  Summarize the whole codebase

  apprentice recall <name>     Show what the Apprentice knows about a function
  apprentice similar <name>    Find similar functions
  apprentice history <name>    Show a function's complexity history

  apprentice daemon            Run as a background watcher
  apprentice hook install      Install git pre-commit hook
  apprentice hook uninstall    Remove the hook
  apprentice hook status       Check if hook is installed

  apprentice config            Show current configuration
  apprentice config --init     Create .apprentice.toml with defaults
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
from ..indexer.python_parser import index_repo, discover_all_files
from ..indexer.embedder import Embedder
from ..analyzer.proactive import run_all_analyzers
from ..config import load_config, save_config, Config
from . import output


# =============================================================================
# Helpers
# =============================================================================

def find_repo_root(start: Optional[str] = None) -> str:
    p = Path(start or os.getcwd()).resolve()
    while True:
        if (p / ".apprentice").is_dir():
            return str(p)
        if (p / ".git").is_dir():
            return str(p)
        if p.parent == p:
            return os.getcwd()
        p = p.parent


def get_store_and_config() -> tuple[str, Store, Config]:
    root = find_repo_root()
    config = load_config(root)
    store = init_store(root)
    return root, store, config


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
    print(f"  Apprentice v{__version__} initialized at {root}")
    print(f"  State: {default_db_path(root)}")
    print(f"  Next: run `apprentice index` to build the codebase model.")


def cmd_index(args):
    root, store, config = get_store_and_config()
    if args.rebuild:
        Path(default_db_path(root)).unlink(missing_ok=True)
        store = init_store(root)
    print(f"  Indexing {root} ...")
    stats = index_repo(root, store, verbose=True, config=config)
    print()
    print(f"  Files indexed:     {stats['files']}")
    print(f"  Files changed:     {stats['changed']}")
    print(f"  Functions found:   {stats['functions']}")
    print(f"  Classes found:     {stats['classes']}")
    print(f"  Dead functions:    {stats['dead_functions']}")
    print(f"  Cliché groups:     {stats['cliches']}")

    print()
    print("  Computing embeddings...")
    embedder = Embedder(backend=config.embedding_backend)
    n_emb = embedder.index_all(store, root, force=args.rebuild)
    print(f"  Embedded {n_emb} functions using backend: {embedder.backend}")


def cmd_status(args):
    root, store, config = get_store_and_config()
    last_snap = store.last_snapshot()
    plans = store.active_plans()
    unacked = store.unacknowledged_observations()

    stats = {
        "version": __version__,
        "repo": root,
        "files": store.file_count(),
        "functions": store.function_count(),
        "plans": len(plans),
        "unacked": len(unacked),
        "last_snapshot": last_snap["run_at"] if last_snap else None,
        "active_plans": [{"id": p.id, "description": p.description} for p in plans],
    }
    print(output.format_status(stats, use_color=config.color))


def cmd_plan(args):
    root, store, config = get_store_and_config()
    if args.list:
        plans = store.all_plans()
        if not plans:
            print("  No plans.")
            return
        for p in plans:
            print(output.format_plan(p, use_color=config.color))
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
        print("  Usage: apprentice plan <description>")
        print("         apprentice plan --list")
        print("         apprentice plan --done <id>")
        return

    plan_id = uuid.uuid4().hex[:8]
    description = " ".join(args.text)
    from ..analyzer.proactive import _extract_drift_keywords
    keywords = sorted(_extract_drift_keywords(description))
    plan = Plan(id=plan_id, description=description, keywords=keywords)
    store.upsert_plan(plan)
    print(f"  Plan [{plan_id}] recorded: {description}")
    print(f"  Detected keywords: {', '.join(keywords) if keywords else '(none)'}")
    print(f"  The Apprentice will check new code against this plan on `apprentice watch`.")


def cmd_watch(args):
    root, store, config = get_store_and_config()

    if args.staged:
        # Git hook mode: only check staged files
        import subprocess
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True, text=True, cwd=root,
        )
        staged = [f for f in result.stdout.strip().split("\n") if f and f.endswith((".py", ".js", ".ts", ".jsx", ".tsx"))]
        if not staged:
            print("  No staged files to analyze.")
            return
        changed = staged
    elif args.all:
        changed = discover_all_files(root, config)
    else:
        changed = changed_files_since_last_index(store, root, config)

    if not changed:
        print("  No changes since last index. Use --all to analyze everything.")
        return

    print(f"  Analyzing {len(changed)} changed files...")
    index_repo(root, store, verbose=False, config=config)

    observations = run_all_analyzers(store, root, changed, config=config)

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
    print(output.format_observations(observations, use_color=config.color))

    # In staged mode, also print severity summary for the hook
    if args.staged:
        errors = [o for o in observations if o.severity == "error"]
        warnings = [o for o in observations if o.severity == "warning"]
        if errors:
            print(f"severity: error ({len(errors)} found)")
        if warnings:
            print(f"severity: warning ({len(warnings)} found)")


def cmd_observations(args):
    root, store, config = get_store_and_config()
    if args.all:
        obs = store.all_observations(limit=200)
    else:
        obs = store.unacknowledged_observations(limit=100)
    if not obs:
        print("  No observations.")
        return
    print(output.format_observations(obs, use_color=config.color))


def cmd_ack(args):
    root, store, config = get_store_and_config()
    for obs_id in args.ids:
        store.acknowledge_observation(obs_id)
        print(f"  Acknowledged: {obs_id}")


def cmd_ask(args):
    """LLM-powered natural-language Q&A."""
    root, store, config = get_store_and_config()
    question = " ".join(args.query)

    from ..llm.client import get_client
    from ..llm.ask import ask

    client = get_client(config={
        "llm_backend": config.llm_backend,
        "llm_model": config.llm_model,
    })
    embedder = Embedder(backend=config.embedding_backend)

    if not client.is_real():
        print("  [no LLM configured — using mock mode]")
        print("  Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or ZAI_API_KEY for real answers.")
        print()

    print(f"  Question: {question}")
    print()
    answer = ask(store, root, question, client=client, embedder=embedder)
    print(answer)


def cmd_fix(args):
    """LLM-powered fix synthesis."""
    root, store, config = get_store_and_config()

    from ..llm.client import get_client
    from ..llm.fix import propose_fix

    client = get_client(config={
        "llm_backend": config.llm_backend,
        "llm_model": config.llm_model,
    })

    result = propose_fix(store, root, args.observation_id, client=client)

    if result["observation"] is None:
        print(f"  {result['explanation']}")
        return

    obs = result["observation"]
    print(f"  Observation: [{obs.kind}] {obs.message}")
    print()

    diff = result["diff"]

    if diff == "NO_FIX_NEEDED":
        print("  No fix needed — this observation is informational.")
        return

    if diff.startswith("INSUFFICIENT_CONTEXT"):
        print(f"  {diff}")
        return

    print("  Proposed fix:")
    print()
    output.print_diff(diff, use_color=config.color)
    print()

    # Save the explanation
    store.save_observation_explanation(
        observation_id=args.observation_id,
        explanation=result["explanation"],
        diff=diff,
        llm_backend=client.backend,
    )

    # Ask if the user wants to apply
    if args.apply:
        print("  --apply not yet implemented. Apply the diff manually.")
    else:
        print("  To apply: pipe the diff to `git apply` or use your editor.")


def cmd_summarize(args):
    """LLM-powered summarization."""
    root, store, config = get_store_and_config()

    from ..llm.client import get_client
    from ..llm.summarize import summarize_function, summarize_codebase

    client = get_client(config={
        "llm_backend": config.llm_backend,
        "llm_model": config.llm_model,
    })

    if args.codebase:
        print("  Summarizing codebase...")
        summary = summarize_codebase(store, root, client=client)
        print()
        print(summary)
    elif args.name:
        print(f"  Summarizing {args.name}...")
        summary = summarize_function(store, root, args.name, client=client)
        print()
        print(summary)
    else:
        print("  Usage: apprentice summarize <function-name>")
        print("         apprentice summarize --codebase")


def cmd_recall(args):
    root, store, config = get_store_and_config()
    qname = args.name
    fn = store.get_function(qname)
    if fn is None:
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

    # Show history if available
    history = store.function_history(fn.qualified_name)
    if history:
        print(f"  history: {len(history)} snapshots")
        for h in history[:3]:
            print(f"    {h['snapshot_at']}: complexity={h['complexity']}, callers={h['caller_count']}")
        if len(history) > 3:
            print(f"    ... and {len(history) - 3} more")


def cmd_similar(args):
    root, store, config = get_store_and_config()
    qname = args.name
    fn = store.get_function(qname)
    if fn is None:
        for f in store.all_functions():
            if f.qualified_name.endswith(qname) or f.name == qname:
                qname = f.qualified_name
                break
    embedder = Embedder(backend=config.embedding_backend)
    existing = store.get_embedding(qname)
    if existing is None:
        print(f"  Computing embeddings...")
        embedder.index_all(store, root)
    similar = embedder.find_similar(store, qname, top_k=10)
    if not similar:
        print(f"  No similar functions found.")
        return
    print(f"  Functions similar to {qname}:")
    for qn, sim in similar:
        print(f"    [{sim:.3f}] {qn}")


def cmd_history(args):
    """Show a function's complexity history."""
    root, store, config = get_store_and_config()
    qname = args.name
    fn = store.get_function(qname)
    if fn is None:
        for f in store.all_functions():
            if f.qualified_name.endswith(qname) or f.name == qname:
                fn = f
                qname = f.qualified_name
                break
    if fn is None:
        print(f"  No function found matching '{qname}'.")
        return

    history = store.function_history(qname)
    if not history:
        print(f"  No history yet for {qname}.")
        print(f"  History is recorded on each `apprentice index` or `apprentice watch`.")
        return

    print(f"  History for {qname}:")
    print(f"  {'snapshot':<28} {'complexity':>11} {'lines':>6} {'callers':>8} {'body_hash':>18}")
    print("  " + "-" * 75)
    for h in reversed(history):
        print(f"  {h['snapshot_at']:<28} {h['complexity']:>11} {h['line_count']:>6} {h['caller_count']:>8} {h['body_hash']:>18}")


def cmd_daemon(args):
    """Run as a background watcher."""
    root, store, config = get_store_and_config()
    from ..daemon import Daemon
    daemon = Daemon(root, config)
    daemon.run(interval=args.interval or config.watch_interval_seconds)


def cmd_hook(args):
    """Manage git hooks."""
    root = find_repo_root()
    from ..hooks import install_hook, uninstall_hook, is_hook_installed

    if args.action == "install":
        path = install_hook(root)
        print(f"  Installed pre-commit hook at {path}")
    elif args.action == "uninstall":
        if uninstall_hook(root):
            print("  Removed pre-commit hook.")
        else:
            print("  No Apprentice hook found.")
    elif args.action == "status":
        if is_hook_installed(root):
            print("  Apprentice pre-commit hook is installed.")
        else:
            print("  No Apprentice hook installed. Run `apprentice hook install`.")
    else:
        print("  Usage: apprentice hook install|uninstall|status")


def cmd_config(args):
    """Show or initialize configuration."""
    root = find_repo_root()
    if args.init:
        config = Config.defaults()
        save_config(config, root)
        print(f"  Created {root}/.apprentice.toml with default settings.")
        print(f"  Edit this file to customize thresholds, LLM backend, etc.")
    else:
        config = load_config(root)
        print("  Current configuration:")
        print(f"    LLM backend:        {config.llm_backend or '(auto)'}")
        print(f"    LLM model:          {config.llm_model or '(default)'}")
        print(f"    Embedding backend:  {config.embedding_backend}")
        print(f"    Complexity warn:    {config.complexity_warn}")
        print(f"    Complexity error:   {config.complexity_error}")
        print(f"    Watch interval:     {config.watch_interval_seconds}s")
        print(f"    Hook block errors:  {config.hook_block_on_error}")
        print(f"    Hook block warns:   {config.hook_block_on_warning}")
        print(f"    Color output:       {config.color}")
        print(f"    Ignore dirs:        {', '.join(config.ignore_dirs[:5])}...")
        print(f"    File extensions:    {', '.join(config.file_extensions)}")
        config_path = Path(root) / ".apprentice.toml"
        print()
        print(f"    Config file: {'exists' if config_path.exists() else 'not found'}")


# =============================================================================
# Helpers
# =============================================================================

def changed_files_since_last_index(store: Store, root: str, config=None) -> List[str]:
    changed = []
    for rel_path in discover_all_files(root, config):
        abs_path = os.path.join(root, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        new_hash = hash_content(content)
        from ..model.entities import hash_content
        existing = store.get_file(rel_path)
        if not existing or existing.content_hash != new_hash:
            changed.append(rel_path)
    return changed


def _print_observations(obs_list: List[Observation], config):
    print(output.format_observations(obs_list, use_color=config.color))


# =============================================================================
# Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apprentice",
        description="The Programmer's Apprentice — persistent, proactive coding agent.",
    )
    p.add_argument("--version", action="version", version=f"apprentice {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize")
    p_init.set_defaults(func=cmd_init)

    p_index = sub.add_parser("index", help="Index the codebase")
    p_index.add_argument("--rebuild", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_status = sub.add_parser("status", help="Show status")
    p_status.set_defaults(func=cmd_status)

    p_plan = sub.add_parser("plan", help="State an intent")
    p_plan.add_argument("text", nargs="*")
    p_plan.add_argument("--list", action="store_true")
    p_plan.add_argument("--done", metavar="ID")
    p_plan.set_defaults(func=cmd_plan)

    p_watch = sub.add_parser("watch", help="Run proactive analyzers")
    p_watch.add_argument("--all", action="store_true")
    p_watch.add_argument("--staged", action="store_true", help="Only check git-staged files")
    p_watch.set_defaults(func=cmd_watch)

    p_obs = sub.add_parser("observations", help="Show observations")
    p_obs.add_argument("--all", action="store_true")
    p_obs.set_defaults(func=cmd_observations)

    p_ack = sub.add_parser("ack", help="Acknowledge observations")
    p_ack.add_argument("ids", nargs="+")
    p_ack.set_defaults(func=cmd_ack)

    p_ask = sub.add_parser("ask", help="Natural-language Q&A (LLM-powered)")
    p_ask.add_argument("query", nargs="+")
    p_ask.set_defaults(func=cmd_ask)

    p_fix = sub.add_parser("fix", help="Propose a fix (LLM-powered)")
    p_fix.add_argument("observation_id")
    p_fix.add_argument("--apply", action="store_true", help="Apply the fix (not yet implemented)")
    p_fix.set_defaults(func=cmd_fix)

    p_summ = sub.add_parser("summarize", help="Summarize a function or codebase")
    p_summ.add_argument("name", nargs="?", help="Function name")
    p_summ.add_argument("--codebase", action="store_true")
    p_summ.set_defaults(func=cmd_summarize)

    p_recall = sub.add_parser("recall", help="Show what the Apprentice knows about a function")
    p_recall.add_argument("name")
    p_recall.set_defaults(func=cmd_recall)

    p_similar = sub.add_parser("similar", help="Find similar functions")
    p_similar.add_argument("name")
    p_similar.set_defaults(func=cmd_similar)

    p_hist = sub.add_parser("history", help="Show a function's complexity history")
    p_hist.add_argument("name")
    p_hist.set_defaults(func=cmd_history)

    p_daemon = sub.add_parser("daemon", help="Run as background watcher")
    p_daemon.add_argument("--interval", type=float, default=None)
    p_daemon.set_defaults(func=cmd_daemon)

    p_hook = sub.add_parser("hook", help="Manage git hooks")
    p_hook.add_argument("action", choices=["install", "uninstall", "status"])
    p_hook.set_defaults(func=cmd_hook)

    p_conf = sub.add_parser("config", help="Show or init configuration")
    p_conf.add_argument("--init", action="store_true")
    p_conf.set_defaults(func=cmd_config)

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
