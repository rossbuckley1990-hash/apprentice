"""
The proactive analyzer — what makes this the Apprentice, not Copilot.

This is the PROACTIVITY: it runs WITHOUT being asked (on `apprentice watch`)
and emits Observations about drift, duplication, dead code, complexity creep,
and TODOs introduced without a plan entry.

Each analyzer is a pure function: (store, root, context) -> List[Observation].
They never modify the store directly — the orchestrator does that.
"""

from __future__ import annotations
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Set, Dict, Tuple

from ..model.entities import (
    File, Function, Class, Plan, Observation, hash_content,
)
from ..model.store import Store


# =============================================================================
# Observation factory
# =============================================================================

def _obs(
    kind: str, severity: str, message: str,
    file_path: Optional[str] = None,
    function_qualified_name: Optional[str] = None,
    line: Optional[int] = None,
    related_plan_id: Optional[str] = None,
    related_function_qualified_name: Optional[str] = None,
) -> Observation:
    return Observation(
        id=uuid.uuid4().hex[:12],
        kind=kind, severity=severity, message=message,
        file_path=file_path,
        function_qualified_name=function_qualified_name,
        line=line,
        related_plan_id=related_plan_id,
        related_function_qualified_name=related_function_qualified_name,
    )


# =============================================================================
# Analyzer 1: Plan drift detection
# =============================================================================

# Heuristic keywords associated with common engineering activities.
# Used to detect when new code doesn't match any active plan's keywords.
DRIFT_KEYWORDS = {
    "refactor": ["refactor", "rename", "extract", "move", "reorganize", "cleanup"],
    "test": ["test", "spec", "fixture", "mock"],
    "auth": ["auth", "login", "session", "password", "jwt", "oauth", "token"],
    "api": ["api", "endpoint", "route", "handler", "request", "response"],
    "ui": ["ui", "component", "render", "css", "layout", "view"],
    "db": ["db", "database", "schema", "migration", "sql", "table", "query"],
    "perf": ["perf", "performance", "optimize", "cache", "speed", "latency"],
    "bug": ["bug", "fix", "regression", "patch"],
    "docs": ["doc", "documentation", "readme", "comment"],
    "config": ["config", "settings", "env", "deploy"],
    "security": ["security", "vulnerability", "cve", "sanitize", "escape"],
}


def _extract_drift_keywords(text: str) -> Set[str]:
    """Extract semantic keywords from a plan description or commit message.
    Uses word-boundary prefix matching: 'auth' matches 'authentication' and
    'authenticate' but NOT 'build' (no word boundary before 'ui' in 'build')."""
    text_lower = text.lower()
    found = set()
    for category, words in DRIFT_KEYWORDS.items():
        for w in words:
            # \b before the keyword ensures it's at the start of a word
            # No trailing \b so it matches as a prefix (auth → authentication)
            if re.search(r"\b" + re.escape(w), text_lower):
                found.add(category)
    return found


def analyze_plan_drift(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """If active plans exist, check whether changed files match the plan's
    keyword domain. Flag drift when new code appears unrelated to any plan.

    This is the Apprentice knowing your intent and checking new code against it.
    """
    obs: List[Observation] = []
    plans = store.active_plans()
    if not plans:
        return obs

    plan_keywords_union: Set[str] = set()
    for p in plans:
        plan_keywords_union |= _extract_drift_keywords(p.description)
        plan_keywords_union |= _extract_drift_keywords(" ".join(p.keywords))

    if not plan_keywords_union:
        # Plans are too generic to detect drift — skip
        return obs

    for rel_path in changed_files:
        fns = store.functions_in_file(rel_path)
        if not fns:
            continue
        # Combine all function names + summaries for keyword analysis
        combined = " ".join(
            fn.name + " " + (fn.ast_summary or "") + " " + " ".join(fn.arg_names)
            for fn in fns
        )
        file_drift_keywords = _extract_drift_keywords(combined)

        if not file_drift_keywords:
            continue  # no signal

        # Find categories in the file but NOT in any active plan
        drift = file_drift_keywords - plan_keywords_union
        if drift:
            # Find the most relevant plan to attach
            related_plan_id = plans[0].id  # most recent active plan
            obs.append(_obs(
                kind="drift",
                severity="info",
                message=(
                    f"File introduces work outside any active plan. "
                    f"Plan categories: {sorted(plan_keywords_union)}. "
                    f"File categories: {sorted(file_drift_keywords)}. "
                    f"Drift: {sorted(drift)}. "
                    f"Either update the plan or this is unintended scope."
                ),
                file_path=rel_path,
                related_plan_id=related_plan_id,
            ))

    return obs


# =============================================================================
# Analyzer 2: Duplication / cliché detection
# =============================================================================

def analyze_duplication(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """Flag newly-introduced duplications. When a function's body_hash matches
    an existing function's body_hash, that's a cliché — the same logic written
    twice. The Apprentice points this out without being asked."""
    obs: List[Observation] = []

    # Get all current clichés (>= 2 instances)
    cliches = store.all_cliches(min_instances=2)

    # Build a set of qualified-name prefixes for changed files
    # so we can tell which cliché instances are in changed files
    changed_qnames: Set[str] = set()
    for rel_path in changed_files:
        # Convert file path to module prefix: "pkg/mod.py" -> "pkg.mod"
        rel_norm = rel_path.replace(os.sep, ".")
        if rel_norm.endswith(".py"):
            changed_qnames.add(rel_norm[:-3])

    for cliche in cliches:
        if len(cliche.instances) < 2:
            continue
        # Only report if at least one instance is in a changed file
        changed_instances = []
        for qn in cliche.instances:
            for prefix in changed_qnames:
                if qn.startswith(prefix + ".") or qn == prefix:
                    changed_instances.append(qn)
                    break
        if not changed_instances:
            continue

        obs.append(_obs(
            kind="duplication",
            severity="info",
            message=(
                f"Found {len(cliche.instances)} instances of the same function body: "
                f"{', '.join(cliche.instances[:5])}"
                f"{' ...' if len(cliche.instances) > 5 else ''}. "
                f"Consider extracting a shared helper."
            ),
            function_qualified_name=cliche.instances[0],
            related_function_qualified_name=cliche.instances[1] if len(cliche.instances) > 1 else None,
        ))

    return obs


# =============================================================================
# Analyzer 3: Dead code detection
# =============================================================================

ENTRY_POINT_NAMES = {"main", "__main__", "run", "app", "create_app", "wsgi", "asgi"}
TEST_PREFIXES = ("test_", "Test")


def analyze_dead_code(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """Flag functions that have no callers and aren't entry points or tests.
    These are candidates for removal."""
    obs: List[Observation] = []

    # Only analyze functions in changed files (the proactive scope)
    changed_fns: List[Function] = []
    for rel_path in changed_files:
        changed_fns.extend(store.functions_in_file(rel_path))

    for fn in changed_fns:
        if not fn.is_dead:
            continue
        if fn.name in ENTRY_POINT_NAMES:
            continue
        if fn.name.startswith(TEST_PREFIXES):
            continue
        if fn.name.startswith("__") and fn.name.endswith("__"):
            continue
        # Don't re-flag the same dead function we already noted
        obs.append(_obs(
            kind="dead_code",
            severity="info",
            message=(
                f"Function '{fn.qualified_name}' has no callers and isn't an entry point "
                f"or test. Candidate for removal, or add a caller."
            ),
            file_path=fn.file_path,
            function_qualified_name=fn.qualified_name,
            line=fn.start_line,
        ))

    return obs


# =============================================================================
# Analyzer 4: Complexity creep
# =============================================================================

# A reasonable threshold; tunable.
COMPLEXITY_WARN = 15
COMPLEXITY_ERROR = 30


def analyze_complexity(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """Flag functions whose complexity exceeds thresholds. Compared to historical
    norms (we'd need to track per-function history; for MVP we just flag absolute
    high complexity in changed files)."""
    obs: List[Observation] = []
    for rel_path in changed_files:
        for fn in store.functions_in_file(rel_path):
            if fn.complexity >= COMPLEXITY_ERROR:
                obs.append(_obs(
                    kind="complexity_creep",
                    severity="error",
                    message=(
                        f"Function '{fn.qualified_name}' has complexity {fn.complexity} "
                        f"(threshold {COMPLEXITY_ERROR}). Strongly consider refactoring."
                    ),
                    file_path=fn.file_path,
                    function_qualified_name=fn.qualified_name,
                    line=fn.start_line,
                ))
            elif fn.complexity >= COMPLEXITY_WARN:
                obs.append(_obs(
                    kind="complexity_creep",
                    severity="warning",
                    message=(
                        f"Function '{fn.qualified_name}' has complexity {fn.complexity} "
                        f"(threshold {COMPLEXITY_WARN}). Worth a look."
                    ),
                    file_path=fn.file_path,
                    function_qualified_name=fn.qualified_name,
                    line=fn.start_line,
                ))
    return obs


# =============================================================================
# Analyzer 5: TODO/FIXME without a plan entry
# =============================================================================

TODO_RE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|BUG)[:\s]*(.+)", re.IGNORECASE)


def analyze_todos_without_plan(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """When you add a TODO/FIXME but have no active plan mentioning the same
    topic, the Apprentice flags it: 'you left a marker but no plan to come back
    to it.' This is the kind of proactivity that catches technical debt at
    introduction time."""
    obs: List[Observation] = []
    plans = store.active_plans()
    plan_text = " ".join(p.description for p in plans).lower()
    plan_keywords = _extract_drift_keywords(plan_text)

    for rel_path in changed_files:
        abs_path = os.path.join(root, rel_path)
        if not os.path.exists(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue

        for i, line in enumerate(lines, start=1):
            m = TODO_RE.search(line)
            if not m:
                continue
            marker = m.group(1).upper()
            text = m.group(2).strip()

            todo_keywords = _extract_drift_keywords(text)
            # If the TODO's topic matches an active plan, that's fine.
            # If not, it's a marker without a plan to come back to it.
            if plan_keywords and (todo_keywords & plan_keywords):
                continue  # aligns with an active plan — OK

            severity = "warning" if marker in ("FIXME", "BUG", "XXX") else "info"
            obs.append(_obs(
                kind="todo_without_plan",
                severity=severity,
                message=(
                    f"{marker} added without matching active plan: '{text}'. "
                    f"Either create a plan for this or it'll be forgotten."
                ),
                file_path=rel_path,
                line=i,
            ))

    return obs


# =============================================================================
# Analyzer 6: New pattern (cliché recognition)
# =============================================================================

# When you write a function whose AST-summary matches the silhouette of a known
# pattern (e.g. "calls request, returns; ifs ×3" looks like a request handler),
# the Apprentice notes it. This is the cliché library in 1988-spec terms.

# For the MVP, we just note functions that share a signature_hash with an
# existing function — i.e., you wrote another "shape" that already exists.
def analyze_new_pattern(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """When a new function has the same signature_hash as existing functions,
    point out the family — the user may not know there's already a clique."""
    obs: List[Observation] = []
    for rel_path in changed_files:
        for fn in store.functions_in_file(rel_path):
            siblings = store.functions_by_signature(fn.signature_hash)
            if len(siblings) <= 1:
                continue
            # Only flag if THIS function is new (first_seen recent) and others are older
            # (heuristic: this is hard without history; just flag if >2 siblings)
            if len(siblings) >= 3:
                others = [s.qualified_name for s in siblings if s.qualified_name != fn.qualified_name]
                obs.append(_obs(
                    kind="new_pattern",
                    severity="info",
                    message=(
                        f"Function '{fn.qualified_name}' shares its signature with "
                        f"{len(siblings)-1} others: {', '.join(others[:3])}. "
                        f"This is a cliché — consider whether they should share an implementation."
                    ),
                    file_path=fn.file_path,
                    function_qualified_name=fn.qualified_name,
                ))
    return obs


# =============================================================================
# Orchestrator
# =============================================================================

ANALYZERS = [
    ("plan_drift", analyze_plan_drift),
    ("duplication", analyze_duplication),
    ("dead_code", analyze_dead_code),
    ("complexity", analyze_complexity),
    ("todos_without_plan", analyze_todos_without_plan),
    ("new_pattern", analyze_new_pattern),
]


def run_all_analyzers(
    store: Store, root: str, changed_files: List[str], config=None
) -> List[Observation]:
    """Run every analyzer and return all observations."""
    all_obs: List[Observation] = []
    for name, analyzer in ANALYZERS:
        try:
            new_obs = analyzer(store, root, changed_files)
            all_obs.extend(new_obs)
        except Exception as e:
            all_obs.append(_obs(
                kind="analyzer_error",
                severity="error",
                message=f"Analyzer '{name}' crashed: {type(e).__name__}: {e}",
            ))

    # Add historical analyzer (needs v0.2.0 schema)
    try:
        from .historical import analyze_complexity_trends
        hist_obs = analyze_complexity_trends(store, root, changed_files)
        all_obs.extend(hist_obs)
    except Exception:
        pass  # history tables might not exist

    # Auto-acknowledge low-severity observations if configured
    if config and config.auto_acknowledge_info:
        for o in all_obs:
            if o.severity == "info":
                o.acknowledged = True

    return all_obs
