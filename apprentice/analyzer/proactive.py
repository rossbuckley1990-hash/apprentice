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
import tokenize
from datetime import datetime, timezone
from io import StringIO
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
    # Deterministic ID derived from (kind, location, message) so that
    # re-running the same analyzer on the same code doesn't produce
    # duplicate observations. The message is included (truncated) because
    # two observations of the same kind at the same location with different
    # messages are genuinely different findings.
    import hashlib as _hashlib
    fingerprint_parts = [
        kind,
        file_path or "",
        function_qualified_name or "",
        str(line or ""),
        message[:200],  # truncate to avoid hash churn from long messages
    ]
    fingerprint = "|".join(fingerprint_parts)
    obs_id = _hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]

    return Observation(
        id=obs_id,
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
    "index": ["index", "indexing", "parser", "parse", "registry", "embedding", "graph"],
    "perf": ["perf", "performance", "optimize", "cache", "speed", "latency"],
    "bug": ["bug", "fix", "regression", "patch"],
    "docs": ["doc", "documentation", "readme", "comment"],
    "config": ["config", "settings", "env", "deploy"],
    "security": ["security", "vulnerability", "cve", "sanitize", "escape"],
}
LOW_SIGNAL_DRIFT = {"docs", "test"}
REFACTOR_COMPANION_DRIFT = {"config"}
TEST_PATH_PARTS = {"test", "tests", "__tests__", "spec", "specs", "fixtures"}


def _drift_tokens(text: str) -> List[str]:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [tok.lower() for tok in re.split(r"[^A-Za-z0-9]+", camel_split) if tok]


def _extract_drift_keywords(text: str) -> Set[str]:
    """Extract semantic keywords from a plan description or code text.

    Prefix-match normalized tokens so 'auth' matches 'authentication' and
    'handle_auth_request', while 'ui' still does not match 'build'.
    """
    tokens = _drift_tokens(text)
    found = set()
    for category, words in DRIFT_KEYWORDS.items():
        for token in tokens:
            if any(token.startswith(w) for w in words):
                found.add(category)
                break
    return found


def _path_parts(rel_path: str) -> Set[str]:
    normalized = rel_path.replace("\\", "/")
    parts = set()
    for part in normalized.split("/"):
        stem = part.rsplit(".", 1)[0]
        parts.add(stem.lower())
    return parts


def _is_test_path(rel_path: str) -> bool:
    parts = _path_parts(rel_path)
    return bool(parts & TEST_PATH_PARTS) or any(p.startswith("test_") for p in parts)


def _drift_after_policy(drift: Set[str], plan_categories: Set[str]) -> Set[str]:
    drift = drift - LOW_SIGNAL_DRIFT
    if "refactor" in plan_categories:
        drift = drift - REFACTOR_COMPANION_DRIFT
    return drift


def _plan_keyword_sets(plans: List[Plan]) -> List[Tuple[Plan, Set[str]]]:
    result = []
    for plan in plans:
        keywords = _extract_drift_keywords(plan.description)
        keywords |= _extract_drift_keywords(" ".join(plan.keywords))
        result.append((plan, keywords))
    return result


def _keyword_union(plan_keywords: List[Tuple[Plan, Set[str]]]) -> Set[str]:
    union: Set[str] = set()
    for _, keywords in plan_keywords:
        union |= keywords
    return union


def _file_drift_keywords(store: Store, rel_path: str) -> Set[str]:
    fns = store.functions_in_file(rel_path)
    if not fns:
        return set()
    combined = " ".join(
        fn.name + " " + (fn.ast_summary or "") + " " + " ".join(fn.arg_names)
        for fn in fns
    )
    return _extract_drift_keywords(f"{rel_path} {combined}")


def _best_matching_plan(
    plans: List[Plan],
    plan_keywords: List[Tuple[Plan, Set[str]]],
    file_keywords: Set[str],
) -> Plan:
    best_plan = plans[0]
    best_overlap = -1
    for plan, keywords in plan_keywords:
        overlap = len(file_keywords & keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best_plan = plan
    return best_plan


def _drift_observation(
    rel_path: str,
    plan: Plan,
    plan_categories: Set[str],
    file_categories: Set[str],
    drift: Set[str],
) -> Observation:
    return _obs(
        kind="drift",
        severity="info",
        message=(
            f"File introduces work outside any active plan. "
            f"Plan categories: {sorted(plan_categories)}. "
            f"File categories: {sorted(file_categories)}. "
            f"Drift: {sorted(drift)}. "
            f"Either update the plan or this is unintended scope."
        ),
        file_path=rel_path,
        related_plan_id=plan.id,
    )


def analyze_plan_drift(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """If active plans exist, check whether changed files match the plan's
    keyword domain. Flag drift when new code appears unrelated to any plan.

    Uses the BEST-MATCHING plan (by keyword overlap) rather than blindly
    attaching to plans[0]. If no plan matches, the observation is still
    emitted but without a related_plan_id.
    """
    obs: List[Observation] = []
    plans = store.active_plans()
    if not plans:
        return obs

    plan_keywords = _plan_keyword_sets(plans)
    plan_keywords_union = _keyword_union(plan_keywords)

    if not plan_keywords_union:
        # Plans are too generic to detect drift — skip
        return obs

    for rel_path in changed_files:
        if _is_test_path(rel_path):
            continue
        file_drift_keywords = _file_drift_keywords(store, rel_path)
        if not file_drift_keywords:
            continue  # no signal

        drift = _drift_after_policy(file_drift_keywords - plan_keywords_union, plan_keywords_union)
        if drift:
            best_plan = _best_matching_plan(plans, plan_keywords, file_drift_keywords)
            obs.append(_drift_observation(
                rel_path, best_plan, plan_keywords_union, file_drift_keywords, drift
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

    changed_qnames = _changed_module_prefixes(changed_files)
    abstract_methods = _abstract_method_qnames(store)

    def function_for(qname: str) -> Optional[Function]:
        return store.get_function(qname)

    for cliche in cliches:
        if len(cliche.instances) < 2:
            continue
        instance_functions = [function_for(qn) for qn in cliche.instances]
        concrete_instances = [fn for fn in instance_functions if fn is not None]
        if _skip_duplication_cliche(concrete_instances, abstract_methods):
            continue
        if not _cliche_touches_changed_file(cliche.instances, changed_qnames):
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
PUBLIC_API_DECORATORS = {"<decorator>", "<export>"}
ABSTRACT_BASE_NAMES = {"ABC", "Protocol"}


def _changed_module_prefixes(changed_files: List[str]) -> Set[str]:
    prefixes: Set[str] = set()
    for rel_path in changed_files:
        rel_norm = rel_path.replace(os.sep, ".")
        for ext in (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            if rel_norm.endswith(ext):
                prefixes.add(rel_norm[: -len(ext)])
                break
    return prefixes


def _cliche_touches_changed_file(instances: List[str], changed_qnames: Set[str]) -> bool:
    for qn in instances:
        for prefix in changed_qnames:
            if qn.startswith(prefix + ".") or qn == prefix:
                return True
    return False


def _abstract_method_qnames(store: Store) -> Set[str]:
    methods = set()
    for cls in store.all_classes():
        if not (set(cls.bases) & ABSTRACT_BASE_NAMES):
            continue
        for method_name in cls.method_names:
            methods.add(f"{cls.qualified_name}.{method_name}")
    return methods


def _skip_duplication_cliche(functions: List[Function], abstract_methods: Set[str]) -> bool:
    if not functions:
        return False
    if all(_is_test_path(fn.file_path) for fn in functions):
        return True
    if all(fn.qualified_name in abstract_methods for fn in functions):
        return True
    return _is_cross_class_method_family(functions[0], functions)


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
    abstract_methods = _abstract_method_qnames(store)

    for fn in changed_fns:
        if not fn.is_dead:
            continue
        if fn.name in ENTRY_POINT_NAMES:
            continue
        if fn.name.startswith(TEST_PREFIXES):
            continue
        if fn.name.startswith("__") and fn.name.endswith("__"):
            continue
        if any(caller in PUBLIC_API_DECORATORS for caller in fn.callers):
            continue
        if fn.qualified_name in abstract_methods:
            continue
        # Don't re-flag the same dead function we already noted
        obs.append(_obs(
            kind="dead_code",
            severity="info",
            message=(
                f"Function '{fn.qualified_name}' has no internal callers in the indexed graph. "
                f"If it is not external API or framework-discovered code, consider removing it "
                f"or adding an explicit caller/registration."
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

TODO_RE = re.compile(r"(?:#|//)\s*(TODO|FIXME|HACK|XXX|BUG)[:\s]*(.+)", re.IGNORECASE)


def _todo_matches_from_text(text: str) -> List[Tuple[int, str, str]]:
    matches = []
    for i, line in enumerate(text.splitlines(), start=1):
        m = TODO_RE.search(line)
        if m:
            matches.append((i, m.group(1).upper(), m.group(2).strip()))
    return matches


def _todo_matches_from_python(text: str) -> List[Tuple[int, str, str]]:
    matches = []
    try:
        tokens = tokenize.generate_tokens(StringIO(text).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            m = TODO_RE.search(tok.string)
            if m:
                matches.append((tok.start[0], m.group(1).upper(), m.group(2).strip()))
    except tokenize.TokenError:
        return _todo_matches_from_text(text)
    return matches


def _todo_matches_for_file(rel_path: str, text: str) -> List[Tuple[int, str, str]]:
    if rel_path.endswith(".py"):
        return _todo_matches_from_python(text)
    return _todo_matches_from_text(text)


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
        if _is_test_path(rel_path):
            continue
        abs_path = os.path.join(root, rel_path)
        if not os.path.exists(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue

        for i, marker, todo_text in _todo_matches_for_file(rel_path, text):
            todo_keywords = _extract_drift_keywords(todo_text)
            # If the TODO's topic matches an active plan, that's fine.
            # If not, it's a marker without a plan to come back to it.
            if plan_keywords and (todo_keywords & plan_keywords):
                continue  # aligns with an active plan — OK

            severity = "warning" if marker in ("FIXME", "BUG", "XXX") else "info"
            obs.append(_obs(
                kind="todo_without_plan",
                severity=severity,
                message=(
                    f"{marker} added without matching active plan: '{todo_text}'. "
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
def _is_method(fn: Function) -> bool:
    return len(fn.qualified_name.split(".")) >= 3


def _is_cross_class_method_family(fn: Function, siblings: List[Function]) -> bool:
    if not siblings or not all(_is_method(s) for s in siblings):
        return False
    owner_classes = {".".join(s.qualified_name.split(".")[:-1]) for s in siblings}
    return len(owner_classes) >= 2 and all(s.name == fn.name for s in siblings)


def analyze_new_pattern(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """When a new function has the same signature_hash as existing functions,
    point out the family — the user may not know there's already a clique.
    Excludes dunder methods (__init__, __str__, etc.) which naturally share
    signatures across classes without being clichés."""
    obs: List[Observation] = []
    for rel_path in changed_files:
        for fn in store.functions_in_file(rel_path):
            # Skip dunders — every class has __init__(self, ...), etc.
            if fn.name.startswith("__") and fn.name.endswith("__"):
                continue
            # Skip __init__ specifically (most common false positive)
            if fn.name == "__init__":
                continue
            siblings = store.functions_by_signature(fn.signature_hash)
            if len(siblings) <= 1:
                continue
            if _is_cross_class_method_family(fn, siblings):
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
