"""
Rich CLI output — colors, tables, formatting.

No external dependencies. Uses ANSI escape codes directly.
"""

from __future__ import annotations
import sys
import os
from collections import Counter
from typing import List, Optional

from ..model.entities import Observation


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


def _supports_color() -> bool:
    if not sys.stderr.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def _c(text: str, color: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Colors.RESET}"


SEVERITY_COLOR = {
    "error": Colors.RED,
    "warning": Colors.YELLOW,
    "info": Colors.CYAN,
}

SEVERITY_SYMBOL = {
    "error": "✗",
    "warning": "⚠",
    "info": "●",
}

KIND_COLOR = {
    "drift": Colors.MAGENTA,
    "duplication": Colors.BLUE,
    "dead_code": Colors.GRAY,
    "complexity_creep": Colors.YELLOW,
    "complexity_trend": Colors.YELLOW,
    "todo_without_plan": Colors.YELLOW,
    "new_pattern": Colors.CYAN,
    "analyzer_error": Colors.RED,
}

SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}
KIND_RANK = {
    "analyzer_error": 0,
    "complexity_creep": 1,
    "complexity_trend": 2,
    "todo_without_plan": 3,
    "drift": 4,
    "duplication": 5,
    "dead_code": 6,
    "new_pattern": 7,
}


def sort_observations(observations: List[Observation]) -> List[Observation]:
    """Stable, user-facing order: most actionable first."""
    return sorted(
        observations,
        key=lambda o: (
            SEVERITY_RANK.get(o.severity, 9),
            KIND_RANK.get(o.kind, 9),
            o.file_path or "",
            o.line or 0,
            o.function_qualified_name or "",
        ),
    )


def format_observation_summary(
    observations: List[Observation], use_color: Optional[bool] = None
) -> str:
    if use_color is None:
        use_color = _supports_color()
    if not observations:
        return ""
    by_severity = Counter(o.severity for o in observations)
    by_kind = Counter(o.kind for o in observations)
    severity_parts = []
    for sev in ("error", "warning", "info"):
        n = by_severity.get(sev, 0)
        if n:
            severity_parts.append(_c(f"{sev}={n}", SEVERITY_COLOR.get(sev, Colors.WHITE), use_color))
    kind_parts = [f"{kind}={n}" for kind, n in sorted(by_kind.items())]
    lines = [f"  Summary: {', '.join(severity_parts)}"]
    lines.append(f"  Kinds: {', '.join(kind_parts)}")
    return "\n".join(lines)


def format_observations(
    observations: List[Observation],
    use_color: Optional[bool] = None,
    max_items: Optional[int] = None,
) -> str:
    """Format observations for display."""
    if use_color is None:
        use_color = _supports_color()

    if not observations:
        return _c("  No observations.", Colors.DIM, use_color)

    lines = []
    if max_items is not None:
        max_items = max(0, max_items)
    ordered = sort_observations(observations)
    displayed = ordered[:max_items] if max_items is not None else ordered
    for obs in displayed:
        sym = SEVERITY_SYMBOL.get(obs.severity, "?")
        sym_color = SEVERITY_COLOR.get(obs.severity, Colors.WHITE)
        kind_color = KIND_COLOR.get(obs.kind, Colors.WHITE)
        ack = _c(" (acknowledged)", Colors.DIM, use_color) if obs.acknowledged else ""

        lines.append(
            f"  {_c(sym, sym_color, use_color)} "
            f"{_c(f'[{obs.kind}]', kind_color, use_color)} "
            f"{_c(obs.id, Colors.DIM, use_color)}{ack}"
        )
        lines.append(f"     {obs.message}")

        loc_parts = []
        if obs.file_path:
            loc_parts.append(_c(obs.file_path, Colors.CYAN, use_color))
        if obs.line:
            loc_parts.append(f"line {obs.line}")
        if obs.function_qualified_name:
            loc_parts.append(f"fn {_c(obs.function_qualified_name, Colors.DIM, use_color)}")
        if loc_parts:
            lines.append(f"     location: {' '.join(loc_parts)}")
        lines.append("")

    if max_items is not None and len(ordered) > max_items:
        hidden = len(ordered) - max_items
        lines.append(_c(
            f"  ... {hidden} more observation(s). Run `apprentice observations --all` to inspect everything.",
            Colors.DIM,
            use_color,
        ))

    return "\n".join(lines)


def format_status(stats: dict, use_color: Optional[bool] = None) -> str:
    """Format the status display."""
    if use_color is None:
        use_color = _supports_color()

    lines = []
    lines.append(_c(f"  Apprentice v{stats['version']}", Colors.BOLD, use_color))
    lines.append(f"  {_c('Repo:', Colors.DIM, use_color)} {stats['repo']}")
    lines.append(f"  {_c('Files in model:', Colors.DIM, use_color)} {stats['files']}")
    lines.append(f"  {_c('Functions in model:', Colors.DIM, use_color)} {stats['functions']}")
    lines.append(f"  {_c('Active plans:', Colors.DIM, use_color)} {stats['plans']}")
    lines.append(f"  {_c('Unacked observations:', Colors.DIM, use_color)} {stats['unacked']}")

    if stats.get('last_snapshot'):
        lines.append(f"  {_c('Last watch:', Colors.DIM, use_color)} {stats['last_snapshot']}")

    if stats.get('active_plans'):
        lines.append("")
        lines.append(_c("  Active plans:", Colors.BOLD, use_color))
        for p in stats['active_plans']:
            plan_id = _c(f"[{p['id']}]", Colors.DIM, use_color)
            lines.append(f"    {plan_id} {p['description'][:80]}")

    return "\n".join(lines)


def format_plan(plan, use_color: Optional[bool] = None) -> str:
    """Format a plan for display."""
    if use_color is None:
        use_color = _supports_color()

    status_marks = {
        "active": _c("●", Colors.GREEN, use_color),
        "completed": _c("✓", Colors.GREEN, use_color),
        "abandoned": _c("✗", Colors.RED, use_color),
    }
    mark = status_marks.get(plan.status, "?")

    lines = [
        f"  {mark} {_c(f'[{plan.id}]', Colors.DIM, use_color)} {plan.description[:80]}"
    ]
    if plan.status == "active":
        if plan.keywords:
            lines.append(f"     {_c('keywords:', Colors.DIM, use_color)} {', '.join(plan.keywords)}")
        lines.append(f"     {_c('created:', Colors.DIM, use_color)} {plan.created}")
    return "\n".join(lines)


def print_diff(diff: str, use_color: Optional[bool] = None):
    """Print a unified diff with colors."""
    if use_color is None:
        use_color = _supports_color()

    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            print(_c(line, Colors.GREEN, use_color))
        elif line.startswith("-") and not line.startswith("---"):
            print(_c(line, Colors.RED, use_color))
        elif line.startswith("@@"):
            print(_c(line, Colors.CYAN, use_color))
        else:
            print(line)
