"""
Historical analyzer — uses function_history to detect complexity trends.

This is the 1988 spec's "complexity creep" made temporal: instead of just
flagging high complexity, we flag functions whose complexity is GROWING.
"""

from __future__ import annotations
import uuid
from typing import List

from ..model.entities import Observation
from ..model.store import Store
from .proactive import _obs


def analyze_complexity_trends(
    store: Store, root: str, changed_files: List[str]
) -> List[Observation]:
    """Flag functions whose complexity has increased significantly over time.
    Requires at least 2 historical snapshots."""
    obs: List[Observation] = []
    trends = store.complexity_trends(min_changes=2)

    for trend in trends[:20]:  # cap at 20 to avoid flooding
        qname = trend["qualified_name"]
        min_c = trend["min_c"]
        max_c = trend["max_c"]
        snapshots = trend["snapshots"]
        delta = max_c - min_c

        # Only flag if complexity is growing (not shrinking)
        if delta <= 0:
            continue

        # Only flag significant growth
        if delta < 3 and max_c < 10:
            continue

        severity = "warning" if max_c >= 15 else "info"
        obs.append(_obs(
            kind="complexity_trend",
            severity=severity,
            message=(
                f"Function '{qname}' complexity grew from {min_c} to {max_c} "
                f"(+{delta} over {snapshots} snapshots). "
                f"This function is getting more complex over time — consider refactoring."
            ),
            function_qualified_name=qname,
        ))

    return obs
