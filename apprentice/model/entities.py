"""
Entity definitions for the Apprentice's codebase model.

These are the structured objects the Apprentice persists across sessions.
The model is NOT just a vector index — it's a typed graph of:
  File → contains → Function / Class
  Function → calls → Function
  Function → defined_in → File
  Plan → references → Function (intent tracking)
  Observation → about → Function (proactive findings)

This structure is what distinguishes the Apprentice from RAG-based tools.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import hashlib
import json


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class File:
    """A source file in the indexed codebase."""
    path: str
    language: str
    content_hash: str
    line_count: int
    last_indexed: str = field(default_factory=utcnow_iso)
    # denormalized for quick "what changed" queries
    function_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)


@dataclass
class Function:
    """A function or method discovered by the indexer."""
    name: str
    qualified_name: str  # e.g. "module.Class.method"
    file_path: str
    start_line: int
    end_line: int
    arg_names: List[str]
    signature_hash: str  # hash of (name, arg_names) — for cliché detection
    body_hash: str       # hash of AST-normalized body — for duplication
    ast_summary: str     # human-readable one-liner, e.g. "calls foo, returns x"
    complexity: int      # cyclomatic complexity (rough)
    docstring: Optional[str] = None
    callers: List[str] = field(default_factory=list)  # qualified_names of callers
    # embedding stored separately (not in this row) — see EmbeddingStore
    first_seen: str = field(default_factory=utcnow_iso)
    last_modified: str = field(default_factory=utcnow_iso)
    is_dead: bool = False  # no callers (best-effort)


@dataclass
class Class:
    """A class discovered by the indexer."""
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    bases: List[str]
    method_names: List[str]
    first_seen: str = field(default_factory=utcnow_iso)


@dataclass
class Plan:
    """A stated intent. The Apprentice checks new code against active plans."""
    id: str  # short uuid
    description: str  # what the user said they're doing
    keywords: List[str]  # extracted for drift matching
    status: str = "active"  # active | completed | abandoned
    created: str = field(default_factory=utcnow_iso)
    updated: str = field(default_factory=utcnow_iso)
    notes: List[str] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)


@dataclass
class Observation:
    """A proactive finding. This is the Apprentice 'piping up'."""
    id: str
    kind: str  # drift | duplication | dead_code | complexity_creep | todo_without_plan | new_pattern
    severity: str  # info | warning | error
    message: str
    file_path: Optional[str] = None
    function_qualified_name: Optional[str] = None
    line: Optional[int] = None
    related_plan_id: Optional[str] = None
    related_function_qualified_name: Optional[str] = None  # e.g. the duplicate
    created: str = field(default_factory=utcnow_iso)
    acknowledged: bool = False


@dataclass
class Cliche:
    """A recognized pattern (cliché) in use across the codebase.
    Two functions with the same signature_hash and similar body_hash are instances
    of the same cliché."""
    signature_hash: str
    body_hash: str
    instances: List[str]  # qualified_names
    first_seen: str = field(default_factory=utcnow_iso)
    note: str = ""


# --- Serialization helpers ---

def to_dict(obj) -> Dict[str, Any]:
    return asdict(obj)


def to_json(obj) -> str:
    return json.dumps(to_dict(obj), default=str, sort_keys=True)
