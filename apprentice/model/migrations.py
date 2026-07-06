"""
Schema migrations for the Apprentice.

Versioned schema with forward-only migrations. Each migration adds/changes
tables. The version is tracked in apprentice_meta.

v0.1.0: initial schema
v0.2.0: add function_history table, observation_explanations, plan_history
"""

from __future__ import annotations
import sqlite3
from typing import List, Tuple


MIGRATIONS: List[Tuple[str, str]] = [
    # (version, SQL)
    ("0.1.0", "-- initial schema (created by init_schema)"),
    ("0.2.0", """
        -- Historical tracking: snapshot of function state per index run
        CREATE TABLE IF NOT EXISTS function_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qualified_name TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            complexity INTEGER NOT NULL,
            line_count INTEGER NOT NULL,
            caller_count INTEGER NOT NULL,
            body_hash TEXT NOT NULL,
            FOREIGN KEY (qualified_name) REFERENCES functions(qualified_name)
        );
        CREATE INDEX IF NOT EXISTS idx_fh_fn ON function_history(qualified_name);
        CREATE INDEX IF NOT EXISTS idx_fh_time ON function_history(snapshot_at);

        -- LLM-generated explanations for observations
        CREATE TABLE IF NOT EXISTS observation_explanations (
            observation_id TEXT PRIMARY KEY,
            explanation TEXT NOT NULL,
            diff TEXT,
            llm_backend TEXT,
            created TEXT NOT NULL,
            FOREIGN KEY (observation_id) REFERENCES observations(id)
        );

        -- Plan history (status changes)
        CREATE TABLE IF NOT EXISTS plan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT NOT NULL,
            old_status TEXT,
            new_status TEXT NOT NULL,
            changed_at TEXT NOT NULL,
            note TEXT,
            FOREIGN KEY (plan_id) REFERENCES plans(id)
        );

        -- Function source snapshots (for diffing over time)
        CREATE TABLE IF NOT EXISTS function_sources (
            qualified_name TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            line_count INTEGER NOT NULL,
            PRIMARY KEY (qualified_name, snapshot_at)
        );
    """),
    ("0.3.0", """
        -- Structured call data on functions (replaces ast_summary regex).
        -- JSON array of raw call names from the AST.
        ALTER TABLE functions ADD COLUMN calls TEXT NOT NULL DEFAULT '[]';
    """),
    ("0.3.1", """
        -- Recreate function_history WITHOUT the foreign key constraint.
        -- History should survive function deletion (so we can track complexity
        -- trends even after a function is renamed/deleted). The original FK
        -- prevented this — deleting a function would fail if history existed.
        CREATE TABLE IF NOT EXISTS function_history_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qualified_name TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            complexity INTEGER NOT NULL,
            line_count INTEGER NOT NULL,
            caller_count INTEGER NOT NULL,
            body_hash TEXT NOT NULL
        );
        INSERT OR IGNORE INTO function_history_new
            SELECT * FROM function_history;
        DROP TABLE IF EXISTS function_history;
        ALTER TABLE function_history_new RENAME TO function_history;
        CREATE INDEX IF NOT EXISTS idx_fh_fn ON function_history(qualified_name);
        CREATE INDEX IF NOT EXISTS idx_fh_time ON function_history(snapshot_at);
    """),
]


def _parse_version(v: str) -> Tuple[int, ...]:
    """Parse a version string like '0.3.1' into a tuple (0, 3, 1).
    Handles versions with any number of dot-separated numeric parts.
    Non-numeric parts (like '0.3.1-rc1') are treated as 0.
    """
    parts = []
    for p in v.split("."):
        # Take only the leading numeric part (handles '1-rc1' → 1)
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts)


def get_schema_version(conn: sqlite3.Connection) -> str:
    """Get the current schema version."""
    try:
        row = conn.execute(
            "SELECT value FROM apprentice_meta WHERE key = 'schema_version'"
        ).fetchone()
        return row[0] if row else "0.0.0"
    except sqlite3.OperationalError:
        return "0.0.0"


def set_schema_version(conn: sqlite3.Connection, version: str):
    conn.execute(
        "INSERT OR REPLACE INTO apprentice_meta(key, value) VALUES (?, ?)",
        ("schema_version", version),
    )


def needs_migration(conn: sqlite3.Connection) -> bool:
    """Check if the database needs migration.
    Uses tuple-based version comparison (not lexicographic) so that
    '0.10.0' correctly sorts after '0.9.0'."""
    current = get_schema_version(conn)
    latest = MIGRATIONS[-1][0]
    return _parse_version(current) < _parse_version(latest)


def migrate(conn: sqlite3.Connection) -> List[str]:
    """Run all pending migrations. Returns list of applied versions.
    Uses tuple-based version comparison."""
    current = _parse_version(get_schema_version(conn))
    applied = []
    for version, sql in MIGRATIONS:
        if _parse_version(version) <= current:
            continue
        # Only run if there's actual SQL (not just comments/whitespace)
        import re
        sql_no_comments = re.sub(r'--.*$', '', sql, flags=re.MULTILINE).strip()
        if sql_no_comments:
            # ALTER TABLE statements can fail if the column already exists.
            # Split into statements and run each, tolerating "duplicate column"
            # errors so re-running migrations is idempotent.
            statements = [s.strip() for s in sql_no_comments.split(';') if s.strip()]
            for stmt in statements:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        pass  # column already exists — idempotent
                    else:
                        raise
        set_schema_version(conn, version)
        applied.append(version)
    return applied
