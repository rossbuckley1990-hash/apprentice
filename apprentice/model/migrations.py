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
]


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
    """Check if the database needs migration."""
    current = get_schema_version(conn)
    return current < MIGRATIONS[-1][0]


def migrate(conn: sqlite3.Connection) -> List[str]:
    """Run all pending migrations. Returns list of applied versions."""
    current = get_schema_version(conn)
    applied = []
    for version, sql in MIGRATIONS:
        if version <= current:
            continue
        # Only run if there's actual SQL (not just comments/whitespace)
        # Strip comments and whitespace to check
        import re
        sql_no_comments = re.sub(r'--.*$', '', sql, flags=re.MULTILINE).strip()
        if sql_no_comments:
            conn.executescript(sql)
        set_schema_version(conn, version)
        applied.append(version)
    return applied
