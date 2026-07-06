"""
SQLite persistence layer.

The store keeps the codebase model across sessions — this is the
PERSISTENCE that distinguishes the Apprentice from Copilot/Cursor.

Schema is intentionally simple and extensible. We use JSON columns for
list-valued fields (callers, keywords, etc.) so we can evolve without
migrations in the MVP.
"""

from __future__ import annotations
import sqlite3
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from .entities import (
    File, Function, Class, Plan, Observation, Cliche,
    to_dict, utcnow_iso, hash_content,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS apprentice_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    line_count INTEGER NOT NULL,
    last_indexed TEXT NOT NULL,
    function_names TEXT NOT NULL,    -- JSON array
    class_names TEXT NOT NULL        -- JSON array
);

CREATE TABLE IF NOT EXISTS functions (
    qualified_name TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    arg_names TEXT NOT NULL,         -- JSON array
    signature_hash TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    ast_summary TEXT NOT NULL,
    complexity INTEGER NOT NULL,
    docstring TEXT,
    callers TEXT NOT NULL,           -- JSON array
    first_seen TEXT NOT NULL,
    last_modified TEXT NOT NULL,
    is_dead INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (file_path) REFERENCES files(path)
);
CREATE INDEX IF NOT EXISTS idx_functions_file ON functions(file_path);
CREATE INDEX IF NOT EXISTS idx_functions_sig ON functions(signature_hash);
CREATE INDEX IF NOT EXISTS idx_functions_body ON functions(body_hash);

CREATE TABLE IF NOT EXISTS classes (
    qualified_name TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    bases TEXT NOT NULL,             -- JSON array
    method_names TEXT NOT NULL,      -- JSON array
    first_seen TEXT NOT NULL,
    FOREIGN KEY (file_path) REFERENCES files(path)
);

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    keywords TEXT NOT NULL,          -- JSON array
    status TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    notes TEXT NOT NULL,             -- JSON array
    related_files TEXT NOT NULL      -- JSON array
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    file_path TEXT,
    function_qualified_name TEXT,
    line INTEGER,
    related_plan_id TEXT,
    related_function_qualified_name TEXT,
    created TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_obs_kind ON observations(kind);
CREATE INDEX IF NOT EXISTS idx_obs_unacked ON observations(acknowledged) WHERE acknowledged = 0;

CREATE TABLE IF NOT EXISTS cliches (
    signature_hash TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    instances TEXT NOT NULL,         -- JSON array
    first_seen TEXT NOT NULL,
    note TEXT NOT NULL,
    PRIMARY KEY (signature_hash, body_hash)
);

CREATE TABLE IF NOT EXISTS embeddings (
    qualified_name TEXT PRIMARY KEY,
    vector TEXT NOT NULL,            -- JSON array
    dim INTEGER NOT NULL,
    backend TEXT NOT NULL,           -- 'tfidf' | 'asthash' | 'sentence-transformers' | 'openai'
    created TEXT NOT NULL,
    FOREIGN KEY (qualified_name) REFERENCES functions(qualified_name)
);

CREATE TABLE IF NOT EXISTS snapshot_log (
    -- records each `apprentice watch` run, so we can see what was checked when
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    files_checked INTEGER NOT NULL,
    observations_emitted INTEGER NOT NULL,
    notes TEXT
);
"""


class Store:
    """SQLite-backed persistence for the codebase model."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @contextmanager
    def conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        yield self._conn
        self._conn.commit()

    def close(self):
        """Close the database connection."""
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def init_schema(self):
        # Create initial schema
        with self.conn() as c:
            c.executescript(SCHEMA)
            c.execute(
                "INSERT OR REPLACE INTO apprentice_meta(key, value) VALUES (?, ?)",
                ("schema_version", "0.1.0"),
            )
        # Run migrations to bring schema up to the latest version
        from .migrations import migrate, needs_migration
        with self.conn() as c:
            if needs_migration(c):
                applied = migrate(c)
                if applied:
                    c.commit()
                    import sys
                    print(f"  [apprentice] schema upgraded: {', '.join(applied)}", file=sys.stderr)
        # Close and reopen to ensure schema changes are visible
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- Files ---

    def upsert_file(self, f: File):
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO files
                   (path, language, content_hash, line_count, last_indexed,
                    function_names, class_names)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f.path, f.language, f.content_hash, f.line_count, f.last_indexed,
                 json.dumps(f.function_names), json.dumps(f.class_names)),
            )

    def get_file(self, path: str) -> Optional[File]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
            if row is None:
                return None
            return File(
                path=row["path"], language=row["language"],
                content_hash=row["content_hash"], line_count=row["line_count"],
                last_indexed=row["last_indexed"],
                function_names=json.loads(row["function_names"]),
                class_names=json.loads(row["class_names"]),
            )

    def all_files(self) -> List[File]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM files ORDER BY path").fetchall()
            return [File(
                path=r["path"], language=r["language"],
                content_hash=r["content_hash"], line_count=r["line_count"],
                last_indexed=r["last_indexed"],
                function_names=json.loads(r["function_names"]),
                class_names=json.loads(r["class_names"]),
            ) for r in rows]

    def file_count(self) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    # --- Functions ---

    def upsert_function(self, fn: Function):
        with self.conn() as c:
            # Check if 'calls' column exists (v0.3.0+); fall back gracefully
            try:
                c.execute(
                    """INSERT OR REPLACE INTO functions
                       (qualified_name, name, file_path, start_line, end_line,
                        arg_names, signature_hash, body_hash, ast_summary,
                        complexity, docstring, calls, callers, first_seen, last_modified, is_dead)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (fn.qualified_name, fn.name, fn.file_path, fn.start_line, fn.end_line,
                     json.dumps(fn.arg_names), fn.signature_hash, fn.body_hash,
                     fn.ast_summary, fn.complexity, fn.docstring,
                     json.dumps(fn.calls), json.dumps(fn.callers),
                     fn.first_seen, fn.last_modified, int(fn.is_dead)),
                )
            except sqlite3.OperationalError:
                # Pre-v0.3.0 schema without 'calls' column
                c.execute(
                    """INSERT OR REPLACE INTO functions
                       (qualified_name, name, file_path, start_line, end_line,
                        arg_names, signature_hash, body_hash, ast_summary,
                        complexity, docstring, callers, first_seen, last_modified, is_dead)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (fn.qualified_name, fn.name, fn.file_path, fn.start_line, fn.end_line,
                     json.dumps(fn.arg_names), fn.signature_hash, fn.body_hash,
                     fn.ast_summary, fn.complexity, fn.docstring,
                     json.dumps(fn.callers), fn.first_seen, fn.last_modified,
                     int(fn.is_dead)),
                )

    def get_function(self, qualified_name: str) -> Optional[Function]:
        with self.conn() as c:
            row = c.execute(
                "SELECT * FROM functions WHERE qualified_name = ?",
                (qualified_name,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_function(row)

    def all_functions(self) -> List[Function]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM functions ORDER BY file_path, start_line").fetchall()
            return [self._row_to_function(r) for r in rows]

    def functions_by_signature(self, sig_hash: str) -> List[Function]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM functions WHERE signature_hash = ?",
                (sig_hash,),
            ).fetchall()
            return [self._row_to_function(r) for r in rows]

    def functions_by_body(self, body_hash: str) -> List[Function]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM functions WHERE body_hash = ?",
                (body_hash,),
            ).fetchall()
            return [self._row_to_function(r) for r in rows]

    def functions_in_file(self, path: str) -> List[Function]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM functions WHERE file_path = ? ORDER BY start_line",
                (path,),
            ).fetchall()
            return [self._row_to_function(r) for r in rows]

    def _row_to_function(self, row) -> Function:
        # 'calls' column may not exist in pre-v0.3.0 databases
        calls = []
        try:
            calls = json.loads(row["calls"]) if row["calls"] else []
        except (IndexError, sqlite3.OperationalError, json.JSONDecodeError):
            calls = []
        return Function(
            qualified_name=row["qualified_name"], name=row["name"],
            file_path=row["file_path"], start_line=row["start_line"],
            end_line=row["end_line"], arg_names=json.loads(row["arg_names"]),
            signature_hash=row["signature_hash"], body_hash=row["body_hash"],
            ast_summary=row["ast_summary"], complexity=row["complexity"],
            docstring=row["docstring"], calls=calls,
            callers=json.loads(row["callers"]),
            first_seen=row["first_seen"], last_modified=row["last_modified"],
            is_dead=bool(row["is_dead"]),
        )

    def function_count(self) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM functions").fetchone()[0]

    def delete_functions_in_file(self, path: str):
        """Delete all function rows for a file. Called before re-indexing a changed file.
        Also cleans up embeddings. History rows are preserved (no FK after v0.3.1)."""
        with self.conn() as c:
            # Get the qualified names of functions being deleted
            rows = c.execute(
                "SELECT qualified_name FROM functions WHERE file_path = ?", (path,)
            ).fetchall()
            qnames = [r["qualified_name"] for r in rows]

            # Delete embeddings (these have a FK to functions)
            for qn in qnames:
                c.execute(
                    "DELETE FROM embeddings WHERE qualified_name = ?", (qn,)
                )

            # Now delete the functions
            c.execute("DELETE FROM functions WHERE file_path = ?", (path,))

    def delete_classes_in_file(self, path: str):
        """Delete all class rows for a file."""
        with self.conn() as c:
            c.execute("DELETE FROM classes WHERE file_path = ?", (path,))

    def delete_file(self, path: str):
        """Delete a file and all its functions, classes, and embeddings.
        History rows are preserved (they outlive the function)."""
        with self.conn() as c:
            # Get function qualified names before deleting (for embedding cleanup)
            rows = c.execute(
                "SELECT qualified_name FROM functions WHERE file_path = ?", (path,)
            ).fetchall()
            for row in rows:
                qn = row["qualified_name"]
                c.execute(
                    "DELETE FROM embeddings WHERE qualified_name = ?",
                    (qn,),
                )
            c.execute("DELETE FROM functions WHERE file_path = ?", (path,))
            c.execute("DELETE FROM classes WHERE file_path = ?", (path,))
            c.execute("DELETE FROM files WHERE path = ?", (path,))

    # --- Classes ---

    def upsert_class(self, cls: Class):
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO classes
                   (qualified_name, name, file_path, start_line, end_line,
                    bases, method_names, first_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (cls.qualified_name, cls.name, cls.file_path, cls.start_line,
                 cls.end_line, json.dumps(cls.bases),
                 json.dumps(cls.method_names), cls.first_seen),
            )

    def all_classes(self) -> List[Class]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM classes ORDER BY file_path, start_line").fetchall()
            return [Class(
                qualified_name=r["qualified_name"], name=r["name"],
                file_path=r["file_path"], start_line=r["start_line"],
                end_line=r["end_line"], bases=json.loads(r["bases"]),
                method_names=json.loads(r["method_names"]),
                first_seen=r["first_seen"],
            ) for r in rows]

    # --- Plans ---

    def upsert_plan(self, plan: Plan):
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO plans
                   (id, description, keywords, status, created, updated,
                    notes, related_files)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (plan.id, plan.description, json.dumps(plan.keywords),
                 plan.status, plan.created, plan.updated,
                 json.dumps(plan.notes), json.dumps(plan.related_files)),
            )

    def active_plans(self) -> List[Plan]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM plans WHERE status = 'active' ORDER BY created DESC"
            ).fetchall()
            return [self._row_to_plan(r) for r in rows]

    def all_plans(self) -> List[Plan]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM plans ORDER BY created DESC").fetchall()
            return [self._row_to_plan(r) for r in rows]

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
            return self._row_to_plan(row) if row else None

    def _row_to_plan(self, row) -> Plan:
        return Plan(
            id=row["id"], description=row["description"],
            keywords=json.loads(row["keywords"]), status=row["status"],
            created=row["created"], updated=row["updated"],
            notes=json.loads(row["notes"]),
            related_files=json.loads(row["related_files"]),
        )

    # --- Observations ---

    def add_observation(self, obs: Observation):
        """Add or update an observation. If an observation with the same ID
        already exists and is acknowledged, preserve the acknowledged status
        (don't re-report something the user already dismissed)."""
        with self.conn() as c:
            # Check if this observation already exists
            existing = c.execute(
                "SELECT acknowledged FROM observations WHERE id = ?", (obs.id,)
            ).fetchone()
            if existing and existing["acknowledged"]:
                # Already acknowledged — don't overwrite (preserve ack status).
                # Update the message in case it changed, but keep acknowledged=1.
                c.execute(
                    """UPDATE observations SET message = ?, severity = ?, created = ?
                       WHERE id = ?""",
                    (obs.message, obs.severity, obs.created, obs.id),
                )
            else:
                # New observation, or update an unacknowledged one
                c.execute(
                    """INSERT OR REPLACE INTO observations
                       (id, kind, severity, message, file_path, function_qualified_name,
                        line, related_plan_id, related_function_qualified_name,
                        created, acknowledged)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (obs.id, obs.kind, obs.severity, obs.message, obs.file_path,
                     obs.function_qualified_name, obs.line, obs.related_plan_id,
                     obs.related_function_qualified_name, obs.created,
                     int(obs.acknowledged)),
                )

    def unacknowledged_observations(self, limit: int = 100) -> List[Observation]:
        with self.conn() as c:
            rows = c.execute(
                """SELECT * FROM observations WHERE acknowledged = 0
                   ORDER BY created DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [self._row_to_observation(r) for r in rows]

    def all_observations(self, limit: int = 200) -> List[Observation]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM observations ORDER BY created DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_observation(r) for r in rows]

    def acknowledge_observation(self, obs_id: str):
        with self.conn() as c:
            c.execute(
                "UPDATE observations SET acknowledged = 1 WHERE id = ?",
                (obs_id,),
            )

    def _row_to_observation(self, row) -> Observation:
        return Observation(
            id=row["id"], kind=row["kind"], severity=row["severity"],
            message=row["message"], file_path=row["file_path"],
            function_qualified_name=row["function_qualified_name"],
            line=row["line"], related_plan_id=row["related_plan_id"],
            related_function_qualified_name=row["related_function_qualified_name"],
            created=row["created"], acknowledged=bool(row["acknowledged"]),
        )

    # --- Clichés ---

    def upsert_cliche(self, cliche: Cliche):
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO cliches
                   (signature_hash, body_hash, instances, first_seen, note)
                   VALUES (?, ?, ?, ?, ?)""",
                (cliche.signature_hash, cliche.body_hash,
                 json.dumps(cliche.instances), cliche.first_seen, cliche.note),
            )

    def all_cliches(self, min_instances: int = 2) -> List[Cliche]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM cliches").fetchall()
            result = []
            for r in rows:
                instances = json.loads(r["instances"])
                if len(instances) >= min_instances:
                    result.append(Cliche(
                        signature_hash=r["signature_hash"],
                        body_hash=r["body_hash"], instances=instances,
                        first_seen=r["first_seen"], note=r["note"],
                    ))
            return result

    # --- Embeddings ---

    def set_embedding(self, qualified_name: str, vector: List[float], backend: str):
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO embeddings
                   (qualified_name, vector, dim, backend, created)
                   VALUES (?, ?, ?, ?, ?)""",
                (qualified_name, json.dumps(vector), len(vector), backend, utcnow_iso()),
            )

    def get_embedding(self, qualified_name: str) -> Optional[Tuple[List[float], str]]:
        with self.conn() as c:
            row = c.execute(
                "SELECT vector, backend FROM embeddings WHERE qualified_name = ?",
                (qualified_name,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["vector"]), row["backend"]

    # --- Snapshot log ---

    def log_snapshot(self, files_checked: int, observations_emitted: int, notes: str = ""):
        with self.conn() as c:
            c.execute(
                """INSERT INTO snapshot_log
                   (run_at, files_checked, observations_emitted, notes)
                   VALUES (?, ?, ?, ?)""",
                (utcnow_iso(), files_checked, observations_emitted, notes),
            )

    def last_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.conn() as c:
            row = c.execute(
                "SELECT * FROM snapshot_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    # --- Historical tracking (v0.2.0) ---

    def snapshot_function(self, fn: Function):
        """Record a snapshot of the function's current state."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self.conn() as c:
            # Check if function_history table exists (v0.2.0 migration)
            try:
                c.execute(
                    """INSERT INTO function_history
                       (qualified_name, snapshot_at, complexity, line_count,
                        caller_count, body_hash)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (fn.qualified_name, now, fn.complexity,
                     fn.end_line - fn.start_line + 1,
                     len(fn.callers), fn.body_hash),
                )
            except sqlite3.OperationalError:
                pass  # table doesn't exist yet — migration will handle it

    def snapshot_all_functions(self):
        """Snapshot all functions. Called at the end of each index run."""
        for fn in self.all_functions():
            self.snapshot_function(fn)

    def function_history(self, qualified_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the history of a function's complexity/size over time."""
        with self.conn() as c:
            try:
                rows = c.execute(
                    """SELECT * FROM function_history
                       WHERE qualified_name = ?
                       ORDER BY snapshot_at DESC LIMIT ?""",
                    (qualified_name, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                return []

    def complexity_trends(self, min_changes: int = 2) -> List[Dict[str, Any]]:
        """Find functions whose complexity has changed over time.
        Returns functions with at least `min_changes` snapshots where
        complexity differs between first and last."""
        with self.conn() as c:
            try:
                rows = c.execute(
                    """SELECT qualified_name,
                              MIN(complexity) as min_c,
                              MAX(complexity) as max_c,
                              COUNT(*) as snapshots
                       FROM function_history
                       GROUP BY qualified_name
                       HAVING snapshots >= ? AND min_c != max_c
                       ORDER BY (max_c - min_c) DESC""",
                    (min_changes,),
                ).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                return []

    def save_observation_explanation(self, observation_id: str, explanation: str,
                                      diff: str, llm_backend: str):
        """Save an LLM-generated explanation/fix for an observation."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self.conn() as c:
            try:
                c.execute(
                    """INSERT OR REPLACE INTO observation_explanations
                       (observation_id, explanation, diff, llm_backend, created)
                       VALUES (?, ?, ?, ?, ?)""",
                    (observation_id, explanation, diff, llm_backend, now),
                )
            except sqlite3.OperationalError:
                pass

    def get_observation_explanation(self, observation_id: str) -> Optional[Dict[str, Any]]:
        with self.conn() as c:
            try:
                row = c.execute(
                    "SELECT * FROM observation_explanations WHERE observation_id = ?",
                    (observation_id,),
                ).fetchone()
                return dict(row) if row else None
            except sqlite3.OperationalError:
                return None


def default_db_path(repo_root: str) -> str:
    """Where the Apprentice stores its model for a given repo."""
    return os.path.join(repo_root, ".apprentice", "apprentice.db")


def init_store(repo_root: str) -> Store:
    db_path = default_db_path(repo_root)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    store = Store(db_path)
    store.init_schema()
    return store
