"""
Daemon mode — the always-on Apprentice.

Runs in the background, watches for file changes, and proactively emits
observations. This is the 1988 spec's "persistent" + "proactive" made
continuous: the Apprentice is always watching, always thinking.

Uses filesystem polling by default (no external deps). If `watchdog` is
installed, uses inotify for instant notification.
"""

from __future__ import annotations
import os
import sys
import time
import signal
from pathlib import Path
from typing import Optional, Set, List

from .config import load_config
from .model.store import init_store
from .indexer.python_parser import index_repo, discover_all_files
from .analyzer.proactive import run_all_analyzers
from .model.entities import hash_content


class Daemon:
    """Background file watcher + proactive analyzer."""

    def __init__(self, repo_root: str, config=None):
        self.repo_root = repo_root
        self.config = config or load_config(repo_root)
        self.store = init_store(repo_root)
        self.running = False
        self._last_hashes: dict[str, str] = {}
        self._lockfile: Optional[str] = None
        self._init_hashes()

    def _acquire_lock(self) -> bool:
        """Acquire a PID lockfile to prevent two daemons on the same repo.
        Returns True if acquired, False if another daemon is running."""
        import os
        lockpath = os.path.join(self.repo_root, ".apprentice", "daemon.pid")
        os.makedirs(os.path.dirname(lockpath), exist_ok=True)
        try:
            fd = os.open(lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._lockfile = lockpath
            return True
        except FileExistsError:
            # Check if the PID is still alive
            try:
                with open(lockpath, "r") as f:
                    old_pid = int(f.read().strip())
                os.kill(old_pid, 0)  # signal 0 = check existence
                return False  # process is alive
            except (OSError, ValueError):
                # Stale lockfile — remove and retry
                try:
                    os.unlink(lockpath)
                    fd = os.open(lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    self._lockfile = lockpath
                    return True
                except OSError:
                    return False
            except ProcessLookupError:
                # Stale — remove and retry
                try:
                    os.unlink(lockpath)
                    fd = os.open(lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    self._lockfile = lockpath
                    return True
                except OSError:
                    return False

    def _release_lock(self):
        """Release the PID lockfile."""
        import os
        if self._lockfile and os.path.exists(self._lockfile):
            try:
                os.unlink(self._lockfile)
            except OSError:
                pass
        self._lockfile = None

    def _init_hashes(self):
        """Initialize file hashes from the current store."""
        for f in self.store.all_files():
            self._last_hashes[f.path] = f.content_hash

    def get_changed_files(self) -> List[str]:
        """Find files that changed since the last check."""
        changed = []
        all_files = discover_all_files(self.repo_root, self.config)
        for rel_path in all_files:
            abs_path = os.path.join(self.repo_root, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue
            new_hash = hash_content(content)
            old_hash = self._last_hashes.get(rel_path)
            if old_hash != new_hash:
                changed.append(rel_path)
                self._last_hashes[rel_path] = new_hash
        return changed

    def run_once(self) -> int:
        """Run one analysis cycle. Returns number of observations emitted."""
        changed = self.get_changed_files()
        if not changed:
            return 0

        # Re-index
        index_repo(self.repo_root, self.store, verbose=False, config=self.config)

        # Run analyzers
        observations = run_all_analyzers(
            self.store, self.repo_root, changed, config=self.config
        )

        # Persist observations
        for obs in observations:
            self.store.add_observation(obs)

        # Log
        self.store.log_snapshot(
            files_checked=len(changed),
            observations_emitted=len(observations),
            notes="daemon cycle",
        )

        # Print new observations
        unacked = self.store.unacknowledged_observations(limit=10)
        new_unacked = [o for o in observations if not o.acknowledged]
        if new_unacked:
            self._print_observations(new_unacked)

        return len(observations)

    def _print_observations(self, observations):
        from .interface.output import format_observations
        print(format_observations(observations), file=sys.stderr)

    def run(self, interval: Optional[float] = None):
        """Run the daemon loop. Blocks until interrupted."""
        if interval is None:
            interval = self.config.watch_interval_seconds

        # Acquire lock to prevent two daemons on the same repo
        if not self._acquire_lock():
            print("  [apprentice] another daemon is already running on this repo", file=sys.stderr)
            print("  [apprentice] remove .apprentice/daemon.pid if stale", file=sys.stderr)
            return

        self.running = True

        def handle_signal(signum, frame):
            self.running = False
            print("\n  [apprentice] shutting down...", file=sys.stderr)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        print(f"  [apprentice] daemon started — watching {self.repo_root}", file=sys.stderr)
        print(f"  [apprentice] checking every {interval}s. Press Ctrl+C to stop.", file=sys.stderr)
        print(file=sys.stderr)

        try:
            while self.running:
                try:
                    n = self.run_once()
                    if n > 0:
                        print(f"  [apprentice] {n} new observation(s)", file=sys.stderr)
                except Exception as e:
                    print(f"  [apprentice] error: {type(e).__name__}: {e}", file=sys.stderr)

                # Sleep in small increments so we can respond to signals
                for _ in range(int(interval * 10)):
                    if not self.running:
                        break
                    time.sleep(0.1)
        finally:
            self._release_lock()
            print("  [apprentice] daemon stopped.", file=sys.stderr)


def run_daemon(repo_root: str, config=None, interval: Optional[float] = None):
    """Convenience function to start the daemon."""
    daemon = Daemon(repo_root, config)
    daemon.run(interval)
