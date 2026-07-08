"""
Configuration management for the Apprentice.

Config sources (in priority order):
  1. CLI flags (highest)
  2. Environment variables
  3. .apprentice.toml in repo root
  4. ~/.apprentice/config.toml (user-level defaults)
  5. Built-in defaults (lowest)

The config controls:
  - LLM backend + model
  - Complexity thresholds
  - Ignored paths
  - Embedding backend
  - Daemon behavior (watch interval, auto-acknowledge)
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Config:
    # LLM
    llm_backend: Optional[str] = None  # auto-detect if None
    llm_model: Optional[str] = None    # use backend default if None

    # Embeddings
    embedding_backend: str = "tfidf"   # tfidf | asthash | sentence-transformers | openai

    # Analyzer thresholds
    complexity_warn: int = 15
    complexity_error: int = 30

    # Indexing
    ignore_dirs: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", ".hg", ".svn", ".venv", "venv", "env",
        "node_modules", ".mypy_cache", ".pytest_cache", ".tox", "build",
        "dist", ".eggs", ".apprentice", "vendor", "third_party",
    ])
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "*.pyo", "*.min.js", "*.map",
    ])
    file_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ])

    # Daemon
    watch_interval_seconds: float = 5.0
    auto_acknowledge_info: bool = False  # auto-ack low-severity observations

    # Git hooks
    hook_block_on_error: bool = True    # block commit on 'error' severity observations
    hook_block_on_warning: bool = False  # block on 'warning' too?

    # Output
    color: bool = True
    verbose: bool = False

    @classmethod
    def defaults(cls) -> "Config":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_toml(self) -> str:
        """Serialize config to TOML format for .apprentice.toml."""
        lines = ["# Apprentice configuration file", ""]
        d = self.to_dict()

        lines.append("[llm]")
        lines.append(f"backend = {_toml_value(d['llm_backend'])}")
        lines.append(f"model = {_toml_value(d['llm_model'])}")
        lines.append("")

        lines.append("[embeddings]")
        lines.append(f"backend = {_toml_value(d['embedding_backend'])}")
        lines.append("")

        lines.append("[analyzer]")
        lines.append(f"complexity_warn = {d['complexity_warn']}")
        lines.append(f"complexity_error = {d['complexity_error']}")
        lines.append("")

        lines.append("[indexing]")
        lines.append(f"ignore_dirs = {_toml_value(d['ignore_dirs'])}")
        lines.append(f"ignore_patterns = {_toml_value(d['ignore_patterns'])}")
        lines.append(f"file_extensions = {_toml_value(d['file_extensions'])}")
        lines.append("")

        lines.append("[daemon]")
        lines.append(f"watch_interval_seconds = {d['watch_interval_seconds']}")
        lines.append(f"auto_acknowledge_info = {str(d['auto_acknowledge_info']).lower()}")
        lines.append("")

        lines.append("[hooks]")
        lines.append(f"block_on_error = {str(d['hook_block_on_error']).lower()}")
        lines.append(f"block_on_warning = {str(d['hook_block_on_warning']).lower()}")
        lines.append("")

        lines.append("[output]")
        lines.append(f"color = {str(d['color']).lower()}")
        lines.append(f"verbose = {str(d['verbose']).lower()}")
        lines.append("")

        return "\n".join(lines)


def _toml_value(v) -> str:
    """Format a Python value as a TOML value."""
    if v is None:
        return '""'
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return f'"{v}"'


def _parse_toml(text: str) -> Dict[str, Any]:
    """Minimal TOML parser (no external deps). Supports the subset we use."""
    result: Dict[str, Any] = {}
    current_section: Dict[str, Any] = result
    current_section_name = ""

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section_name = line[1:-1]
            current_section = result.setdefault(current_section_name, {})
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Remove inline comments
        if " #" in value:
            value = value.split(" #")[0].strip()
        # Parse value
        current_section[key] = _parse_toml_value(value)

    return result


def _parse_toml_value(value: str):
    """Parse a TOML scalar value."""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [_parse_toml_value(p.strip()) for p in inner.split(",")]
        return parts
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(repo_root: Optional[str] = None) -> Config:
    """Load config from all sources, merged in priority order."""
    config = Config.defaults()

    # User-level config: ~/.apprentice/config.toml
    user_config_path = Path.home() / ".apprentice" / "config.toml"
    if user_config_path.exists():
        _merge_config(config, _load_toml_file(user_config_path))

    # Repo-level config: .apprentice.toml in repo root
    if repo_root is None:
        repo_root = os.getcwd()
    repo_config_path = Path(repo_root) / ".apprentice.toml"
    if repo_config_path.exists():
        _merge_config(config, _load_toml_file(repo_config_path))

    # Environment variable overrides
    if os.environ.get("APPRENTICE_LLM_BACKEND"):
        config.llm_backend = os.environ["APPRENTICE_LLM_BACKEND"]
    if os.environ.get("APPRENTICE_LLM_MODEL"):
        config.llm_model = os.environ["APPRENTICE_LLM_MODEL"]
    if os.environ.get("APPRENTICE_EMBEDDING_BACKEND"):
        config.embedding_backend = os.environ["APPRENTICE_EMBEDDING_BACKEND"]
    if os.environ.get("APPRENTICE_VERBOSE"):
        config.verbose = True

    return config


def _load_toml_file(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return _parse_toml(f.read())
    except OSError:
        return {}


def _merge_config(config: Config, toml_data: Dict[str, Any]):
    """Merge TOML data into a Config object."""
    if "llm" in toml_data:
        llm = toml_data["llm"]
        if "backend" in llm and llm["backend"]:
            config.llm_backend = llm["backend"]
        if "model" in llm and llm["model"]:
            config.llm_model = llm["model"]
    if "embeddings" in toml_data:
        emb = toml_data["embeddings"]
        if "backend" in emb:
            config.embedding_backend = emb["backend"]
    if "analyzer" in toml_data:
        an = toml_data["analyzer"]
        if "complexity_warn" in an:
            config.complexity_warn = int(an["complexity_warn"])
        if "complexity_error" in an:
            config.complexity_error = int(an["complexity_error"])
    if "indexing" in toml_data:
        idx = toml_data["indexing"]
        if "ignore_dirs" in idx:
            config.ignore_dirs = idx["ignore_dirs"]
        if "ignore_patterns" in idx:
            config.ignore_patterns = idx["ignore_patterns"]
        if "file_extensions" in idx:
            config.file_extensions = idx["file_extensions"]
    if "daemon" in toml_data:
        da = toml_data["daemon"]
        if "watch_interval_seconds" in da:
            config.watch_interval_seconds = float(da["watch_interval_seconds"])
        if "auto_acknowledge_info" in da:
            config.auto_acknowledge_info = bool(da["auto_acknowledge_info"])
    if "hooks" in toml_data:
        hk = toml_data["hooks"]
        if "block_on_error" in hk:
            config.hook_block_on_error = bool(hk["block_on_error"])
        if "block_on_warning" in hk:
            config.hook_block_on_warning = bool(hk["block_on_warning"])
    if "output" in toml_data:
        out = toml_data["output"]
        if "color" in out:
            config.color = bool(out["color"])
        if "verbose" in out:
            config.verbose = bool(out["verbose"])


def save_config(config: Config, repo_root: str):
    """Write config to .apprentice.toml in the repo root."""
    path = Path(repo_root) / ".apprentice.toml"
    with open(path, "w") as f:
        f.write(config.to_toml())
