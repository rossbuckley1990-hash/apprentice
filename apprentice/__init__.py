"""
The Programmer's Apprentice — a persistent, proactive coding agent.

After Rich & Waters, "The Programmer's Apprentice" (IEEE Computer, 1988):
  - PERSISTENT: maintains a living model of your codebase across sessions
  - PROACTIVE:  flags bugs, drift, and repeated patterns WITHOUT being asked
  - PLAN-AWARE: knows your current intent and checks new code against it
  - CLICHÉ-AWARE: recognizes patterns in use across the codebase

v0.2.0 adds:
  - LLM integration (ask, fix, summarize) — the 1988 "synthesis engine"
  - Historical tracking (complexity trends over time)
  - Configuration system (.apprentice.toml)
  - Multi-language framework (Python + JavaScript/TypeScript)
  - Git hook integration (pre-commit proactive checks)
  - Daemon mode (always-on background watcher)
  - Rich CLI output (colors, formatting)
  - Schema migrations (versioned, forward-only)
"""

__version__ = "0.3.1"
