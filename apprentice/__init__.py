"""
The Programmer's Apprentice — a persistent, proactive coding agent.

After Rich & Waters, "The Programmer's Apprentice" (IEEE Computer, 1988):
  - PERSISTENT: maintains a living model of your codebase across sessions
  - PROACTIVE:  flags bugs, drift, and repeated patterns WITHOUT being asked
  - PLAN-AWARE: knows your current intent and checks new code against it
  - CLICHÉ-AWARE: recognizes patterns in use across the codebase

This is what Copilot, Cursor, and Devin are NOT. They are stateless
prompt-response tools. The Apprentice remembers.

MVP scope:
  - Python AST-based indexer (extensible to other languages)
  - SQLite persistence (no external services required)
  - Pluggable embeddings (default: AST-hash + TF-IDF for offline operation)
  - Proactive analyzers: drift, duplication, dead code, complexity creep,
    TODO-without-plan
  - CLI: init / index / plan / watch / ask / recall
"""

__version__ = "0.1.0"
