# The Programmer's Apprentice

**A persistent, proactive coding agent — the 1988 vision, finally buildable.**

> "The Apprentice would be a persistent, proactive assistant that maintains a
> model of the codebase, recognizes the clichés in use, tracks the programmer's
> plan, and pipes up when it sees bugs, drift, or repeated patterns — without
> being asked."
>
> — Rich & Waters, "The Programmer's Apprentice," IEEE Computer, 1988

For 35 years this was a research vision. The AI didn't exist. Now it does.

## What makes this different

| Tool | Persistent model? | Proactive? | Plan-aware? | Cliché-aware? | LLM synthesis? |
|---|---|---|---|---|---|
| GitHub Copilot | partial (rules) | ❌ | ❌ | ❌ | ✓ |
| Cursor | partial (rules) | partial (PR) | ❌ | ❌ | ✓ |
| Devin | ❌ (per-task) | partial (incidents) | ❌ | ❌ | ✓ |
| Claude Code | partial (sleep) | ❌ | ❌ | ❌ | ✓ |
| **Apprentice** | **✓ (SQLite)** | **✓** | **✓** | **✓** | **✓** |

Most coding tools maintain *some* workspace context (Copilot rules, Cursor
memories, Claude Code's CLAUDE.md). The Apprentice's difference is a
**structured, queryable model** — not just rules or notes, but a typed graph
of functions, call edges, clichés, and plans — plus **proactive analysis**
that runs without being asked. The dead-code and duplication analyzers use
an AST-based call graph (not regex) and deterministic observation IDs for
deduplication.

## Install

```bash
pip install -e .

# Optional: LLM backends
pip install -e ".[openai]"      # OpenAI / Z.ai GLM
pip install -e ".[anthropic]"   # Anthropic Claude
pip install -e ".[embeddings]"  # sentence-transformers
pip install -e ".[daemon]"      # watchdog for file watching
pip install -e ".[dev]"         # pytest for development
```

Requires Python 3.9+. No external services needed for the core — embeddings
default to offline TF-IDF, LLM defaults to mock mode.

## Quick start

```bash
cd your-repo
apprentice init              # one-time setup
apprentice index             # build the codebase model
apprentice status            # see what the Apprentice knows

# State an intent
apprentice plan "refactor authentication to use JWT tokens"

# Make changes, then:
apprentice watch             # proactive analysis

# LLM-powered commands (set an API key first)
export OPENAI_API_KEY=sk-...  # or ANTHROPIC_API_KEY or ZAI_API_KEY
apprentice ask "where do we handle login?"
apprentice fix <observation-id>
apprentice summarize --codebase

# Always-on mode
apprentice daemon            # watches for changes, proactive in background

# Git integration
apprentice hook install      # pre-commit hook runs proactive checks
```

## Commands (v0.2.0)

| Command | Description |
|---|---|
| `apprentice init` | Initialize the Apprentice in this repo |
| `apprentice index [--rebuild]` | Index the codebase (multi-language) |
| `apprentice status` | Show what the Apprentice knows |
| `apprentice plan <text>` | State an intent |
| `apprentice plan --list` | List plans |
| `apprentice plan --done ID` | Mark a plan completed |
| `apprentice watch [--all] [--staged]` | Run proactive analyzers |
| `apprentice observations [--all]` | Show observations |
| `apprentice ack <ID>` | Acknowledge observations |
| `apprentice ask <question>` | **LLM-powered** natural-language Q&A |
| `apprentice fix <obs-id>` | **LLM-powered** fix synthesis |
| `apprentice summarize <name>` | **LLM-powered** function summary |
| `apprentice summarize --codebase` | **LLM-powered** codebase summary |
| `apprentice recall <name>` | Show what the Apprentice knows about a function |
| `apprentice similar <name>` | Find similar functions (embedding-based) |
| `apprentice history <name>` | Show a function's complexity history |
| `apprentice daemon` | Run as a background watcher |
| `apprentice hook install\|uninstall\|status` | Git hook management |
| `apprentice config [--init]` | Show or initialize configuration |

## The proactive analyzers

When you run `apprentice watch`, the Apprentice runs seven analyzers:

1. **Plan drift** — flags work outside any active plan
2. **Duplication** — detects same-body functions (clichés)
3. **Dead code** — functions with no callers
4. **Complexity creep** — functions exceeding complexity thresholds
5. **TODO without plan** — TODOs that don't match any active plan
6. **New pattern** — functions sharing signatures with existing ones
7. **Complexity trend** (v0.2.0) — functions whose complexity is *growing over time*

## Architecture

```
apprentice/
├── config.py              # .apprentice.toml configuration
├── model/
│   ├── entities.py        # File, Function, Class, Plan, Observation, Cliche
│   ├── store.py           # SQLite persistence with historical tracking
│   └── migrations.py      # Versioned schema migrations
├── indexer/
│   ├── base.py            # LanguageParser interface
│   ├── registry.py        # Language detection + parser selection
│   ├── python_parser.py   # Python AST parser
│   ├── javascript_parser.py # JS/TS lightweight parser
│   └── embedder.py        # Pluggable: TF-IDF | sentence-transformers | OpenAI
├── analyzer/
│   ├── proactive.py       # 6 proactive analyzers
│   └── historical.py      # Complexity trend analyzer
├── llm/                   # v0.2.0: LLM integration
│   ├── client.py          # Pluggable: OpenAI | Anthropic | Z.ai | mock
│   ├── ask.py             # Natural-language Q&A
│   ├── fix.py             # Fix synthesis (diff generation)
│   └── summarize.py       # Function/codebase summarization
├── interface/
│   ├── cli.py             # Full CLI
│   └── output.py          # Rich colored output
├── daemon.py              # Background file watcher
└── hooks.py               # Git pre-commit hook
```

## Configuration

Create `.apprentice.toml` in your repo root:

```toml
[llm]
backend = "openai"  # or "anthropic", "zai", or omit for auto-detect
model = "gpt-4o-mini"

[embeddings]
backend = "tfidf"  # or "sentence-transformers", "openai"

[analyzer]
complexity_warn = 15
complexity_error = 30

[daemon]
watch_interval_seconds = 5.0
auto_acknowledge_info = false

[hooks]
block_on_error = true
block_on_warning = false

[indexing]
ignore_dirs = ["__pycache__", ".git", "node_modules", "vendor"]
file_extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]
```

Or run `apprentice config --init` to create one with defaults.

## LLM backends

The Apprentice supports multiple LLM backends, auto-detected from environment:

| Backend | Env var | Models |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini, gpt-4o, etc. |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514, claude-opus, etc. |
| Z.ai | `ZAI_API_KEY` | glm-4-flash, glm-4, etc. |
| Mock | (none) | Deterministic offline responses |

The LLM is **optional**. The core (persistence + proactivity) works without
any LLM. The LLM adds: natural-language `ask`, `fix` synthesis, `summarize`.

## Multi-language support

| Language | Parser | Status |
|---|---|---|
| Python | AST (ast module) | Full support |
| JavaScript | Lightweight scanner | Beta — finds functions, arrows, classes, methods, and calls |
| TypeScript | Lightweight scanner | Beta — handles common TS generics and type annotations |

To add a new language, implement the `LanguageParser` interface in `indexer/base.py`
and register it in `registry.py`.

## What this is NOT (yet)

- **Not an IDE plugin.** CLI only. The store is designed for IDE integration.
- **No `--apply` for fixes.** The `fix` command generates diffs; applying them is manual (or pipe to `git apply`).
- **JS parser is lightweight.** Less accurate than a full AST. Replace with tree-sitter for production.
- **No semantic drift.** Plan drift is keyword-based. Embedding-based semantic drift is the next step.

## Roadmap

- [x] Persistent codebase model (SQLite)
- [x] Python AST indexer
- [x] Seven proactive analyzers (AST-based call graph, deterministic IDs)
- [x] Pluggable embeddings (offline default: hashed-TF)
- [x] Plan tracking + drift detection
- [x] CLI
- [x] **LLM integration** (ask, fix, summarize)
- [x] **Historical tracking** (complexity trends over time)
- [x] **Configuration** (.apprentice.toml)
- [x] **Multi-language** (Python + JS/TS)
- [x] **Git hooks** (pre-commit proactive checks)
- [x] **Daemon mode** (always-on background watcher)
- [x] **Rich CLI** (colors, formatting)
- [x] **Schema migrations** (versioned, forward-only)
- [x] **CI/CD** (GitHub Actions)
- [x] **Self-hosting tests** (index own repo, verify analyzers don't false-positive)
- [ ] `apprentice fix --apply` (auto-apply patches)
- [ ] VS Code extension
- [ ] Tree-sitter for JS/TS (replacing the lightweight scanner)
- [ ] Semantic drift detection (embedding-based)
- [ ] Real IDF computation (currently hashed-TF, not full TF-IDF)
- [ ] Sleep-time consolidation (forward-looking, like Claude Code's Auto Dream but proactive)

## Why now

The 1988 spec required:
- Recognizing clichés → LLMs do this trivially
- Inferring plans → LLMs do this from natural language
- Synthesizing routine code → LLMs do this well
- Persistent codebase model → SQLite + embeddings make this cheap
- Proactivity → just a daemon that runs the analyzers

Every ingredient exists. Nobody has assembled them. This is that assembly.

## Origin

The Programmer's Apprentice was specified in:
- Rich, C. & Waters, R. C. (1988). "The Programmer's Apprentice." *IEEE Computer*, 21(11), 10-25.
- Shrobe, H. & Katz, A. (2015). "Towards a Programmer's Apprentice (Again)." AAAI Workshop.

Closest existing work: Shadow-Frog (Microsoft Research), Claude Code's Auto Dream,
Cursor Bugbot. None implements the full vision. See `docs/competition_audit.md`.

## License

MIT. See [LICENSE](LICENSE).
