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

| Tool | Persistent model? | Proactive? | Plan-aware? | Cliché-aware? |
|---|---|---|---|---|
| GitHub Copilot | ❌ | ❌ | ❌ | ❌ |
| Cursor | partial (rules) | partial (PR only) | ❌ | ❌ |
| Devin | ❌ (per-task) | partial (incidents) | ❌ | ❌ |
| Claude Code | partial (sleep) | ❌ | ❌ | ❌ |
| **Apprentice** | **✓** | **✓** | **✓** | **✓** |

Copilot, Cursor, and Devin are **stateless prompt-response tools**. They forget
everything between sessions. They only respond when asked. They don't know what
you're trying to do. They don't notice when you repeat yourself.

The Apprentice is the opposite:
- **Persistent** — maintains a living model of your codebase in SQLite. Remembers across sessions.
- **Proactive** — runs `apprentice watch` and flags issues *without being asked*.
- **Plan-aware** — you state an intent; the Apprentice checks new code against it.
- **Cliché-aware** — detects when you've written the same function twice.

## Install

```bash
pip install -e .
```

Requires Python 3.9+. No external services needed for the MVP — embeddings
default to a fully offline TF-IDF backend.

## Quick start

```bash
cd your-repo
apprentice init              # one-time setup
apprentice index             # build the codebase model
apprentice status            # see what the Apprentice knows

# State an intent — the Apprentice tracks it
apprentice plan "refactor authentication to use JWT tokens"

# Make some changes to your code, then:
apprentice watch             # proactive analysis — flags drift, duplication, etc.

# Ask about your codebase using the persistent model
apprentice ask "where do we handle login"
apprentice recall mymodule.myfunction
apprentice similar mymodule.myfunction

# Review and acknowledge observations
apprentice observations
apprentice ack <observation-id>
```

## The proactive analyzers

When you run `apprentice watch`, the Apprentice runs six analyzers over the
files that changed since the last index:

1. **Plan drift** — if you have an active plan about "auth" but new code
   introduces "ui" work, the Apprentice flags it: *File introduces work outside
   any active plan.*

2. **Duplication** — when a new function's body-hash matches an existing
   function's, the Apprentice points out both: *Found 2 instances of the same
   function body. Consider extracting a shared helper.*

3. **Dead code** — functions with no callers, that aren't entry points or
   tests: *Function 'foo' has no callers and isn't an entry point. Candidate
   for removal.*

4. **Complexity creep** — functions whose cyclomatic complexity exceeds
   thresholds: *Function 'bar' has complexity 32 (threshold 30). Strongly
   consider refactoring.*

5. **TODO without plan** — TODO/FIXME markers that don't match any active
   plan: *FIXME added without matching active plan. Either create a plan for
   this or it'll be forgotten.*

6. **New pattern (cliché recognition)** — when you write a function whose
   signature matches a family of existing functions: *Function 'baz' shares
   its signature with 3 others. This is a cliché — consider whether they
   should share an implementation.*

None of these require an LLM call. They run on the structured codebase model.
The LLM is for the *next* layer — generating fixes, explaining observations,
synthesizing routine code — and is pluggable.

## Architecture

```
apprentice/
├── model/
│   ├── entities.py     # File, Function, Class, Plan, Observation, Cliche
│   └── store.py        # SQLite persistence (the cross-session memory)
├── indexer/
│   ├── python_parser.py # AST-based; pluggable for other languages
│   └── embedder.py      # Pluggable: TF-IDF (default, offline) | sentence-transformers | OpenAI
├── analyzer/
│   └── proactive.py     # The 6 proactive analyzers — what makes this the Apprentice
└── interface/
    └── cli.py            # init / index / plan / watch / ask / recall / similar
```

The **store** is the persistence layer — a SQLite database in `.apprentice/`
that survives across sessions. This is the structural primitive that
Copilot/Cursor/Devin lack.

The **indexer** parses Python files into typed entities (File, Function, Class)
with structural hashes (signature_hash, body_hash) for cliché detection. The
embedding backend is pluggable: default is offline TF-IDF; you can opt into
sentence-transformers or OpenAI embeddings by installing the extras.

The **analyzer** is the proactivity. Six pure functions, each taking the store
+ a list of changed files and returning Observations. They never modify state
directly — the orchestrator persists observations.

The **interface** is a CLI today. An IDE plugin (VS Code, Neovim) is the
natural next step — same store, same analyzers, just a different front-end.

## What this is NOT (yet)

- **Not an LLM-powered code generator.** The MVP is the *memory + proactivity*
  layer. LLM synthesis (the 1988 spec's "synthesis engine") is the next layer.
- **Not an IDE plugin.** CLI only for now. The store is designed for IDE
  integration.
- **Not multi-language.** Python only. The `python_parser.py` interface makes
  adding JS/TS/Rust straightforward.
- **Not a remote service.** Local-first by design. Your codebase model lives
  in `.apprentice/apprentice.db` and never leaves your machine.

## Roadmap

- [x] Persistent codebase model (SQLite)
- [x] Python AST indexer
- [x] Six proactive analyzers
- [x] Pluggable embeddings (offline default)
- [x] Plan tracking + drift detection
- [x] CLI
- [ ] LLM-backed `apprentice ask` (currently keyword search)
- [ ] LLM-backed `apprentice fix <observation-id>` (synthesize a fix)
- [ ] VS Code extension (uses the same store)
- [ ] Git hook integration (run `watch` on every commit)
- [ ] Multi-language support (JS/TS, Rust, Go)
- [ ] Historical tracking (per-function complexity over time)
- [ ] Sleep-time consolidation (like Claude Code's Auto Dream, but forward-looking)

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
Cursor Bugbot. None implements the full vision. See `docs/competition_audit.md`
for the full landscape.

## License

MIT. See [LICENSE](LICENSE).
