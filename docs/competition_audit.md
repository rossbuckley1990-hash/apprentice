# Competition Audit — Mid-2026

## Verdict: The full Apprentice is unbuilt.

No shipped product has all four features (persistence, proactivity, plan tracking, cliché recognition). Two of the four (plan tracking, cliché recognition) are essentially absent from shipped products entirely.

## Per-Feature Scorecard

| Feature | Status |
|---|---|
| **Persistence** | Partial — memory is everywhere, but factoid/markdown/preferences, NOT a structured model |
| **Proactivity** | Partial + narrow — Cursor Bugbot (PR review) and Devin Auto-Triage (incidents) are real, but event-triggered in narrow domains |
| **Plan tracking** | Effectively unbuilt — only per-session plan mode exists |
| **Cliché recognition** | Unbuilt — agents pattern-match badly when prompted |

## Per-Product Detail

### Stateless tools (still stateless or thin memory)

- **GitHub Copilot** — Agentic Memory (Jan 2026) is factoid memory, not a model. Coding Agent is async for assigned tasks. No proactivity.
- **Cursor** — Has "Memories" (auto-generated project rules) and Cursor Automations (March 2026) including Bugbot, which proactively reviews every PR. Closest shipped thing to Apprentice proactivity, but PR-event-triggered, not intent-driven.
- **Sourcegraph Cody** — Still stateless (RAG over code search, no agent memory).
- **Codeium/Windsurf** — Has "Memories" but users call them "passive reference points." Cognition acquired Windsurf (Dec 2025) and rebranded as "Devin Desktop" (June 2026).
- **Continue.dev** — Acquired by Cursor. Stateless; memory is an open feature request.
- **Devin** — Auto-Triage genuinely proactively investigates incoming alerts/incidents and opens PRs. But narrowly scoped to incident response.
- **SWE-agent / OpenHands** — Stateless benchmark runners.
- **Aider** — Stateless terminal pair programmer.
- **Claude Code** — Most sophisticated persistence in any shipped product: 4 layers (CLAUDE.md → auto-memory → Auto Dream sleep-time consolidation → KAIROS unreleased always-on daemon). But Auto Dream "looks backward, not forward." Hooks are user-defined plumbing. No proactivity out of the box.

### Research projects (none shipped as products)

- **Stanford Amanuensis** — Was a 2018 course iteration (CS379C), never commercialized.
- **"Towards a Programmer's Apprentice (Again)"** — AAAI 2015 (not 2020 as sometimes cited), Shrobe & Katz (MIT), about REASON. Never shipped.
- **Shadow-Frog (Microsoft Research)** — Closest thing to the Apprentice vision that exists. Open-source (github.com/microsoft/ShadowFrog). Builds a shadow knowledge base in idle time, has a "meditate" skill that finds duplicate/near-duplicate discoveries (real cliché recognition!). But it's a research skill suite, not a product, and lacks plan tracking.
- **ToM-SWE** (arxiv 2510.21903) — Research variant of SWE-agent that maintains persistent memory of user goals. Not shipped.
- **"Agentic Coding Needs Proactivity, Not Just Autonomy"** (arxiv 2605.06717, May 2026) — Academic argument for the gap the Apprentice fills.

### Memory tools (none cross into proactivity)

- **Mem0, Zep, Letta** — General-purpose agent memory frameworks, not coding-specific.
- **Cognee** — Indexes codebases into knowledge graphs (closest to structured persistence) but it's a library, not an agent. No proactivity.
- Third-party MCP servers (Claude-Mem, agentmemory, agora-code, Hindsight, Recallium) — All recall layers, none proactive.

## The Green Field

Nobody has integrated: **structured codebase model + cross-session intent tracking + proactive drift detection + unprompted cliché recognition**. The primitives exist (Cognee for graphs, Letta for sleep-time compute, Shadow-Frog for discovery loops, Cursor Automations for event triggers) but no one has assembled them into a single Apprentice-grade product.

## Strongest reference implementations to study

1. **Shadow-Frog** (Microsoft Research) — closest overall
2. **Cognee** — structured graph
3. **Claude Code's Auto Dream** — sleep-time consolidation
4. **Cursor Bugbot** — event-triggered proactivity
5. **ToM-SWE** — plan/intent tracking
