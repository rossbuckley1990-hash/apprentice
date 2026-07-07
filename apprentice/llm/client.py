"""
Pluggable LLM client for the Apprentice.

The 1988 spec called for a "synthesis engine" that could generate routine code,
explain in terms of intent, and recognize clichés. That's the LLM layer.

Backends (auto-detected, configurable):
  - 'openai':       uses OPENAI_API_KEY (or any OpenAI-compatible endpoint)
  - 'anthropic':    uses ANTHROPIC_API_KEY
  - 'zai':          uses ZAI_API_KEY (Z.ai GLM models)
  - 'mock':         deterministic offline responses (for tests / dev)

The client is ALWAYS optional. The Apprentice's core (persistence + proactivity)
works without any LLM. The LLM adds: natural-language ask, fix synthesis,
function summarization.
"""

from __future__ import annotations
import os
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    backend: str
    usage: Dict[str, int]  # {"prompt_tokens": N, "completion_tokens": M}
    ok: bool = True          # False if the call failed (error message in `text`)
    error: Optional[str] = None  # the exception message if ok=False


class LLMClient:
    """Pluggable LLM client. Picks the best available backend."""

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None):
        self.backend = backend or self._auto_backend()
        self.model = model or self._default_model()
        self._client = None  # lazy

    @staticmethod
    def _auto_backend() -> str:
        # Check for the z-ai CLI first (available in Z.ai environments)
        import shutil
        if shutil.which("z-ai"):
            return "zaicli"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("ZAI_API_KEY"):
            return "zai"
        return "mock"

    def _default_model(self) -> str:
        return {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "zai": "glm-4-flash",
            "zaicli": "glm-4-plus",  # the z-ai CLI uses glm-4-plus by default
            "mock": "mock",
        }.get(self.backend, "mock")

    def is_real(self) -> bool:
        """True if this is a real LLM backend (not mock)."""
        return self.backend != "mock"

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        if self.backend == "mock":
            return self._mock_complete(system, user)
        elif self.backend == "zaicli":
            return self._zaicli_complete(system, user, max_tokens)
        elif self.backend == "openai":
            return self._openai_complete(system, user, max_tokens)
        elif self.backend == "anthropic":
            return self._anthropic_complete(system, user, max_tokens)
        elif self.backend == "zai":
            return self._zai_complete(system, user, max_tokens)
        else:
            return self._mock_complete(system, user)

    def _zaicli_complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        """Use the z-ai CLI (z-ai-web-dev-sdk) to call GLM models.
        This is the backend used in Z.ai environments where the z-ai CLI
        is available but no API key is set as an env var."""
        import subprocess
        import json as _json
        import tempfile

        try:
            # Write the user prompt to a temp file to avoid shell escaping issues
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(user)
                user_file = f.name

            try:
                result = subprocess.run(
                    ["z-ai", "chat", "--prompt", user, "--system", system],
                    capture_output=True, text=True, timeout=120,
                )
            finally:
                import os as _os
                _os.unlink(user_file)

            if result.returncode != 0:
                return LLMResponse(
                    text=f"[LLM error: z-ai CLI returned {result.returncode}: {result.stderr[:200]}]",
                    backend="zaicli",
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                    ok=False,
                    error=f"z-ai CLI error: {result.stderr[:200]}",
                )

            # The z-ai CLI outputs JSON with some status lines before it.
            # Find the JSON object in the output.
            output = result.stdout
            # Strip the status lines (🚀 ...)
            json_start = output.find("{")
            if json_start == -1:
                return LLMResponse(
                    text=f"[LLM error: no JSON in z-ai output]",
                    backend="zaicli",
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                    ok=False,
                    error="no JSON in z-ai output",
                )

            data = _json.loads(output[json_start:])
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return LLMResponse(
                text=text,
                backend="zaicli",
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            )
        except subprocess.TimeoutExpired:
            return LLMResponse(
                text="[LLM error: z-ai CLI timed out after 120s]",
                backend="zaicli",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                ok=False,
                error="z-ai CLI timed out",
            )
        except Exception as e:
            return LLMResponse(
                text=f"[LLM error: {type(e).__name__}: {e}]",
                backend="zaicli",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )

    def _openai_complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        try:
            import openai
            if self._client is None:
                self._client = openai.OpenAI()
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return LLMResponse(
                text=resp.choices[0].message.content,
                backend="openai",
                usage={
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                },
            )
        except Exception as e:
            return LLMResponse(
                text=f"[LLM error: {type(e).__name__}: {e}]",
                ok=False,
                error=f"{type(e).__name__}: {e}",
                backend="openai",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )

    def _anthropic_complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        try:
            import anthropic
            if self._client is None:
                self._client = anthropic.Anthropic()
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
            return LLMResponse(
                text=resp.content[0].text,
                backend="anthropic",
                usage={
                    "prompt_tokens": resp.usage.input_tokens,
                    "completion_tokens": resp.usage.output_tokens,
                },
            )
        except Exception as e:
            return LLMResponse(
                text=f"[LLM error: {type(e).__name__}: {e}]",
                ok=False,
                error=f"{type(e).__name__}: {e}",
                backend="anthropic",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )

    def _zai_complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        """Z.ai GLM models via the OpenAI-compatible API."""
        try:
            import openai
            if self._client is None:
                self._client = openai.OpenAI(
                    api_key=os.environ["ZAI_API_KEY"],
                    base_url="https://open.bigmodel.cn/api/paas/v4/",
                )
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return LLMResponse(
                text=resp.choices[0].message.content,
                backend="zai",
                usage={
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                },
            )
        except Exception as e:
            return LLMResponse(
                text=f"[LLM error: {type(e).__name__}: {e}]",
                ok=False,
                error=f"{type(e).__name__}: {e}",
                backend="zai",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )

    def _mock_complete(self, system: str, user: str) -> LLMResponse:
        """Deterministic offline responses for dev/tests."""
        text = self._mock_response(system, user)
        return LLMResponse(text=text, backend="mock", usage={"prompt_tokens": 0, "completion_tokens": 0})

    @staticmethod
    def _mock_response(system: str, user: str) -> str:
        """Heuristic mock: produces a plausible-shaped response without an LLM."""
        if "summarize" in system.lower():
            return (
                "This function implements a core operation. It takes inputs, "
                "processes them through several steps, and returns a result. "
                "Key dependencies: standard library functions. "
                "[Mock LLM — set OPENAI_API_KEY, ANTHROPIC_API_KEY, or ZAI_API_KEY for real responses.]"
            )
        if "fix" in system.lower() or "patch" in system.lower():
            return (
                "# Suggested fix (mock mode — no LLM configured):\n"
                "# Review the observation and apply the recommended change manually.\n"
                "# Set an API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or ZAI_API_KEY)\n"
                "# to get AI-generated patches.\n"
                "# No code changes generated in mock mode.\n"
            )
        if "ask" in system.lower() or "question" in system.lower():
            return (
                "Based on the codebase model, here's what I found:\n"
                "(Mock mode — no LLM configured. Set an API key for natural-language answers.)\n"
            )
        return "[Mock LLM response — configure a real backend for actual output.]"


def get_client(config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """Factory: create an LLM client from config or environment."""
    if config:
        return LLMClient(
            backend=config.get("llm_backend"),
            model=config.get("llm_model"),
        )
    return LLMClient()
