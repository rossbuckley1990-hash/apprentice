"""
Pluggable embedding backend.

The Apprentice needs embeddings to answer "show me functions like this one"
and to find semantically similar code. The backend is pluggable so the MVP
works fully offline, with hooks for production-grade models.

Default backends (no external dependencies):
  - 'asthash':   bag-of-AST-node-types — extremely fast, fully offline
  - 'tfidf':     TF-IDF over tokenized code — better similarity, offline

Production backends (optional, lazy-loaded):
  - 'sentence-transformers': if `sentence_transformers` is installed
  - 'openai':                 if `OPENAI_API_KEY` is set
"""

from __future__ import annotations
import os
import re
import math
import hashlib
from collections import Counter
from typing import List, Dict, Optional, Tuple
import ast

from ..model.entities import Function
from ..model.store import Store


# =============================================================================
# Tokenizer
# =============================================================================

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[+\-*/%=<>!&|^~]+|[.,;:()\[\]{}]")


def tokenize_code(source: str) -> List[str]:
    """Tokenize Python source for TF-IDF."""
    return TOKEN_RE.findall(source)


# =============================================================================
# AST-hash backend
# =============================================================================

def ast_hash_vector(func_source: str, dim: int = 256) -> List[float]:
    """Bag-of-AST-node-types, hashed into a fixed-dim vector.
    Extremely fast, fully offline, decent for structural similarity."""
    try:
        tree = ast.parse(func_source)
    except SyntaxError:
        return [0.0] * dim

    counts: Counter = Counter()
    for node in ast.walk(tree):
        counts[type(node).__name__] += 1

    vec = [0.0] * dim
    for node_type, count in counts.items():
        h = int(hashlib.md5(node_type.encode()).hexdigest(), 16) % dim
        vec[h] += count

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# =============================================================================
# Hashed-TF backend (corpus-aware, offline)
# =============================================================================
# NOTE: This is NOT full TF-IDF — it does not compute inverse document
# frequency. It's a hashed term-frequency vector with log-dampening.
# We keep the backend label "tfidf" for backward compatibility with
# existing databases, but the class name is honest. Real IDF computation
# over the indexed corpus is on the roadmap.

class HashedTFBackend:
    """Hashed term-frequency vectorizer. No IDF (despite the 'tfidf' label
    stored in the DB for backward compat). Uses a hashing trick with a fixed
    salt so the same token maps to the same dimension across runs."""

    def __init__(self, dim: int = 1024, salt: int = 0):
        self.dim = dim
        self.salt = salt

    def vectorize(self, tokens: List[str]) -> List[float]:
        counts: Counter = Counter()
        for tok in tokens:
            counts[tok] += 1

        vec = [0.0] * self.dim
        for tok, count in counts.items():
            h = int(hashlib.md5(f"{tok}|{self.salt}".encode()).hexdigest(), 16) % self.dim
            # log(1+count) dampening
            vec[h] += 1.0 + math.log(count)

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# Backward-compat alias
TFIDFBackend = HashedTFBackend


# =============================================================================
# Cosine similarity
# =============================================================================

def cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# =============================================================================
# Manager
# =============================================================================

class Embedder:
    """Pluggable embedding manager. Picks the best available backend.

    The TF-IDF (hashed-TF) backend is ALWAYS constructed as the fallback,
    so if a real backend (sentence-transformers, openai) fails, we degrade
    gracefully instead of crashing.
    """

    def __init__(self, backend: Optional[str] = None):
        if backend is None:
            backend = self._auto_backend()
        self.backend = backend
        # Always construct the fallback — it's cheap and ensures we never
        # crash with AttributeError on the fallback path.
        self._tfidf = TFIDFBackend()
        self._st_model = None  # lazy-loaded sentence-transformers model

    @staticmethod
    def _auto_backend() -> str:
        # Prefer sentence-transformers if available
        try:
            import sentence_transformers  # noqa: F401
            return "sentence-transformers"
        except ImportError:
            pass
        # Fall back to TF-IDF (offline)
        return "tfidf"

    def vectorize_function(self, fn: Function, source: str) -> List[float]:
        if self.backend == "asthash":
            return ast_hash_vector(source)
        elif self.backend == "tfidf":
            tokens = tokenize_code(source)
            return self._tfidf.vectorize(tokens)
        elif self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                if self._st_model is None:
                    self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                vec = self._st_model.encode(source, normalize_embeddings=True)
                return vec.tolist()
            except Exception as e:
                # Degrade gracefully — log the real error, fall back to TF-IDF
                import sys
                print(f"  [embedder] sentence-transformers failed ({type(e).__name__}: {e}), "
                      f"falling back to tfidf", file=sys.stderr)
                tokens = tokenize_code(source)
                return self._tfidf.vectorize(tokens)
        elif self.backend == "openai":
            try:
                import openai
                client = openai.OpenAI()
                resp = client.embeddings.create(
                    input=source, model="text-embedding-3-small"
                )
                return resp.data[0].embedding
            except Exception as e:
                # Degrade gracefully — log the real error, fall back to TF-IDF
                import sys
                print(f"  [embedder] openai failed ({type(e).__name__}: {e}), "
                      f"falling back to tfidf", file=sys.stderr)
                tokens = tokenize_code(source)
                return self._tfidf.vectorize(tokens)
        else:
            tokens = tokenize_code(source)
            return self._tfidf.vectorize(tokens)

    def index_all(self, store: Store, root: str, force: bool = False) -> int:
        """(Re)compute embeddings for all functions in the store."""
        n = 0
        for fn in store.all_functions():
            if not force:
                existing = store.get_embedding(fn.qualified_name)
                if existing and existing[1] == self.backend:
                    continue
            # Read source — use errors="replace" and catch UnicodeDecodeError
            # (which OSError doesn't catch)
            try:
                with open(os.path.join(root, fn.file_path), "r",
                          encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            # Extract function source via line range
            lines = content.splitlines()
            start = max(0, fn.start_line - 1)
            end = min(len(lines), fn.end_line)
            source = "\n".join(lines[start:end])
            vec = self.vectorize_function(fn, source)
            store.set_embedding(fn.qualified_name, vec, self.backend)
            n += 1
        return n

    def find_similar(
        self, store: Store, qualified_name: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find the top-k most similar functions to the given one."""
        target = store.get_embedding(qualified_name)
        if target is None:
            return []
        target_vec, _ = target

        results: List[Tuple[str, float]] = []
        for fn in store.all_functions():
            if fn.qualified_name == qualified_name:
                continue
            other = store.get_embedding(fn.qualified_name)
            if other is None:
                continue
            other_vec, _ = other
            sim = cosine(target_vec, other_vec)
            results.append((fn.qualified_name, sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]
