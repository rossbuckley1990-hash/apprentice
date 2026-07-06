"""
Python AST-based indexer.

Parses each .py file into Function and Class entities, computes:
  - signature_hash: identifies clichés (functions with same name + arg shape)
  - body_hash: identifies duplication (AST-normalized body hash)
  - ast_summary: one-line human-readable description
  - complexity: rough cyclomatic complexity
  - callers: best-effort call graph (resolved at index time, updated incrementally)

Multi-language support: the indexer has a pluggable language registry.
Python is implemented; the interface for adding JS/TS/Rust/etc. is in
`LanguageParser`. The Apprentice degrades gracefully on unknown languages
(indexes the File only, with a `language='unknown'` marker).
"""

from __future__ import annotations
import ast
import hashlib
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any

from ..model.entities import File, Function, Class, hash_content
from ..model.store import Store


# =============================================================================
# AST utilities
# =============================================================================

def _norm_ast(node: ast.AST) -> str:
    """Canonical string form of an AST, ignoring docstrings, comments, names of
    private locals. Used for body_hash (duplication detection)."""
    # Strip docstrings (Expr nodes whose value is a constant string at the start)
    class DocStripper(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return node
        def visit_Name(self, node):
            # Normalize non-public names to a placeholder so two functions
            # with the same structure but different local variable names
            # hash to the same body.
            if node.id.startswith("_"):
                return ast.copy_location(
                    ast.Name(id="_local", ctx=node.ctx), node
                )
            return node
        def visit_arg(self, node):
            # Keep arg names — they're part of the signature, not the body
            return node

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Take only the body, strip docstring, normalize
        body = [DocStripper().visit(stmt) for stmt in node.body]
        body = [b for b in body if b is not None]
        # Use ast.dump on the body to get a canonical form
        return ast.dump(ast.Module(body=body, type_ignores=[]))
    return ast.dump(node)


def _signature_hash(name: str, arg_names: List[str]) -> str:
    """Hash of (name, arg_names) — identifies functions that share a 'shape'."""
    return hash_content(f"{name}({','.join(arg_names)})")


def _body_hash(func_node: ast.FunctionDef) -> str:
    """Hash of normalized AST body — identifies duplication."""
    return hash_content(_norm_ast(func_node))


def _ast_summary(func_node: ast.FunctionDef) -> str:
    """One-line human-readable summary: what does this function do?
    e.g. 'calls foo, bar; returns x; raises ValueError'
    """
    calls: List[str] = []
    returns = 0
    raises: List[str] = []
    ifs = 0
    loops = 0

    class Walker(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(f"*.{node.func.attr}")
            self.generic_visit(node)
        def visit_Return(self, node):
            nonlocal returns
            returns += 1
            self.generic_visit(node)
        def visit_Raise(self, node):
            if node.exc and isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name):
                    raises.append(node.exc.func.id)
            self.generic_visit(node)
        def visit_If(self, node):
            nonlocal ifs
            ifs += 1
            self.generic_visit(node)
        def visit_For(self, node):
            nonlocal loops
            loops += 1
            self.generic_visit(node)
        def visit_While(self, node):
            nonlocal loops
            loops += 1
            self.generic_visit(node)

    Walker().visit(func_node)

    parts = []
    if calls:
        unique = list(dict.fromkeys(calls))  # preserve order, dedupe
        parts.append(f"calls {', '.join(unique[:5])}")
    if returns:
        parts.append(f"returns ×{returns}")
    if raises:
        parts.append(f"raises {', '.join(set(raises))}")
    if ifs:
        parts.append(f"ifs ×{ifs}")
    if loops:
        parts.append(f"loops ×{loops}")
    return "; ".join(parts) if parts else "no-op"


def _complexity(func_node: ast.FunctionDef) -> int:
    """Rough cyclomatic complexity: 1 + count of branching constructs."""
    complexity = 1
    for node in ast.walk(func_node):
        if isinstance(node, (ast.If, ast.IfExp)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, (ast.For, ast.While)):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            complexity += 1
        elif isinstance(node, ast.Assert):
            complexity += 1
    return complexity


# =============================================================================
# Call graph builder
# =============================================================================

def _collect_calls(func_node: ast.FunctionDef) -> List[str]:
    """Names of functions called inside this function (best-effort, unresolved)."""
    called_names: List[str] = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                # method call — we can't fully resolve without type info
                # but we record the attribute name as a hint
                called_names.append(f".{node.func.attr}")
    return list(dict.fromkeys(called_names))


def _build_call_graph(functions: List[Function]) -> Dict[str, List[str]]:
    """Given a list of Function entities (with .ast_summary containing 'calls X'),
    resolve callers for each.

    Returns: qualified_name -> list of qualified_names that call it.
    """
    # Build name -> qualified_name index (handles simple cases)
    name_to_qualified: Dict[str, List[str]] = {}
    for fn in functions:
        name_to_qualified.setdefault(fn.name, []).append(fn.qualified_name)

    callers: Dict[str, List[str]] = {fn.qualified_name: [] for fn in functions}

    for fn in functions:
        # Parse the ast_summary for "calls X, Y"
        m = re.search(r"calls ([^;]+)", fn.ast_summary)
        if not m:
            continue
        called_str = m.group(1)
        for called_name in [c.strip() for c in called_str.split(",")]:
            # Direct function name
            if called_name in name_to_qualified:
                for target_qn in name_to_qualified[called_name]:
                    if target_qn != fn.qualified_name:
                        callers.setdefault(target_qn, []).append(fn.qualified_name)
            # Method call like "*.foo" or ".foo" — skip for now (no type info)

    # Dedupe
    for k in callers:
        callers[k] = list(dict.fromkeys(callers[k]))
    return callers


# =============================================================================
# Main indexer
# =============================================================================

PY_FILE_EXTS = {".py"}
IGNORE_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", ".venv", "venv", "env",
    "node_modules", ".mypy_cache", ".pytest_cache", ".tox", "build",
    "dist", ".eggs", ".apprentice",  # don't index our own state
}
IGNORE_FILE_PATTERNS = (".pyc", ".pyo")


def discover_python_files(root: str) -> List[str]:
    files = []
    root_path = Path(root)
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in PY_FILE_EXTS:
            continue
        # Skip ignored dirs
        parts = set(path.relative_to(root_path).parts)
        if parts & IGNORE_DIRS:
            continue
        if any(str(path).endswith(pat) for pat in IGNORE_FILE_PATTERNS):
            continue
        files.append(str(path.relative_to(root_path)))
    return sorted(files)


def index_python_file(file_path: str, content: str, root: str) -> Tuple[File, List[Function], List[Class]]:
    """Parse a Python file into entities."""
    rel_path = os.path.relpath(file_path, root)
    content_hash = hash_content(content)
    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

    function_names: List[str] = []
    class_names: List[str] = []
    functions: List[Function] = []
    classes: List[Class] = []

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        # Unparseable file — record it as a File with no entities
        return File(
            path=rel_path, language="python", content_hash=content_hash,
            line_count=line_count, function_names=[], class_names=[],
        ), [], []

    # Build module qualified-name prefix
    # rel_path is like "mymod.py" or "pkg/sub/mod.py"
    # We want "mymod" or "pkg.sub.mod"
    rel_norm = rel_path.replace(os.sep, ".")
    if rel_norm.endswith(".py"):
        module_qname = rel_norm[:-3]
    else:
        module_qname = rel_norm

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.parent_stack: List[str] = []

        def _qualified(self, name: str) -> str:
            if self.parent_stack:
                return ".".join(self.parent_stack + [name])
            # Use module-qualified name
            return f"{module_qname}.{name}" if module_qname else name

        def visit_ClassDef(self, node: ast.ClassDef):
            qname = self._qualified(node.name)
            class_names.append(node.name)

            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(f"{ast.dump(b)}")

            method_names = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_names.append(item.name)

            classes.append(Class(
                name=node.name, qualified_name=qname,
                file_path=rel_path,
                start_line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno),
                bases=bases, method_names=method_names,
            ))

            self.parent_stack.append(node.name)
            self.generic_visit(node)
            self.parent_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._handle_func(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._handle_func(node)

        def _handle_func(self, node):
            # Skip if nested inside another function (we only track top-level + methods)
            # Actually, let's track them all but with qualified names
            qname = self._qualified(node.name)
            function_names.append(node.name)

            arg_names = [a.arg for a in node.args.args]
            if node.args.vararg:
                arg_names.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                arg_names.append(f"**{node.args.kwarg.arg}")

            docstring = ast.get_docstring(node)
            sig_hash = _signature_hash(node.name, arg_names)
            body_h = _body_hash(node)
            summary = _ast_summary(node)
            cx = _complexity(node)

            end_line = getattr(node, "end_lineno", node.lineno)

            functions.append(Function(
                name=node.name, qualified_name=qname,
                file_path=rel_path,
                start_line=node.lineno, end_line=end_line,
                arg_names=arg_names,
                signature_hash=sig_hash, body_hash=body_h,
                ast_summary=summary, complexity=cx,
                docstring=docstring,
            ))

            # Don't recurse into function bodies for nested defs to keep it simple
            # (we'd need a more sophisticated qualified-name scheme)

    Visitor().visit(tree)

    file_entity = File(
        path=rel_path, language="python", content_hash=content_hash,
        line_count=line_count,
        function_names=function_names, class_names=class_names,
    )
    return file_entity, functions, classes


def index_repo(root: str, store: Store, verbose: bool = True) -> Dict[str, int]:
    """Index all Python files in the repo. Updates the store incrementally."""
    files = discover_python_files(root)
    n_files = 0
    n_functions = 0
    n_classes = 0
    n_changed = 0

    all_functions: List[Function] = []

    for rel_path in files:
        abs_path = os.path.join(root, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue

        existing = store.get_file(rel_path)
        new_hash = hash_content(content)
        if existing and existing.content_hash == new_hash:
            # Unchanged — still pull its functions for the call-graph pass
            all_functions.extend(store.functions_in_file(rel_path))
            n_files += 1
            continue

        file_entity, functions, classes = index_python_file(abs_path, content, root)
        store.upsert_file(file_entity)
        for fn in functions:
            store.upsert_function(fn)
            all_functions.append(fn)
            n_functions += 1
        for cls in classes:
            store.upsert_class(cls)
            n_classes += 1

        n_files += 1
        n_changed += 1

        if verbose and n_changed % 20 == 0:
            print(f"  indexed {n_files}/{len(files)} files...")

    # Build call graph and update callers
    if verbose:
        print(f"  building call graph over {len(all_functions)} functions...")
    callers_map = _build_call_graph(all_functions)
    n_dead = 0
    for fn in all_functions:
        callers = callers_map.get(fn.qualified_name, [])
        fn.callers = callers
        # A function is "dead" if it has no callers AND it's not a dunder method
        # AND it's not a module entry point (main, etc.)
        is_entry = fn.name in ("main", "__main__") or fn.name.startswith("test_")
        is_dunder = fn.name.startswith("__") and fn.name.endswith("__")
        fn.is_dead = (not callers) and (not is_entry) and (not is_dunder)
        if fn.is_dead:
            n_dead += 1
        store.upsert_function(fn)

    # Detect clichés (duplication)
    # Two kinds:
    #   1. Same signature_hash AND same body_hash → exact same function written twice
    #   2. Same body_hash (different signature) → same logic, different name (the common case)
    if verbose:
        print("  detecting clichés (duplication)...")
    body_groups: Dict[str, List[str]] = {}
    for fn in all_functions:
        body_groups.setdefault(fn.body_hash, []).append(fn.qualified_name)
    for body_hash, instances in body_groups.items():
        if len(instances) >= 2:
            from ..model.entities import Cliche
            # Find the signature_hash of the first instance (for grouping)
            first_fn = next(f for f in all_functions if f.qualified_name == instances[0])
            sig_hash = first_fn.signature_hash
            store.upsert_cliche(Cliche(
                signature_hash=sig_hash, body_hash=body_hash,
                instances=instances,
                note=f"{len(instances)} instances of the same function body",
            ))

    # Count cliché groups for stats
    n_cliche_groups = sum(1 for insts in body_groups.values() if len(insts) >= 2)

    return {
        "files": n_files,
        "functions": n_functions,
        "classes": n_classes,
        "changed": n_changed,
        "dead_functions": n_dead,
        "cliches": n_cliche_groups,
    }
