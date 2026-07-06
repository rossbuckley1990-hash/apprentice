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
    """Canonical string form of an AST, ignoring docstrings and normalizing
    local variable names. Used for body_hash (duplication detection).

    IMPORTANT: deep-copies the node before transforming, so the original AST
    is not mutated. (Previous version mutated in place, corrupting the
    ast_summary and call graph computed downstream.)
    """
    import copy

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ast.dump(node)

    # Deep-copy so we don't mutate the original tree
    node_copy = copy.deepcopy(node)

    # Strip ONLY the leading docstring (first stmt if it's a bare string constant)
    body = list(node_copy.body)
    if (body and isinstance(body[0], ast.Expr) and
            isinstance(body[0].value, ast.Constant) and
            isinstance(body[0].value.value, str)):
        body = body[1:]

    # Normalize local variable names: any Name that is loaded/stored but is NOT
    # a function argument and is not a global/builtin gets renamed to _local_N.
    # We collect arg names first so we don't rename them.
    arg_names: set = set()
    for arg in node_copy.args.args:
        arg_names.add(arg.arg)
    if node_copy.args.vararg:
        arg_names.add(node_copy.args.vararg.arg)
    if node_copy.args.kwarg:
        arg_names.add(node_copy.args.kwarg.arg)

    # Walk and normalize. We rename ALL Name nodes to _local (losing the
    # distinction between different locals) — this is intentionally aggressive
    # for duplication detection: two functions with the same structure but
    # different variable names should hash the same.
    class NameNormalizer(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in arg_names:
                return n  # keep arg names
            if n.id.startswith("__") and n.id.endswith("__"):
                return n  # keep dunder names (language primitives)
            return ast.copy_location(ast.Name(id="_local", ctx=n.ctx), n)

    body = [NameNormalizer().visit(stmt) for stmt in body]
    return ast.dump(ast.Module(body=body, type_ignores=[]))


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
    """Collect all call targets from the function body, structured for the
    call graph. Returns a list of call descriptors:

      - 'foo'           : a direct call to a name 'foo' (resolved by name lookup)
      - '.method'       : a method call on some object (resolved by class context)
      - 'Class.method'  : a qualified call (resolved by class lookup)
      - 'module.func'   : a dotted call (best-effort, often unresolvable)

    The call graph builder (_build_call_graph) resolves these to qualified names.
    """
    calls: List[str] = []
    seen = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            # Direct call: foo()
            name = func.id
            if name not in seen:
                seen.add(name)
                calls.append(name)
        elif isinstance(func, ast.Attribute):
            # Method or dotted call: obj.method() or module.func()
            attr = func.attr
            if isinstance(func.value, ast.Name):
                # obj.method() — record as '.method' for within-class resolution
                # Also record the receiver variable name for type inference
                receiver = func.value.id
                call_desc = f".{attr}"
                if call_desc not in seen:
                    seen.add(call_desc)
                    calls.append(call_desc)
                # If the receiver is 'self', this is a method call within the class
                if receiver == "self":
                    self_call = f"self.{attr}"
                    if self_call not in seen:
                        seen.add(self_call)
                        calls.append(self_call)
            elif isinstance(func.value, ast.Attribute):
                # module.Class.method() — record as 'Class.method'
                if isinstance(func.value.value, ast.Name):
                    qualified = f"{func.value.value.id}.{func.value.attr}.{attr}"
                    if qualified not in seen:
                        seen.add(qualified)
                        calls.append(qualified)
                else:
                    call_desc = f".{attr}"
                    if call_desc not in seen:
                        seen.add(call_desc)
                        calls.append(call_desc)
            else:
                # Complex receiver: (expr).method()
                call_desc = f".{attr}"
                if call_desc not in seen:
                    seen.add(call_desc)
                    calls.append(call_desc)
    return calls


def _build_call_graph(functions: List[Function]) -> Dict[str, List[str]]:
    """Build the caller map from structured call data.

    Uses the `calls` field on each Function (populated from the AST by
    _collect_calls). Resolves:
      - Direct calls ('foo') by name → qualified name lookup
      - self.method calls by class context → Class.method
      - .method calls by best-effort class context (if the function is a method,
        try resolving .method against sibling methods in the same class)

    Returns: qualified_name -> list of qualified_names that call it.
    """
    # Build name -> qualified_name index (handles direct calls)
    name_to_qualified: Dict[str, List[str]] = {}
    # Build method_name -> list of qualified names (handles method calls)
    # e.g. 'method' -> ['module.Class1.method', 'module.Class2.method']
    method_to_qualified: Dict[str, List[str]] = {}
    # Build class_name -> set of method qualified_names (for self.method resolution)
    class_methods: Dict[str, List[str]] = {}

    for fn in functions:
        name_to_qualified.setdefault(fn.name, []).append(fn.qualified_name)
        # If this function is a method (qualified_name has 3+ parts), index it
        parts = fn.qualified_name.rsplit(".", 2)
        if len(parts) >= 2:
            # Could be module.Class.method or Class.method
            method_name = fn.name
            method_to_qualified.setdefault(method_name, []).append(fn.qualified_name)
            if len(parts) >= 3:
                class_name = parts[-2]
                class_methods.setdefault(class_name, []).append(fn.qualified_name)

    callers: Dict[str, List[str]] = {fn.qualified_name: [] for fn in functions}

    for fn in functions:
        # Determine the class context of this function (if it's a method)
        fn_parts = fn.qualified_name.rsplit(".", 2)
        fn_class = None
        if len(fn_parts) >= 3:
            fn_class = fn_parts[-2]

        for call_desc in fn.calls:
            target_qualified = None

            if call_desc.startswith("self."):
                # self.method() — resolve within the same class
                method_name = call_desc[5:]
                if fn_class and fn_class in class_methods:
                    for qn in class_methods[fn_class]:
                        if qn.rsplit(".", 1)[-1] == method_name:
                            target_qualified = qn
                            break

            elif call_desc.startswith("."):
                # .method() — try within-class first, then global
                method_name = call_desc[1:]
                if fn_class and fn_class in class_methods:
                    for qn in class_methods[fn_class]:
                        if qn.rsplit(".", 1)[-1] == method_name:
                            target_qualified = qn
                            break
                if target_qualified is None:
                    # Fall back to global method lookup
                    candidates = method_to_qualified.get(method_name, [])
                    if len(candidates) == 1:
                        target_qualified = candidates[0]
                    # If multiple, we can't resolve without type info — skip

            elif "." in call_desc:
                # Class.method or module.func — try direct qualified lookup
                if call_desc in name_to_qualified:
                    target_qualified = call_desc
                else:
                    # Try suffix match: 'Class.method' might match 'module.Class.method'
                    for qn_list in [name_to_qualified.get(call_desc.split(".")[-1], [])]:
                        for qn in qn_list:
                            if qn.endswith(call_desc):
                                target_qualified = qn
                                break

            else:
                # Direct call: foo()
                candidates = name_to_qualified.get(call_desc, [])
                if len(candidates) == 1:
                    target_qualified = candidates[0]
                elif len(candidates) > 1:
                    # Ambiguous — skip (can't resolve without module context)
                    pass

            if target_qualified and target_qualified != fn.qualified_name:
                callers.setdefault(target_qualified, []).append(fn.qualified_name)

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


class PythonParser:
    """Python AST-based parser. Implements the LanguageParser interface."""
    from .base import LanguageParser as _Base

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def file_extensions(self):
        return [".py"]

    def parse_file(self, file_path: str, content: str, root: str):
        return index_python_file(file_path, content, root)

    def should_ignore(self, rel_path: str) -> bool:
        return False


def discover_python_files(root: str) -> List[str]:
    """Deprecated: use discover_all_files for multi-language support."""
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


def discover_all_files(root: str, config=None) -> List[str]:
    """Discover all supported source files using the language registry."""
    from .registry import supported_extensions

    if config is None:
        ignore_dirs = IGNORE_DIRS
        exts = set(supported_extensions())
    else:
        ignore_dirs = set(config.ignore_dirs)
        exts = set(config.file_extensions) & set(supported_extensions())

    files = []
    root_path = Path(root)
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        try:
            parts = set(path.relative_to(root_path).parts)
        except ValueError:
            continue
        if parts & ignore_dirs:
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
            calls = _collect_calls(node)

            end_line = getattr(node, "end_lineno", node.lineno)

            functions.append(Function(
                name=node.name, qualified_name=qname,
                file_path=rel_path,
                start_line=node.lineno, end_line=end_line,
                arg_names=arg_names,
                signature_hash=sig_hash, body_hash=body_h,
                ast_summary=summary, complexity=cx,
                docstring=docstring,
                calls=calls,
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


def index_repo(root: str, store: Store, verbose: bool = True, config=None) -> Dict[str, int]:
    """Index all supported files in the repo. Updates the store incrementally.
    Uses the language registry to find the right parser for each file.

    Stale-row cleanup: when a file changes, its old function/class rows are
    deleted before inserting new ones. Files no longer on disk are removed
    entirely (including their embeddings).
    """
    from .registry import get_parser_for_file, supported_extensions

    # Discover all supported files currently on disk
    files_on_disk = set(discover_all_files(root, config))

    # Clean up files that are in the store but no longer on disk
    stored_files = store.all_files()
    n_deleted = 0
    for f in stored_files:
        if f.path not in files_on_disk:
            store.delete_file(f.path)
            n_deleted += 1
    if verbose and n_deleted > 0:
        print(f"  removed {n_deleted} deleted file(s) from the model")

    n_files = 0
    n_functions = 0
    n_classes = 0
    n_changed = 0

    all_functions: List[Function] = []

    for rel_path in sorted(files_on_disk):
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

        # File changed (or is new) — delete stale rows before inserting
        if existing:
            store.delete_functions_in_file(rel_path)
            store.delete_classes_in_file(rel_path)

        # Find the right parser
        parser = get_parser_for_file(rel_path)
        if parser is None:
            # Unknown language — record as File only
            line_count = content.count("\n") + 1
            file_entity = File(
                path=rel_path, language="unknown", content_hash=new_hash,
                line_count=line_count, function_names=[], class_names=[],
            )
            store.upsert_file(file_entity)
            n_files += 1
            continue

        file_entity, functions, classes = parser.parse_file(abs_path, content, root)
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
            print(f"  indexed {n_files}/{len(files_on_disk)} files...")

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

    # Snapshot all functions for historical tracking (v0.2.0)
    try:
        store.snapshot_all_functions()
    except Exception:
        pass  # history tables might not exist yet

    return {
        "files": n_files,
        "functions": n_functions,
        "classes": n_classes,
        "changed": n_changed,
        "dead_functions": n_dead,
        "cliches": n_cliche_groups,
    }
