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
    local variable AND parameter names. Used for body_hash (duplication detection).

    IMPORTANT: deep-copies the node before transforming, so the original AST
    is not mutated. (Previous version mutated in place, corrupting the
    ast_summary and call graph computed downstream.)

    Parameters are normalized positionally (_arg0, _arg1, ...) so that two
    functions with identical logic but different parameter names (e.g.
    `def f(x): return x+1` and `def g(y): return y+1`) hash the same.
    This catches the most common form of duplication.
    """
    import copy

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ast.dump(node)

    # Deep-copy so we don't mutate the original tree
    node_copy = copy.deepcopy(node)

    # Strip ONLY the leading docstring
    body = list(node_copy.body)
    if (body and isinstance(body[0], ast.Expr) and
            isinstance(body[0].value, ast.Constant) and
            isinstance(body[0].value.value, str)):
        body = body[1:]

    # Build a mapping from arg names to positional placeholders (_arg0, _arg1, ...)
    arg_names: set = set()
    arg_name_map: Dict[str, str] = {}
    for i, arg in enumerate(node_copy.args.args):
        arg_names.add(arg.arg)
        arg_name_map[arg.arg] = f"_arg{i}"
    if node_copy.args.vararg:
        arg_names.add(node_copy.args.vararg.arg)
        arg_name_map[node_copy.args.vararg.arg] = "_vararg"
    if node_copy.args.kwarg:
        arg_names.add(node_copy.args.kwarg.arg)
        arg_name_map[node_copy.args.kwarg.arg] = "_kwarg"

    # Walk and normalize. Rename ALL Name nodes:
    # - args → _arg0, _arg1, ... (positional)
    # - other locals → _local (anonymous)
    class NameNormalizer(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in arg_name_map:
                new_id = arg_name_map[n.id]
            elif n.id.startswith("__") and n.id.endswith("__"):
                return n  # keep dunder names (language primitives)
            else:
                new_id = "_local"
            return ast.copy_location(ast.Name(id=new_id, ctx=n.ctx), n)

        # Also rename arg nodes themselves so the hash is consistent
        def visit_arg(self, n):
            if n.arg in arg_name_map:
                return ast.copy_location(
                    ast.arg(arg=arg_name_map[n.arg], annotation=n.annotation), n
                )
            return n

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
    """Collect all call targets AND value references from the function body.

    Call targets are used for call-graph edge resolution.
    Value references (Name nodes in Load context, Attribute nodes in Load context)
    are used for liveness: a function referenced as a value (e.g. in a list of
    analyzers, or passed as a callback) is NOT dead even if it's never called
    directly.

    Returns a list of descriptors:
      - 'call:foo'       : a direct call to name 'foo'
      - 'call:.method'   : a method call (resolved by class context)
      - 'call:self.method': a self.method call (within-class)
      - 'call:Class.method': a qualified call
      - 'ref:foo'        : a value reference to name 'foo' (liveness signal)
      - 'ref:.method'    : a value reference to an attribute (e.g. obj.method used as value)
    """
    calls: List[str] = []
    seen_calls = set()
    seen_refs = set()

    for node in ast.walk(func_node):
        # Collect call targets
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
                key = f"call:{name}"
                if key not in seen_calls:
                    seen_calls.add(key)
                    calls.append(key)
            elif isinstance(func, ast.Attribute):
                attr = func.attr
                if isinstance(func.value, ast.Name):
                    receiver = func.value.id
                    call_desc = f"call:.{attr}"
                    if call_desc not in seen_calls:
                        seen_calls.add(call_desc)
                        calls.append(call_desc)
                    if receiver == "self":
                        self_call = f"call:self.{attr}"
                        if self_call not in seen_calls:
                            seen_calls.add(self_call)
                            calls.append(self_call)
                elif isinstance(func.value, ast.Attribute):
                    if isinstance(func.value.value, ast.Name):
                        qualified = f"call:{func.value.value.id}.{func.value.attr}.{attr}"
                        if qualified not in seen_calls:
                            seen_calls.add(qualified)
                            calls.append(qualified)
                else:
                    call_desc = f"call:.{attr}"
                    if call_desc not in seen_calls:
                        seen_calls.add(call_desc)
                        calls.append(call_desc)

        # Collect value references (Name in Load context) for liveness
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            key = f"ref:{name}"
            if key not in seen_refs:
                seen_refs.add(key)
                calls.append(key)

        # Also collect attribute references (obj.attr used as value, not called)
        # e.g. set_defaults(func=cmd_init) — cmd_init is an Attribute.value
        # in Load context, not a call.
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            # Only collect if the attribute itself is used as a value
            # (not when it's part of a call, which is handled above)
            if not isinstance(node, ast.Call):  # this check isn't quite right
                attr = node.attr
                key = f"ref:.{attr}"
                if key not in seen_refs:
                    seen_refs.add(key)
                    calls.append(key)

    return calls


def _build_call_graph(functions: List[Function]) -> Dict[str, List[str]]:
    """Build the caller map from structured call data.

    Uses the `calls` field on each Function. Resolves:
      - Direct calls ('call:foo') by name → qualified name lookup
      - self.method calls by class context → Class.method
      - .method calls by best-effort class context
      - Value references ('ref:foo') for liveness — a function referenced
        as a value is NOT dead even if never called directly

    Returns: qualified_name -> list of qualified_names that call OR reference it.
    """
    # Build name -> qualified_name index (handles direct calls + refs)
    name_to_qualified: Dict[str, List[str]] = {}
    # Build method_name -> list of qualified names (handles method calls)
    method_to_qualified: Dict[str, List[str]] = {}
    # Build class_name -> set of method qualified_names (for self.method resolution)
    # Key is the FULL class qualified name (module.Class), not just Class
    class_methods: Dict[str, List[str]] = {}
    # Map short class name -> list of full class qualified names
    # (for resolving self.method when we only know the short name)
    short_to_full_class: Dict[str, List[str]] = {}

    for fn in functions:
        name_to_qualified.setdefault(fn.name, []).append(fn.qualified_name)

        # Determine if this is a method by checking if it has a class context
        # Methods now have qualified names like module.Class.method (3+ parts)
        parts = fn.qualified_name.split(".")
        if len(parts) >= 3:
            # module.Class.method
            method_name = fn.name
            method_to_qualified.setdefault(method_name, []).append(fn.qualified_name)
            # Full class qualified name is everything except the method
            class_full_qname = ".".join(parts[:-1])
            class_methods.setdefault(class_full_qname, []).append(fn.qualified_name)
            # Also index by short class name
            short_class = parts[-2]
            short_to_full_class.setdefault(short_class, []).append(class_full_qname)

    callers: Dict[str, List[str]] = {fn.qualified_name: [] for fn in functions}

    for fn in functions:
        # Determine the class context of this function (if it's a method)
        fn_parts = fn.qualified_name.split(".")
        fn_class_full = None
        fn_class_short = None
        if len(fn_parts) >= 3:
            fn_class_full = ".".join(fn_parts[:-1])  # module.Class
            fn_class_short = fn_parts[-2]  # Class

        for call_desc in fn.calls:
            target_qualified = None

            if call_desc.startswith("call:self."):
                # self.method() — resolve within the same class
                method_name = call_desc[len("call:self."):]
                if fn_class_full and fn_class_full in class_methods:
                    for qn in class_methods[fn_class_full]:
                        if qn.rsplit(".", 1)[-1] == method_name:
                            target_qualified = qn
                            break

            elif call_desc.startswith("call:."):
                # .method() — try within-class first, then global unique lookup
                method_name = call_desc[len("call:."):]
                if fn_class_full and fn_class_full in class_methods:
                    for qn in class_methods[fn_class_full]:
                        if qn.rsplit(".", 1)[-1] == method_name:
                            target_qualified = qn
                            break
                if target_qualified is None:
                    # Fall back to global method lookup
                    candidates = method_to_qualified.get(method_name, [])
                    if len(candidates) == 1:
                        target_qualified = candidates[0]
                    # If multiple, we can't resolve without type info — skip

            elif call_desc.startswith("call:") and "." in call_desc:
                # Class.method or module.func — try direct qualified lookup
                name_part = call_desc[len("call:"):]
                if name_part in name_to_qualified:
                    target_qualified = name_part
                else:
                    # Try suffix match
                    for qn in name_to_qualified.get(name_part.split(".")[-1], []):
                        if qn.endswith(name_part):
                            target_qualified = qn
                            break

            elif call_desc.startswith("call:"):
                # Direct call: foo()
                name_part = call_desc[len("call:"):]
                candidates = name_to_qualified.get(name_part, [])
                if len(candidates) == 1:
                    target_qualified = candidates[0]
                elif len(candidates) > 1:
                    # Ambiguous — skip
                    pass

            elif call_desc.startswith("ref:"):
                # Value reference — for liveness only.
                # A function referenced as a value (e.g. in a list, or as a
                # callback argument) is NOT dead. We add a "reference" edge
                # which counts as being alive.
                name_part = call_desc[len("ref:"):]
                if name_part.startswith("."):
                    # Attribute reference (obj.method as value)
                    method_name = name_part[1:]
                    candidates = method_to_qualified.get(method_name, [])
                    if len(candidates) == 1:
                        target_qualified = candidates[0]
                else:
                    # Name reference (foo as value)
                    candidates = name_to_qualified.get(name_part, [])
                    if len(candidates) == 1:
                        target_qualified = candidates[0]
                    elif len(candidates) > 1:
                        # For references, we can be more lenient — mark ALL
                        # candidates as "referenced" since we don't know which
                        # one is meant. This avoids false dead-code for
                        # overloaded names.
                        for c in candidates:
                            if c != fn.qualified_name:
                                callers.setdefault(c, []).append(fn.qualified_name)
                        continue

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


def _collect_module_refs(tree: ast.Module) -> List[str]:
    """Collect all Name/Attribute references at module scope (outside function/class bodies).
    These are used for liveness: a function referenced at module scope (e.g. in a list
    like ANALYZERS = [(..., analyze_dead_code), ...], or as set_defaults(func=cmd_init))
    is NOT dead even if never called directly.
    """
    refs: List[str] = []
    seen = set()
    for node in ast.iter_child_nodes(tree):
        # Skip function and class definitions — their bodies are handled by _collect_calls
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                key = f"ref:{sub.id}"
                if key not in seen:
                    seen.add(key)
                    refs.append(key)
            elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                key = f"ref:.{sub.attr}"
                if key not in seen:
                    seen.add(key)
                    refs.append(key)
    return refs


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
            """Build a fully-qualified name including the module prefix.
            e.g. for a method 'foo' in class 'Bar' in module 'pkg.mod':
              returns 'pkg.mod.Bar.foo'
            This ensures methods from different modules never collide
            and the call graph can resolve self.method() calls reliably."""
            parts = []
            if module_qname:
                parts.append(module_qname)
            parts.extend(self.parent_stack)
            parts.append(name)
            return ".".join(parts)

        def _class_context(self) -> Optional[str]:
            """Return the current class name (for self.method resolution),
            or None if we're at module level."""
            if self.parent_stack:
                return self.parent_stack[-1]
            return None

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

    # Collect module-level value references for liveness.
    # Functions referenced at module scope (in lists like ANALYZERS, or as
    # arguments to set_defaults(func=...)) are NOT dead. We collect these
    # and add them to every function in this file's calls list. This is
    # slightly imprecise (it adds edges from every function in the file to
    # the referenced names) but correct for liveness: the referenced function
    # IS alive, regardless of which specific function "calls" it.
    module_refs = _collect_module_refs(tree)
    if module_refs:
        for fn in functions:
            fn.calls.extend(module_refs)

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
