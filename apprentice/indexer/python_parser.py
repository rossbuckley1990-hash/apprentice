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


def _collect_import_aliases(tree: ast.Module) -> Dict[str, str]:
    """Map module-local import aliases back to the imported symbol name."""
    aliases: Dict[str, str] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                aliases[local] = alias.name.split(".")[-1]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = alias.name.split(".")[-1]
    return aliases


def _with_aliases(descriptors: List[str], aliases: Dict[str, str]) -> List[str]:
    """Add descriptors resolved through import aliases, preserving originals."""
    expanded = list(descriptors)
    seen = set(expanded)
    for desc in descriptors:
        if ":" not in desc:
            continue
        prefix, name = desc.split(":", 1)
        if prefix not in {"call", "ref", "module_ref"} or name.startswith("."):
            continue

        target = aliases.get(name)
        if target is None and "." in name:
            head, rest = name.split(".", 1)
            if head in aliases:
                target = f"{aliases[head]}.{rest}"
        if target is None:
            continue

        key = f"{prefix}:{target}"
        if key not in seen:
            seen.add(key)
            expanded.append(key)
    return expanded


def _module_liveness_descriptors(content: str) -> List[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    aliases = _collect_import_aliases(tree)
    return _with_aliases(_collect_module_refs(tree), aliases)


def _build_call_indexes(functions: List[Function]) -> Dict[str, Dict[str, List[str]]]:
    name_to_qualified: Dict[str, List[str]] = {}
    method_to_qualified: Dict[str, List[str]] = {}
    class_methods: Dict[str, List[str]] = {}
    for fn in functions:
        name_to_qualified.setdefault(fn.name, []).append(fn.qualified_name)
        parts = fn.qualified_name.split(".")
        if len(parts) >= 3:
            method_to_qualified.setdefault(fn.name, []).append(fn.qualified_name)
            class_full_qname = ".".join(parts[:-1])
            class_methods.setdefault(class_full_qname, []).append(fn.qualified_name)
    return {
        "name_to_qualified": name_to_qualified,
        "method_to_qualified": method_to_qualified,
        "class_methods": class_methods,
    }


def _function_class_context(fn: Function) -> Optional[str]:
    parts = fn.qualified_name.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:-1])
    return None


def _resolve_name(
    name_part: str, indexes: Dict[str, Dict[str, List[str]]], callers: Dict[str, List[str]]
) -> List[str]:
    if name_part in callers:
        return [name_part]
    if name_part.startswith("."):
        candidates = indexes["method_to_qualified"].get(name_part[1:], [])
        return candidates if len(candidates) == 1 else []
    candidates = indexes["name_to_qualified"].get(name_part, [])
    if candidates:
        return candidates
    if "." not in name_part:
        return []
    return [
        qn for qn in indexes["name_to_qualified"].get(name_part.split(".")[-1], [])
        if qn.endswith(name_part)
    ]


def _resolve_class_method(
    method_name: str, fn_class_full: Optional[str], indexes: Dict[str, Dict[str, List[str]]]
) -> Optional[str]:
    if not fn_class_full:
        return None
    for qn in indexes["class_methods"].get(fn_class_full, []):
        if qn.rsplit(".", 1)[-1] == method_name:
            return qn
    return None


def _resolve_method_call(
    method_name: str, fn_class_full: Optional[str], indexes: Dict[str, Dict[str, List[str]]]
) -> Optional[str]:
    target = _resolve_class_method(method_name, fn_class_full, indexes)
    if target:
        return target
    candidates = indexes["method_to_qualified"].get(method_name, [])
    return candidates[0] if len(candidates) == 1 else None


def _resolve_call_targets(
    call_desc: str,
    fn_class_full: Optional[str],
    indexes: Dict[str, Dict[str, List[str]]],
    callers: Dict[str, List[str]],
) -> List[str]:
    if call_desc.startswith("call:self."):
        target = _resolve_class_method(call_desc[len("call:self."):], fn_class_full, indexes)
        return [target] if target else []
    if call_desc.startswith("call:."):
        target = _resolve_method_call(call_desc[len("call:."):], fn_class_full, indexes)
        return [target] if target else []
    if call_desc.startswith("call:"):
        candidates = _resolve_name(call_desc[len("call:"):], indexes, callers)
        return candidates if len(candidates) == 1 else []
    if call_desc.startswith("ref:"):
        return _resolve_name(call_desc[len("ref:"):], indexes, callers)
    return []


def _add_live_marker(call_desc: str, callers: Dict[str, List[str]]) -> bool:
    if not call_desc.startswith("live:"):
        return False
    parts = call_desc.split(":", 2)
    if len(parts) >= 3 and parts[1] in callers:
        callers[parts[1]].append(f"<{parts[2]}>")
    return True


def _add_module_ref(
    call_desc: str, callers: Dict[str, List[str]], indexes: Dict[str, Dict[str, List[str]]]
) -> bool:
    if not call_desc.startswith("module_ref:"):
        return False
    for target in _resolve_name(call_desc[len("module_ref:"):], indexes, callers):
        if target in callers:
            callers[target].append("<module>")
    return True


def _apply_call_desc(
    fn: Function,
    call_desc: str,
    callers: Dict[str, List[str]],
    indexes: Dict[str, Dict[str, List[str]]],
) -> None:
    if _add_live_marker(call_desc, callers) or _add_module_ref(call_desc, callers, indexes):
        return
    fn_class_full = _function_class_context(fn)
    for target in _resolve_call_targets(call_desc, fn_class_full, indexes, callers):
        if target != fn.qualified_name:
            callers.setdefault(target, []).append(fn.qualified_name)


def _build_call_graph(
    functions: List[Function], external_calls: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """Build the caller map from structured call and liveness descriptors."""
    indexes = _build_call_indexes(functions)
    callers: Dict[str, List[str]] = {fn.qualified_name: [] for fn in functions}
    for fn in functions:
        for call_desc in fn.calls:
            _apply_call_desc(fn, call_desc, callers, indexes)
    for call_desc in external_calls or []:
        _add_module_ref(call_desc, callers, indexes)
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
                module_key = f"module_ref:{sub.id}"
                if module_key not in seen:
                    seen.add(module_key)
                    refs.append(module_key)
            elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                key = f"ref:.{sub.attr}"
                if key not in seen:
                    seen.add(key)
                    refs.append(key)
                module_key = f"module_ref:.{sub.attr}"
                if module_key not in seen:
                    seen.add(module_key)
                    refs.append(module_key)
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
            if node.decorator_list:
                calls.append(f"live:{qname}:decorator")

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
    import_aliases = _collect_import_aliases(tree)
    if import_aliases:
        for fn in functions:
            fn.calls = _with_aliases(fn.calls, import_aliases)

    # Collect module-level value references for liveness.
    # Functions referenced at module scope (in lists like ANALYZERS, or as
    # arguments to set_defaults(func=...)) are NOT dead. We collect these
    # and add them to every function in this file's calls list. This is
    # slightly imprecise (it adds edges from every function in the file to
    # the referenced names) but correct for liveness: the referenced function
    # IS alive, regardless of which specific function "calls" it.
    module_refs = _with_aliases(_collect_module_refs(tree), import_aliases)
    if module_refs:
        for fn in functions:
            fn.calls.extend(module_refs)

    file_entity = File(
        path=rel_path, language="python", content_hash=content_hash,
        line_count=line_count,
        function_names=function_names, class_names=class_names,
    )
    return file_entity, functions, classes


def _read_source_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _remove_deleted_files(store: Store, files_on_disk: Set[str], verbose: bool) -> None:
    n_deleted = 0
    for f in store.all_files():
        if f.path not in files_on_disk:
            store.delete_file(f.path)
            n_deleted += 1
    if verbose and n_deleted > 0:
        print(f"  removed {n_deleted} deleted file(s) from the model")


def _external_calls_for_file(parser: Any, content: str) -> List[str]:
    if parser is not None and parser.language_name == "python":
        return _module_liveness_descriptors(content)
    return []


def _index_one_file(
    root: str,
    store: Store,
    rel_path: str,
    content: str,
    parser: Any,
) -> Tuple[int, int, int, int, List[Function]]:
    abs_path = os.path.join(root, rel_path)
    existing = store.get_file(rel_path)
    new_hash = hash_content(content)
    if existing and existing.content_hash == new_hash:
        return 1, 0, 0, 0, store.functions_in_file(rel_path)

    if existing:
        store.delete_functions_in_file(rel_path)
        store.delete_classes_in_file(rel_path)

    if parser is None:
        file_entity = File(
            path=rel_path, language="unknown", content_hash=new_hash,
            line_count=content.count("\n") + 1, function_names=[], class_names=[],
        )
        store.upsert_file(file_entity)
        return 1, 0, 0, 0, []

    file_entity, functions, classes = parser.parse_file(abs_path, content, root)
    store.upsert_file(file_entity)
    for fn in functions:
        store.upsert_function(fn)
    for cls in classes:
        store.upsert_class(cls)
    return 1, len(functions), len(classes), 1, functions


def _apply_call_graph(store: Store, functions: List[Function], external_calls: List[str]) -> int:
    callers_map = _build_call_graph(functions, external_calls)
    n_dead = 0
    for fn in functions:
        callers = callers_map.get(fn.qualified_name, [])
        fn.callers = callers
        is_entry = fn.name in ("main", "__main__") or fn.name.startswith("test_")
        is_dunder = fn.name.startswith("__") and fn.name.endswith("__")
        fn.is_dead = (not callers) and (not is_entry) and (not is_dunder)
        if fn.is_dead:
            n_dead += 1
        store.upsert_function(fn)
    return n_dead


def _detect_cliches(store: Store, functions: List[Function], verbose: bool) -> int:
    if verbose:
        print("  detecting clichés (duplication)...")
    body_groups: Dict[str, List[str]] = {}
    for fn in functions:
        body_groups.setdefault(fn.body_hash, []).append(fn.qualified_name)
    for body_hash, instances in body_groups.items():
        if len(instances) >= 2:
            from ..model.entities import Cliche
            first_fn = next(f for f in functions if f.qualified_name == instances[0])
            store.upsert_cliche(Cliche(
                signature_hash=first_fn.signature_hash,
                body_hash=body_hash,
                instances=instances,
                note=f"{len(instances)} instances of the same function body",
            ))
    return sum(1 for insts in body_groups.values() if len(insts) >= 2)


def _snapshot_functions(store: Store) -> None:
    try:
        store.snapshot_all_functions()
    except Exception:
        pass


def index_repo(root: str, store: Store, verbose: bool = True, config=None) -> Dict[str, int]:
    """Index all supported files in the repo. Updates the store incrementally.
    Uses the language registry to find the right parser for each file.

    Stale-row cleanup: when a file changes, its old function/class rows are
    deleted before inserting new ones. Files no longer on disk are removed
    entirely (including their embeddings).
    """
    from .registry import get_parser_for_file

    # Discover all supported files currently on disk
    files_on_disk = set(discover_all_files(root, config))

    _remove_deleted_files(store, files_on_disk, verbose)

    n_files = 0
    n_functions = 0
    n_classes = 0
    n_changed = 0

    all_functions: List[Function] = []
    external_calls: List[str] = []

    for rel_path in sorted(files_on_disk):
        abs_path = os.path.join(root, rel_path)
        content = _read_source_file(abs_path)
        if content is None:
            continue

        parser = get_parser_for_file(rel_path)
        external_calls.extend(_external_calls_for_file(parser, content))
        file_count, function_count, class_count, changed_count, functions = _index_one_file(
            root, store, rel_path, content, parser
        )
        n_files += file_count
        n_functions += function_count
        n_classes += class_count
        n_changed += changed_count
        all_functions.extend(functions)
        if verbose and changed_count and n_changed % 20 == 0:
            print(f"  indexed {n_files}/{len(files_on_disk)} files...")

    # Build call graph and update callers
    if verbose:
        print(f"  building call graph over {len(all_functions)} functions...")
    n_dead = _apply_call_graph(store, all_functions, external_calls)
    n_cliche_groups = _detect_cliches(store, all_functions, verbose)
    _snapshot_functions(store)

    return {
        "files": n_files,
        "functions": n_functions,
        "classes": n_classes,
        "changed": n_changed,
        "dead_functions": n_dead,
        "cliches": n_cliche_groups,
    }
