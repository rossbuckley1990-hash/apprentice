"""
JavaScript/TypeScript parser.

Uses regex-based extraction (not a full AST) for the MVP. This is less
accurate than the Python AST parser but functional — it finds function
declarations, arrow functions, and class definitions.

For production use, replace with a tree-sitter-based parser.
"""

from __future__ import annotations
import re
import os
import hashlib
from typing import List, Tuple

from ..model.entities import File, Function, Class, hash_content


# Regex patterns for JS/TS function detection
# Matches: function foo(a, b) { ... }
FUNCTION_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)
# Matches: const foo = (a, b) => { ... } and const foo = function(a, b) { ... }
ARROW_FUNCTION_RE = re.compile(
    r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>\s*(?:\{|=>)",
    re.MULTILINE,
)
# Matches: class Foo extends Bar { ... }
CLASS_RE = re.compile(
    r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{",
    re.MULTILINE,
)
# Matches: method(a, b) { ... } inside classes
METHOD_RE = re.compile(
    r"(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)
# Rough complexity: count if/for/while/switch/case/&&/||/?/
BRANCH_RE = re.compile(
    r"\b(?:if|for|while|switch|case|catch)\b|&&|\|\||\?[^.:]",
    re.MULTILINE,
)


class JavaScriptParser:
    """Regex-based JS/TS parser. Implements the LanguageParser interface."""

    @property
    def language_name(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> List[str]:
        return [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]

    def parse_file(self, file_path: str, content: str, root: str) -> Tuple[File, List[Function], List[Class]]:
        rel_path = os.path.relpath(file_path, root)
        content_hash = hash_content(content)
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        functions: List[Function] = []
        classes: List[Class] = []
        function_names: List[str] = []
        class_names: List[str] = []

        # Determine if it's TypeScript
        lang = "typescript" if file_path.endswith((".ts", ".tsx")) else "javascript"

        # Build module qualified-name
        rel_norm = rel_path.replace(os.sep, ".")
        for ext in self.file_extensions:
            if rel_norm.endswith(ext):
                rel_norm = rel_norm[: -len(ext)]
                break
        module_qname = rel_norm

        # Find regular functions
        for m in FUNCTION_RE.finditer(content):
            name = m.group(1)
            args_str = m.group(2).strip()
            arg_names = [a.strip().split(":")[0].split("=")[0].strip() for a in args_str.split(",") if a.strip()]
            start_line = content[: m.start()].count("\n") + 1
            brace_end_offset = self._find_matching_brace_end(content, m.end())
            end_line = self._offset_to_line(content, brace_end_offset) if brace_end_offset > m.end() else start_line

            body = content[m.end():brace_end_offset] if brace_end_offset > m.end() else ""
            body_hash = hash_content(body)
            sig_hash = hash_content(f"{name}({','.join(arg_names)})")
            complexity = 1 + len(BRANCH_RE.findall(body))
            summary = self._make_summary(body)

            function_names.append(name)
            functions.append(Function(
                name=name,
                qualified_name=f"{module_qname}.{name}",
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                arg_names=arg_names,
                signature_hash=sig_hash,
                body_hash=body_hash,
                ast_summary=summary,
                complexity=complexity,
                docstring=None,
                calls=[],
            ))

        # Find arrow functions
        for m in ARROW_FUNCTION_RE.finditer(content):
            name = m.group(1)
            args_str = m.group(2).strip()
            arg_names = [a.strip().split(":")[0].split("=")[0].strip() for a in args_str.split(",") if a.strip()]
            start_line = content[: m.start()].count("\n") + 1
            brace_end_offset = self._find_matching_brace_end(content, m.end())
            end_line = self._offset_to_line(content, brace_end_offset) if brace_end_offset > m.end() else start_line

            body = content[m.end():brace_end_offset] if brace_end_offset > m.end() else ""
            body_hash = hash_content(body)
            sig_hash = hash_content(f"{name}({','.join(arg_names)})")
            complexity = 1 + len(BRANCH_RE.findall(body))
            summary = self._make_summary(body)

            function_names.append(name)
            functions.append(Function(
                name=name,
                qualified_name=f"{module_qname}.{name}",
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                arg_names=arg_names,
                signature_hash=sig_hash,
                body_hash=body_hash,
                ast_summary=summary,
                complexity=complexity,
                docstring=None,
                calls=[],
            ))

        # Find classes
        for m in CLASS_RE.finditer(content):
            name = m.group(1)
            base = m.group(2) if m.group(2) else ""
            start_line = content[: m.start()].count("\n") + 1
            end_line = self._find_matching_brace_end(content, m.end()) or start_line

            # Find methods within the class
            class_body = content[m.end():end_line] if end_line > m.end() else ""
            method_names = []
            for mm in METHOD_RE.finditer(class_body):
                method_names.append(mm.group(1))

            class_names.append(name)
            classes.append(Class(
                name=name,
                qualified_name=f"{module_qname}.{name}",
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                bases=[base] if base else [],
                method_names=method_names,
            ))

        file_entity = File(
            path=rel_path,
            language=lang,
            content_hash=content_hash,
            line_count=line_count,
            function_names=function_names,
            class_names=class_names,
        )
        return file_entity, functions, classes

    def _find_matching_brace_end(self, content: str, start: int) -> int:
        """Find the character offset of the matching closing brace.
        Returns the offset AFTER the closing brace, or `start` if no match."""
        depth = 1
        i = start
        while i < len(content) and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return i  # character offset after the closing brace
        return start  # no match found

    def _offset_to_line(self, content: str, offset: int) -> int:
        """Convert a character offset to a line number."""
        return content[:offset].count("\n") + 1

    def _make_summary(self, body: str) -> str:
        """Make a rough AST summary from function body."""
        calls = []
        for m in re.finditer(r"(\w+)\s*\(", body):
            calls.append(m.group(1))
        returns = body.count("return ")
        ifs = len(re.findall(r"\bif\b", body))
        loops = len(re.findall(r"\b(?:for|while)\b", body))

        parts = []
        if calls:
            unique = list(dict.fromkeys(calls))
            parts.append(f"calls {', '.join(unique[:5])}")
        if returns:
            parts.append(f"returns ×{returns}")
        if ifs:
            parts.append(f"ifs ×{ifs}")
        if loops:
            parts.append(f"loops ×{loops}")
        return "; ".join(parts) if parts else "no-op"

    def should_ignore(self, rel_path: str) -> bool:
        # Skip minified files
        return ".min." in rel_path
