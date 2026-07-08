"""
JavaScript/TypeScript parser.

This is intentionally lightweight and dependency-free, but it is not a plain
regex parser: source is first masked so strings and comments cannot confuse
brace matching or function-boundary detection.
"""

from __future__ import annotations
import os
import re
from typing import List, Optional, Tuple

from ..model.entities import File, Function, Class, hash_content


IDENT = r"[A-Za-z_$][\w$]*"
CONTROL_WORDS = {
    "if", "for", "while", "switch", "catch", "function", "return", "typeof",
    "new", "class", "import", "export", "await", "async", "super",
}

FUNCTION_RE = re.compile(
    rf"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+"
    rf"(?P<name>{IDENT})(?:\s*<[^>{{}}()]*>)?\s*"
    rf"\((?P<args>[^()]*)\)\s*(?::\s*[^{{;]+)?\s*\{{",
    re.MULTILINE,
)
FUNCTION_EXPR_RE = re.compile(
    rf"(?:export\s+)?(?:const|let|var)\s+(?P<name>{IDENT})"
    rf"(?:\s*:\s*[^=\n;]+)?\s*=\s*(?:async\s+)?function"
    rf"(?:\s+{IDENT})?(?:\s*<[^>{{}}()]*>)?\s*"
    rf"\((?P<args>[^()]*)\)\s*(?::\s*[^{{;]+)?\s*\{{",
    re.MULTILINE,
)
ARROW_FUNCTION_RE = re.compile(
    rf"(?:export\s+)?(?:const|let|var)\s+(?P<name>{IDENT})"
    rf"(?:\s*:\s*[^=\n;]+)?\s*=\s*(?:async\s*)?(?:<[^>{{}}()]*>\s*)?"
    rf"(?P<args>\([^()]*\)|{IDENT}(?:\s*:\s*[^=;,\n]+)?)\s*"
    rf"(?::\s*[^=;{{]+)?=>",
    re.MULTILINE,
)
CLASS_RE = re.compile(
    rf"(?:export\s+)?(?:default\s+)?class\s+(?P<name>{IDENT})"
    rf"(?:\s+extends\s+(?P<base>{IDENT}))?\s*\{{",
    re.MULTILINE,
)
METHOD_RE = re.compile(
    rf"(?:(?:public|private|protected|static|async|override|readonly|get|set)\s+)*"
    rf"(?P<name>{IDENT})(?:\s*<[^>{{}}()]*>)?\s*"
    rf"\((?P<args>[^()]*)\)\s*(?::\s*[^{{;]+)?\s*\{{",
    re.MULTILINE,
)
BRANCH_RE = re.compile(r"\b(?:if|for|while|switch|case|catch)\b|&&|\|\||\?[^.:]")
DIRECT_CALL_RE = re.compile(rf"(?<![\w$.])({IDENT})\s*\(")
METHOD_CALL_RE = re.compile(rf"\.\s*({IDENT})\s*\(")


class JavaScriptParser:
    """Lightweight JS/TS parser. Implements the LanguageParser interface."""

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
        lang = "typescript" if file_path.endswith((".ts", ".tsx")) else "javascript"
        module_qname = self._module_qname(rel_path)
        masked = self._mask_non_code(content)

        functions: List[Function] = []
        classes: List[Class] = []
        function_names: List[str] = []
        class_names: List[str] = []

        for m in CLASS_RE.finditer(masked):
            open_pos = m.end() - 1
            close_end = self._find_matching_brace_end(masked, open_pos + 1)
            if close_end <= open_pos + 1:
                continue
            class_name = m.group("name")
            base = m.group("base") or ""
            class_is_exported = self._is_exported(masked, m.start(), m.end())
            class_names.append(class_name)

            body_start = open_pos + 1
            body_end = close_end - 1
            method_names = []
            for mm in METHOD_RE.finditer(masked, body_start, body_end):
                if self._brace_depth(masked, body_start, mm.start()) != 0:
                    continue
                method_name = mm.group("name")
                if method_name in CONTROL_WORDS:
                    continue
                method_names.append(method_name)
                method_open = mm.end() - 1
                method_close_end = self._find_matching_brace_end(masked, method_open + 1)
                if method_close_end <= method_open + 1:
                    continue
                if method_name != "constructor":
                    fn = self._make_function(
                        name=method_name,
                        qualified_name=f"{module_qname}.{class_name}.{method_name}",
                        file_path=rel_path,
                        start=mm.start(),
                        args=mm.group("args"),
                        body_start=method_open + 1,
                        body_end=method_close_end - 1,
                        content=content,
                        masked=masked,
                    )
                    if class_is_exported:
                        fn.calls.append(f"live:{fn.qualified_name}:export")
                    functions.append(fn)
                    function_names.append(method_name)

            classes.append(Class(
                name=class_name,
                qualified_name=f"{module_qname}.{class_name}",
                file_path=rel_path,
                start_line=self._offset_to_line(content, m.start()),
                end_line=self._offset_to_line(content, close_end),
                bases=[base] if base else [],
                method_names=method_names,
            ))

        for pattern in (FUNCTION_RE, FUNCTION_EXPR_RE):
            for m in pattern.finditer(masked):
                open_pos = m.end() - 1
                close_end = self._find_matching_brace_end(masked, open_pos + 1)
                if close_end <= open_pos + 1:
                    continue
                fn = self._make_function(
                    name=m.group("name"),
                    qualified_name=f"{module_qname}.{m.group('name')}",
                    file_path=rel_path,
                    start=m.start(),
                    args=m.group("args"),
                    body_start=open_pos + 1,
                    body_end=close_end - 1,
                    content=content,
                    masked=masked,
                )
                if self._is_exported(masked, m.start(), m.end()):
                    fn.calls.append(f"live:{fn.qualified_name}:export")
                functions.append(fn)
                function_names.append(m.group("name"))

        for m in ARROW_FUNCTION_RE.finditer(masked):
            body_start = self._skip_ws(masked, m.end())
            if body_start >= len(masked):
                continue
            if masked[body_start] == "{":
                close_end = self._find_matching_brace_end(masked, body_start + 1)
                if close_end <= body_start + 1:
                    continue
                body_end = close_end - 1
            else:
                body_end = self._find_expression_end(masked, body_start)
            fn = self._make_function(
                name=m.group("name"),
                qualified_name=f"{module_qname}.{m.group('name')}",
                file_path=rel_path,
                start=m.start(),
                args=m.group("args"),
                body_start=body_start,
                body_end=body_end,
                content=content,
                masked=masked,
            )
            if self._is_exported(masked, m.start(), m.end()):
                fn.calls.append(f"live:{fn.qualified_name}:export")
            functions.append(fn)
            function_names.append(m.group("name"))

        functions = self._dedupe_functions(functions)
        function_names = [f.name for f in functions]

        file_entity = File(
            path=rel_path,
            language=lang,
            content_hash=content_hash,
            line_count=line_count,
            function_names=function_names,
            class_names=class_names,
        )
        return file_entity, functions, classes

    def _make_function(
        self,
        name: str,
        qualified_name: str,
        file_path: str,
        start: int,
        args: str,
        body_start: int,
        body_end: int,
        content: str,
        masked: str,
    ) -> Function:
        body = content[body_start:body_end]
        masked_body = masked[body_start:body_end]
        arg_names = self._parse_args(args)
        return Function(
            name=name,
            qualified_name=qualified_name,
            file_path=file_path,
            start_line=self._offset_to_line(content, start),
            end_line=self._offset_to_line(content, body_end),
            arg_names=arg_names,
            signature_hash=hash_content(f"{name}({','.join(arg_names)})"),
            body_hash=hash_content(body.strip()),
            ast_summary=self._make_summary(masked_body),
            complexity=1 + len(BRANCH_RE.findall(masked_body)),
            docstring=None,
            calls=self._collect_calls(masked_body),
        )

    def _module_qname(self, rel_path: str) -> str:
        rel_norm = rel_path.replace(os.sep, ".")
        for ext in self.file_extensions:
            if rel_norm.endswith(ext):
                return rel_norm[: -len(ext)]
        return rel_norm

    def _is_exported(self, masked: str, start: int, end: int) -> bool:
        return masked[start:end].lstrip().startswith("export ")

    def _parse_args(self, args: str) -> List[str]:
        args = args.strip()
        if args.startswith("(") and args.endswith(")"):
            args = args[1:-1]
        names = []
        for part in self._split_args(args):
            part = part.strip()
            if not part or part == "void":
                continue
            part = part.split("=", 1)[0].strip()
            part = part.removeprefix("...")
            if part.startswith("{") or part.startswith("["):
                names.append("destructured")
                continue
            part = part.split(":", 1)[0].strip().rstrip("?")
            if part and re.match(rf"^{IDENT}$", part):
                names.append(part)
        return names

    def _split_args(self, args: str) -> List[str]:
        parts = []
        start = 0
        depth = 0
        for i, ch in enumerate(args):
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                parts.append(args[start:i])
                start = i + 1
        if args[start:].strip():
            parts.append(args[start:])
        return parts

    def _collect_calls(self, masked_body: str) -> List[str]:
        calls = []
        seen = set()
        for m in METHOD_CALL_RE.finditer(masked_body):
            name = m.group(1)
            key = f"call:.{name}"
            if key not in seen:
                seen.add(key)
                calls.append(key)
        for m in DIRECT_CALL_RE.finditer(masked_body):
            name = m.group(1)
            if name in CONTROL_WORDS:
                continue
            key = f"call:{name}"
            if key not in seen:
                seen.add(key)
                calls.append(key)
        return calls

    def _dedupe_functions(self, functions: List[Function]) -> List[Function]:
        by_key = {}
        for fn in sorted(functions, key=lambda f: (f.file_path, f.start_line, f.end_line, f.qualified_name)):
            by_key.setdefault((fn.qualified_name, fn.start_line), fn)
        return list(by_key.values())

    def _mask_non_code(self, content: str) -> str:
        chars = list(content)
        state = "code"
        i = 0
        while i < len(chars):
            ch = chars[i]
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if state == "code":
                if ch == "/" and nxt == "/":
                    chars[i] = chars[i + 1] = " "
                    i += 2
                    state = "line_comment"
                    continue
                if ch == "/" and nxt == "*":
                    chars[i] = chars[i + 1] = " "
                    i += 2
                    state = "block_comment"
                    continue
                if ch in ("'", '"', "`"):
                    quote = ch
                    chars[i] = " "
                    i += 1
                    state = quote
                    continue
            elif state == "line_comment":
                if ch == "\n":
                    state = "code"
                else:
                    chars[i] = " "
            elif state == "block_comment":
                if ch == "*" and nxt == "/":
                    chars[i] = chars[i + 1] = " "
                    i += 2
                    state = "code"
                    continue
                if ch != "\n":
                    chars[i] = " "
            else:
                quote = state
                if ch == "\\":
                    chars[i] = " "
                    if i + 1 < len(chars) and chars[i + 1] != "\n":
                        chars[i + 1] = " "
                    i += 2
                    continue
                if ch == quote:
                    chars[i] = " "
                    state = "code"
                elif ch != "\n":
                    chars[i] = " "
            i += 1
        return "".join(chars)

    def _find_matching_brace_end(self, content: str, start: int) -> int:
        """Find the offset after the matching closing brace.

        `start` is the offset immediately after the opening brace, matching the
        original public helper's calling convention.
        """
        open_pos = start - 1
        if open_pos < 0 or open_pos >= len(content) or content[open_pos] != "{":
            return start
        depth = 1
        i = start
        while i < len(content) and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1
        return i if depth == 0 else start

    def _brace_depth(self, masked: str, start: int, end: int) -> int:
        depth = 0
        for ch in masked[start:end]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)
        return depth

    def _skip_ws(self, text: str, pos: int) -> int:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    def _find_expression_end(self, masked: str, start: int) -> int:
        depth = 0
        i = start
        while i < len(masked):
            ch = masked[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and ch in ";\n":
                break
            i += 1
        return i

    def _offset_to_line(self, content: str, offset: int) -> int:
        return content[:offset].count("\n") + 1

    def _make_summary(self, masked_body: str) -> str:
        calls = []
        for call_desc in self._collect_calls(masked_body):
            if call_desc.startswith("call:."):
                calls.append(call_desc[len("call:."):])
            elif call_desc.startswith("call:"):
                calls.append(call_desc[len("call:"):])
        returns = len(re.findall(r"\breturn\b", masked_body))
        ifs = len(re.findall(r"\bif\b", masked_body))
        loops = len(re.findall(r"\b(?:for|while)\b", masked_body))

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
        return ".min." in rel_path
