"""
Language parser registry.

Maps file extensions to parsers. To add a language, register its parser here.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Type
from pathlib import Path

from .base import LanguageParser
from .python_parser import PythonParser
from .javascript_parser import JavaScriptParser


_REGISTRY: Dict[str, LanguageParser] = {}
_EXTENSION_MAP: Dict[str, str] = {}  # extension -> language_name


def register_parser(parser: LanguageParser):
    """Register a language parser."""
    _REGISTRY[parser.language_name] = parser
    for ext in parser.file_extensions:
        _EXTENSION_MAP[ext] = parser.language_name


def get_parser_for_file(file_path: str) -> Optional[LanguageParser]:
    """Find the parser for a given file path."""
    ext = Path(file_path).suffix.lower()
    lang = _EXTENSION_MAP.get(ext)
    if lang is None:
        return None
    return _REGISTRY.get(lang)


def get_parser(language_name: str) -> Optional[LanguageParser]:
    return _REGISTRY.get(language_name)


def supported_extensions() -> List[str]:
    return sorted(_EXTENSION_MAP.keys())


def supported_languages() -> List[str]:
    return sorted(_REGISTRY.keys())


# Register built-in parsers
register_parser(PythonParser())
register_parser(JavaScriptParser())
