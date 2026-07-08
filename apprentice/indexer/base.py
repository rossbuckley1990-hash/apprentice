"""
Base interface for language parsers.

Each language parser implements this interface. The indexer uses the registry
to find the right parser for each file extension.

To add a new language:
  1. Create a parser class implementing LanguageParser
  2. Register it in registry.py
  3. Add tests

Python is fully implemented. JavaScript/TypeScript support is lightweight,
dependency-free, and less accurate than a full AST.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path

from ..model.entities import File, Function, Class


class LanguageParser(ABC):
    """Abstract base for language-specific parsers."""

    @property
    @abstractmethod
    def language_name(self) -> str:
        """e.g. 'python', 'javascript'"""

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """e.g. ['.py'], ['.js', '.ts']"""

    @abstractmethod
    def parse_file(
        self, file_path: str, content: str, root: str
    ) -> Tuple[File, List[Function], List[Class]]:
        """Parse a file into entities.

        Args:
            file_path: absolute path to the file
            content: file contents as string
            root: repo root (for computing relative paths)

        Returns:
            (File entity, list of Function entities, list of Class entities)
        """
        ...

    def should_ignore(self, rel_path: str) -> bool:
        """Override to add language-specific ignore rules."""
        return False
