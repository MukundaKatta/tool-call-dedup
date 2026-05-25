"""Session-scoped exact-duplicate detection for LLM tool calls."""

from __future__ import annotations

from tool_call_dedup.core import DuplicateToolCallError, ToolCallDedup

__all__ = ["ToolCallDedup", "DuplicateToolCallError"]
__version__ = "0.1.0"
