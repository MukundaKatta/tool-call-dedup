"""
tool_call_dedup — session-scoped exact duplicate tool-call detection.

Tracks (tool_name, input_hash) pairs and raises or warns when the same
call is attempted again. Zero dependencies (stdlib: hashlib, json, dataclasses).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DuplicateCall(Exception):
    """Raised when a tool call is detected as a duplicate."""

    def __init__(self, tool_name: str, call_count: int, input_repr: str) -> None:
        self.tool_name = tool_name
        self.call_count = call_count
        self.input_repr = input_repr
        super().__init__(
            f"Duplicate call to {tool_name!r} (seen {call_count} time(s)): {input_repr[:80]}"
        )


# ---------------------------------------------------------------------------
# Call record
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """Record of a single seen call."""

    tool_name: str
    input_hash: str
    count: int = 1
    input_repr: str = ""


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def _hash_input(input_data: Any) -> str:
    """Stable hash of tool input — key-order insensitive."""
    try:
        serialized = json.dumps(input_data, sort_keys=True, default=str)
    except Exception:
        serialized = str(input_data)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _input_repr(input_data: Any) -> str:
    try:
        return json.dumps(input_data, sort_keys=True, default=str)[:120]
    except Exception:
        return str(input_data)[:120]


class CallDedup:
    """
    Session-scoped exact duplicate tool-call detector.

    Usage::

        dedup = CallDedup()

        for tool_use in response.tool_uses:
            if dedup.is_duplicate(tool_use.name, tool_use.input):
                continue   # skip; we already did this
            dedup.record(tool_use.name, tool_use.input)
            result = call_tool(tool_use)
    """

    def __init__(
        self,
        *,
        raise_on_duplicate: bool = False,
        track_per_tool: bool = True,
    ) -> None:
        """
        Args:
            raise_on_duplicate: If True, :meth:`check_and_record` raises
                                :class:`DuplicateCall` on a dup.
                                If False (default), it returns False.
            track_per_tool: If True (default), the (tool_name, input_hash)
                            pair is tracked. If False, only input_hash is
                            tracked (same input to different tools counts as dup).
        """
        self.raise_on_duplicate = raise_on_duplicate
        self.track_per_tool = track_per_tool
        # key → CallRecord
        self._seen: dict[str, CallRecord] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _make_key(self, tool_name: str, input_hash: str) -> str:
        if self.track_per_tool:
            return f"{tool_name}:{input_hash}"
        return input_hash

    def is_duplicate(self, tool_name: str, input_data: Any) -> bool:
        """
        Return True if this (tool_name, input) has been seen before.

        Does NOT record the call.

        Args:
            tool_name: Tool name.
            input_data: Tool input (dict, str, or any JSON-serializable value).

        Returns:
            bool
        """
        h = _hash_input(input_data)
        key = self._make_key(tool_name, h)
        return key in self._seen

    def record(self, tool_name: str, input_data: Any) -> CallRecord:
        """
        Record a call (incrementing count if already seen).

        Does NOT check for duplicates or raise.

        Args:
            tool_name: Tool name.
            input_data: Tool input.

        Returns:
            :class:`CallRecord`
        """
        h = _hash_input(input_data)
        key = self._make_key(tool_name, h)
        if key in self._seen:
            self._seen[key].count += 1
        else:
            self._seen[key] = CallRecord(
                tool_name=tool_name,
                input_hash=h,
                count=1,
                input_repr=_input_repr(input_data),
            )
        return self._seen[key]

    def check_and_record(self, tool_name: str, input_data: Any) -> bool:
        """
        Check if this call is a duplicate, then record it.

        Args:
            tool_name: Tool name.
            input_data: Tool input.

        Returns:
            True if this is the FIRST time this call is seen (not a dup).
            False if it is a duplicate (and ``raise_on_duplicate`` is False).

        Raises:
            :class:`DuplicateCall` if it is a dup and ``raise_on_duplicate`` is True.
        """
        h = _hash_input(input_data)
        key = self._make_key(tool_name, h)
        is_new = key not in self._seen
        rec = self.record(tool_name, input_data)
        if not is_new:
            if self.raise_on_duplicate:
                raise DuplicateCall(tool_name, rec.count, rec.input_repr)
            return False
        return True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def count(self, tool_name: str, input_data: Any) -> int:
        """
        Return how many times this call has been seen.

        Returns:
            0 if never seen, else the count.
        """
        h = _hash_input(input_data)
        key = self._make_key(tool_name, h)
        rec = self._seen.get(key)
        return rec.count if rec else 0

    def seen_calls(self, tool_name: str | None = None) -> list[CallRecord]:
        """
        Return all recorded calls.

        Args:
            tool_name: If given, filter to this tool only.

        Returns:
            List of :class:`CallRecord`.
        """
        records = list(self._seen.values())
        if tool_name is not None:
            records = [r for r in records if r.tool_name == tool_name]
        return records

    def unique_call_count(self, tool_name: str | None = None) -> int:
        """Number of distinct (tool, input) pairs seen."""
        return len(self.seen_calls(tool_name))

    def total_call_count(self, tool_name: str | None = None) -> int:
        """Total number of calls recorded (including duplicates)."""
        return sum(r.count for r in self.seen_calls(tool_name))

    def duplicate_count(self, tool_name: str | None = None) -> int:
        """Number of duplicate calls (count - 1 for each multi-seen entry)."""
        return sum(max(0, r.count - 1) for r in self.seen_calls(tool_name))

    def summary(self) -> dict[str, Any]:
        """Return a summary dict."""
        return {
            "unique_calls": self.unique_call_count(),
            "total_calls": self.total_call_count(),
            "duplicate_calls": self.duplicate_count(),
            "tools_seen": list({r.tool_name for r in self._seen.values()}),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, tool_name: str | None = None) -> None:
        """
        Clear recorded calls.

        Args:
            tool_name: If given, only clear calls for this tool.
                       If None, clear all.
        """
        if tool_name is None:
            self._seen.clear()
        else:
            to_delete = [k for k, r in self._seen.items() if r.tool_name == tool_name]
            for k in to_delete:
                del self._seen[k]

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, item: tuple[str, Any]) -> bool:
        """Support ``(tool_name, input_data) in dedup`` syntax."""
        if isinstance(item, tuple) and len(item) == 2:
            return self.is_duplicate(item[0], item[1])
        return False
