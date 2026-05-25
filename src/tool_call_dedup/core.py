"""Session-scoped exact-duplicate detection for LLM tool calls.

LLM agents sometimes get stuck in loops, calling the same tool with the
same arguments repeatedly.  :class:`ToolCallDedup` detects exact
duplicates within a session (or globally) and lets callers decide whether
to skip, error, or log.

Keys are stable canonical hashes over ``(tool_name, sorted_kwargs)``.

Example::

    from tool_call_dedup import ToolCallDedup

    dedup = ToolCallDedup()

    # Inside your tool dispatch loop:
    if dedup.is_duplicate("search", session="sess1", query="climate change"):
        print("Skipping duplicate tool call")
    else:
        result = search(query="climate change")
        dedup.record("search", session="sess1", query="climate change")

    # Or use allow() which records on first call
    if dedup.allow("search", session="sess1", query="climate change"):
        result = search(query="climate change")
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable


def _make_key(tool_name: str, kwargs: dict[str, Any]) -> str:
    """Build a stable canonical key for a tool call.

    Sorts kwargs by key so that argument order does not matter.
    Uses SHA-256 truncated to 16 hex chars for the kwargs fingerprint.
    """
    canonical = json.dumps(
        {"tool": tool_name, "args": kwargs}, sort_keys=True, default=str
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{tool_name}:{digest}"


class DuplicateToolCallError(Exception):
    """Raised by :meth:`ToolCallDedup.require_unique` on a duplicate."""

    def __init__(self, tool_name: str, call_key: str) -> None:
        self.tool_name = tool_name
        self.call_key = call_key
        super().__init__(
            f"Duplicate tool call detected: {tool_name!r} (key={call_key!r})"
        )


class ToolCallDedup:
    """Session-scoped exact-duplicate detector for tool calls.

    Calls are identified by a stable hash of ``(tool_name, kwargs)``.
    Each session maintains independent call history; the default session
    key is ``None`` (a single global bucket).

    Args:
        ttl:   Optional time-to-live in seconds.  Entries older than *ttl*
               are considered expired and no longer count as duplicates.
               Pass ``None`` (default) to keep entries forever.
        clock: Optional callable returning current time as a float
               (default: ``time.monotonic``).  Useful in tests.
    """

    def __init__(
        self,
        *,
        ttl: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl is not None and ttl <= 0:
            raise ValueError(f"ttl must be > 0 or None, got {ttl}")
        self._ttl = ttl
        self._clock = clock or time.monotonic
        # session -> {call_key -> timestamp}
        self._history: dict[str | None, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def is_duplicate(
        self,
        tool_name: str,
        *,
        session: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Return ``True`` if this exact call was already seen in *session*.

        Does **not** record the call.

        Args:
            tool_name: Name of the tool being called.
            session:   Session identifier.  Defaults to ``None``.
            **kwargs:  Tool arguments (any JSON-serialisable values).
        """
        key = _make_key(tool_name, kwargs)
        bucket = self._bucket(session)
        self._prune(bucket)
        return key in bucket

    def allow(
        self,
        tool_name: str,
        *,
        session: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Return ``True`` if this call is **not** a duplicate, and record it.

        If the call is a duplicate, nothing is recorded and ``False`` is
        returned.

        Args:
            tool_name: Name of the tool being called.
            session:   Session identifier.  Defaults to ``None``.
            **kwargs:  Tool arguments.

        Returns:
            ``True`` on the first call with these arguments, ``False`` on
            any subsequent duplicate.
        """
        key = _make_key(tool_name, kwargs)
        bucket = self._bucket(session)
        self._prune(bucket)
        if key in bucket:
            return False
        bucket[key] = self._clock()
        return True

    def record(
        self,
        tool_name: str,
        *,
        session: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool call **without** checking for duplicates.

        Use this when the call already happened externally.

        Args:
            tool_name: Name of the tool being called.
            session:   Session identifier.  Defaults to ``None``.
            **kwargs:  Tool arguments.
        """
        key = _make_key(tool_name, kwargs)
        bucket = self._bucket(session)
        bucket[key] = self._clock()

    def require_unique(
        self,
        tool_name: str,
        *,
        session: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Like :meth:`allow`, but raises :class:`DuplicateToolCallError`.

        Records the call if it is new; raises if it is a duplicate.

        Args:
            tool_name: Name of the tool being called.
            session:   Session identifier.  Defaults to ``None``.
            **kwargs:  Tool arguments.

        Raises:
            DuplicateToolCallError: If the call is a duplicate.
        """
        key = _make_key(tool_name, kwargs)
        bucket = self._bucket(session)
        self._prune(bucket)
        if key in bucket:
            raise DuplicateToolCallError(tool_name, key)
        bucket[key] = self._clock()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def call_count(self, session: str | None = None) -> int:
        """Return the number of distinct tool calls recorded in *session*."""
        bucket = self._bucket(session)
        self._prune(bucket)
        return len(bucket)

    def key_for(self, tool_name: str, **kwargs: Any) -> str:
        """Return the canonical dedup key for *tool_name* + *kwargs*.

        Useful for inspection and testing.
        """
        return _make_key(tool_name, kwargs)

    def sessions(self) -> list[str | None]:
        """Return a sorted list of all sessions that have history.

        ``None`` (the default session) appears first if present.
        """
        keys = list(self._history.keys())
        nones = [k for k in keys if k is None]
        strings = sorted(k for k in keys if k is not None)
        return nones + strings

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self, session: str | None = "_ALL_") -> ToolCallDedup:
        """Clear the call history for *session*, or all sessions.

        Args:
            session: Session to clear.  Pass ``None`` to clear the default
                     (``None``) session.  Pass the sentinel string
                     ``"_ALL_"`` (default) to clear everything.

        Returns:
            ``self`` for chaining.
        """
        if session == "_ALL_":
            self._history.clear()
        elif session in self._history:
            del self._history[session]
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bucket(self, session: str | None) -> dict[str, float]:
        if session not in self._history:
            self._history[session] = {}
        return self._history[session]

    def _prune(self, bucket: dict[str, float]) -> None:
        if self._ttl is None:
            return
        cutoff = self._clock() - self._ttl
        expired = [k for k, ts in bucket.items() if ts <= cutoff]
        for k in expired:
            del bucket[k]

    def __repr__(self) -> str:
        return f"ToolCallDedup(ttl={self._ttl}, sessions={len(self._history)})"
