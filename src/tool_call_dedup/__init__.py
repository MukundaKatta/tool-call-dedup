"""tool-call-dedup - session-scoped exact-duplicate tool call detection.

Within one agent session, if the same tool is called with identical args,
return the cached result instead of calling again. No TTL, no eviction —
the cache is session-lived and grows until explicitly cleared.

    from tool_call_dedup import ToolCallDedup

    dedup = ToolCallDedup(on_dup="return_cached")

    @dedup.deduplicated("search_web")
    def search_web(query: str) -> list[str]:
        return expensive_search(query)

    search_web(query="python")   # calls expensive_search
    search_web(query="python")   # returns cached result, skips expensive_search

Different from LRU+TTL caches (tool-result-cache): this library is purely
session-scoped with no eviction policy — the first result wins for the
lifetime of the ToolCallDedup instance.
"""

import functools
import hashlib
import inspect
import json
import threading
from dataclasses import dataclass
from typing import Any, Literal

__version__ = "0.1.0"


@dataclass
class DedupResult:
    """Outcome of a cache lookup."""

    hit: bool
    result: Any  # None when hit=False


class DuplicateToolCallError(Exception):
    """Raised when on_dup='raise' and the same (tool, args) pair is seen again."""

    def __init__(self, tool_name: str, args: dict) -> None:
        self.tool_name = tool_name
        self.call_args = args  # avoid shadowing BaseException.args
        super().__init__(f"Duplicate tool call: '{tool_name}' with args {args}")


def _canonical_key(tool_name: str, args: dict) -> str:
    """Return a stable SHA-256 hex digest for (tool_name, args).

    Args are serialized with sorted keys so {"a": 1, "b": 2} and
    {"b": 2, "a": 1} produce the same key.
    """
    canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ToolCallDedup:
    """Session-scoped exact-duplicate tool call detector.

    Args:
        on_dup: What to do when a duplicate call is detected.
            - "return_cached": return the stored result (default)
            - "skip": return None without calling the function
            - "raise": raise DuplicateToolCallError
    """

    def __init__(self, on_dup: Literal["return_cached", "skip", "raise"] = "return_cached") -> None:
        if on_dup not in ("return_cached", "skip", "raise"):
            raise ValueError(f"on_dup must be 'return_cached', 'skip', or 'raise'; got {on_dup!r}")
        self._on_dup = on_dup
        # Maps canonical key -> stored result
        self._cache: dict[str, Any] = {}
        # Maps canonical key -> tool_name (for clear(tool_name))
        self._key_to_tool: dict[str, str] = {}
        # Per-tool hit/miss counters: {tool_name: {"hits": N, "misses": N}}
        self._tool_stats: dict[str, dict[str, int]] = {}
        self._total_hits = 0
        self._total_misses = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(self, tool_name: str, args: dict, result: Any) -> None:
        """Store result for the given (tool_name, args) pair.

        Idempotent: recording the same key twice keeps the first result.
        """
        key = _canonical_key(tool_name, args)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = result
                self._key_to_tool[key] = tool_name

    def check(self, tool_name: str, args: dict) -> DedupResult:
        """Return a DedupResult indicating whether this call was seen before."""
        key = _canonical_key(tool_name, args)
        with self._lock:
            if key in self._cache:
                return DedupResult(hit=True, result=self._cache[key])
            return DedupResult(hit=False, result=None)

    def seen(self, tool_name: str, args: dict) -> bool:
        """Return True if this (tool_name, args) pair has already been recorded."""
        key = _canonical_key(tool_name, args)
        with self._lock:
            return key in self._cache

    def clear(self, tool_name: str | None = None) -> None:
        """Clear cached entries.

        Args:
            tool_name: If given, clear only entries for that tool.
                       If None, clear all entries.
        """
        with self._lock:
            if tool_name is None:
                self._cache.clear()
                self._key_to_tool.clear()
            else:
                keys_to_delete = [k for k, t in self._key_to_tool.items() if t == tool_name]
                for k in keys_to_delete:
                    del self._cache[k]
                    del self._key_to_tool[k]

    def stats(self) -> dict:
        """Return hit/miss statistics accumulated since creation (or last clear).

        Returns:
            {
                "total_calls": N,
                "hits": N,
                "misses": N,
                "tools": {tool_name: {"hits": N, "misses": N}},
            }
        """
        with self._lock:
            return {
                "total_calls": self._total_hits + self._total_misses,
                "hits": self._total_hits,
                "misses": self._total_misses,
                "tools": {
                    name: dict(counts) for name, counts in self._tool_stats.items()
                },
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track(self, tool_name: str, *, hit: bool) -> None:
        """Update counters. Must be called while holding self._lock."""
        if hit:
            self._total_hits += 1
        else:
            self._total_misses += 1
        entry = self._tool_stats.setdefault(tool_name, {"hits": 0, "misses": 0})
        if hit:
            entry["hits"] += 1
        else:
            entry["misses"] += 1

    def _handle_hit(self, tool_name: str, args: dict, cached_result: Any) -> Any:
        """Apply on_dup policy for a cache hit. Returns value or raises."""
        if self._on_dup == "return_cached":
            return cached_result
        if self._on_dup == "skip":
            return None
        # "raise"
        raise DuplicateToolCallError(tool_name, args)

    def _build_args_dict(self, fn: Any, pos_args: tuple, kw_args: dict) -> dict:
        """Map positional + keyword args to a normalised kwargs dict."""
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        # Map positional args to param names
        merged: dict[str, Any] = {}
        for i, val in enumerate(pos_args):
            if i < len(params):
                merged[params[i]] = val
        merged.update(kw_args)
        return merged

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def deduplicated(self, tool_name: str):
        """Sync decorator that deduplicates calls to the wrapped function."""

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                call_args = self._build_args_dict(fn, args, kwargs)
                key = _canonical_key(tool_name, call_args)
                with self._lock:
                    hit = key in self._cache
                    cached = self._cache[key] if hit else None
                    if hit:
                        self._track(tool_name, hit=True)

                if hit:
                    return self._handle_hit(tool_name, call_args, cached)

                # Miss: call the real function (outside the lock so it can block freely)
                result = fn(*args, **kwargs)
                with self._lock:
                    self._track(tool_name, hit=False)
                    if key not in self._cache:
                        self._cache[key] = result
                        self._key_to_tool[key] = tool_name
                return result

            return wrapper

        return decorator

    def deduplicated_async(self, tool_name: str):
        """Async decorator that deduplicates calls to the wrapped coroutine."""

        def decorator(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                call_args = self._build_args_dict(fn, args, kwargs)
                key = _canonical_key(tool_name, call_args)
                with self._lock:
                    hit = key in self._cache
                    cached = self._cache[key] if hit else None
                    if hit:
                        self._track(tool_name, hit=True)

                if hit:
                    return self._handle_hit(tool_name, call_args, cached)

                # Miss: await the real coroutine
                result = await fn(*args, **kwargs)
                with self._lock:
                    self._track(tool_name, hit=False)
                    if key not in self._cache:
                        self._cache[key] = result
                        self._key_to_tool[key] = tool_name
                return result

            return wrapper

        return decorator


__all__ = [
    "DedupResult",
    "DuplicateToolCallError",
    "ToolCallDedup",
    "__version__",
]
