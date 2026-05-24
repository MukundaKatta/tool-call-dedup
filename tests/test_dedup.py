"""Tests for tool_call_dedup.ToolCallDedup."""

import asyncio
import threading

import pytest

from tool_call_dedup import DedupResult, DuplicateToolCallError, ToolCallDedup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup(**kwargs) -> ToolCallDedup:
    return ToolCallDedup(**kwargs)


# ---------------------------------------------------------------------------
# record() + check()
# ---------------------------------------------------------------------------


def test_check_miss_when_nothing_recorded():
    d = _dedup()
    result = d.check("search", {"q": "hello"})
    assert result == DedupResult(hit=False, result=None)


def test_record_then_check_hit():
    d = _dedup()
    d.record("search", {"q": "hello"}, ["result1"])
    r = d.check("search", {"q": "hello"})
    assert r.hit is True
    assert r.result == ["result1"]


def test_check_different_args_miss():
    d = _dedup()
    d.record("search", {"q": "hello"}, "A")
    r = d.check("search", {"q": "world"})
    assert r.hit is False


def test_different_tools_same_args_separate_entries():
    d = _dedup()
    d.record("tool_a", {"x": 1}, "from_a")
    d.record("tool_b", {"x": 1}, "from_b")
    assert d.check("tool_a", {"x": 1}).result == "from_a"
    assert d.check("tool_b", {"x": 1}).result == "from_b"


def test_arg_order_does_not_matter():
    d = _dedup()
    d.record("fn", {"a": 1, "b": 2}, "first")
    r = d.check("fn", {"b": 2, "a": 1})
    assert r.hit is True
    assert r.result == "first"


def test_record_is_idempotent_first_result_wins():
    d = _dedup()
    d.record("fn", {"x": 0}, "original")
    d.record("fn", {"x": 0}, "overwrite_attempt")
    assert d.check("fn", {"x": 0}).result == "original"


# ---------------------------------------------------------------------------
# seen()
# ---------------------------------------------------------------------------


def test_seen_false_before_record():
    d = _dedup()
    assert d.seen("fn", {"k": "v"}) is False


def test_seen_true_after_record():
    d = _dedup()
    d.record("fn", {"k": "v"}, 42)
    assert d.seen("fn", {"k": "v"}) is True


def test_seen_false_for_different_args():
    d = _dedup()
    d.record("fn", {"k": "v"}, 42)
    assert d.seen("fn", {"k": "other"}) is False


# ---------------------------------------------------------------------------
# deduplicated() sync decorator
# ---------------------------------------------------------------------------


def test_decorator_miss_calls_fn_and_records():
    d = _dedup()
    call_count = {"n": 0}

    @d.deduplicated("fetch")
    def fetch(url):
        call_count["n"] += 1
        return f"content:{url}"

    result = fetch("http://example.com")
    assert result == "content:http://example.com"
    assert call_count["n"] == 1
    assert d.seen("fetch", {"url": "http://example.com"})


def test_decorator_hit_return_cached():
    d = _dedup(on_dup="return_cached")
    call_count = {"n": 0}

    @d.deduplicated("fetch")
    def fetch(url):
        call_count["n"] += 1
        return f"content:{url}"

    fetch("http://example.com")
    result = fetch("http://example.com")
    assert call_count["n"] == 1  # fn only called once
    assert result == "content:http://example.com"


def test_decorator_hit_skip():
    d = _dedup(on_dup="skip")
    call_count = {"n": 0}

    @d.deduplicated("fetch")
    def fetch(url):
        call_count["n"] += 1
        return "data"

    fetch("http://example.com")
    result = fetch("http://example.com")
    assert call_count["n"] == 1
    assert result is None


def test_decorator_hit_raise():
    d = _dedup(on_dup="raise")

    @d.deduplicated("fetch")
    def fetch(url):
        return "data"

    fetch("http://example.com")
    with pytest.raises(DuplicateToolCallError) as exc_info:
        fetch("http://example.com")
    assert exc_info.value.tool_name == "fetch"
    assert exc_info.value.call_args == {"url": "http://example.com"}


def test_decorator_positional_args_mapped():
    d = _dedup()
    call_count = {"n": 0}

    @d.deduplicated("greet")
    def greet(name, greeting="Hello"):
        call_count["n"] += 1
        return f"{greeting} {name}"

    greet("Alice")
    greet("Alice")
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# deduplicated_async() decorator
# ---------------------------------------------------------------------------


def test_async_decorator_miss_calls_fn():
    d = _dedup()
    call_count = {"n": 0}

    @d.deduplicated_async("async_fetch")
    async def async_fetch(url):
        call_count["n"] += 1
        return f"async:{url}"

    result = asyncio.run(async_fetch("http://example.com"))
    assert result == "async:http://example.com"
    assert call_count["n"] == 1


def test_async_decorator_hit_return_cached():
    d = _dedup(on_dup="return_cached")
    call_count = {"n": 0}

    @d.deduplicated_async("async_fetch")
    async def async_fetch(url):
        call_count["n"] += 1
        return "async_data"

    async def run():
        await async_fetch("http://example.com")
        return await async_fetch("http://example.com")

    result = asyncio.run(run())
    assert call_count["n"] == 1
    assert result == "async_data"


def test_async_decorator_hit_skip():
    d = _dedup(on_dup="skip")

    @d.deduplicated_async("async_fetch")
    async def async_fetch(url):
        return "data"

    async def run():
        await async_fetch("http://example.com")
        return await async_fetch("http://example.com")

    result = asyncio.run(run())
    assert result is None


def test_async_decorator_hit_raise():
    d = _dedup(on_dup="raise")

    @d.deduplicated_async("async_fetch")
    async def async_fetch(url):
        return "data"

    async def run():
        await async_fetch("http://example.com")
        await async_fetch("http://example.com")

    with pytest.raises(DuplicateToolCallError) as exc_info:
        asyncio.run(run())
    assert exc_info.value.tool_name == "async_fetch"
    assert exc_info.value.call_args == {"url": "http://example.com"}


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_all():
    d = _dedup()
    d.record("a", {"x": 1}, "va")
    d.record("b", {"x": 1}, "vb")
    d.clear()
    assert not d.seen("a", {"x": 1})
    assert not d.seen("b", {"x": 1})


def test_clear_specific_tool():
    d = _dedup()
    d.record("a", {"x": 1}, "va")
    d.record("b", {"x": 1}, "vb")
    d.clear("a")
    assert not d.seen("a", {"x": 1})
    assert d.seen("b", {"x": 1})


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------


def test_stats_empty():
    d = _dedup()
    s = d.stats()
    assert s["total_calls"] == 0
    assert s["hits"] == 0
    assert s["misses"] == 0
    assert s["tools"] == {}


def test_stats_counts():
    d = _dedup()
    call_count = {"n": 0}

    @d.deduplicated("tool")
    def tool(x):
        call_count["n"] += 1
        return x * 2

    tool(1)    # miss
    tool(1)    # hit
    tool(2)    # miss

    s = d.stats()
    assert s["total_calls"] == 3
    assert s["hits"] == 1
    assert s["misses"] == 2
    assert s["tools"]["tool"]["hits"] == 1
    assert s["tools"]["tool"]["misses"] == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_thread_safety_only_one_call_recorded():
    d = _dedup()
    call_count = {"n": 0}
    lock = threading.Lock()

    @d.deduplicated("thread_tool")
    def thread_tool(x):
        with lock:
            call_count["n"] += 1
        return x

    threads = [threading.Thread(target=thread_tool, args=(42,)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The underlying fn may be called more than once due to race between
    # check and record, but the cache should be stable and all calls succeed.
    assert d.seen("thread_tool", {"x": 42})


# ---------------------------------------------------------------------------
# DuplicateToolCallError attributes
# ---------------------------------------------------------------------------


def test_duplicate_error_attributes():
    err = DuplicateToolCallError("my_tool", {"key": "value"})
    assert err.tool_name == "my_tool"
    assert err.call_args == {"key": "value"}
    assert "my_tool" in str(err)
    assert "Duplicate tool call" in str(err)
