"""Tests for tool_call_dedup."""

from __future__ import annotations

import pytest

from tool_call_dedup import DuplicateToolCallError, ToolCallDedup

# ---------------------------------------------------------------------------
# Fake clock
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# ---------------------------------------------------------------------------
# Constructor / repr
# ---------------------------------------------------------------------------


def test_repr():
    d = ToolCallDedup()
    assert "ttl=None" in repr(d)
    assert "sessions=0" in repr(d)


def test_invalid_ttl():
    with pytest.raises(ValueError):
        ToolCallDedup(ttl=0)
    with pytest.raises(ValueError):
        ToolCallDedup(ttl=-1)


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------


def test_is_duplicate_false_on_first():
    d = ToolCallDedup()
    assert d.is_duplicate("search", query="hello") is False


def test_is_duplicate_false_without_record():
    d = ToolCallDedup()
    d.is_duplicate("search", query="hello")  # doesn't record
    assert d.is_duplicate("search", query="hello") is False


def test_is_duplicate_true_after_record():
    d = ToolCallDedup()
    d.record("search", query="hello")
    assert d.is_duplicate("search", query="hello") is True


def test_is_duplicate_different_args():
    d = ToolCallDedup()
    d.record("search", query="hello")
    assert d.is_duplicate("search", query="world") is False


def test_is_duplicate_different_tool():
    d = ToolCallDedup()
    d.record("search", query="hello")
    assert d.is_duplicate("browse", query="hello") is False


def test_is_duplicate_session_isolation():
    d = ToolCallDedup()
    d.record("search", session="s1", query="hello")
    assert d.is_duplicate("search", session="s2", query="hello") is False


def test_is_duplicate_kwarg_order_invariant():
    d = ToolCallDedup()
    d.record("search", query="q", limit=10)
    assert d.is_duplicate("search", limit=10, query="q") is True


# ---------------------------------------------------------------------------
# allow
# ---------------------------------------------------------------------------


def test_allow_first_call_returns_true():
    d = ToolCallDedup()
    assert d.allow("search", query="hello") is True


def test_allow_records_on_first_call():
    d = ToolCallDedup()
    d.allow("search", query="hello")
    assert d.is_duplicate("search", query="hello") is True


def test_allow_second_call_returns_false():
    d = ToolCallDedup()
    d.allow("search", query="hello")
    assert d.allow("search", query="hello") is False


def test_allow_does_not_record_duplicate():
    d = ToolCallDedup()
    d.allow("search", query="hello")
    d.allow("search", query="hello")  # duplicate — not recorded again
    assert d.call_count() == 1


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def test_record_increments_count():
    d = ToolCallDedup()
    d.record("a")
    d.record("b")
    assert d.call_count() == 2


def test_record_same_key_twice_updates_timestamp():
    clock = FakeClock()
    d = ToolCallDedup(ttl=10, clock=clock)
    d.record("search", query="q")
    clock.advance(5)
    d.record("search", query="q")  # re-record with fresh timestamp
    clock.advance(6)  # t=11: first timestamp expired but second hasn't
    # Still a duplicate because the re-record timestamp is at t=5 -> expires at t=15
    assert d.is_duplicate("search", query="q") is True


# ---------------------------------------------------------------------------
# require_unique
# ---------------------------------------------------------------------------


def test_require_unique_ok():
    d = ToolCallDedup()
    d.require_unique("search", query="hello")  # should not raise


def test_require_unique_raises_on_duplicate():
    d = ToolCallDedup()
    d.require_unique("search", query="hello")
    with pytest.raises(DuplicateToolCallError) as exc_info:
        d.require_unique("search", query="hello")
    err = exc_info.value
    assert err.tool_name == "search"
    assert "search" in str(err)


def test_require_unique_records_on_first():
    d = ToolCallDedup()
    d.require_unique("search", query="q")
    assert d.is_duplicate("search", query="q") is True


def test_duplicate_tool_call_error_has_key():
    d = ToolCallDedup()
    d.record("search", q="x")
    with pytest.raises(DuplicateToolCallError) as exc_info:
        d.require_unique("search", q="x")
    assert exc_info.value.call_key


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


def test_ttl_expiry():
    clock = FakeClock()
    d = ToolCallDedup(ttl=10, clock=clock)
    d.record("search", query="q")
    assert d.is_duplicate("search", query="q") is True
    clock.advance(11)
    assert d.is_duplicate("search", query="q") is False


def test_ttl_not_yet_expired():
    clock = FakeClock()
    d = ToolCallDedup(ttl=10, clock=clock)
    d.record("search", query="q")
    clock.advance(9)
    assert d.is_duplicate("search", query="q") is True


def test_ttl_refreshes_on_allow():
    clock = FakeClock()
    d = ToolCallDedup(ttl=10, clock=clock)
    d.allow("search", query="q")  # t=0
    clock.advance(5)
    # Second allow at t=5 returns False (duplicate) but could note timing
    d.allow("search", query="q")
    clock.advance(6)
    # t=11: original entry at t=0 would have expired, but dedup only
    # records fresh entries — the second allow did NOT re-record
    # so at t=11 the entry is still the one from t=0, which expired
    assert d.is_duplicate("search", query="q") is False


# ---------------------------------------------------------------------------
# call_count
# ---------------------------------------------------------------------------


def test_call_count_default_session():
    d = ToolCallDedup()
    d.record("a")
    d.record("b")
    assert d.call_count() == 2


def test_call_count_per_session():
    d = ToolCallDedup()
    d.record("a", session="s1")
    d.record("b", session="s1")
    d.record("c", session="s2")
    assert d.call_count("s1") == 2
    assert d.call_count("s2") == 1


def test_call_count_zero_for_new_session():
    d = ToolCallDedup()
    assert d.call_count("new") == 0


# ---------------------------------------------------------------------------
# key_for
# ---------------------------------------------------------------------------


def test_key_for_same_args():
    d = ToolCallDedup()
    k1 = d.key_for("search", query="q")
    k2 = d.key_for("search", query="q")
    assert k1 == k2


def test_key_for_different_tool():
    d = ToolCallDedup()
    k1 = d.key_for("search", query="q")
    k2 = d.key_for("browse", query="q")
    assert k1 != k2


def test_key_for_kwarg_order_invariant():
    d = ToolCallDedup()
    k1 = d.key_for("f", a=1, b=2)
    k2 = d.key_for("f", b=2, a=1)
    assert k1 == k2


# ---------------------------------------------------------------------------
# sessions / clear
# ---------------------------------------------------------------------------


def test_sessions_none_first():
    d = ToolCallDedup()
    d.record("a")  # default (None) session
    d.record("b", session="s1")
    sessions = d.sessions()
    assert sessions[0] is None
    assert "s1" in sessions


def test_clear_default_session():
    d = ToolCallDedup()
    d.record("a")
    d.clear(session=None)
    assert d.call_count() == 0


def test_clear_specific_session():
    d = ToolCallDedup()
    d.record("a", session="s1")
    d.record("b", session="s2")
    d.clear(session="s1")
    assert d.call_count("s1") == 0
    assert d.call_count("s2") == 1


def test_clear_all():
    d = ToolCallDedup()
    d.record("a", session="s1")
    d.record("b", session="s2")
    d.clear()  # clears all
    assert d.call_count("s1") == 0
    assert d.call_count("s2") == 0


def test_clear_returns_self():
    d = ToolCallDedup()
    assert d.clear() is d
