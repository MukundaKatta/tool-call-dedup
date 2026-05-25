import pytest
from tool_call_dedup import DuplicateCall, CallRecord, CallDedup


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------

def test_first_call_not_duplicate():
    d = CallDedup()
    assert not d.is_duplicate("search", {"query": "cats"})

def test_second_same_call_is_duplicate():
    d = CallDedup()
    d.record("search", {"query": "cats"})
    assert d.is_duplicate("search", {"query": "cats"})

def test_different_input_not_duplicate():
    d = CallDedup()
    d.record("search", {"query": "cats"})
    assert not d.is_duplicate("search", {"query": "dogs"})

def test_different_tool_not_duplicate():
    d = CallDedup()
    d.record("search", {"query": "cats"})
    assert not d.is_duplicate("read_file", {"query": "cats"})

def test_input_key_order_insensitive():
    d = CallDedup()
    d.record("tool", {"b": 2, "a": 1})
    assert d.is_duplicate("tool", {"a": 1, "b": 2})

def test_string_input():
    d = CallDedup()
    d.record("tool", "hello")
    assert d.is_duplicate("tool", "hello")
    assert not d.is_duplicate("tool", "world")

def test_none_input():
    d = CallDedup()
    d.record("tool", None)
    assert d.is_duplicate("tool", None)

def test_empty_dict_input():
    d = CallDedup()
    d.record("tool", {})
    assert d.is_duplicate("tool", {})


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def test_record_returns_call_record():
    d = CallDedup()
    r = d.record("search", {"q": "cats"})
    assert isinstance(r, CallRecord)

def test_record_increments_count():
    d = CallDedup()
    d.record("search", {"q": "cats"})
    r = d.record("search", {"q": "cats"})
    assert r.count == 2

def test_record_different_tools():
    d = CallDedup()
    d.record("search", {"q": "cats"})
    d.record("read", {"q": "cats"})
    assert len(d) == 2


# ---------------------------------------------------------------------------
# check_and_record
# ---------------------------------------------------------------------------

def test_check_and_record_first_is_true():
    d = CallDedup()
    assert d.check_and_record("tool", {"q": "test"}) is True

def test_check_and_record_second_is_false():
    d = CallDedup()
    d.check_and_record("tool", {"q": "test"})
    assert d.check_and_record("tool", {"q": "test"}) is False

def test_check_and_record_raise_mode():
    d = CallDedup(raise_on_duplicate=True)
    d.check_and_record("tool", {"q": "test"})
    with pytest.raises(DuplicateCall) as exc_info:
        d.check_and_record("tool", {"q": "test"})
    assert exc_info.value.tool_name == "tool"
    assert exc_info.value.call_count == 2

def test_duplicate_call_exception_attrs():
    d = CallDedup(raise_on_duplicate=True)
    d.check_and_record("search", {"q": "cats"})
    try:
        d.check_and_record("search", {"q": "cats"})
    except DuplicateCall as e:
        assert e.tool_name == "search"
        assert e.call_count == 2
        assert "cats" in e.input_repr


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_zero_unseen():
    d = CallDedup()
    assert d.count("tool", {"q": "test"}) == 0

def test_count_after_record():
    d = CallDedup()
    d.record("tool", {"q": "test"})
    assert d.count("tool", {"q": "test"}) == 1

def test_count_multiple():
    d = CallDedup()
    d.record("tool", {"q": "x"})
    d.record("tool", {"q": "x"})
    d.record("tool", {"q": "x"})
    assert d.count("tool", {"q": "x"}) == 3


# ---------------------------------------------------------------------------
# track_per_tool=False
# ---------------------------------------------------------------------------

def test_cross_tool_dedup():
    d = CallDedup(track_per_tool=False)
    d.record("search", {"q": "cats"})
    assert d.is_duplicate("read_file", {"q": "cats"})  # same input, different tool

def test_cross_tool_different_input_not_dup():
    d = CallDedup(track_per_tool=False)
    d.record("search", {"q": "cats"})
    assert not d.is_duplicate("read_file", {"q": "dogs"})


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def test_seen_calls():
    d = CallDedup()
    d.record("search", {"q": "a"})
    d.record("read", {"f": "b"})
    assert len(d.seen_calls()) == 2

def test_seen_calls_filter():
    d = CallDedup()
    d.record("search", {"q": "a"})
    d.record("read", {"f": "b"})
    assert len(d.seen_calls("search")) == 1

def test_unique_call_count():
    d = CallDedup()
    d.record("tool", {"q": "a"})
    d.record("tool", {"q": "b"})
    d.record("tool", {"q": "a"})  # dup
    assert d.unique_call_count() == 2

def test_total_call_count():
    d = CallDedup()
    d.record("tool", {"q": "a"})
    d.record("tool", {"q": "b"})
    d.record("tool", {"q": "a"})  # dup
    assert d.total_call_count() == 3

def test_duplicate_count():
    d = CallDedup()
    d.record("tool", {"q": "a"})
    d.record("tool", {"q": "a"})  # dup
    d.record("tool", {"q": "a"})  # dup
    assert d.duplicate_count() == 2

def test_summary():
    d = CallDedup()
    d.record("search", {"q": "a"})
    d.record("search", {"q": "a"})
    d.record("read", {"f": "x"})
    s = d.summary()
    assert s["unique_calls"] == 2
    assert s["total_calls"] == 3
    assert s["duplicate_calls"] == 1
    assert "search" in s["tools_seen"]


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_all():
    d = CallDedup()
    d.record("tool", {"q": "x"})
    d.reset()
    assert len(d) == 0

def test_reset_by_tool():
    d = CallDedup()
    d.record("search", {"q": "x"})
    d.record("read", {"f": "y"})
    d.reset("search")
    assert len(d) == 1
    assert not d.is_duplicate("search", {"q": "x"})
    assert d.is_duplicate("read", {"f": "y"})


# ---------------------------------------------------------------------------
# __contains__ / __len__
# ---------------------------------------------------------------------------

def test_contains_syntax():
    d = CallDedup()
    d.record("search", {"q": "cats"})
    assert ("search", {"q": "cats"}) in d
    assert ("search", {"q": "dogs"}) not in d

def test_len():
    d = CallDedup()
    assert len(d) == 0
    d.record("a", {})
    d.record("b", {})
    assert len(d) == 2
