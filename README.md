# tool-call-dedup

Session-scoped exact-duplicate detection for LLM tool calls.

Detects when an LLM agent calls the same tool with the same arguments twice, helping break infinite loops.

## Install

```bash
pip install tool-call-dedup
```

## Usage

```python
from tool_call_dedup import ToolCallDedup, DuplicateToolCallError

dedup = ToolCallDedup()

# Check before calling
if dedup.allow("search", session="sess:1", query="climate"):
    result = search(query="climate")  # only runs once
else:
    print("Skipping duplicate")

# Or raise automatically
try:
    dedup.require_unique("search", session="sess:1", query="climate")
    result = search(query="climate")
except DuplicateToolCallError as e:
    print(f"Duplicate blocked: {e}")
```

## API

### `ToolCallDedup(*, ttl=None, clock=None)`

| Parameter | Description |
|-----------|-------------|
| `ttl` | Optional time-to-live in seconds. Entries older than `ttl` are treated as new. |
| `clock` | Optional `() -> float` for deterministic testing. |

### Core

| Method | Returns | Description |
|--------|---------|-------------|
| `allow(tool, *, session=None, **kwargs)` | `bool` | `True` on first call; records it. `False` on duplicate. |
| `require_unique(tool, *, session, **kwargs)` | `None` | Like `allow()` but raises `DuplicateToolCallError`. |
| `is_duplicate(tool, *, session, **kwargs)` | `bool` | Check without recording. |
| `record(tool, *, session, **kwargs)` | `None` | Record without checking. |

### Queries

| Method | Returns | Description |
|--------|---------|-------------|
| `call_count(session=None)` | `int` | Distinct calls in this session. |
| `key_for(tool, **kwargs)` | `str` | Canonical dedup key (for inspection). |
| `sessions()` | `list` | All sessions with history. |

### Management

| Method | Returns | Description |
|--------|---------|-------------|
| `clear(session="_ALL_")` | `self` | Clear one session or all. |

## Note on kwargs

Arguments are hashed with `json.dumps(..., sort_keys=True)`, so order doesn't matter and the hash is stable across Python versions for JSON-serialisable values.

## License

MIT
