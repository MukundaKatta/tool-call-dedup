# tool-call-dedup

Detect exact duplicate tool calls within an agent session. Zero dependencies.

```python
from tool_call_dedup import CallDedup

dedup = CallDedup()

for tool_use in response.tool_uses:
    if not dedup.check_and_record(tool_use.name, tool_use.input):
        continue   # already ran this exact call
    result = call_tool(tool_use)
```

## Install

```bash
pip install tool-call-dedup
```
