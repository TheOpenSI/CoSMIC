"""ADK ``before_tool_callback`` safeguards shared across every sub-agent.

The loop guard counts how many times each ``(tool_name, arguments)`` pair has
been invoked within the current turn (using ``tool_context.state``). ADK
``>=1.31`` passes ``tool_context=`` (see ``functions.handle_function_calls``).
Once the count
exceeds ``MAX_CALLS_PER_TOOL_PER_TURN``, the callback returns a synthetic
error dict instead of executing the tool again. ADK feeds that dict back to
the model as the tool response, breaking the runaway tool-call loop.

This works under both ``adk web`` and the project's own ``Runner`` in
``agents/runner.py`` because callbacks run inside the agent itself.

``agents/runner.py`` additionally enforces separate limits
(``RUNNER_MAX_CALLS_PER_TOOL_NAME_PER_TURN``, ``RUNNER_MAX_TOTAL_TOOL_CALLS``
in ``agents.config``): those count calls by tool name / total regardless of args.
"""

import hashlib
import json
from typing import Any, Optional

from agents.config import MAX_CALLS_PER_TOOL_PER_TURN

_STATE_KEY = "__tool_call_counts__"


def _signature(tool_name: str, args: dict[str, Any]) -> str:
    """Hash the (tool, args) pair to a short signature for counting."""
    try:
        blob = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        blob = repr(args)
    return f"{tool_name}:{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]}"


def loop_guard_before_tool(tool, args: dict[str, Any], tool_context) -> Optional[dict]:
    """Short-circuit duplicate (tool, args) calls in the same turn.

    Signature matches ADK 1.31+ ``canonical_before_tool_callbacks``:
    ``callback(tool=..., args=..., tool_context=...)``.

    Returns ``None`` to let the tool run, or an error dict to stop it.
    """
    counts = tool_context.state.get(_STATE_KEY)
    if not isinstance(counts, dict):
        counts = {}

    sig = _signature(getattr(tool, "name", "<unknown>"), args or {})
    counts[sig] = counts.get(sig, 0) + 1
    tool_context.state[_STATE_KEY] = counts

    if counts[sig] > MAX_CALLS_PER_TOOL_PER_TURN:
        return {
            "status": "error",
            "error": (
                f"Loop guard: '{getattr(tool, 'name', 'tool')}' has already been called "
                f"{counts[sig]} times this turn with the same arguments. "
                "Do NOT call this tool again. Reply to the user using the "
                "previous tool result already in this conversation."
            ),
        }
    return None
