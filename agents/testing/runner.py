# -------------------------------------------------------------------------------------------------------------
# Scalability test runner: ADK Runner bound to tier-specific coordinators (not production root_agent).
# -------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import os
import sys
import uuid

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.config import RUNNER_MAX_CALLS_PER_TOOL_NAME_PER_TURN, RUNNER_MAX_TOTAL_TOOL_CALLS
from agents.testing.coordinator_factory import load_tier

APP_NAME = "cosmic-scalability-test"

_session_service = InMemorySessionService()
_runners: dict[int, Runner] = {}


def _resolve_tier(tier: int | None) -> int:
    if tier is not None:
        return tier
    raw = os.environ.get("COSMIC_TEST_TIER", "3").strip()
    return int(raw)


def _get_runner(tier: int) -> Runner:
    if tier not in _runners:
        agent = load_tier(tier)
        _runners[tier] = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=_session_service,
        )
    return _runners[tier]


def clear_runner_cache() -> None:
    """Drop cached runners (e.g. after presets or specialists change)."""
    _runners.clear()


async def run_agent(
    user_id: str,
    query: str,
    context: str = "",
    vector_db_path: str = "",
    tier: int | None = None,
) -> str:
    """Run the scalability-test coordinator for ``tier`` (3, 5, or 8).

    If ``tier`` is None, uses environment variable ``COSMIC_TEST_TIER`` (default ``3``).
    """
    t = _resolve_tier(tier)
    if t not in (3, 5, 8):
        raise ValueError("tier must be 3, 5, or 8")

    runner = _get_runner(t)
    session_id = str(uuid.uuid4())

    initial_state: dict = {}
    if vector_db_path:
        initial_state["vector_db_path"] = vector_db_path

    await _session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    if context:
        message_text = f"Previous conversation context:\n{context}\n\nCurrent question: {query}"
    else:
        message_text = query

    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message_text)],
    )

    final_text = ""
    tool_call_counts: dict[str, int] = {}
    total_tool_calls = 0
    aborted_reason = ""

    runner_iter = runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    )

    async for event in runner_iter:
        for fc in event.get_function_calls():
            tool_name = fc.name or "<unknown>"
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            total_tool_calls += 1

            if tool_call_counts[tool_name] > RUNNER_MAX_CALLS_PER_TOOL_NAME_PER_TURN:
                aborted_reason = (
                    f"Aborted: tool '{tool_name}' was called "
                    f"{tool_call_counts[tool_name]} times in a single turn "
                    f"(limit: {RUNNER_MAX_CALLS_PER_TOOL_NAME_PER_TURN}). The model appears to be stuck "
                    "in a loop. Please rephrase your request."
                )
                break
            if total_tool_calls > RUNNER_MAX_TOTAL_TOOL_CALLS:
                aborted_reason = (
                    f"Aborted: exceeded total tool-call limit "
                    f"({RUNNER_MAX_TOTAL_TOOL_CALLS}) in a single turn. The model appears "
                    "to be stuck in a loop. Please rephrase your request."
                )
                break

        if aborted_reason:
            break

        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text += part.text

    if aborted_reason:
        try:
            await runner_iter.aclose()
        except Exception:
            pass
        return final_text + ("\n\n" if final_text else "") + aborted_reason

    return final_text or "No response generated."


def run_agent_sync(
    user_id: str,
    query: str,
    context: str = "",
    vector_db_path: str = "",
    tier: int | None = None,
) -> str:
    """Synchronous wrapper around :func:`run_agent` for scalability harness."""
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None:
        raise RuntimeError(
            "run_agent_sync() cannot be called from a running event loop. "
            "Use 'await run_agent(...)' instead."
        )

    return asyncio.run(
        run_agent(
            user_id=user_id,
            query=query,
            context=context,
            vector_db_path=vector_db_path,
            tier=tier,
        )
    )
