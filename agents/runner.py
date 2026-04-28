# -------------------------------------------------------------------------------------------------------------
# File: runner.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
#
# Copyright (c) 2024 Open Source Institute
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import asyncio
import os
import sys
import uuid

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.config import RUNNER_MAX_CALLS_PER_TOOL_NAME_PER_TURN, RUNNER_MAX_TOTAL_TOOL_CALLS
from agents.coordinator import root_agent

APP_NAME = "cosmic"

_session_service = InMemorySessionService()
_runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=_session_service,
)

# Limits are ``RUNNER_*`` from ``agents.config``: per tool *name* (any arguments) and
# total tool calls per ``run_async`` stream. They complement ``loop_guard_before_tool``
# in ``agents/safeguards.py``, which limits repeated identical (tool, arguments) pairs
# (``MAX_CALLS_PER_TOOL_PER_TURN``).


async def run_agent(
    user_id: str,
    query: str,
    context: str = "",
    vector_db_path: str = "",
) -> str:
    """Run the ADK coordinator agent and return the final text response.

    Args:
        user_id: Unique user identifier (used for session scoping).
        query: The user's question.
        context: Optional chat history context to prepend.
        vector_db_path: Optional per-user vector DB path for session state.

    Returns:
        The agent's final text response.
    """
    session_id = str(uuid.uuid4())

    initial_state = {}
    if vector_db_path:
        initial_state["vector_db_path"] = vector_db_path

    await _session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    if context:
        message_text = (
            f"Previous conversation context:\n{context}\n\n"
            f"Current question: {query}"
        )
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

    runner_iter = _runner.run_async(
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
) -> str:
    """Synchronous wrapper around :func:`run_agent` for non-async callers.

    Use this from CLI entry points (``main.py``, ``demo.py``,
    ``modules/docker/main_docker.py``) and from synchronous code paths in
    ``OpenSICoSMIC.__call__`` when routing through the ADK pipeline. Internally
    starts a fresh event loop via ``asyncio.run``; therefore this MUST NOT be
    called from inside an already-running event loop (FastAPI request handlers,
    asyncio tasks). Async callers should ``await run_agent(...)`` directly.

    Args:
        user_id: Unique user identifier (used for session scoping).
        query: The user's question.
        context: Optional chat history context to prepend.
        vector_db_path: Optional per-user vector DB path for session state.

    Returns:
        The agent's final text response.

    Raises:
        RuntimeError: If invoked while an asyncio event loop is already running
            in the current thread.
    """
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
        )
    )
