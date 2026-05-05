# -------------------------------------------------------------------------------------------------------------
# Scalability test runner: ADK Runner bound to tier-specific coordinators (not production root_agent).
# -------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass

_log = logging.getLogger(__name__)

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agents.deprecation_filters import apply_known_deprecation_filters

apply_known_deprecation_filters()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.config import RUNNER_MAX_CALLS_PER_TOOL_NAME_PER_TURN, RUNNER_MAX_TOTAL_TOOL_CALLS
from agents.testing.coordinator import load_tier

APP_NAME = "cosmic_scalability_test"

_session_service = InMemorySessionService()
_runners: dict[int, Runner] = {}

_VALID_TIERS = frozenset({3, 5, 8, 10})


def _ensure_agents_testing_log_handler() -> None:
    """If nothing configured ``agents.testing``, attach stderr INFO logging (library-style default)."""
    pkg = logging.getLogger("agents.testing")
    if pkg.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(fmt)
    pkg.addHandler(h)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False


@dataclass(frozen=True)
class RoutingOutcome:
    """Result of running the scalability-test coordinator once."""

    final_text: str
    predicted_service: str | None
    latency_ms: float
    tokens_total: int
    aborted_reason: str


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


def _function_call_args_to_dict(fc) -> dict[str, object]:
    args = getattr(fc, "args", None)
    if args is None:
        return {}
    if isinstance(args, dict):
        return args  # type: ignore[return-value]
    if hasattr(args, "model_dump"):
        return args.model_dump()  # type: ignore[no-any-return]
    if hasattr(args, "items"):
        try:
            return dict(args.items())
        except Exception:
            pass
    return {}


def _transfer_target_from_event(event) -> str | None:
    actions = getattr(event, "actions", None)
    if actions:
        transferred = getattr(actions, "transfer_to_agent", None)
        if transferred:
            return str(transferred)
    for fc in event.get_function_calls():
        if fc.name != "transfer_to_agent":
            continue
        args_map = _function_call_args_to_dict(fc)
        raw_name = args_map.get("agent_name")
        if raw_name:
            return str(raw_name)
    return None


async def _run_agent_inner(
    user_id: str,
    query: str,
    context: str = "",
    vector_db_path: str = "",
    tier: int | None = None,
    *,
    verbose: bool = False,
) -> RoutingOutcome:
    """Run coordinator once; compute predicted routing transfer, timings, and token totals."""
    _ensure_agents_testing_log_handler()
    t = _resolve_tier(tier)
    if t not in _VALID_TIERS:
        raise ValueError(f"tier must be one of {sorted(_VALID_TIERS)}, got {t}")
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

    predicted_service: str | None = None
    tokens_accum = 0

    t_start = time.perf_counter()

    runner_iter = runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    )

    try:
        async for event in runner_iter:
            xfer = _transfer_target_from_event(event)
            if xfer is not None and predicted_service is None:
                predicted_service = xfer

            if verbose and _log.isEnabledFor(logging.DEBUG):
                tool_names = [getattr(fc, "name", None) or "<unknown>" for fc in event.get_function_calls()]
                _log.debug(
                    "ADK event final=%s transfer_target=%s tools=%s tokens_this_event=%s",
                    event.is_final_response(),
                    xfer,
                    tool_names,
                    getattr(getattr(event, "usage_metadata", None), "total_token_count", None),
                )

            um = getattr(event, "usage_metadata", None)
            if um is not None:
                tt = getattr(um, "total_token_count", None)
                if isinstance(tt, int) and tt > 0:
                    tokens_accum += tt

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
    except ValueError as exc:
        aborted_reason = f"Routing error (hallucinated agent name): {exc}"
        _log.warning("%s", aborted_reason)

    latency_ms = (time.perf_counter() - t_start) * 1000

    if aborted_reason:
        try:
            await runner_iter.aclose()
        except Exception:
            pass

    if aborted_reason:
        text_out = final_text + ("\n\n" if final_text else "") + aborted_reason
    else:
        text_out = final_text or "No response generated."

    return RoutingOutcome(
        final_text=text_out,
        predicted_service=predicted_service,
        latency_ms=latency_ms,
        tokens_total=tokens_accum,
        aborted_reason=aborted_reason,
    )


async def run_agent(
    user_id: str,
    query: str,
    context: str = "",
    vector_db_path: str = "",
    tier: int | None = None,
) -> str:
    """Run the scalability-test coordinator for ``tier`` (3, 5, 8, or 10).

    If ``tier`` is None, uses environment variable ``COSMIC_TEST_TIER`` (default ``3``).
    """
    outcome = await _run_agent_inner(
        user_id=user_id,
        query=query,
        context=context,
        vector_db_path=vector_db_path,
        tier=tier,
        verbose=False,
    )
    return outcome.final_text


async def run_agent_routing(
    user_id: str,
    query: str,
    context: str = "",
    vector_db_path: str = "",
    tier: int | None = None,
    *,
    verbose: bool = False,
) -> RoutingOutcome:
    """Like :func:`run_agent` but returns structured routing metrics (predicted service, latency, tokens)."""
    return await _run_agent_inner(
        user_id=user_id,
        query=query,
        context=context,
        vector_db_path=vector_db_path,
        tier=tier,
        verbose=verbose,
    )


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
