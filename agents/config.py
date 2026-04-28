"""Shared configuration for the ADK agents.

The model used by every sub-agent and the coordinator is centralised here so it
can be overridden in one place via the ``COSMIC_AGENT_MODEL`` environment
variable.

The default matches ``llm_name`` in ``scripts/configs/config.yaml``
(``llama3.1:8b``) so a typical CoSMIC / Ollama setup already has the model. For
stronger tool calling you can ``ollama pull qwen2.5:7b-instruct`` and set
``COSMIC_AGENT_MODEL=ollama_chat/qwen2.5:7b-instruct``.

The coordinator default matches specialists; override routing-only with
``COSMIC_COORDINATOR_MODEL`` if you want a different model.

Duplicate-tool-call limits: ``COSMIC_MAX_CALLS_PER_TOOL`` caps repeated identical
(tool, arguments) pairs in ``agents.safeguards``; ``COSMIC_RUNNER_MAX_CALLS_PER_TOOL_NAME``
and ``COSMIC_RUNNER_MAX_TOTAL_TOOL_CALLS`` tune the stream guard in ``agents.runner``.
"""

import os

# LiteLLM format: ollama_chat/<name> — name must exist in `ollama list`.
DEFAULT_MODEL = "ollama_chat/llama3.1:8b"

AGENT_MODEL = os.environ.get("COSMIC_AGENT_MODEL", DEFAULT_MODEL)

# Coordinator defaults to the same Ollama tag as specialists.
DEFAULT_COORDINATOR_MODEL = "ollama_chat/llama3.1:8b"
COORDINATOR_MODEL = os.environ.get("COSMIC_COORDINATOR_MODEL", DEFAULT_COORDINATOR_MODEL)

# ``loop_guard_before_tool`` (``agents/safeguards.py``): max invocations of the same
# (tool name + arguments) pair per turn. Set via ``COSMIC_MAX_CALLS_PER_TOOL``.
MAX_CALLS_PER_TOOL_PER_TURN = int(os.environ.get("COSMIC_MAX_CALLS_PER_TOOL", "3"))

# ``run_agent`` stream guard (``agents/runner.py``): separate limits that count every
# tool invocation by tool *name* (any arguments) and total calls across tools. These
# catch runaway loops the duplicate-args guard might miss. Defaults align the
# per-name cap with ``MAX_CALLS_PER_TOOL_PER_TURN`` for operator consistency only;
# the two mechanisms measure different things.
RUNNER_MAX_CALLS_PER_TOOL_NAME_PER_TURN = int(
    os.environ.get(
        "COSMIC_RUNNER_MAX_CALLS_PER_TOOL_NAME",
        str(MAX_CALLS_PER_TOOL_PER_TURN),
    )
)
RUNNER_MAX_TOTAL_TOOL_CALLS = int(os.environ.get("COSMIC_RUNNER_MAX_TOTAL_TOOL_CALLS", "10"))
