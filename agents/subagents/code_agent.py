import os
import sys

from google.adk.agents.llm_agent import Agent
from google.genai import types

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from modules.code_generation.code_generation import CodeGenerator
from pydantic import ValidationError

from agents.config import DEFAULT_MODEL
from agents.safeguards import loop_guard_before_tool
from agents.tool_schemas import GenerateCodeInput, validation_error_response

# Lazy singleton for the code generator to avoid connecting at import time.
_code_generator = None


def _get_code_generator() -> CodeGenerator:
    """Lazily initialise the shared CodeGenerator instance."""
    global _code_generator
    if _code_generator is None:
        _code_generator = CodeGenerator()
    return _code_generator


# Tool functions

# Keep tool output bounded so the LLM context is not flooded by long generations.
_MAX_CODE_CHARS = 12_000


def generate_code(query: str) -> dict:
    """Generate Python or C++ code from a natural-language request.

    Args:
        query: What to build or change. Required.

    Returns:
        dict with status, code, explanation, and an optional truncated flag.

    Example:
        ``query``: ``"Read a CSV of numbers and print the row sums as a list."``
    """
    try:
        valid = GenerateCodeInput.model_validate({"query": query})
    except ValidationError as e:
        return validation_error_response(e)
    try:
        generator = _get_code_generator()
        code, raw_response = generator(valid.query)
        if not code or not code.strip():
            return {
                "status": "error",
                "error": "The code generator returned an empty response. Ask the user to rephrase.",
            }
        explanation = raw_response.replace(code, "").strip()
        truncated = False
        if len(code) > _MAX_CODE_CHARS:
            code = code[:_MAX_CODE_CHARS] + "\n\n# ... truncated ..."
            truncated = True
            explanation = (
                (explanation + " " if explanation else "")
                + "Output exceeded size limit; only the first portion is returned."
            ).strip()
        payload = {
            "status": "success",
            "code": code,
            "explanation": explanation,
        }
        if truncated:
            payload["truncated"] = True
        return payload
    except Exception as e:
        return {
            "status": "error",
            "error": f"Code generation failed: {e}. Ask the user to rephrase or try a simpler query.",
        }


# Agent definition

code_agent = Agent(
    model=DEFAULT_MODEL,
    name="code_agent",
    description=(
        "Specialist for code generation and improvement. Handles requests to write, "
        "generate, improve, debug, or refactor Python or C++ code."
    ),
    instruction=(
        "You help users with Python and C++: new code, fixes, refactors, and small programs.\n"
        "If the request is unclear, ask one clarifying question before using tools.\n"
        "After a tool returns status=success, present the code in a fenced block with a short summary, then "
        "stop. Do NOT call any tool again in the same turn.\n"
        "If a tool returns status=error or truncated=true, follow the message and do not repeat the call."
    ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=2048,
    ),
    tools=[generate_code],
    before_tool_callback=loop_guard_before_tool,
)
