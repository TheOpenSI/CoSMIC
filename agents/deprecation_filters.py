# -------------------------------------------------------------------------------------------------------------
# Centralized warning filters for predictable dependency deprecations during ADK / CLI runs.
# -------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import warnings

_initialized = False


def apply_known_deprecation_filters() -> None:
    """Install ``warnings`` filters once per process.

    CoSMIC pulls in Google ADK, client libraries, Pydantic, LiteLLM, and HTTP stacks that
    may emit ``DeprecationWarning`` for upcoming API changes. Those are not actionable in
    this repository; filtering keeps logs readable without hiding ``UserWarning`` or other
    categories by default.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    for mod in (
        r"google\.genai",
        r"google\.api_core",
        r"google\.auth",
        r"google\.cloud",
        r"google\.protobuf",
        "pydantic",
        "pydantic_core",
        "litellm",
        "httpx",
        "httpcore",
    ):
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=mod,
        )
