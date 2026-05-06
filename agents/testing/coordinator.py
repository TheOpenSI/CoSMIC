"""Build a scalability-test coordinator with a subset of domain specialists."""

from __future__ import annotations

import os
import sys

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agents.deprecation_filters import apply_known_deprecation_filters

apply_known_deprecation_filters()

from google.adk.agents.llm_agent import Agent
from google.genai import types

import yaml

from agents.config import COORDINATOR_MODEL
from agents.testing.specialists import SPECIALISTS, get_specialist

_PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.yaml")
_TESTING_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_TESTING_ROOT, "data")

_VALID_TIERS = frozenset({3, 5, 8, 10})

# Appended automatically by ``build_root_agent``; coordinator agent_name must match exactly.
FALLBACK_ROUTING_SERVICE = "general_qa"


def _load_presets() -> dict:
    with open(_PRESETS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def tier_keys(tier: int) -> list[str]:
    """Return specialist service identifiers for tier 3, 5, 8, or 10 from presets."""
    if tier not in _VALID_TIERS:
        raise ValueError(f"tier must be one of {sorted(_VALID_TIERS)}, got {tier}")
    data = _load_presets()
    key = f"tier_{tier}"
    keys = data.get(key)
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"presets.yaml missing or empty '{key}'")
    out: list[str] = []
    for k in keys:
        if not isinstance(k, str):
            continue
        s = k.strip()
        if s and s not in out:
            out.append(s)
    if not out:
        raise ValueError(f"presets.yaml '{key}' has no valid services")
    for s in out:
        if s not in SPECIALISTS:
            raise KeyError(f"Unknown service identifier in presets.yaml {key}: {s!r}")
        if s == FALLBACK_ROUTING_SERVICE:
            raise ValueError(
                f"presets.yaml {key}: {FALLBACK_ROUTING_SERVICE!r} is reserved — omit it (added automatically)"
            )
    return out


def default_csv_path(tier: int) -> str:
    """Default CSV path for a tier; may not exist yet."""
    if tier not in _VALID_TIERS:
        raise ValueError(f"tier must be one of {sorted(_VALID_TIERS)}, got {tier}")
    data = _load_presets()
    override_key = f"csv_tier_{tier}"
    override = data.get(override_key)
    if isinstance(override, str) and override.strip():
        p = override.strip()
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(_project_root, p))
        return p
    name = f"prompts_tier{tier}.csv"
    return os.path.join(_DATA_DIR, name)


def tier_prediction_labels(tier: int) -> list[str]:
    """Row/column order for routing metrics: tier specialists then ``FALLBACK_ROUTING_SERVICE``."""
    return list(tier_keys(tier)) + [FALLBACK_ROUTING_SERVICE]


def build_root_agent(keys: list[str]) -> Agent:
    """Coordinator that routes to exactly one sub-agent: tier specialists plus ``general_qa`` fallback."""
    seen: list[str] = []
    for k in keys:
        if k not in SPECIALISTS:
            raise KeyError(f"Unknown service identifier: {k!r} (not in SPECIALISTS)")
        if k not in seen:
            seen.append(k)

    if FALLBACK_ROUTING_SERVICE in seen:
        raise ValueError(
            f"{FALLBACK_ROUTING_SERVICE!r} is reserved for the coordinator fallback; remove it from tier keys"
        )

    sub_agents: list[Agent] = [get_specialist(k) for k in seen]
    sub_agents.append(get_specialist(FALLBACK_ROUTING_SERVICE))

    all_names = [a.name for a in sub_agents]
    name_list = ", ".join(all_names)

    lines = "\n".join(
        f"- {a.name}: {a.description}"
        for a in sub_agents
    )
    instruction = (
        "You are a routing coordinator. Your ONLY job is to transfer the user's message "
        "to exactly one of the sub-agents listed below.\n\n"
        "## Available sub-agents\n"
        f"{lines}\n\n"
        "## Rules\n"
        f"1. You MUST transfer using `transfer_to_agent` with `agent_name` exactly matching one of: {name_list} "
        "(same spelling and underscores; never paraphrase or rename).\n"
        "2. Do NOT invent agent names or human-readable titles that are not listed above.\n"
        "3. Prefer the single best-matching domain specialist when the user's question clearly fits one domain.\n"
        "4. If no listed domain specialist clearly fits—or the question is off-topic or mixed—transfer to "
        f"`{FALLBACK_ROUTING_SERVICE}` exactly (fallback).\n"
        "5. Do NOT answer the question yourself — always delegate via one transfer.\n"
        "6. Once a specialist replies, return that reply to the user verbatim."
    )

    descriptions = "; ".join(f"{k}" for k in seen)

    return Agent(
        model=COORDINATOR_MODEL,
        name="scalability_coordinator",
        description=f"Routes to one domain specialist: {descriptions}",
        instruction=instruction,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            top_p=0.1,
            max_output_tokens=1024,
        ),
        sub_agents=sub_agents,
    )


def load_tier(tier: int) -> Agent:
    """Load presets for tier 3, 5, 8, or 10 and build the coordinator."""
    return build_root_agent(tier_keys(tier))
