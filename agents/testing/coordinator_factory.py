"""Build a scalability-test coordinator with a subset of domain specialists."""

from __future__ import annotations

import os
import sys

from google.adk.agents.llm_agent import Agent
from google.genai import types

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml

from agents.config import COORDINATOR_MODEL
from agents.testing.specialists import SPECIALISTS

_PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.yaml")
_TESTING_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_TESTING_ROOT, "data")

_VALID_TIERS = frozenset({3, 5, 8})


def _load_presets() -> dict:
    with open(_PRESETS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def tier_keys(tier: int) -> list[str]:
    """Return specialist slug list for tier 3, 5, or 8 from presets."""
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
        raise ValueError(f"presets.yaml '{key}' has no valid slugs")
    for s in out:
        if s not in SPECIALISTS:
            raise KeyError(f"Unknown specialist slug in presets.yaml {key}: {s!r}")
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


def build_root_agent(keys: list[str]) -> Agent:
    """Coordinator that routes to exactly one of the given specialist slugs."""
    seen: list[str] = []
    for k in keys:
        if k not in SPECIALISTS:
            raise KeyError(f"Unknown specialist slug: {k!r} (not in SPECIALISTS)")
        if k not in seen:
            seen.append(k)

    lines = "\n".join(f"- {k}: use for questions matching that specialist's domain (see sub-agent description)." for k in seen)
    instruction = (
        "Scalability-test coordinator: route each user message to exactly one specialist below. "
        "Pick the single best-matching domain.\n"
        f"{lines}\n"
        "Do not answer yourself. Once a specialist replies, return that reply to the user verbatim and "
        "do not call another agent for the same turn."
    )

    sub_agents = [SPECIALISTS[k] for k in seen]
    descriptions = "; ".join(f"{k}" for k in seen)

    return Agent(
        model=COORDINATOR_MODEL,
        name="scalability_coordinator",
        description=f"Routes to one domain specialist: {descriptions}",
        instruction=instruction,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.9,
            max_output_tokens=1024,
        ),
        sub_agents=sub_agents,
    )


def load_tier(tier: int) -> Agent:
    """Load presets for tier 3, 5, or 8 and build the coordinator."""
    return build_root_agent(tier_keys(tier))
