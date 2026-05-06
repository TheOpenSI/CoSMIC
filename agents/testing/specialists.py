"""Scalability-testing domain specialists (LLM-only, no tools).

Each agent mirrors one evaluation subject. Keys in :data:`ALL_KEYS` follow the
nested-prefix tier convention in ``presets.yaml``.
"""

import os
import sys

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agents.deprecation_filters import apply_known_deprecation_filters

apply_known_deprecation_filters()

from google.adk.agents.llm_agent import Agent
from google.genai import types

from agents.config import AGENT_MODEL

_GEN = types.GenerateContentConfig(
    temperature=0.1,
    top_p=0.9,
    max_output_tokens=1024,
)


def _make_agent(name: str, description: str, domain_line: str) -> Agent:
    return Agent(
        model=AGENT_MODEL,
        name=name,
        description=description,
        instruction=(
            f"You are a specialist in {domain_line}. Answer clearly and concisely from that domain only. "
            "If the question is outside your domain, say briefly that it is not your specialty."
        ),
        generate_content_config=_GEN,
    )


def get_specialist(key: str) -> Agent:
    """Return a fresh instance of a specialist agent to avoid parent-conflict errors."""
    if key == "abstract_algebra":
        return _make_agent(
            "abstract_algebra",
            (
                "Contains theoretical mathematics problems focused on algebraic structures, used to evaluate "
                "abstract reasoning routing."
            ),
            "abstract algebra and algebraic structures",
        )
    if key == "anatomy":
        return _make_agent(
            "anatomy",
            (
                "Includes questions about human body structure and systems, helping identify life science "
                "and medical queries."
            ),
            "human anatomy, body structure, and organ systems",
        )
    if key == "astronomy":
        return _make_agent(
            "astronomy",
            (
                "Covers celestial objects and space-related concepts, supporting routing for physics-oriented queries."
            ),
            "astronomy, celestial mechanics, and space science",
        )
    if key == "business_ethics":
        return _make_agent(
            "business_ethics",
            (
                "Consists of ethical decision-making scenarios in business contexts, useful for social science reasoning."
            ),
            "business ethics and professional ethical decision-making",
        )
    if key == "clinical_knowledge":
        return _make_agent(
            "clinical_knowledge",
            (
                "Covers patient-centered medical scenarios involving diagnosis, symptoms, or treatment decisions, "
                "used to route clinical reasoning tasks."
            ),
            "clinical medicine: diagnosis, symptoms, and treatment decisions in patient scenarios",
        )
    if key == "college_biology":
        return _make_agent(
            "college_biology",
            (
                "Focuses on theoretical and conceptual biology such as genetics, evolution, and cellular processes, "
                "without clinical or patient context."
            ),
            "college-level biology: genetics, evolution, cell biology, and related theory (not bedside clinical care)",
        )
    if key == "college_chemistry":
        return _make_agent(
            "college_chemistry",
            (
                "Includes chemistry problems involving reactions, equations, and physical or organic principles, "
                "supporting chemistry-specific query routing."
            ),
            "college chemistry: reactions, stoichiometry, physical and organic principles",
        )
    if key == "college_computer_science":
        return _make_agent(
            "college_computer_science",
            (
                "Covers algorithms and data structures, used for routing technical and computational queries."
            ),
            "algorithms, data structures, and theoretical computer science",
        )
    if key == "mathematics":
        return _make_agent(
            "mathematics",
            (
                "Contains general math problems across topics, supporting quantitative reasoning routing."
            ),
            "general mathematics and quantitative reasoning",
        )
    if key == "medicine":
        return _make_agent(
            "medicine",
            (
                "Includes broad medical knowledge questions, enabling routing for healthcare-related queries."
            ),
            "general medicine and healthcare knowledge",
        )
    if key == "general_qa":
        return Agent(
            model=AGENT_MODEL,
            name="general_qa",
            description=(
                "Fallback when no domain specialist is a clear match: mixed or off-topic queries, casual questions, "
                "or topics that do not fit the listed domains."
            ),
            instruction=(
                "You receive questions that the router could not assign to a single domain specialist. "
                "Answer helpfully and concisely in plain language."
            ),
            generate_content_config=_GEN,
        )
    raise KeyError(f"Unknown specialist: {key}")


# Fixed table order (nested-prefix tiers use prefixes of this list).
ALL_KEYS: list[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "mathematics",
    "medicine",
]

# For backwards compatibility in coordinator.py lookup checks
SPECIALISTS = frozenset(ALL_KEYS + ["general_qa"])
