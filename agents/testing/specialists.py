"""Scalability-testing domain specialists (LLM-only, no tools).

Each agent mirrors one evaluation subject. Keys in :data:`ALL_KEYS` follow the
nested-prefix tier convention in ``presets.yaml``.
"""

import os
import sys

from google.adk.agents.llm_agent import Agent
from google.genai import types

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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


abstract_algebra_agent = _make_agent(
    "abstract_algebra",
    (
        "Contains theoretical mathematics problems focused on algebraic structures, used to evaluate "
        "abstract reasoning routing."
    ),
    "abstract algebra and algebraic structures",
)

anatomy_agent = _make_agent(
    "anatomy",
    (
        "Includes questions about human body structure and systems, helping identify life science "
        "and medical queries."
    ),
    "human anatomy, body structure, and organ systems",
)

astronomy_agent = _make_agent(
    "astronomy",
    (
        "Covers celestial objects and space-related concepts, supporting routing for physics-oriented queries."
    ),
    "astronomy, celestial mechanics, and space science",
)

business_ethics_agent = _make_agent(
    "business_ethics",
    (
        "Consists of ethical decision-making scenarios in business contexts, useful for social science reasoning."
    ),
    "business ethics and professional ethical decision-making",
)

clinical_knowledge_agent = _make_agent(
    "clinical_knowledge",
    (
        "Covers patient-centered medical scenarios involving diagnosis, symptoms, or treatment decisions, "
        "used to route clinical reasoning tasks."
    ),
    "clinical medicine: diagnosis, symptoms, and treatment decisions in patient scenarios",
)

college_biology_agent = _make_agent(
    "college_biology",
    (
        "Focuses on theoretical and conceptual biology such as genetics, evolution, and cellular processes, "
        "without clinical or patient context."
    ),
    "college-level biology: genetics, evolution, cell biology, and related theory (not bedside clinical care)",
)

college_chemistry_agent = _make_agent(
    "college_chemistry",
    (
        "Includes chemistry problems involving reactions, equations, and physical or organic principles, "
        "supporting chemistry-specific query routing."
    ),
    "college chemistry: reactions, stoichiometry, physical and organic principles",
)

college_computer_science_agent = _make_agent(
    "college_computer_science",
    (
        "Covers algorithms and data structures, used for routing technical and computational queries."
    ),
    "algorithms, data structures, and theoretical computer science",
)

mathematics_agent = _make_agent(
    "mathematics",
    (
        "Contains general math problems across topics, supporting quantitative reasoning routing."
    ),
    "general mathematics and quantitative reasoning",
)

medicine_agent = _make_agent(
    "medicine",
    (
        "Includes broad medical knowledge questions, enabling routing for healthcare-related queries."
    ),
    "general medicine and healthcare knowledge",
)

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

SPECIALISTS: dict[str, Agent] = {
    "abstract_algebra": abstract_algebra_agent,
    "anatomy": anatomy_agent,
    "astronomy": astronomy_agent,
    "business_ethics": business_ethics_agent,
    "clinical_knowledge": clinical_knowledge_agent,
    "college_biology": college_biology_agent,
    "college_chemistry": college_chemistry_agent,
    "college_computer_science": college_computer_science_agent,
    "mathematics": mathematics_agent,
    "medicine": medicine_agent,
}
