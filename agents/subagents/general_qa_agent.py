import os
import sys
import yaml
import torch

from google.adk.agents.llm_agent import Agent
from google.adk.agents.context import Context
from google.genai import types

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pydantic import ValidationError

from box import Box
from src.services.vector_database import VectorDatabase
from src.services.rag import RAGBase
from agents.config import DEFAULT_MODEL
from agents.safeguards import loop_guard_before_tool
from agents.tool_schemas import RetrieveContextInput, validation_error_response

# Load config.
_CONFIG_PATH = os.path.join(_project_root, "scripts", "configs", "config.yaml")
_config = Box.from_yaml(filename=_CONFIG_PATH, Loader=yaml.FullLoader) if os.path.exists(_CONFIG_PATH) else None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_default_vector_db_path = _config.rag.vector_db_path if _config else ""

# Per-user RAG instances keyed by vector db path.
_rag_instances: dict[str, RAGBase] = {}


def _get_rag(vector_db_path: str = "") -> RAGBase:
    """Get or create a RAG instance for the given vector DB path."""
    path = vector_db_path or _default_vector_db_path
    if path not in _rag_instances:
        vdb = VectorDatabase(local_database_path=path, device=_device)
        topk = _config.rag.topk if _config else 10
        threshold = _config.rag.retrieve_score_threshold if _config else 0.7
        _rag_instances[path] = RAGBase(
            vector_database=vdb, retrieve_score_threshold=threshold, topk=topk
        )
    return _rag_instances[path]


def _resolve_vector_db_path(tool_context: Context) -> str:
    """Read per-user vector_db_path from session state, falling back to config."""
    return tool_context.state.get("vector_db_path", _default_vector_db_path)


# Tool functions

_MAX_CONTEXT_CHARS_PER_PASSAGE = 500
_MAX_PASSAGES = 5


def retrieve_context(query: str, tool_context: Context) -> dict:
    """Retrieve up to five short passages from the user's saved knowledge.

    Args:
        query: The user's question in plain language.
        tool_context: Session context (injected by ADK).

    Returns:
        dict with status, passages (may be empty), and matching scores.

    Example:
        ``query``: ``"How does ingestion work in this project?"`` when the user's library may contain the answer.
    """
    try:
        valid = RetrieveContextInput.model_validate({"query": query})
    except ValidationError as e:
        return validation_error_response(e)
    try:
        rag = _get_rag(_resolve_vector_db_path(tool_context))
        context, scores = rag(valid.query)

        passages = []
        if context:
            for i, chunk in enumerate(context.split("Document ")[1:]):
                if i >= _MAX_PASSAGES:
                    break
                text = chunk.split(": ", 1)[-1].strip()
                if len(text) > _MAX_CONTEXT_CHARS_PER_PASSAGE:
                    text = text[:_MAX_CONTEXT_CHARS_PER_PASSAGE] + "..."
                passages.append(text)

        filtered_scores = [round(s, 4) for s in scores[:_MAX_PASSAGES]]
        return {"status": "success", "passages": passages, "scores": filtered_scores}
    except Exception as e:
        return {
            "status": "error",
            "error": (
                f"Knowledge base retrieval failed: {e}. "
                "Answer from your own knowledge and let the user know the knowledge base was unavailable."
            ),
        }


# Agent definition

general_qa_agent = Agent(
    model=DEFAULT_MODEL,
    name="general_qa_agent",
    description=(
        "Specialist for general knowledge, OpenSI-CoSMIC product questions, and conversational answers "
        "grounded in the user's saved documents (use retrieve_context for that—not the database agent's "
        "search_database tool)."
    ),
    instruction=(
        "You are OpenSI-CoSMIC: general Q&A, system information, and answers grounded in the user's saved "
        "knowledge when relevant. Project link: https://github.com/TheOpenSI/CoSMIC (Open Source Institute, "
        "University of Canberra).\n"
        "When a stored-documents answer matters, call retrieve_context once. After it returns, reply in your "
        "own words using the passages and cite that excerpts came from the user's library if applicable. Do "
        "NOT call any tool again in the same turn.\n"
        "If retrieval fails or is empty, say so and answer from general knowledge where appropriate."
    ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=2048,
    ),
    tools=[retrieve_context],
    before_tool_callback=loop_guard_before_tool,
)
