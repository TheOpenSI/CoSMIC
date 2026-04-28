import os
import sys

import yaml
import torch
from pydantic import ValidationError

from google.adk.agents.llm_agent import Agent
from google.adk.agents.context import Context
from google.genai import types

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from box import Box
from src.services.vector_database import VectorDatabase
from agents.config import DEFAULT_MODEL
from agents.safeguards import loop_guard_before_tool
from agents.tool_schemas import (
    SearchDatabaseInput,
    UpdateDatabaseDocumentInput,
    UpdateDatabaseTextInput,
    validation_error_response,
)

# Load config.
_CONFIG_PATH = os.path.join(_project_root, "scripts", "configs", "config.yaml")
_config = Box.from_yaml(filename=_CONFIG_PATH, Loader=yaml.FullLoader) if os.path.exists(_CONFIG_PATH) else None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_default_vector_db_path = _config.rag.vector_db_path if _config else ""

# Per-user vector database instances keyed by db path.
_vector_databases: dict[str, VectorDatabase] = {}


def _get_vector_database(vector_db_path: str = "") -> VectorDatabase:
    """Get or create a VectorDatabase for the given path."""
    path = vector_db_path or _default_vector_db_path
    if path not in _vector_databases:
        _vector_databases[path] = VectorDatabase(
            local_database_path=path,
            device=_device,
        )
    return _vector_databases[path]


def _resolve_vector_db_path(tool_context: Context) -> str:
    """Read per-user vector_db_path from session state, falling back to config."""
    return tool_context.state.get("vector_db_path", _default_vector_db_path)


# Tool functions

def update_database_from_document(document_path: str, tool_context: Context) -> dict:
    """Index one PDF into the user's knowledge base.

    Args:
        document_path: Path to a .pdf file (absolute or project-relative).
        tool_context: Session context (injected by ADK).

    Returns:
        dict with status and a message; on success, the file is searchable via search_database.

    Example:
        ``document_path``: ``"backend/data/paper.pdf"`` or an absolute Windows path starting with ``C:\\...``.
    """
    try:
        valid_doc = UpdateDatabaseDocumentInput.model_validate({"document_path": document_path})
    except ValidationError as e:
        return validation_error_response(e)
    try:
        vdb = _get_vector_database(_resolve_vector_db_path(tool_context))
        path_value = valid_doc.document_path
        if not os.path.isabs(path_value):
            path_value = os.path.join(_project_root, path_value)
        vdb.update_database_from_document(document_path=path_value)
        return {
            "status": "success",
            "message": f"Document indexed: {path_value}",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to ingest document: {e}. Verify the file path is a valid PDF.",
        }


def update_database_from_text(text: str, tool_context: Context) -> dict:
    """Store plain text in the user's knowledge base.

    Args:
        text: Content to remember.
        tool_context: Session context (injected by ADK).

    Returns:
        dict with status and a message; near-duplicate text is reported as status=skipped.

    Example:
        ``text``: ``"Key idea: rollout policy uses softmax over legal moves."``.
    """
    try:
        valid = UpdateDatabaseTextInput.model_validate({"text": text})
    except ValidationError as e:
        return validation_error_response(e)
    try:
        vdb = _get_vector_database(_resolve_vector_db_path(tool_context))
        result = vdb.update_database_from_text(text=valid.text)
        if result == -1:
            return {"status": "skipped", "message": "Similar content already exists in the knowledge base."}
        return {
            "status": "success",
            "message": "Text added to the knowledge base.",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to add text: {e}. Ask the user to try again.",
        }


_MAX_SEARCH_RESULT_CHARS = 500


def search_database(
    query: str,
    topk: int = 5,
    *,
    tool_context: Context,
) -> dict:
    """Find short passages from the user's knowledge base matching a query.

    Uses session state for which vector DB to query; ``tool_context`` is required (injected).

    Args:
        query: Question or keywords.
        topk: How many passages to return (default 5, max 50).
        tool_context: Session context (injected by ADK).

    Returns:
        dict with status, query, and a list of {content, score} results (may be empty).

    Example:
        ``query``: ``"vector database ingestion"``, ``topk``: ``3``.
    """
    try:
        valid = SearchDatabaseInput.model_validate({"query": query, "topk": topk})
    except ValidationError as e:
        return validation_error_response(e)
    try:
        db_path = _resolve_vector_db_path(tool_context)
        vdb = _get_vector_database(db_path)
        results = vdb.similarity_search_with_relevance_scores(query=valid.query, k=valid.topk)
        documents = []
        for doc, score in results:
            content = doc.page_content
            if len(content) > _MAX_SEARCH_RESULT_CHARS:
                content = content[:_MAX_SEARCH_RESULT_CHARS] + "..."
            documents.append({"content": content, "score": round(score, 4)})
        if not documents:
            return {
                "status": "success",
                "query": valid.query,
                "results": [],
                "message": "No matching documents found.",
            }
        return {"status": "success", "query": valid.query, "results": documents}
    except Exception as e:
        return {
            "status": "error",
            "error": f"Knowledge base search failed: {e}. Ask the user to try a different query.",
        }


# Agent definition

database_agent = Agent(
    model=DEFAULT_MODEL,
    name="database_agent",
    description=(
        "Specialist for ingesting PDFs or text into the user's vector knowledge base, and "
        "for keyword-style similarity search that returns scored passages when the user "
        "explicitly asks to browse, search indexed content, or pull snippets—not for normal "
        "conversational Q&A grounded in documents (routing uses general_qa for that)."
    ),
    instruction=(
        "Focus on ingestion (PDF or text notes) or explicit similarity search over indexed content.\n"
        "Use search_database when the user wants passages listed or inspected by keyword; conversational "
        "questions that merely need facts from stored docs are handled by general_qa.\n"
        "Prefer one tool call per user intent.\n"
        "After a tool returns status=success or status=skipped, reply in plain text summarising the outcome "
        "(or the matching snippets). Do NOT call any tool again in the same turn.\n"
        "On errors or empty results, say so clearly and suggest a next step."
    ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=2048,
    ),
    tools=[update_database_from_document, update_database_from_text, search_database],
    before_tool_callback=loop_guard_before_tool,
)
