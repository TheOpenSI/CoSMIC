import os
import sys

# Ensure the project root is on sys.path so specialist agent imports resolve.
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk.agents.llm_agent import Agent
from google.genai import types

from agents.config import COORDINATOR_MODEL
from agents.subagents.chess_agent import chess_agent
from agents.subagents.database_agent import database_agent
from agents.subagents.code_agent import code_agent
from agents.subagents.general_qa_agent import general_qa_agent

# Coordinator agent

root_agent = Agent(
    model=COORDINATOR_MODEL,
    name="coordinator",
    description=(
        "OpenSI-CoSMIC Coordinator: routes each user turn to exactly one of chess, database, code, or general_qa."
    ),
    instruction=(
        "Route each user message to exactly one specialist:\n"
        "- chess: a FEN string and/or a move sequence from the start position (best next moves).\n"
        "- database: indexing a PDF or saving text into the knowledge base, or an explicit keyword/search request "
        "to list matching passages (browse or admin-style search). Do not use database for normal "
        "conversational Q&A that should read the library to answer a question.\n"
        "- code: write, change, debug, or refactor programs (Python or C++).\n"
        "- general_qa: everything else—general knowledge, CoSMIC system info, and answering questions using the "
        "user's saved documents (the specialist uses retrieve_context for that).\n"
        "Do not answer the topic yourself. Once a specialist replies, pass that reply "
        "back to the user verbatim and do NOT re-delegate the same task or call any other agent." 
    ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=1024,
    ),
    sub_agents=[chess_agent, database_agent, code_agent, general_qa_agent],
)
