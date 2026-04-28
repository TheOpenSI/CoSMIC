import os
import sys
import yaml

from google.adk.agents.llm_agent import Agent
from google.genai import types
from pydantic import ValidationError

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from box import Box
from src.services.chess import StockfishFENNextMove, StockfishSequenceNextMove
from agents.config import DEFAULT_MODEL
from agents.safeguards import loop_guard_before_tool
from agents.tool_schemas import ChessFENInput, ChessSequenceInput, validation_error_response

# Load config for stockfish binary path.
_CONFIG_PATH = os.path.join(_project_root, "scripts", "configs", "config.yaml")
_config = Box.from_yaml(filename=_CONFIG_PATH, Loader=yaml.FullLoader) if os.path.exists(_CONFIG_PATH) else None
_stockfish_path = _config.chess.stockfish_path if _config else ""


def _resolve_stockfish_binary() -> str:
    """Resolve the Stockfish binary path, returning empty string if not found."""
    if _stockfish_path:
        path = _stockfish_path if os.path.isabs(_stockfish_path) else os.path.join(_project_root, _stockfish_path)
        if os.path.exists(path):
            return path

    candidates = [
        os.path.join(_project_root, "third_party", "stockfish", "stockfish", "stockfish-windows-x86-64-avx2.exe"),
        os.path.join(_project_root, "third_party", "stockfish", "stockfish-windows-x86-64-avx2.exe"),
        os.path.join(_project_root, "third_party", "stockfish", "stockfish-ubuntu-x86-64-avx2"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def predict_next_move_from_fen(fen: str, move_mode: str = "algebric", topk: int = 5) -> dict:
    """Predict best next chess moves from a FEN board position.

    Args:
        fen: Full FEN string of the position.
        move_mode: Algebraic San style (``"algebric"`` legacy spelling or ``"algebraic"``) or UCI-style
            ``"coordinate"``.
        topk: Number of candidate moves to return (default 5).

    Returns:
        dict with status and next_moves on success, or status and error otherwise.

    Example:
        ``fen``: ``"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"``, ``move_mode``: ``"algebraic"``.
    """
    try:
        inp = ChessFENInput.model_validate({"fen": fen, "move_mode": move_mode, "topk": topk})
    except ValidationError as e:
        return validation_error_response(e)
    binary = _resolve_stockfish_binary()
    if not binary:
        return {
            "status": "error",
            "error": "Stockfish engine is not installed. Configure chess.stockfish_path in config.yaml.",
        }
    try:
        predictor = StockfishFENNextMove(binary_path=binary)
        next_moves = predictor(fen=inp.fen, move_mode=inp.move_mode, topk=inp.topk)
        if not next_moves:
            return {
                "status": "error",
                "error": f"No moves for FEN '{inp.fen}'. The position may be invalid, checkmate, or stalemate.",
            }
        return {"status": "success", "fen": inp.fen, "next_moves": next_moves}
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to predict move from FEN: {e}. Verify the FEN is valid and complete.",
        }


def predict_next_move_from_sequence(moves: str, move_mode: str = "algebric", topk: int = 5) -> dict:
    """Predict best next chess moves from a move sequence starting at the initial position.

    Args:
        moves: Space-separated complete moves (e.g. algebraic ``"e4 e5 Nf3"`` or coordinate ``"e2e4 e7e5 g1f3"``).
        move_mode: ``"algebric"`` / ``"algebraic"`` vs ``"coordinate"``; must match the input notation style.
        topk: Number of candidate moves to return (default 5).

    Returns:
        dict with status and next_moves on success, or status and error otherwise.

    Example:
        ``moves``: ``"e4 e5"``, ``move_mode``: ``"algebraic"``, ``topk``: ``3``.
    """
    try:
        inp = ChessSequenceInput.model_validate({"moves": moves, "move_mode": move_mode, "topk": topk})
    except ValidationError as e:
        return validation_error_response(e)
    binary = _resolve_stockfish_binary()
    if not binary:
        return {
            "status": "error",
            "error": "Stockfish engine is not installed. Configure chess.stockfish_path in config.yaml.",
        }
    try:
        predictor = StockfishSequenceNextMove(binary_path=binary)
        next_moves = predictor(moves=inp.moves, move_mode=inp.move_mode, topk=inp.topk)
        if not next_moves:
            return {
                "status": "error",
                "error": (
                    f"Could not compute a next move from sequence '{inp.moves}' (mode: {inp.move_mode}). "
                    "One or more moves are illegal or malformed."
                ),
            }
        return {"status": "success", "moves": inp.moves, "next_moves": next_moves}
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to predict move from sequence: {e}. Verify the sequence is valid.",
        }


chess_agent = Agent(
    model=DEFAULT_MODEL,
    name="chess_agent",
    description=(
        "Specialist for chess move prediction. Handles queries about predicting "
        "the next best chess move given a FEN position or a sequence of moves."
    ),
    instruction=(
        "Recommend strong next moves from a position or move list, and briefly say why the best option matters.\n"
        "Use a tool only when the user supplied a clear FEN or a legal move list from the start; otherwise ask "
        "for that input.\n"
        "After a tool returns status=success, immediately reply to the user in plain text with a one-line "
        "recommendation and a short list of the candidate moves. Do NOT call any tool again in the same turn.\n"
        "If a tool returns status=error, explain the error simply and ask the user to fix the input. Do NOT "
        "retry the same call."
    ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=2048,
    ),
    tools=[predict_next_move_from_fen, predict_next_move_from_sequence],
    before_tool_callback=loop_guard_before_tool,
)
