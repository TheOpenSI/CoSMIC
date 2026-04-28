"""Pydantic validators for agent tool inputs. Returns are normalised values for downstream use."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


def validation_error_response(exc: ValidationError) -> dict:
    """Maps Pydantic errors to LLM-guidance-shaped tool results."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"])
        ctx = err.get("ctx", {})
        extra = ""
        if isinstance(ctx, dict) and "error" in ctx:
            extra = f" ({ctx['error']})"
        parts.append(f"{loc}: {err['msg']}{extra}")
    return {
        "status": "error",
        "error": "Invalid arguments. " + " ".join(parts) + " Fix inputs and retry once.",
    }


class RetrieveContextInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=16_384)

    @field_validator("query", mode="before")
    @classmethod
    def strip_nonempty(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("query must be a string")
        s = v.strip()
        if not s:
            raise ValueError("query cannot be empty or whitespace-only")
        return s


class SearchDatabaseInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    topk: int = Field(default=5, ge=1, le=50)

    @field_validator("query", mode="before")
    @classmethod
    def strip_nonempty(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("query must be a string")
        s = v.strip()
        if not s:
            raise ValueError("query cannot be empty or whitespace-only")
        return s


class UpdateDatabaseTextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=512_000)

    @field_validator("text", mode="before")
    @classmethod
    def strip_nonempty(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("text must be a string")
        s = v.strip()
        if not s:
            raise ValueError("text cannot be empty or whitespace-only")
        return s


class UpdateDatabaseDocumentInput(BaseModel):
    document_path: str = Field(..., min_length=1, max_length=8192)

    @field_validator("document_path", mode="before")
    @classmethod
    def strip_nonempty(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("document_path must be a string")
        s = v.strip()
        if not s:
            raise ValueError("document_path cannot be empty or whitespace-only")
        return s


class GenerateCodeInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=32_768)

    @field_validator("query", mode="before")
    @classmethod
    def strip_nonempty(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("query must be a string")
        s = v.strip()
        if not s:
            raise ValueError("query cannot be empty or whitespace-only")
        return s


class ChessFENInput(BaseModel):
    fen: str = Field(..., min_length=10, max_length=512)
    move_mode: str = Field(default="algebric")
    topk: int = Field(default=5, ge=1, le=50)

    @field_validator("fen", mode="before")
    @classmethod
    def strip_fen(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("fen must be a string")
        s = v.strip()
        if not s:
            raise ValueError("fen cannot be empty")
        return s

    @field_validator("move_mode", mode="before")
    @classmethod
    def normalize_move_mode(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "algebric"
        if not isinstance(v, str):
            raise ValueError("move_mode must be a string")
        s = v.strip().lower()
        if s == "algebraic":
            return "algebric"
        return s

    @field_validator("move_mode")
    @classmethod
    def allowed_move_modes(cls, v: str) -> str:
        if v not in ("algebric", "coordinate"):
            raise ValueError("must be 'algebric', 'algebraic', or 'coordinate'")
        return v


class ChessSequenceInput(BaseModel):
    moves: str = Field(..., min_length=1, max_length=4096)
    move_mode: str = Field(default="algebric")
    topk: int = Field(default=5, ge=1, le=50)

    @field_validator("moves", mode="before")
    @classmethod
    def strip_moves(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("moves must be a string")
        s = v.strip()
        if not s:
            raise ValueError("moves cannot be empty")
        return s

    @field_validator("move_mode", mode="before")
    @classmethod
    def normalize_move_mode(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "algebric"
        if not isinstance(v, str):
            raise ValueError("move_mode must be a string")
        s = v.strip().lower()
        if s == "algebraic":
            return "algebric"
        return s

    @field_validator("move_mode")
    @classmethod
    def allowed_move_modes(cls, v: str) -> str:
        if v not in ("algebric", "coordinate"):
            raise ValueError("must be 'algebric', 'algebraic', or 'coordinate'")
        return v
