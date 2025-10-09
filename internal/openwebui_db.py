"""Secondary DB connection to OpenWebUI database for synchronization.

This module creates a SQLAlchemy engine/session to the existing OpenWebUI DB
(assumed reachable via env vars prefixed with OPENWEBUI_ or by reusing DATABASE_URL
from OpenWebUI environment variables passed into this container).

Env precedence:
1. OPENWEBUI_DATABASE_URL (explicit full URL)
2. OPENWEBUI_DB_HOST + OPENWEBUI_DB_PORT + OPENWEBUI_DB_USER + OPENWEBUI_DB_PASSWORD + OPENWEBUI_DB_NAME
3. FALLBACK: None (module will raise to surface misconfiguration)

We intentionally do NOT auto-create this database; it must already exist and is
owned by the OpenWebUI service.
"""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sqlalchemy.orm import sessionmaker, Session as SASession

# Build URL

def _build_openwebui_url() -> str:
    direct = os.getenv("OPENWEBUI_DATABASE_URL") or os.getenv("OPENWEBUI_DB_URL")
    if direct:
        return direct
    # Fallback to generic DATABASE_URL if explicitly provided (useful in dev)
    generic = os.getenv("DATABASE_URL")
    if generic:
        return generic
    host = os.getenv("OPENWEBUI_DB_HOST")
    user = os.getenv("OPENWEBUI_DB_USER")
    password = os.getenv("OPENWEBUI_DB_PASSWORD")
    name = os.getenv("OPENWEBUI_DB_NAME")
    port = os.getenv("OPENWEBUI_DB_PORT", "5432")
    if all([host, user, password, name]):
        # URL-encode password to handle special characters
        pw = quote_plus(password)
        return f"postgresql://{user}:{pw}@{host}:{port}/{name}"
    raise RuntimeError(
        "OpenWebUI DB connection not configured; set OPENWEBUI_DATABASE_URL or host/user/password/name env vars"
    )

# Lazily populated connection URL and engine/session references.
_OPENWEBUI_URL: str | None = None
openwebui_engine = None
OpenWebUISessionLocal = None

def _ensure_engine():
    global _OPENWEBUI_URL, openwebui_engine, OpenWebUISessionLocal
    if openwebui_engine is not None and OpenWebUISessionLocal is not None:
        return
    # Build URL only when needed; if not configured, raise with clear message
    if _OPENWEBUI_URL is None:
        _OPENWEBUI_URL = _build_openwebui_url()
    openwebui_engine = create_engine(_OPENWEBUI_URL, pool_pre_ping=True)
    OpenWebUISessionLocal = sessionmaker(bind=openwebui_engine, autoflush=False, autocommit=False, expire_on_commit=False)

def get_openwebui_db() -> Generator[SASession, None, None]:
    _ensure_engine()
    db = OpenWebUISessionLocal()
    try:
        yield db
    finally:
        db.close()

get_openwebui_session = contextmanager(get_openwebui_db)

def get_latest_model_id() -> str | None:
    """Return the most recently updated model (id) from OpenWebUI DB.

    Used to select the active LLM in CoSMIC when not explicitly configured.
    Returns None if the model table is absent or query fails.
    """
    from sqlalchemy import text
    try:
        with get_openwebui_session() as db:
            row = db.execute(text("SELECT id FROM model ORDER BY updated_at DESC LIMIT 1")).fetchone()
            return row[0] if row else None
    except Exception:
        return None
