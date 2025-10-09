import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

"""
Database configuration logic

Priority:
1. If COSMIC_DB_URL is provided, use it directly.
2. Else, if COSMIC_DB_HOST (and related env vars) are provided, build a Postgres URL:
   postgresql://user:password@host:port/database
   Database name defaults to value of COSMIC_DB_NAME (default: cosmic_db)
3. Else, fallback to DATABASE_URL (legacy) or local sqlite file cosmic.db

Automatic Postgres Database Creation:
If CREATE_COSMIC_DB=1 and the target Postgres database does not yet exist, attempt to
connect to the server-level database (usually 'postgres') and create it prior to binding the engine.

Environment Variables:
  COSMIC_DB_URL            Full SQLAlchemy URL (overrides all below)
  COSMIC_DB_HOST           Postgres host (e.g., cosmic_db_server or postgres)
  COSMIC_DB_PORT           Postgres port (default: 5432)
  COSMIC_DB_USER           Username (default: value of DATABASE_USER or 'cosmic')
  COSMIC_DB_PASSWORD       Password (default: value of DATABASE_PASSWORD or empty)
  COSMIC_DB_NAME           Database name (default: cosmic_db)
  CREATE_COSMIC_DB         If '1' or 'true', attempt to create database if missing

Backward Compatibility:
  DATABASE_URL respected if no dedicated COSMIC_* vars are set.
"""

def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def build_db_url() -> str:
    explicit = _env("COSMIC_DB_URL")
    if explicit:
        return explicit

    host = _env("COSMIC_DB_HOST")
    if host:
        port = _env("COSMIC_DB_PORT", "5432")
        user = _env("COSMIC_DB_USER", _env("DATABASE_USER", "cosmic"))
        password = _env("COSMIC_DB_PASSWORD", _env("DATABASE_PASSWORD", "")) or ""
        db_name = _env("COSMIC_DB_NAME", "cosmic_db")
        # URL encode password in case of special chars
        pw_enc = quote_plus(password)
        return f"postgresql://{user}:{pw_enc}@{host}:{port}/{db_name}"

    # Legacy path
    legacy = _env("DATABASE_URL")
    if legacy:
        return legacy
    return "sqlite:///./cosmic.db"


def ensure_postgres_database(url: str):
    if not url.startswith("postgresql://"):
        return
    create_flag = (_env("CREATE_COSMIC_DB", "0").lower() in ("1", "true", "yes"))
    if not create_flag:
        return

    # Parse out db name by splitting last path component
    # Reconnect to 'postgres' maintenance db to create if missing
    try:
        without_scheme = url[len("postgresql://"):]
        creds_host, db_name = without_scheme.rsplit("/", 1)
        if not db_name:
            return
        maintenance_url = f"postgresql://{creds_host}/postgres"
        maint_engine = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")
        with maint_engine.connect() as conn:
            exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname=:d"), {"d": db_name}).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    except Exception as e:
        # Non-fatal; proceed even if creation fails (likely permission issue)
        print(f"[cosmic][db] Database auto-creation skipped or failed: {e}")


DATABASE_URL = build_db_url()
ensure_postgres_database(DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
)

# Ensure SQLite enforces foreign keys in dev/local
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()