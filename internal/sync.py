"""Data synchronization utilities from OpenWebUI DB into Cosmic DB.

Currently implemented:
- sync_users: mirror user basic identity & role info from openwebui.user table
  into cosmic.users table (upsert by email, update name/role/profile_image_url).

Design notes:
- OpenWebUI user primary key is a string id (uuid). Cosmic users table currently
  may have its own integer/serial PK (depends on existing model). We map by email
  as the stable natural key; if email collision arises with differing openwebui id,
  we update existing cosmic row.
- If cosmic user row has additional domain fields, we leave them untouched.
- Idempotent: running multiple times yields same end state.

Future extensions:
- sync_services: derive service usage from OpenWebUI events.
- incremental sync using updated_at timestamps.
"""
from __future__ import annotations
import logging
from typing import List
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from internal.openwebui_db import get_openwebui_session
from internal.db import SessionLocal as CosmicSessionLocal
from internal import models as cosmic_models

log = logging.getLogger(__name__)

# Reflect minimal OpenWebUI user via lightweight ORM mapping to avoid importing entire package.
from sqlalchemy.orm import registry, Mapped, mapped_column
from sqlalchemy import String, Text, BigInteger

mapper_registry = registry()

@mapper_registry.mapped
class OpenWebUIUser:  # type: ignore
    __tablename__ = "user"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    profile_image_url: Mapped[str] = mapped_column(Text)
    last_active_at: Mapped[int] = mapped_column(BigInteger)
    updated_at: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[int] = mapped_column(BigInteger)
    api_key: Mapped[str | None] = mapped_column(String)
    # settings, info, oauth_sub omitted (not needed for sync)


def sync_users() -> dict:
    """Synchronize users from OpenWebUI DB into cosmic users table.

    Returns summary dict: {inserted: int, updated: int, total_source: int}
    """
    inserted = 0
    updated = 0
    total = 0

    with get_openwebui_session() as src_db, CosmicSessionLocal() as dst_db:
        try:
            rows: List[OpenWebUIUser] = list(src_db.scalars(select(OpenWebUIUser)))
            total = len(rows)
            for row in rows:
                # Find by email in cosmic DB
                existing = (
                    dst_db.query(cosmic_models.User)
                    .filter(cosmic_models.User.email == row.email)
                    .first()
                )
                if existing:
                    changed = False
                    # set openweb_id if missing
                    if hasattr(existing, "openweb_id") and not getattr(existing, "openweb_id"):
                        existing.openweb_id = row.id
                        changed = True
                    if hasattr(existing, "name") and row.name and getattr(existing, "name", None) != row.name:
                        existing.name = row.name
                        changed = True
                    if hasattr(existing, "role") and row.role and getattr(existing, "role", None) != row.role:
                        existing.role = row.role
                        changed = True
                    # profile_image_url not present in cosmic model; skip safely
                    if changed:
                        updated += 1
                else:
                    # Create new cosmic user, mapping available fields.
                    try:
                        kwargs = {}
                        if hasattr(cosmic_models.User, 'email'):
                            kwargs['email'] = row.email
                        if hasattr(cosmic_models.User, 'role'):
                            kwargs['role'] = row.role
                        if hasattr(cosmic_models.User, 'openweb_id'):
                            kwargs['openweb_id'] = row.id
                        if hasattr(cosmic_models.User, 'name'):
                            kwargs['name'] = row.name
                        new_user = cosmic_models.User(**kwargs)
                        dst_db.add(new_user)
                        inserted += 1
                    except Exception as inner_e:  # pragma: no cover - defensive
                        log.error("Failed to stage new user %s: %s", row.email, inner_e)
                # Flush batched changes periodically (could optimize for large sets)
            dst_db.commit()
        except SQLAlchemyError as e:
            dst_db.rollback()
            log.error("User sync failed: %s", e)
            raise

    summary = {"inserted": inserted, "updated": updated, "total_source": total}
    log.info("User sync summary: %s", summary)
    return summary


@mapper_registry.mapped
class OpenWebUIModel:  # type: ignore
    __tablename__ = "model"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String)
    base_model_id: Mapped[str | None] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    updated_at: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[int] = mapped_column(BigInteger)


def sync_llms() -> dict:
    """Synchronize LLM models from OpenWebUI DB into cosmic llms table.

    Upsert by openweb_model_id; update name/base_model_id/updated_at/created_at when changed.
    """
    inserted = 0
    updated = 0
    total = 0

    with get_openwebui_session() as src_db, CosmicSessionLocal() as dst_db:
        try:
            rows: List[OpenWebUIModel] = list(src_db.scalars(select(OpenWebUIModel)))
            total = len(rows)
            for row in rows:
                existing = (
                    dst_db.query(cosmic_models.LLM)
                    .filter(cosmic_models.LLM.openweb_model_id == row.id)
                    .first()
                )
                if existing:
                    changed = False
                    if existing.name != row.name:
                        existing.name = row.name
                        changed = True
                    if existing.base_model_id != (row.base_model_id or None):
                        existing.base_model_id = row.base_model_id
                        changed = True
                    if existing.updated_at != row.updated_at:
                        existing.updated_at = row.updated_at
                        changed = True
                    if existing.created_at != row.created_at:
                        existing.created_at = row.created_at
                        changed = True
                    if changed:
                        updated += 1
                else:
                    model = cosmic_models.LLM(
                        openweb_model_id=row.id,
                        name=row.name,
                        base_model_id=row.base_model_id,
                        updated_at=row.updated_at,
                        created_at=row.created_at,
                    )
                    dst_db.add(model)
                    inserted += 1
            dst_db.commit()
        except SQLAlchemyError as e:
            dst_db.rollback()
            log.error("LLM sync failed: %s", e)
            raise

    summary = {"inserted": inserted, "updated": updated, "total_source": total}
    log.info("LLM sync summary: %s", summary)
    return summary


__all__ = ["sync_users", "sync_llms"]
