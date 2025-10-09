"""add description and active columns to services

Revision ID: 20251008_add_service_cols
Revises: 20251008_add_llm
Create Date: 2025-10-08 00:25:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20251008_add_service_cols'
down_revision: Union[str, None] = '20251008_add_llm'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('services', sa.Column('description', sa.String(length=255), nullable=True))
    op.add_column('services', sa.Column('active', sa.Boolean(), nullable=True))

    # Set defaults for existing rows: mark all active and set simple descriptions
    bind = op.get_bind()
    try:
        bind.execute(sa.text("UPDATE services SET active = TRUE WHERE active IS NULL"))
        bind.execute(sa.text("UPDATE services SET description = COALESCE(description, title)"))
    except Exception:
        pass


def downgrade() -> None:
    op.drop_column('services', 'active')
    op.drop_column('services', 'description')
