"""add llm table to track OpenWebUI models

Revision ID: 20251008_add_llm
Revises: 20251008_ensure_fks
Create Date: 2025-10-08 00:10:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20251008_add_llm'
down_revision: Union[str, None] = '20251008_ensure_fks'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table('llms'):
        op.create_table(
            'llms',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('openweb_model_id', sa.String(), nullable=True),
            sa.Column('name', sa.String(), nullable=True),
            sa.Column('base_model_id', sa.String(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=True),
            sa.Column('updated_at', sa.BigInteger(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_llms_id'), 'llms', ['id'], unique=True)
        op.create_index(op.f('ix_llms_openweb_model_id'), 'llms', ['openweb_model_id'], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table('llms'):
        # Drop indexes if they exist
        try:
            op.drop_index(op.f('ix_llms_openweb_model_id'), table_name='llms')
        except Exception:
            pass
        try:
            op.drop_index(op.f('ix_llms_id'), table_name='llms')
        except Exception:
            pass
        op.drop_table('llms')
