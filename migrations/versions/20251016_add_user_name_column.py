"""add name column to users

Revision ID: 20251016_add_user_name
Revises: 20251008_ensure_fks
Create Date: 2025-10-16 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '20251016_add_user_name'
down_revision: Union[str, None] = '20251009_drop_service_value'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    # Add column if missing (idempotent for Postgres)
    if dialect == 'postgresql':
        bind.execute(sa.text(
            """
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='users' AND column_name='name'
                ) THEN
                    ALTER TABLE users ADD COLUMN name VARCHAR;
                END IF;
            END $$;
            """
        ))
    else:
        with op.batch_alter_table('users') as batch_op:
            batch_op.add_column(sa.Column('name', sa.String(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == 'postgresql':
        bind.execute(sa.text(
            """
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='users' AND column_name='name'
                ) THEN
                    ALTER TABLE users DROP COLUMN name;
                END IF;
            END $$;
            """
        ))
    else:
        with op.batch_alter_table('users') as batch_op:
            batch_op.drop_column('name')
