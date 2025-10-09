"""drop obsolete value column from services

Revision ID: 20251009_drop_service_value
Revises: 20251008_add_service_cols
Create Date: 2025-10-09 00:05:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20251009_drop_service_value'
down_revision: Union[str, None] = '20251008_add_service_cols'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c['name'] for c in inspector.get_columns('services')}
    if 'value' in cols:
        try:
            op.drop_index('ix_services_value', table_name='services')
        except Exception:
            pass
        op.drop_column('services', 'value')


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c['name'] for c in inspector.get_columns('services')}
    if 'value' not in cols:
        op.add_column('services', sa.Column('value', sa.Integer(), nullable=True))
        try:
            op.create_index('ix_services_value', 'services', ['value'], unique=True)
        except Exception:
            pass
"""drop obsolete value column from services

Revision ID: 20251009_drop_service_value
Revises: 20251008_add_service_cols
Create Date: 2025-10-09 00:05:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20251009_drop_service_value'
down_revision: Union[str, None] = '20251008_add_service_cols'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c['name'] for c in inspector.get_columns('services')}
    if 'value' in cols:
        # Drop dependent index if present
        try:
            op.drop_index('ix_services_value', table_name='services')
        except Exception:
            pass
        op.drop_column('services', 'value')


def downgrade() -> None:
    # Best-effort restore (without uniqueness constraint semantics)
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c['name'] for c in inspector.get_columns('services')}
    if 'value' not in cols:
        op.add_column('services', sa.Column('value', sa.Integer(), nullable=True))
        try:
            op.create_index('ix_services_value', 'services', ['value'], unique=True)
        except Exception:
            pass
