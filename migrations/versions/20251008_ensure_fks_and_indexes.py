"""ensure foreign keys and indexes exist with proper cascade

Revision ID: 20251008_ensure_fks
Revises: b3ed4b5451ce
Create Date: 2025-10-08 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20251008_ensure_fks'
down_revision: Union[str, None] = 'b3ed4b5451ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _pg_execute(sql: str):
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        # No-op for non-Postgres engines
        return
    bind.execute(sa.text(sql))


def upgrade() -> None:
    # Create indexes if they don't exist
    _pg_execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'ix_configs_user_id'
            ) THEN
                CREATE INDEX ix_configs_user_id ON configs (user_id);
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'ix_configs_service_id'
            ) THEN
                CREATE INDEX ix_configs_service_id ON configs (service_id);
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'ix_statistics_user_id'
            ) THEN
                CREATE INDEX ix_statistics_user_id ON statistics (user_id);
            END IF;
        END $$;
        """
    )

    # Add missing foreign keys with explicit names and ON DELETE CASCADE where appropriate
    _pg_execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                     ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_name = 'configs'
                  AND kcu.column_name = 'user_id'
            ) THEN
                ALTER TABLE configs
                ADD CONSTRAINT fk_configs_user_id_users
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
            END IF;

            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                     ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_name = 'configs'
                  AND kcu.column_name = 'service_id'
            ) THEN
                ALTER TABLE configs
                ADD CONSTRAINT fk_configs_service_id_services
                FOREIGN KEY (service_id) REFERENCES services(id);
            END IF;

            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                     ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_name = 'statistics'
                  AND kcu.column_name = 'user_id'
            ) THEN
                ALTER TABLE statistics
                ADD CONSTRAINT fk_statistics_user_id_users
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    # Downgrade best-effort: drop the explicitly named constraints and indexes if present
    _pg_execute(
        """
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_configs_user_id_users'
                  AND table_name = 'configs'
            ) THEN
                ALTER TABLE configs DROP CONSTRAINT fk_configs_user_id_users;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_configs_service_id_services'
                  AND table_name = 'configs'
            ) THEN
                ALTER TABLE configs DROP CONSTRAINT fk_configs_service_id_services;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_statistics_user_id_users'
                  AND table_name = 'statistics'
            ) THEN
                ALTER TABLE statistics DROP CONSTRAINT fk_statistics_user_id_users;
            END IF;

            IF EXISTS (
                SELECT 1 FROM pg_class WHERE relname = 'ix_configs_user_id'
            ) THEN
                DROP INDEX ix_configs_user_id;
            END IF;

            IF EXISTS (
                SELECT 1 FROM pg_class WHERE relname = 'ix_configs_service_id'
            ) THEN
                DROP INDEX ix_configs_service_id;
            END IF;

            IF EXISTS (
                SELECT 1 FROM pg_class WHERE relname = 'ix_statistics_user_id'
            ) THEN
                DROP INDEX ix_statistics_user_id;
            END IF;
        END $$;
        """
    )
