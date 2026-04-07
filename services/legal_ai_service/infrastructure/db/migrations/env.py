from __future__ import annotations

import logging
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.legal_ai_service.config import settings  # noqa: E402
from services.legal_ai_service.infrastructure.db.models import Base  # noqa: E402

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")
config.set_main_option("sqlalchemy.url", settings.legal_ai_database_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.legal_ai_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
