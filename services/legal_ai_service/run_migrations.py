from __future__ import annotations

import argparse
from pathlib import Path

from shap_service_runtime.db.alembic import (
    MigrationTarget,
    build_migration_parser,
    get_expected_heads as load_expected_heads,
    run_migration,
)
from services.legal_ai_service.config import settings


def get_migration_target() -> MigrationTarget:
    return MigrationTarget(
        service_name="legal_ai_service",
        alembic_config_path=str(
            Path(__file__).resolve().parent / "infrastructure/db/alembic.ini"
        ),
        database_url=settings.legal_ai_database_url,
    )


def get_expected_heads() -> tuple[str, ...]:
    return load_expected_heads(get_migration_target().alembic_config_path)


def parse_args() -> argparse.Namespace:
    parser = build_migration_parser(
        description="Run Alembic migrations for the legal_ai_service database."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_migration(
        get_migration_target(),
        retries=args.retries,
        delay_seconds=args.delay_seconds,
        allow_create_database=args.allow_create_database,
    )


if __name__ == "__main__":
    main()
