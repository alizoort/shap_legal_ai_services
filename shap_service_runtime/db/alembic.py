from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MigrationTarget:
    service_name: str
    alembic_config_path: str
    database_url: str


def build_migration_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--retries",
        type=int,
        default=30,
        help="How many database readiness checks to attempt before failing.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=2.0,
        help="Delay between readiness checks.",
    )
    parser.add_argument(
        "--allow-create-database",
        action="store_true",
        help="Create the target service database if it does not already exist.",
    )
    return parser


def resolve_alembic_config_path(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Alembic config not found at {resolved}")
    return resolved


def get_expected_heads(config_path: str | Path) -> tuple[str, ...]:
    config_path = resolve_alembic_config_path(config_path)
    script = ScriptDirectory.from_config(Config(str(config_path)))
    return tuple(sorted(script.get_heads()))


def get_current_database_heads(connection: Connection) -> tuple[str, ...]:
    context = MigrationContext.configure(connection)
    return tuple(sorted(context.get_current_heads()))


def upgrade_target(target: MigrationTarget) -> None:
    config_path = resolve_alembic_config_path(target.alembic_config_path)
    logger.info(
        "Applying %s Alembic migrations using %s",
        target.service_name,
        config_path,
    )
    command.upgrade(Config(str(config_path)), "head")


def run_migration(
    target: MigrationTarget,
    *,
    retries: int,
    delay_seconds: float,
    allow_create_database: bool,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [service-migrations] %(levelname)s %(message)s",
    )
    logger.info("Waiting for %s database connectivity.", target.service_name)
    wait_for_database(
        target.database_url,
        service_name=target.service_name,
        retries=retries,
        delay=delay_seconds,
        allow_create_database=allow_create_database,
    )
    upgrade_target(target)
    logger.info("Completed migrations for %s.", target.service_name)


def wait_for_database(
    url: str,
    *,
    service_name: str,
    retries: int = 10,
    delay: float = 1.0,
    allow_create_database: bool = False,
) -> None:
    engine = create_engine(url, echo=False, future=True)
    try:
        for attempt in range(1, retries + 1):
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                    return
            except OperationalError as exc:  # pragma: no cover - startup timing only
                lowered_error = str(exc).lower()
                if allow_create_database and "does not exist" in lowered_error:
                    try:
                        _create_database_if_missing(url, service_name=service_name)
                    except OperationalError as create_exc:
                        logger.info(
                            "%s DB create check deferred (attempt %s/%s): %s",
                            service_name,
                            attempt,
                            retries,
                            create_exc,
                        )
                    except Exception as create_exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"{service_name} database bootstrap failed while ensuring database exists."
                        ) from create_exc

                logger.info(
                    "%s DB not ready (attempt %s/%s): %s",
                    service_name,
                    attempt,
                    retries,
                    exc,
                )
                time.sleep(delay)
    finally:
        engine.dispose()

    raise RuntimeError(f"{service_name} database not ready after waiting for availability.")


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _resolve_admin_db_name(default_database_url: str) -> str:
    configured = os.getenv("ADMIN_DBNAME", "").strip()
    if configured:
        return configured

    try:
        fallback = make_url(default_database_url).database
    except Exception:  # noqa: BLE001
        fallback = None

    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return "postgres"


def _build_admin_database_url(target_database_url: str) -> str:
    parsed = make_url(target_database_url)
    return parsed.set(database=_resolve_admin_db_name(target_database_url)).render_as_string(
        hide_password=False
    )


def _create_database_if_missing(target_database_url: str, *, service_name: str) -> bool:
    target_url = make_url(target_database_url)
    target_db_name = target_url.database
    if not target_db_name:
        raise RuntimeError(f"{service_name} database bootstrap could not determine database name.")

    admin_url = _build_admin_database_url(target_database_url)
    admin_engine = create_engine(admin_url, echo=False, future=True, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :database_name"),
                {"database_name": target_db_name},
            ).scalar()
            if exists:
                return False
            connection.execute(text(f"CREATE DATABASE {_quote_identifier(target_db_name)}"))
            logger.info("%s database %s created.", service_name, target_db_name)
            return True
    finally:
        admin_engine.dispose()
