from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from services.legal_ai_service.config import settings

engine = create_engine(settings.legal_ai_database_url, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()
