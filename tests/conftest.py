from __future__ import annotations

from pathlib import Path
import pytest

from app.database import db


@pytest.fixture(autouse=True)
def isolate_test_database(tmp_path: Path):
    """Ensure every test runs against a clean, isolated SQLite database in tmp_path.
    This completely prevents test execution from reading, wiping, or polluting production data/trading_setup.db."""
    orig_path = db.db_path
    test_db_path = tmp_path / "test_trading.db"
    db.db_path = test_db_path
    db._init_schema()

    try:
        yield test_db_path
    finally:
        db.db_path = orig_path
