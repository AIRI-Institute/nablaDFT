from pathlib import Path

import pytest

import nablaDFT


@pytest.fixture()
def db_path():
    db_path = Path(nablaDFT.__path__[0]) / ".." / "tests/data/raw/test_database.db"
    return str(db_path)


@pytest.fixture()
def hamiltonian_db_path():
    db_path = (
        Path(nablaDFT.__path__[0])
        / ".."
        / "tests/data/raw/test_hamiltonian_database.db"
    )
    return str(db_path)
