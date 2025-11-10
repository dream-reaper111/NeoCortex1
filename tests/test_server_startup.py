"""Tests for server startup and authentication database handling."""

from __future__ import annotations

import importlib
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest


MODULE_NAME = "server"


def _purge_server_modules() -> None:
    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith(f"{MODULE_NAME}."):
            sys.modules.pop(name, None)


@pytest.fixture
def import_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Import the server module with a temporary AUTH_DB_PATH."""

    auth_db = tmp_path / "auth" / "auth.db"
    monkeypatch.setenv("AUTH_DB_PATH", str(auth_db))
    monkeypatch.delenv("TMPDIR", raising=False)
    _purge_server_modules()
    module = importlib.import_module(MODULE_NAME)

    yield module

    _purge_server_modules()
    monkeypatch.delenv("AUTH_DB_PATH", raising=False)


def test_server_app_initializes(import_server: ModuleType) -> None:
    """The FastAPI app should be importable and expose an app instance."""

    server = import_server
    assert hasattr(server, "app"), "server.app should be defined"
    assert server.AUTH_DB_PATH.is_file() or not server.AUTH_DB_PATH.exists()

    conn = server._db_conn()
    try:
        conn.execute("SELECT 1")
    finally:
        conn.close()

    assert server.AUTH_DB_PATH.exists(), "auth database file should be created"


def test_auth_db_falls_back_when_directory_unwritable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If the configured directory is not writable, the server should use a fallback location."""

    auth_dir = tmp_path / "readonly"
    auth_db = auth_dir / "auth.db"

    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()

    original_mkdir = Path.mkdir

    def fake_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if self == auth_dir:
            raise PermissionError("read-only directory")
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    monkeypatch.setenv("TMPDIR", str(fallback_root))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fallback_root))
    monkeypatch.setenv("AUTH_DB_PATH", str(auth_db))

    _purge_server_modules()
    try:
        server = importlib.import_module(MODULE_NAME)

        expected = fallback_root / "neocortex" / "auth" / "auth.db"
        assert server.AUTH_DB_PATH == expected.resolve()
        assert expected.parent.is_dir()

        conn = sqlite3.connect(expected)
        try:
            conn.execute("SELECT 1")
        finally:
            conn.close()
    finally:
        _purge_server_modules()
        monkeypatch.delenv("AUTH_DB_PATH", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
