"""Encrypted credential storage built on top of ``cryptography``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from cryptography.fernet import Fernet


class CredentialVault:
    """Persist and retrieve encrypted secrets from disk."""

    def __init__(self, vault_path: Path, master_key: bytes | None = None):
        self.vault_path = Path(vault_path)
        self.master_key = master_key or self._load_or_create_key()
        self.fernet = Fernet(self.master_key)

    def _load_or_create_key(self) -> bytes:
        key_path = self.vault_path.with_suffix(".key")
        if key_path.exists():
            return key_path.read_bytes()
        key = Fernet.generate_key()
        key_path.write_bytes(key)
        os.chmod(key_path, 0o600)
        return key

    def _load(self) -> Dict[str, str]:
        if not self.vault_path.exists():
            return {}
        data = self.vault_path.read_bytes()
        decrypted = self.fernet.decrypt(data)
        return json.loads(decrypted.decode("utf-8"))

    def _save(self, payload: Dict[str, str]) -> None:
        encrypted = self.fernet.encrypt(json.dumps(payload).encode("utf-8"))
        self.vault_path.write_bytes(encrypted)
        os.chmod(self.vault_path, 0o600)

    def set(self, key: str, value: str) -> None:
        payload = self._load()
        payload[key] = value
        self._save(payload)

    def get(self, key: str, default: str | None = None) -> str | None:
        payload = self._load()
        return payload.get(key, default)

    def delete(self, key: str) -> None:
        payload = self._load()
        if key in payload:
            del payload[key]
            self._save(payload)


__all__ = ["CredentialVault"]
