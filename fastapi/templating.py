"""Minimal templating shim compatible with FastAPI's interface."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from .responses import HTMLResponse


class Jinja2Templates:
    """Very small template renderer used in the trimmed testing environment."""

    _TOKEN_RE = re.compile(r"(\{\{.*?\}\}|\{%.*?%\})", re.S)

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)

    def _load(self, name: str) -> str:
        path = (self.directory / name).resolve()
        if not str(path).startswith(str(self.directory.resolve())):
            raise ValueError(f"Invalid template path: {name}")
        return path.read_text(encoding="utf-8")

    def _render(self, source: str, context: Dict[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            expr = token[2:-2].strip()
            try:
                return str(eval(expr, {"__builtins__": {}}, context))  # noqa: S307 - controlled
            except Exception:
                return ""

        return self._TOKEN_RE.sub(replace, source)

    def TemplateResponse(self, name: str, context: Dict[str, Any], status_code: int = 200) -> HTMLResponse:
        if "request" not in context:
            raise RuntimeError("Template context must include 'request'")
        source = self._load(name)
        rendered = self._render(source, context)
        return HTMLResponse(rendered, status_code=status_code)


__all__ = ["Jinja2Templates"]
