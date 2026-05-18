"""Minimal templating shim compatible with FastAPI's interface."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .responses import HTMLResponse


class Jinja2Templates:
    """Very small template renderer used in the trimmed testing environment."""

    _TOKEN_RE = re.compile(r"(\{\{.*?\}\}|\{%.*?%\})", re.S)
    _EXTENDS_RE = re.compile(r"\{%\s*extends\s+['\"](.+?)['\"]\s*%\}")
    _BLOCK_RE = re.compile(r"\{%\s*block\s+(\w+)\s*%\}(.*?)\{%\s*endblock\s*%\}", re.S)
    _INCLUDE_RE = re.compile(r"\{%\s*include\s+['\"](.+?)['\"]\s*%\}")
    _IF_RE = re.compile(r"\{%\s*if\s+(.+?)\s*%\}(.*?)(\{%\s*else\s*%\}(.*?))?\{%\s*endif\s*%\}", re.S)

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self._jinja_env = self._build_jinja_env()

    def _load(self, name: str) -> str:
        path = (self.directory / name).resolve()
        if not str(path).startswith(str(self.directory.resolve())):
            raise ValueError(f"Invalid template path: {name}")
        return path.read_text(encoding="utf-8")

    def _build_jinja_env(self) -> Optional[Tuple[Any, Callable[[str, Dict[str, Any]], str]]]:
        if importlib.util.find_spec("jinja2") is None:
            return None
        from jinja2 import Environment, FileSystemLoader  # type: ignore

        env = Environment(loader=FileSystemLoader(str(self.directory)), autoescape=False)
        return env, lambda name, context: env.get_template(name).render(context)

    def _eval_expr(self, expr: str, context: Dict[str, Any]) -> Any:
        parts = [part.strip() for part in expr.split("|") if part.strip()]
        if not parts:
            return ""
        base = parts[0]
        value = eval(base, {"__builtins__": {}}, context)  # noqa: S307 - controlled
        for filt in parts[1:]:
            name, args = self._parse_filter(filt, context)
            if name == "join":
                sep = args[0] if args else ""
                value = (sep or "").join(str(item) for item in (value or []))
            else:
                raise ValueError(f"Unsupported filter: {name}")
        return value

    def _parse_filter(self, token: str, context: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
        match = re.match(r"(\w+)\s*(?:\((.*)\))?$", token)
        if not match:
            return token, ()
        name = match.group(1)
        raw_args = match.group(2)
        if not raw_args:
            return name, ()
        args = eval(f"({raw_args},)", {"__builtins__": {}}, context)  # noqa: S307 - controlled
        if not isinstance(args, tuple):
            args = (args,)
        return name, args

    def _render_if_blocks(self, source: str, context: Dict[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            expr = match.group(1)
            truthy = False
            try:
                truthy = bool(self._eval_expr(expr, context))
            except Exception:
                truthy = False
            if truthy:
                return match.group(2)
            return match.group(4) or ""

        while True:
            updated, count = self._IF_RE.subn(replace, source)
            if count == 0:
                return updated
            source = updated

    def _render_includes(self, source: str, context: Dict[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            return self._render_template_source(self._load(name), context)

        while True:
            updated, count = self._INCLUDE_RE.subn(replace, source)
            if count == 0:
                return updated
            source = updated

    def _render_blocks(self, base: str, blocks: Dict[str, str]) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            fallback = match.group(2)
            return blocks.get(name, fallback)

        return self._BLOCK_RE.sub(replace, base)

    def _render_template_source(self, source: str, context: Dict[str, Any]) -> str:
        source = self._render_includes(source, context)
        source = self._render_if_blocks(source, context)
        return self._render(source, context)

    def _render(self, source: str, context: Dict[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            expr = token[2:-2].strip()
            try:
                return str(self._eval_expr(expr, context))
            except Exception:
                return ""

        return self._TOKEN_RE.sub(replace, source)

    def TemplateResponse(self, name: str, context: Dict[str, Any], status_code: int = 200) -> HTMLResponse:
        if "request" not in context:
            raise RuntimeError("Template context must include 'request'")
        if self._jinja_env:
            env, renderer = self._jinja_env
            rendered = renderer(name, context)
            return HTMLResponse(rendered, status_code=status_code)
        source = self._load(name)
        extends = self._EXTENDS_RE.search(source)
        blocks = {name: content for name, content in self._BLOCK_RE.findall(source)}
        if extends:
            base = self._load(extends.group(1))
            combined = self._render_blocks(base, blocks)
            rendered = self._render_template_source(combined, context)
        else:
            rendered = self._render_template_source(source, context)
        return HTMLResponse(rendered, status_code=status_code)


__all__ = ["Jinja2Templates"]
