from __future__ import annotations

from pathlib import Path
from typing import Optional

from .responses import FileResponse, HTMLResponse, Response


class StaticFiles:
    def __init__(self, *, directory: str, html: bool = False) -> None:
        self.directory = Path(directory).resolve()
        self.html = html

    def _safe_path(self, path: str) -> Optional[Path]:
        candidate = (self.directory / path.lstrip("/")).resolve()
        try:
            candidate.relative_to(self.directory)
        except ValueError:
            return None
        return candidate

    def get_response(self, path: str, request=None) -> Response:
        subpath = path
        if not subpath or subpath == "/":
            if self.html:
                subpath = "/index.html"
            else:
                return HTMLResponse("<h1>Not Found</h1>", status_code=404)
        target = self._safe_path(subpath)
        if target is None:
            return HTMLResponse("<h1>Not Found</h1>", status_code=404)
        if target.is_dir():
            if self.html:
                index = target / "index.html"
                if index.exists():
                    target = index
                else:
                    return HTMLResponse("<h1>Not Found</h1>", status_code=404)
            else:
                return HTMLResponse("<h1>Not Found</h1>", status_code=404)
        if not target.exists():
            return HTMLResponse("<h1>Not Found</h1>", status_code=404)
        return FileResponse(target)

