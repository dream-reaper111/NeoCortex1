from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Union


class Response:
    def __init__(
        self,
        content: Union[str, bytes] = b"",
        status_code: int = 200,
        media_type: str = "text/plain",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if isinstance(content, str):
            body = content.encode("utf-8")
        else:
            body = bytes(content)
        self.body = body
        self.status_code = status_code
        self.media_type = media_type
        self.headers: Dict[str, str] = dict(headers or {})
        if media_type and "content-type" not in {k.lower() for k in self.headers}:
            self.headers.setdefault("Content-Type", media_type)
        self.streaming = False
        self._cookies: list[str] = []

    def set_cookie(
        self,
        key: str,
        value: str = "",
        *,
        max_age: Optional[int] = None,
        expires: Optional[Union[str, datetime]] = None,
        path: str = "/",
        domain: Optional[str] = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: Optional[str] = None,
    ) -> None:
        parts = [f"{key}={value}"]
        if path:
            parts.append(f"Path={path}")
        if domain:
            parts.append(f"Domain={domain}")
        if max_age is not None:
            parts.append(f"Max-Age={int(max_age)}")
        if expires is not None:
            if isinstance(expires, datetime):
                expires = expires.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
            parts.append(f"Expires={expires}")
        if secure:
            parts.append("Secure")
        if httponly:
            parts.append("HttpOnly")
        if samesite:
            parts.append(f"SameSite={samesite}")
        self._cookies.append("; ".join(parts))

    def delete_cookie(
        self,
        key: str,
        *,
        path: str = "/",
        domain: Optional[str] = None,
        samesite: Optional[str] = None,
        secure: bool = False,
    ) -> None:
        self.set_cookie(
            key,
            value="",
            path=path,
            domain=domain,
            secure=secure,
            httponly=True,
            samesite=samesite,
            expires="Thu, 01 Jan 1970 00:00:00 GMT",
            max_age=0,
        )

    @property
    def cookies(self) -> Iterable[str]:
        return list(self._cookies)


class JSONResponse(Response):
    def __init__(self, content: Any, status_code: int = 200, headers: Optional[Dict[str, str]] = None, media_type: str = "application/json") -> None:
        body = json.dumps(content, ensure_ascii=False).encode("utf-8")
        super().__init__(body, status_code=status_code, media_type=media_type, headers=headers)


class HTMLResponse(Response):
    def __init__(self, content: str, status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        super().__init__(content, status_code=status_code, media_type="text/html; charset=utf-8", headers=headers)


class RedirectResponse(Response):
    def __init__(self, url: str, status_code: int = 307, headers: Optional[Dict[str, str]] = None) -> None:
        headers = dict(headers or {})
        headers["Location"] = url
        super().__init__(b"", status_code=status_code, media_type="text/plain", headers=headers)


class FileResponse(Response):
    def __init__(self, path: Union[str, Path], *, media_type: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        file_path = Path(path)
        data = file_path.read_bytes()
        if media_type is None:
            media_type, _ = mimetypes.guess_type(str(file_path))
        super().__init__(data, status_code=200, media_type=media_type or "application/octet-stream", headers=headers)
        self.headers.setdefault("Content-Length", str(len(data)))


class StreamingResponse(Response):
    def __init__(
        self,
        content: Union[Iterable[Union[str, bytes]], Iterator[Union[str, bytes]], Any],
        status_code: int = 200,
        media_type: str = "text/plain",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(b"", status_code=status_code, media_type=media_type, headers=headers)
        self.streaming = True
        self.content = content

