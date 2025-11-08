from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import Any, Iterable, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

__all__ = ["run"]


class _Server:
    def __init__(self, app: Any, host: str, port: int, log_level: str) -> None:
        self.app = app
        self.host = host
        self.port = port
        self.log_level = log_level

    async def serve(self) -> None:
        await self.app._startup()
        server = await asyncio.start_server(self._handle_client, self.host, self.port)
        addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        print(f"* Serving on {addresses}")
        try:
            await server.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover - manual interruption
            pass
        finally:
            server.close()
            await server.wait_closed()
            await self.app._shutdown()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername") or ("127.0.0.1", 0)
        try:
            request_data = await self._read_headers(reader)
            if request_data is None:
                writer.close()
                await writer.wait_closed()
                return
            method, target, headers, body = request_data
            request = Request(method, target, headers, body, peer)
            response = await self.app._handle(request)
            await self._send_response(writer, response)
        except HTTPException as exc:
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            await self._send_response(writer, response)
        except Exception as exc:  # pragma: no cover - unexpected error path
            response = JSONResponse({"detail": f"Internal Server Error: {exc}"}, status_code=500)
            await self._send_response(writer, response)
        finally:
            if not writer.is_closing():
                writer.close()
            await writer.wait_closed()

    async def _read_headers(self, reader: asyncio.StreamReader) -> Optional[Tuple[str, str, Iterable[Tuple[str, str]], bytes]]:
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError:
            return None
        header_text = raw.decode("latin-1")
        lines = header_text.split("\r\n")
        if not lines or len(lines[0].split()) < 3:
            return None
        method, target, _ = lines[0].split()[:3]
        headers = []
        content_length = 0
        for line in lines[1:]:
            if not line:
                continue
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            value = value.strip()
            headers.append((name.strip(), value))
            if name.lower() == "content-length":
                try:
                    content_length = int(value)
                except ValueError:
                    content_length = 0
        body = b""
        if content_length > 0:
            body = await reader.readexactly(content_length)
        return method, target, headers, body

    async def _send_response(self, writer: asyncio.StreamWriter, response: Response) -> None:
        status = HTTPStatus(response.status_code)
        headers = dict(response.headers)
        body = response.body
        if isinstance(response, StreamingResponse) and response.streaming:
            headers.setdefault("Transfer-Encoding", "chunked")
            header_lines = [f"HTTP/1.1 {response.status_code} {status.phrase}"]
            for key, value in headers.items():
                header_lines.append(f"{key}: {value}")
            for cookie in response.cookies:
                header_lines.append(f"Set-Cookie: {cookie}")
            header_lines.append("Connection: close")
            header_lines.append("")
            header_data = "\r\n".join(header_lines).encode("latin-1")
            writer.write(header_data + b"\r\n")
            async for chunk in _iterate_stream(response.content):
                if isinstance(chunk, str):
                    data = chunk.encode("utf-8")
                else:
                    data = bytes(chunk)
                if not data:
                    continue
                writer.write(f"{len(data):X}\r\n".encode("latin-1") + data + b"\r\n")
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            return
        headers.setdefault("Content-Length", str(len(body)))
        header_lines = [f"HTTP/1.1 {response.status_code} {status.phrase}"]
        for key, value in headers.items():
            header_lines.append(f"{key}: {value}")
        for cookie in response.cookies:
            header_lines.append(f"Set-Cookie: {cookie}")
        header_lines.append("Connection: close")
        header_lines.append("")
        header_data = "\r\n".join(header_lines).encode("latin-1")
        writer.write(header_data + b"\r\n" + body)
        await writer.drain()


def run(app: Any, host: str = "127.0.0.1", port: int = 8000, log_level: str = "info", reload: bool = False) -> None:
    server = _Server(app, host, port, log_level)
    asyncio.run(server.serve())


async def _iterate_stream(content: Any):
    if hasattr(content, "__aiter__"):
        async for chunk in content:
            yield chunk
        return
    for chunk in content:
        yield chunk

