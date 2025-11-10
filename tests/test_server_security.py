import asyncio
import importlib
import os
import sys
import unittest
from typing import Dict, Iterable, Tuple
from urllib.parse import urlsplit

from fastapi import Request


class ServerSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        self.loop.close()
        asyncio.set_event_loop(None)

    def reload_server(self, env: Dict[str, str]) -> object:
        previous: Dict[str, str | None] = {}
        for key, value in env.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value

        for name in list(sys.modules):
            if name == "server" or name.startswith("server."):
                sys.modules.pop(name)

        try:
            module = importlib.import_module("server")
        finally:
            def _restore() -> None:
                for key, old_value in previous.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value

            self.addCleanup(_restore)

        return module

    def make_request(self, method: str, url: str, *, scheme: str | None = None, headers: Dict[str, str] | None = None, body: bytes = b"") -> Request:
        parts = urlsplit(url)
        target = parts.path or "/"
        if parts.query:
            target = f"{target}?{parts.query}"
        header_map = {k.lower(): v for k, v in (headers or {}).items()}
        host = parts.netloc or header_map.get("host", "localhost")
        header_map.setdefault("host", host)
        header_list: Iterable[Tuple[str, str]] = [(key.title(), value) for key, value in header_map.items()]
        return Request(method, target, header_list, body, ("127.0.0.1", 12345), scheme=scheme or parts.scheme or "http")

    def test_server_imports_successfully(self) -> None:
        server = self.reload_server({"FORCE_HTTPS_REDIRECT": "0"})
        self.assertTrue(hasattr(server, "app"))

    def test_http_request_redirects_to_https(self) -> None:
        server = self.reload_server({"FORCE_HTTPS_REDIRECT": "1"})
        request = self.make_request("GET", "http://example.com/admin/login", scheme="http")
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers.get("Location"), "https://example.com/admin/login")

    def test_admin_login_page_served_over_https(self) -> None:
        server = self.reload_server({"FORCE_HTTPS_REDIRECT": "1"})
        request = self.make_request("GET", "https://example.com/admin/login", scheme="https")
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Admin Sign In", response.body)

    def test_security_headers_without_hsts(self) -> None:
        server = self.reload_server(
            {
                "ENABLE_SECURITY_HEADERS": "1",
                "ENABLE_HSTS": "0",
            }
        )
        request = self.make_request("GET", "https://example.com/admin/login", scheme="https")
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertNotIn("Strict-Transport-Security", response.headers)

    def test_hsts_header_applied_over_https(self) -> None:
        server = self.reload_server(
            {
                "ENABLE_SECURITY_HEADERS": "1",
                "ENABLE_HSTS": "1",
            }
        )
        request = self.make_request("GET", "https://example.com/admin/login", scheme="https")
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("Strict-Transport-Security"),
            "max-age=31536000; includeSubDomains",
        )


if __name__ == "__main__":
    unittest.main()
