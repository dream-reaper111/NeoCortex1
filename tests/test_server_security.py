import asyncio
import base64
import hashlib
import hmac
import importlib
import json
import os
import sys
import time
import unittest
from datetime import datetime, timezone
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

    def sign_webhook(self, secret: str, timestamp: str, body: bytes) -> str:
        message = timestamp.encode("utf-8") + b"." + body
        digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
        return base64.b64encode(digest).decode("ascii")

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

    def test_login_rejects_credentials_in_query_params(self) -> None:
        server = self.reload_server({"FORCE_HTTPS_REDIRECT": "0"})
        request = self.make_request(
            "POST",
            "https://example.com/login?username=alice&password=secret",
            scheme="https",
        )
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"request body", response.body)

    def test_alpaca_webhook_requires_secret_configuration(self) -> None:
        server = self.reload_server({"FORCE_HTTPS_REDIRECT": "0"})
        request = self.make_request(
            "POST",
            "https://example.com/alpaca/webhook",
            scheme="https",
            headers={"Content-Type": "application/json"},
            body=b"{}",
        )
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"webhook secret", response.body)

    def test_alpaca_webhook_valid_signature_and_replay_detection(self) -> None:
        secret = "topsecret"
        server = self.reload_server(
            {
                "FORCE_HTTPS_REDIRECT": "0",
                "ALPACA_WEBHOOK_SECRET": secret,
                "ALPACA_WEBHOOK_TOLERANCE_SECONDS": "600",
            }
        )
        payload = {"event": "trade_update", "symbol": "AAPL"}
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        timestamp = datetime.now(timezone.utc).isoformat()
        signature = self.sign_webhook(secret, timestamp, body)
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": timestamp,
        }
        request = self.make_request(
            "POST",
            "https://example.com/alpaca/webhook",
            scheme="https",
            headers=headers,
            body=body,
        )
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 200)

        second_request = self.make_request(
            "POST",
            "https://example.com/alpaca/webhook",
            scheme="https",
            headers=headers,
            body=body,
        )
        replay_response = self.loop.run_until_complete(server.app._handle(second_request))
        self.assertEqual(replay_response.status_code, 409)
        self.assertIn(b"replay", replay_response.body.lower())

    def test_alpaca_webhook_rejects_missing_timestamp(self) -> None:
        secret = "anothersecret"
        server = self.reload_server(
            {
                "FORCE_HTTPS_REDIRECT": "0",
                "ALPACA_WEBHOOK_SECRET": secret,
            }
        )
        body = json.dumps({"foo": "bar"}, separators=(",", ":"), sort_keys=True).encode()
        timestamp = str(time.time())
        signature = self.sign_webhook(secret, timestamp, body)
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
        }
        request = self.make_request(
            "POST",
            "https://example.com/alpaca/webhook",
            scheme="https",
            headers=headers,
            body=body,
        )
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"timestamp", response.body.lower())

    def test_alpaca_webhook_test_requires_admin_gate(self) -> None:
        server = self.reload_server(
            {
                "FORCE_HTTPS_REDIRECT": "0",
                "ALPACA_WEBHOOK_TEST_REQUIRE_AUTH": "1",
            }
        )
        request = self.make_request(
            "POST",
            "https://example.com/alpaca/webhook/test",
            scheme="https",
            headers={"Content-Type": "application/json"},
            body=json.dumps({}).encode(),
        )
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 503)

    def test_alpaca_webhook_test_enforces_gate_token(self) -> None:
        gate_token = "s3cret"
        server = self.reload_server(
            {
                "FORCE_HTTPS_REDIRECT": "0",
                "ADMIN_PORTAL_GATE_TOKEN": gate_token,
            }
        )
        body = json.dumps({}).encode()
        request = self.make_request(
            "POST",
            "https://example.com/alpaca/webhook/test",
            scheme="https",
            headers={"Content-Type": "application/json"},
            body=body,
        )
        response = self.loop.run_until_complete(server.app._handle(request))
        self.assertEqual(response.status_code, 401)

        authed_request = self.make_request(
            "POST",
            "https://example.com/alpaca/webhook/test",
            scheme="https",
            headers={
                "Content-Type": "application/json",
                "X-Admin-Portal-Key": gate_token,
            },
            body=body,
        )
        authed_response = self.loop.run_until_complete(server.app._handle(authed_request))
        self.assertEqual(authed_response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
