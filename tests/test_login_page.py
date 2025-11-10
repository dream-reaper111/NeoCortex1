import server
from fastapi import Request


def _make_request() -> Request:
    return Request(
        "GET",
        "http://testserver/login",
        [("host", "testserver"), ("user-agent", "pytest")],
        b"",
        ("127.0.0.1", 0),
    )


def test_login_page_returns_html():
    request = _make_request()
    response = server.login_page(request)

    assert response.status_code == 200
    assert response.media_type.startswith("text/html")

    body = response.body.decode("utf-8")
    assert "NeoCortex" in body
    assert "auth-shell" in body
    assert "form" in body
