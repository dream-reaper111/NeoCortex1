"""
Run the NeoAI API server with an ngrok tunnel.

This script uses the `pyngrok` library to expose your local FastAPI server
running on ``API_PORT`` (default 8000) to the internet via ngrok. When
executed, it will start an ngrok tunnel, print the public URL, and then
launch Uvicorn to serve the FastAPI ``app`` defined in ``server.py``.

Usage:

    python run_with_ngrok.py

Set the environment variable ``NGROK_AUTH_TOKEN`` if you have an ngrok
account and want to authenticate the tunnel. The script reads ``API_PORT``
from the environment (default ``8000``) so you can override the port if
necessary.

After running, copy the printed ngrok URL (e.g. ``https://1234.ngrok.io``)
into your Alpaca dashboard as the webhook URL. For example, set:

    https://1234.ngrok.io/alpaca/webhook

under the "Webhook URL" field in your Alpaca paper trading settings.

Note: This script blocks until the server shuts down. To stop, press Ctrl+C.
"""

import os
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency in tests
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        """Fallback no-op when python-dotenv is unavailable."""

        return False

try:
    from pyngrok import ngrok
    from pyngrok.exception import PyngrokNgrokHTTPError
except ModuleNotFoundError:  # pragma: no cover - optional dependency in tests
    ngrok = None  # type: ignore

    class PyngrokNgrokHTTPError(Exception):
        """Placeholder error type when pyngrok is unavailable."""

        pass

try:
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover - optional dependency in tests
    uvicorn = None  # type: ignore

try:
    # Import the FastAPI app from server.py
    from server import app, DISABLE_ACCESS_LOGS  # type: ignore
except Exception as e:
    raise ImportError(
        "Unable to import FastAPI app from server.py. Please ensure that the "
        "project has been compiled and that server.py is in the same directory. "
        f"Original error: {e}"
    ) from e

def _normalize_domain(raw: str | None) -> str:
    """Return an ngrok-compatible domain string without protocol or slashes."""

    if not raw:
        return ""

    cleaned = raw.strip()
    if not cleaned:
        return ""

    if "://" in cleaned:
        parsed = urlparse(cleaned)
        host = parsed.netloc or parsed.path
    else:
        host = cleaned

    return host.strip().strip("/")


def main() -> None:
    load_dotenv(override=False)
    port = int(os.getenv("API_PORT", "8000"))

    if uvicorn is None:
        raise RuntimeError(
            "uvicorn is required to run the FastAPI server. Install it with 'pip install uvicorn'."
        )

    if ngrok is None:
        print(
            "* pyngrok is not installed. Starting the FastAPI app locally on port {} without an ngrok tunnel.".format(port),
            flush=True,
        )
        print("  Install pyngrok to enable tunnelling: pip install pyngrok", flush=True)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=not DISABLE_ACCESS_LOGS,
        )
        return

    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        raise RuntimeError(
            "NGROK_AUTH_TOKEN is required to start the tunnel. "
            "Generate a token from the ngrok dashboard and set it in your environment "
            "or .env file before running this script."
        )

    ngrok.set_auth_token(auth_token)

    connect_kwargs = {"bind_tls": True}

    basic_auth = os.getenv("NGROK_BASIC_AUTH")
    basic_auth_applied = False
    if not basic_auth:
        username = os.getenv("NGROK_BASIC_AUTH_USER", "").strip()
        password = os.getenv("NGROK_BASIC_AUTH_PASS", "").strip()
        if username and password:
            basic_auth = f"{username}:{password}"

    if basic_auth and ":" in basic_auth:
        connect_kwargs["basic_auth"] = basic_auth
        basic_auth_applied = True
    else:
        print(
            "* Warning: ngrok tunnel will be launched without HTTP basic auth.\n"
            "  Set NGROK_BASIC_AUTH or NGROK_BASIC_AUTH_USER/NGROK_BASIC_AUTH_PASS to secure the tunnel.",
            flush=True,
        )
        basic_auth = ""

    domain_env = os.getenv("NGROK_DOMAIN")
    domain = _normalize_domain(domain_env)
    if domain:
        connect_kwargs["domain"] = domain
    allow_cidrs = [cidr.strip() for cidr in (os.getenv("NGROK_ALLOWED_CIDRS") or "").split(",") if cidr.strip()]
    if allow_cidrs:
        connect_kwargs["ip_restriction"] = {"allow_cidrs": allow_cidrs}

    try:
        tunnel = ngrok.connect(port, "http", **connect_kwargs)
    except PyngrokNgrokHTTPError as error:
        error_text = str(error)
        if domain and "ERR_NGROK_15013" in error_text:
            print(
                "* Requested ngrok dev domain not found. Falling back to a random domain.\n"
                "  To use a different domain, reserve one in the ngrok dashboard and set NGROK_DOMAIN.\n"
                "  Set NGROK_DOMAIN=\"\" to skip requesting a reserved domain.",
                flush=True,
            )
            fallback_kwargs = {k: v for k, v in connect_kwargs.items() if k != "domain"}
            tunnel = ngrok.connect(port, "http", **fallback_kwargs)
        else:
            raise RuntimeError(
                "Unable to start ngrok tunnel.\n"
                "Original error:\n"
                f"{error_text}"
            ) from error
    public_url = tunnel.public_url
    print("* ngrok tunnel established: {} -> http://127.0.0.1:{}".format(public_url, port), flush=True)
    if basic_auth_applied:
        print("  (basic auth enabled via NGROK_BASIC_AUTH)", flush=True)
    if allow_cidrs:
        print("  (IP allow list: {})".format(", ".join(allow_cidrs)), flush=True)
    print("* To test Alpaca webhooks, configure your Alpaca dashboard to use:")
    print(f"  {public_url}/alpaca/webhook", flush=True)

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=not DISABLE_ACCESS_LOGS,
        )
    finally:
        try:
            ngrok.disconnect(tunnel.public_url)
        finally:
            ngrok.kill()


if __name__ == "__main__":
    main()
