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

DEFAULT_NGROK_AUTH_TOKEN = "33aDTBUq8xRsoeabQ5HE1rWk0U3_3hvYA6JV8MPegF1DyXMAT"

from dotenv import load_dotenv
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokHTTPError
import uvicorn

try:
    # Import the FastAPI app from server.py
    from server import app  # type: ignore
except Exception as e:
    raise ImportError(
        "Unable to import FastAPI app from server.py. Please ensure that the "
        "project has been compiled and that server.py is in the same directory. "
        f"Original error: {e}"
    ) from e

def main() -> None:
    load_dotenv(override=False)
    # read port and auth token from environment
    port = int(os.getenv("API_PORT", "8000"))
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        auth_token = DEFAULT_NGROK_AUTH_TOKEN
    if auth_token:
        ngrok.set_auth_token(auth_token)

    connect_kwargs = {"bind_tls": True}
    basic_auth = os.getenv("NGROK_BASIC_AUTH")
    if basic_auth:
        connect_kwargs["basic_auth"] = basic_auth
    domain = os.getenv("NGROK_DOMAIN")
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
                "  To use a custom domain, reserve one in the ngrok dashboard and set NGROK_DOMAIN.",
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
    if basic_auth:
        print("  (basic auth enabled via NGROK_BASIC_AUTH)", flush=True)
    if allow_cidrs:
        print("  (IP allow list: {})".format(", ".join(allow_cidrs)), flush=True)
    print("* To test Alpaca webhooks, configure your Alpaca dashboard to use:")
    print(f"  {public_url}/alpaca/webhook", flush=True)

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    finally:
        try:
            ngrok.disconnect(tunnel.public_url)
        finally:
            ngrok.kill()


if __name__ == "__main__":
    main()
