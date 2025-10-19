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
from pyngrok import ngrok
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
    # read port and auth token from environment
    port = int(os.getenv("API_PORT", "8000"))
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)

    # open the tunnel
    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    print("* ngrok tunnel established: {} -> http://127.0.0.1:{}".format(public_url, port), flush=True)
    print("* To test Alpaca webhooks, configure your Alpaca dashboard to use:")
    print(f"  {public_url}/alpaca/webhook", flush=True)

    # run uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()