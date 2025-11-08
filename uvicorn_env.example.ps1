# Copy to uvicorn_env.ps1 and edit:
$env:API_HOST = "0.0.0.0"
$env:API_PORT = "8000"
$env:TV_WEBHOOK_SECRET = "super-secret-here"
$env:PAPER = "true"
$env:NGROK_AUTH_TOKEN = "<your-ngrok-auth-token>"
$env:NGROK_BASIC_AUTH_USER = "<strong-username>"
$env:NGROK_BASIC_AUTH_PASS = "<strong-random-password>"
# optional: request a reserved domain (leave blank for a random allocation)
$env:NGROK_DOMAIN = ""
