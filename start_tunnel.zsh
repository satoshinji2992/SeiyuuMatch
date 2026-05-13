#!/bin/zsh
set -e

cd "$(dirname "$0")"

PORT="${PORT:-3724}"
TARGET_URL="${1:-http://127.0.0.1:${PORT}}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed."
  echo ""
  echo "Install it on macOS with:"
  echo "  brew install cloudflared"
  echo ""
  echo "Then run:"
  echo "  ./start_server.zsh"
  echo "  ./start_tunnel.zsh"
  exit 1
fi

echo "Starting Cloudflare Quick Tunnel -> ${TARGET_URL}"
echo "Keep this terminal open. Copy the https://*.trycloudflare.com URL printed below."
cloudflared tunnel --url "${TARGET_URL}"
