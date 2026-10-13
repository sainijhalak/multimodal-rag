#!/usr/bin/env bash
# ==============================================================================
# Cloudflare Tunnel for Multimodal RAG
# 100% Free Forever, Zero Configuration, Secure HTTPS Bridge for Vercel
# ==============================================================================

set -e

echo "=================================================================="
echo "🌐 Starting Free Cloudflare Tunnel for Local Multimodal RAG"
echo "=================================================================="

# 1. Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "• Downloading official cloudflared binary to ~/.local/bin/..."
    mkdir -p ~/.local/bin
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared
    chmod +x ~/.local/bin/cloudflared
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ cloudflared installed successfully!"
fi

echo ""
echo "🚀 Exposing http://localhost:8000 to the world via Cloudflare..."
echo "👉 Look for the line below containing: https://xxxx.trycloudflare.com"
echo "👉 Copy that URL and paste it into your Vercel Environment Variables: NEXT_PUBLIC_API_URL"
echo ""

cloudflared tunnel --url http://localhost:8000
