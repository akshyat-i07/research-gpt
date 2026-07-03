#!/bin/bash
# ResearchGPT - start the backend server
# Usage: ./start.sh

set -e

echo "🔬 ResearchGPT Backend"
echo "────────────────────────"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Install from https://python.org"
    exit 1
fi

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "❌ GEMINI_API_KEY is not set."
    echo "   Copy .env.example to .env and add your Gemini API key:"
    echo "   cp .env.example .env"
    exit 1
fi

if ! python3 -c "import fastapi, requests, faiss, fitz, google.genai, dotenv" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

if [ ! -f frontend/dist/index.html ]; then
    echo "📦 Building frontend (first run)…"
    if ! command -v npm &> /dev/null; then
        echo "❌ frontend/dist is missing and npm is not installed."
        echo "   Install Node.js 18+, or run: docker compose up --build -d"
        exit 1
    fi
    (cd frontend && npm ci && npm run build)
fi

echo "✅ Starting server on http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

python3 -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload \
  --reload-exclude 'frontend/node_modules/*' \
  --reload-exclude 'frontend/dist/*'
