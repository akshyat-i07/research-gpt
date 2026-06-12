#!/bin/bash
# ResearchGPT - start the backend server
# Usage: ./start.sh
# Then open the React frontend (ResearchGPT.jsx) in Claude or your React app.

set -e

echo "🔬 ResearchGPT Backend"
echo "────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Install from https://python.org"
    exit 1
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Starting server on http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

python3 -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
