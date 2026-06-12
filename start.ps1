# ResearchGPT - start the backend server (Windows PowerShell)
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"

Write-Host "ResearchGPT Backend" -ForegroundColor Cyan
Write-Host "-------------------"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python 3 is required. Install from https://python.org" -ForegroundColor Red
    exit 1
}

try {
    python -c "import fastapi" 2>$null
} catch {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
    Write-Host "Tip: Copy .env.example to .env and set GEMINI_API_KEY" -ForegroundColor Yellow
}

Write-Host "Starting server at http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop"
Write-Host ""

python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
