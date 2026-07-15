# scripts/start_services.ps1
# Launch every DataPilot service locally (Windows dev, no Docker).
# Each opens in its own window so logs are visible.
# Prereq: pip install -r requirements.txt  (and Ollama running for LLM output)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

Write-Host "Starting FastAPI backend  -> http://localhost:8000/docs"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root'; python -m uvicorn api.main:app --port 8000"

Write-Host "Starting Celery worker (solo pool for Windows)"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root'; celery -A tasks.worker worker --pool=solo --loglevel=info"

Write-Host "Starting Streamlit UI     -> http://localhost:8501"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root'; python -m streamlit run ui/app.py --server.port 8501 --server.headless true"

Write-Host "Starting MLflow UI        -> http://localhost:5000"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root'; python -m mlflow ui --port 5000"

Write-Host "`nAll services launched. Ollama must be started separately:"
Write-Host "  & `"$env:USERPROFILE\ollama-portable\ollama.exe`" serve"
