@echo off
title PDF RAG

echo Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Ollama is not running!
    echo.
    echo  1. Download from https://ollama.com and install
    echo  2. Open a terminal and run:  ollama pull llama3.2
    echo  3. Double-click this file again
    echo.
    pause
    exit /b 1
)

cd /d "%~dp0"
echo  Ollama found. Starting PDF RAG...
echo  Open http://localhost:8000 in your browser.
echo  Press Ctrl+C to stop.
echo.
python -m uvicorn app:app --port 8000
pause
