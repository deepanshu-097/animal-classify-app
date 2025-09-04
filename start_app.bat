@echo off
echo Starting Cow & Buffalo AI Recognition App...
cd /d "%~dp0"
.\.venv\Scripts\streamlit run app.py --server.port=8502 --server.address=127.0.0.1
pause


