@echo off
echo ========================================================
echo   Iniciando SafeVision AI - Monitoramento de Seguranca
echo ========================================================
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python -m streamlit run app.py
pause
