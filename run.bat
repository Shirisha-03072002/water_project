@echo off
echo 🚰 Water Quality Prediction System with Authentication
echo ===================================================
echo.
echo Activating virtual environment...
call test\Scripts\activate.bat
echo.
echo Starting Streamlit application with login features...
echo.
echo 📌 First time setup:
echo    1. Make sure you've installed requirements: pip install -r requirements.txt
echo    2. Make sure the model is trained: python main.py train
echo    3. Register a new account to start tracking your water quality history
echo.
echo 🌐 The app will open in your default browser
echo    If not, go to: http://localhost:8501
echo.
echo ⏹️  Press Ctrl+C to stop the server
echo.
streamlit run streamlit_app_new.py
pause
call test\Scripts\deactivate.bat
