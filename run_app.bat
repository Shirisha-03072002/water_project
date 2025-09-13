@echo off
echo 🚰 Water Quality Prediction System
echo ================================
echo.
echo Activating virtual environment...
call test\Scripts\activate.bat
echo.
echo Starting Streamlit application...
echo.
echo 📌 Make sure you have:
echo    1. Installed requirements: pip install -r requirements.txt
echo    2. Trained the model: python main.py train
echo.
echo 🌐 The app will open in your default browser
echo    If not, go to: http://localhost:8501
echo.
echo ⏹️  Press Ctrl+C to stop the server
echo.
streamlit run streamlit_app.py
pause
call test\Scripts\deactivate.bat
