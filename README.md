# 🚰 Water Quality Prediction System

An AI-powered web application for predicting water quality safety with detailed explanations and treatment recommendations.

## 🌟 Features

### 🔍 **Water Quality Analysis**
- **Instant Predictions**: Get immediate safe/unsafe classification
- **99%+ Accuracy**: Powered by XGBoost machine learning model
- **Confidence Scores**: Know how certain the prediction is

### 📊 **Detailed Explanations**
- **Parameter Analysis**: See which parameters are problematic
- **Health Impact Assessment**: Understand potential health effects
- **Visual Indicators**: Easy-to-read gauges and charts

### 💡 **Treatment Recommendations**
- **Priority-based Suggestions**: High, medium, and low priority issues
- **Specific Treatments**: Actionable recommendations for each problem
- **Professional Standards**: Based on WHO/EPA guidelines

### 📈 **Interactive Dashboard**
- **Model Performance**: View training metrics and accuracy
- **Data Visualization**: Explore parameter distributions
- **Report Generation**: Download detailed analysis reports

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Generate Training Data**
```bash
python generate_synthetic_data.py
```

### 3. **Train the Model**
```bash
python main.py train
```

### 4. **Launch Web Application**
```bash
streamlit run streamlit_app.py
```
**Or use the launcher:**
```bash
run_app.bat
```

### 5. **Open in Browser**
Navigate to: `http://localhost:8501`

## 📱 How to Use

### **Main Prediction Interface**
1. **Enter Parameters**: Input your water quality measurements
2. **Analyze**: Click "Analyze Water Quality" button
3. **Review Results**: Get instant safety classification
4. **Explore Details**: Check parameter analysis and recommendations

### **Sample Data**
- Use "Load Safe Water Sample" for testing with good water
- Use "Load Unsafe Water Sample" for testing with problematic water

## 🧪 Water Quality Parameters

| Parameter | Unit | Safe Range | Description |
|-----------|------|------------|-------------|
| **pH** | - | 6.5 - 8.5 | Acidity/alkalinity level |
| **Hardness** | mg/L | 60 - 120 | Calcium and magnesium content |
| **Total Dissolved Solids** | ppm | 100 - 500 | Dissolved minerals and salts |
| **Chloramines** | ppm | 0.2 - 4.0 | Disinfectant residual |
| **Sulfate** | mg/L | 50 - 250 | Sulfur compound content |
| **Conductivity** | μS/cm | 200 - 600 | Electrical conductivity |
| **Organic Carbon** | ppm | 2 - 8 | Total organic matter |
| **Trihalomethanes** | μg/L | 5 - 80 | Disinfection byproducts |
| **Turbidity** | NTU | 0.1 - 1.0 | Water clarity measure |

## 🏗️ Project Structure

```
water_quality/
│
├── 📁 data/                    # Data loading and preprocessing
│   ├── __init__.py
│   └── data_loader.py
│
├── 📁 features/                # Feature engineering
│   ├── __init__.py
│   └── feature_engineering.py
│
├── 📁 models/                  # XGBoost model implementation
│   ├── __init__.py
│   └── xgboost_model.py
│
├── 📁 utils/                   # Explanations and suggestions
│   ├── __init__.py
│   └── explain.py
│
├── 📄 streamlit_app.py         # Web application
├── 📄 main.py                  # Training pipeline
├── 📄 generate_synthetic_data.py # Data generation
├── 📄 requirements.txt         # Dependencies
├── 📄 run_app.bat             # Windows launcher
└── 📄 README.md               # This file
```

## 🤖 Model Performance

- **Training Accuracy**: 99.92%
- **Validation Accuracy**: 99.10%
- **Cross-validation**: 99.43% ± 0.33%
- **AUC Score**: 99.99%

### **Top Important Features**
1. Turbidity-Trihalomethanes interaction (14.96%)
2. Hardness (10.82%)
3. pH (10.15%)
4. Trihalomethanes (9.27%)
5. Sulfate (9.00%)

## 🛠️ Technology Stack

- **🤖 Machine Learning**: XGBoost, Scikit-learn
- **🌐 Web Framework**: Streamlit
- **📊 Data Processing**: Pandas, NumPy
- **📈 Visualization**: Plotly, Matplotlib, Seaborn
- **🔍 Explainability**: SHAP, Feature Importance
- **⚡ Progress Tracking**: tqdm

## 📋 Commands Reference

### **Training Pipeline**
```bash
python main.py train          # Full training pipeline
python main.py demo           # Test with sample data
```

### **Data Generation**
```bash
python generate_synthetic_data.py    # Generate training dataset
```

### **Web Application**
```bash
streamlit run streamlit_app.py       # Start web server
run_app.bat                          # Windows launcher
```

## ⚠️ Important Notes

### **Before First Use**
1. ✅ Install all dependencies
2. ✅ Generate training data
3. ✅ Train the model
4. ✅ Launch the web app

### **Troubleshooting**
- **Model not found**: Run `python main.py train` first
- **Import errors**: Check if all dependencies are installed
- **Port issues**: Streamlit uses port 8501 by default

## 📊 Example Results

### **Safe Water Sample**
```
✅ WATER IS SAFE FOR CONSUMPTION
Confidence: 98.5%
All parameters within safe ranges
```

### **Unsafe Water Sample**
```
⚠️ WATER IS NOT SAFE FOR CONSUMPTION
Confidence: 97.2%
Issues found: 5 parameters
High priority: pH (too low), Trihalomethanes (too high)
```

## 🔮 Future Enhancements

- 🌊 Real-time sensor integration
- 📱 Mobile app version
- 🌍 Multi-language support
- 📡 API endpoints for integration
- 🎨 Advanced visualization dashboards

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the project structure
3. Ensure all steps are followed correctly

## ⚖️ Disclaimer

This tool is for educational and preliminary analysis purposes. Always consult certified water quality professionals for official testing and treatment decisions.

---

**Built with ❤️ using modern ML and web technologies**
