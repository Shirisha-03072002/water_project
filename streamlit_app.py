"""
Streamlit Web Application for Water Quality Prediction
=====================================================

This app provides a user-friendly interface for:
1. Predicting water quality safety
2. Explaining predictions with detailed analysis
3. Providing treatment recommendations
4. Visualizing model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import load_water_quality_data, prepare_features_target
from features.feature_engineering import engineer_features
from models.xgboost_model import WaterQualityXGBoostModel
from utils.explain import WaterQualityExplainer
from main import predict_water_quality

# Page configuration
st.set_page_config(
    page_title="Water Quality Prediction System",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
    }
    
    .safe-water {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .unsafe-water {
        background: linear-gradient(90deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model"""
    try:
        model = WaterQualityXGBoostModel()
        model.load_model("models/water_quality_xgboost.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the model is trained first by running 'python main.py train'")
        return None

def create_parameter_input_form():
    """Create input form for water quality parameters"""
    st.subheader("🧪 Enter Water Quality Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph = st.number_input(
            "pH Level",
            min_value=0.0,
            max_value=14.0,
            value=7.0,
            step=0.1,
            help="Measure of acidity/alkalinity (6.5-8.5 is safe)"
        )
        
        hardness = st.number_input(
            "Hardness (mg/L)",
            min_value=0.0,
            max_value=500.0,
            value=180.0,
            step=1.0,
            help="Calcium and magnesium content (60-120 mg/L is safe)"
        )
        
        solids = st.number_input(
            "Total Dissolved Solids (ppm)",
            min_value=0.0,
            max_value=2000.0,
            value=350.0,
            step=1.0,
            help="Total dissolved solids (100-500 ppm is safe)"
        )
    
    with col2:
        chloramines = st.number_input(
            "Chloramines (ppm)",
            min_value=0.0,
            max_value=12.0,
            value=2.5,
            step=0.1,
            help="Disinfectant level (0.2-4.0 ppm is safe)"
        )
        
        sulfate = st.number_input(
            "Sulfate (mg/L)",
            min_value=0.0,
            max_value=600.0,
            value=150.0,
            step=1.0,
            help="Sulfate content (50-250 mg/L is safe)"
        )
        
        conductivity = st.number_input(
            "Conductivity (μS/cm)",
            min_value=0.0,
            max_value=1200.0,
            value=400.0,
            step=1.0,
            help="Electrical conductivity (200-600 μS/cm is safe)"
        )
    
    with col3:
        organic_carbon = st.number_input(
            "Organic Carbon (ppm)",
            min_value=0.0,
            max_value=25.0,
            value=5.0,
            step=0.1,
            help="Total organic carbon (2-8 ppm is safe)"
        )
        
        trihalomethanes = st.number_input(
            "Trihalomethanes (μg/L)",
            min_value=0.0,
            max_value=200.0,
            value=45.0,
            step=1.0,
            help="Disinfection byproducts (5-80 μg/L is safe)"
        )
        
        turbidity = st.number_input(
            "Turbidity (NTU)",
            min_value=0.0,
            max_value=15.0,
            value=0.5,
            step=0.1,
            help="Water clarity (0.1-1.0 NTU is safe)"
        )
    
    return {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }

def display_prediction_results(prediction_result):
    """Display prediction results with visual indicators"""
    
    if prediction_result['safety_status'] == 'SAFE':
        st.markdown(f"""
        <div class="safe-water">
            <h2>✅ WATER IS SAFE FOR CONSUMPTION</h2>
            <p>Confidence: {prediction_result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="unsafe-water">
            <h2>⚠️ WATER IS NOT SAFE FOR CONSUMPTION</h2>
            <p>Confidence: {prediction_result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_result['probabilities']['safe'] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Safety Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_parameter_analysis(detailed_report):
    """Display detailed parameter analysis"""
    st.subheader("📊 Parameter Analysis")
    
    param_analysis = detailed_report['parameter_analysis']
    
    # Create parameter status table
    status_data = []
    for param, analysis in param_analysis.items():
        status_data.append({
            'Parameter': param.replace('_', ' ').title(),
            'Value': f"{analysis['value']:.2f}",
            'Safe Range': analysis['safe_range'],
            'Status': '✅ Safe' if analysis['status'] == 'safe' else '❌ Unsafe',
            'Issue': analysis['issue_type'].replace('_', ' ').title() if analysis['issue_type'] else 'None'
        })
    
    df_status = pd.DataFrame(status_data)
    st.dataframe(df_status, use_container_width=True)
    
    # Visualization of parameters vs safe ranges
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[param.replace('_', ' ').title() for param in param_analysis.keys()],
        specs=[[{"secondary_y": True}]*3]*3
    )
    
    row, col = 1, 1
    for param, analysis in param_analysis.items():
        # Extract safe range
        safe_range = analysis['safe_range'].split('-')
        if len(safe_range) == 2:
            safe_min = float(safe_range[0])
            safe_max = float(safe_range[1].split()[0])  # Remove unit
            
            # Add safe range as a bar
            fig.add_trace(
                go.Bar(x=[param], y=[safe_max - safe_min], 
                      base=[safe_min], name='Safe Range',
                      marker_color='lightgreen', opacity=0.3,
                      showlegend=(row==1 and col==1)),
                row=row, col=col
            )
            
            # Add actual value as a point
            color = 'green' if analysis['status'] == 'safe' else 'red'
            fig.add_trace(
                go.Scatter(x=[param], y=[analysis['value']], 
                          mode='markers', name='Actual Value',
                          marker=dict(color=color, size=10),
                          showlegend=(row==1 and col==1)),
                row=row, col=col
            )
        
        col += 1
        if col > 3:
            col = 1
            row += 1
    
    fig.update_layout(height=800, title="Parameter Values vs Safe Ranges")
    st.plotly_chart(fig, use_container_width=True)

def display_suggestions(detailed_report):
    """Display treatment suggestions"""
    suggestions = detailed_report['suggestions']
    
    if not suggestions:
        st.success("🎉 No treatment needed! All parameters are within safe ranges.")
        return
    
    st.subheader("💡 Treatment Recommendations")
    
    # Group suggestions by priority
    high_priority = [s for s in suggestions if s['priority'] == 'high']
    medium_priority = [s for s in suggestions if s['priority'] == 'medium']
    low_priority = [s for s in suggestions if s['priority'] == 'low']
    
    if high_priority:
        st.error("🚨 HIGH PRIORITY ISSUES")
        for suggestion in high_priority:
            with st.expander(f"⚠️ {suggestion['parameter']} - {suggestion['severity'].title()} Issue"):
                st.write(f"**Problem:** {suggestion['issue']}")
                st.write(f"**Health Impact:** {suggestion['health_impact']}")
                st.write("**Recommended Treatments:**")
                for i, treatment in enumerate(suggestion['treatments'], 1):
                    st.write(f"{i}. {treatment}")
    
    if medium_priority:
        st.warning("⚡ MEDIUM PRIORITY ISSUES")
        for suggestion in medium_priority:
            with st.expander(f"⚡ {suggestion['parameter']} - {suggestion['severity'].title()} Issue"):
                st.write(f"**Problem:** {suggestion['issue']}")
                st.write(f"**Health Impact:** {suggestion['health_impact']}")
                st.write("**Recommended Treatments:**")
                for i, treatment in enumerate(suggestion['treatments'], 1):
                    st.write(f"{i}. {treatment}")
    
    if low_priority:
        st.info("ℹ️ LOW PRIORITY ISSUES")
        for suggestion in low_priority:
            with st.expander(f"ℹ️ {suggestion['parameter']} - {suggestion['severity'].title()} Issue"):
                st.write(f"**Problem:** {suggestion['issue']}")
                st.write(f"**Health Impact:** {suggestion['health_impact']}")
                st.write("**Recommended Treatments:**")
                for i, treatment in enumerate(suggestion['treatments'], 1):
                    st.write(f"{i}. {treatment}")

def display_next_steps(detailed_report):
    """Display next steps"""
    st.subheader("🎯 Next Steps")
    
    next_steps = detailed_report['next_steps']
    
    for i, step in enumerate(next_steps, 1):
        if step.startswith("DO NOT"):
            st.error(f"{i}. {step}")
        elif step.startswith("Address high-priority"):
            st.warning(f"{i}. {step}")
        elif step.startswith("  -"):
            st.write(f"   {step}")
        else:
            st.info(f"{i}. {step}")

def load_model_performance():
    """Load and display model performance metrics"""
    try:
        # Try to load dataset for visualization
        df = load_water_quality_data()
        return df
    except:
        return None

def display_model_info():
    """Display model information and performance"""
    st.subheader("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Model Details:**
        - Algorithm: XGBoost Classifier
        - Features: 18 (9 original + 9 engineered)
        - Training Accuracy: ~99.9%
        - Validation Accuracy: ~99.1%
        - Cross-validation: 99.4% ± 0.3%
        """)
    
    with col2:
        st.info("""
        **Key Features:**
        - Turbidity-Trihalomethanes interaction
        - Hardness
        - pH Level
        - Trihalomethanes
        - Sulfate
        """)
    
    # Display feature correlations with larger size
    st.subheader("📊 Feature Correlations")
    
    # Create a larger figure for better visibility
    fig = plt.figure(figsize=(15, 12))
    
    # Load dataset for visualization
    try:
        df = load_water_quality_data()
        if df is not None:
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap with larger size
            sns.heatmap(
                corr_matrix,
                annot=True,  # Show correlation values
                cmap='RdBu',  # Use Red-Blue colormap
                center=0,     # Center the colormap at 0
                fmt='.2f',   # Format correlation values to 2 decimal places
                square=True,  # Make the plot square-shaped
                linewidths=0.5, # Add lines between cells
                cbar_kws={"shrink": .8} # Adjust colorbar size
            )
            
            plt.title('Feature Correlation Matrix', pad=20, size=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Use streamlit's pyplot function with the figure
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading or displaying correlation matrix: {e}")
        plt.close(fig)  # Close the figure if there's an error

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚰 Water Quality Prediction System</h1>
        <p>AI-powered water safety analysis with treatment recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["🧪 Predict Water Quality", "📊 Model Performance", "📚 About"]
    )
    
    if app_mode == "🧪 Predict Water Quality":
        st.header("Water Quality Prediction")
        
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Input form
        water_sample = create_parameter_input_form()
        
        # Prediction button
        if st.button("🔍 Analyze Water Quality", type="primary"):
            with st.spinner("Analyzing water quality..."):
                try:
                    # Make prediction
                    prediction_result = predict_water_quality(
                        water_sample, 
                        detailed_explanation=True
                    )
                    
                    # Debug information (can be removed later)
                    st.write("**Debug Information:**")
                    st.write(f"Raw prediction: {prediction_result['prediction']}")
                    st.write(f"Probabilities: Safe={prediction_result['probabilities']['safe']:.3f}, Unsafe={prediction_result['probabilities']['unsafe']:.3f}")
                    
                    # Display results
                    display_prediction_results(prediction_result)
                    
                    # Detailed analysis
                    if 'detailed_report' in prediction_result:
                        detailed_report = prediction_result['detailed_report']
                        
                        # Create tabs for different sections
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "📊 Parameter Analysis", 
                            "💡 Recommendations", 
                            "🎯 Next Steps",
                            "📋 Summary Report"
                        ])
                        
                        with tab1:
                            display_parameter_analysis(detailed_report)
                        
                        with tab2:
                            display_suggestions(detailed_report)
                        
                        with tab3:
                            display_next_steps(detailed_report)
                        
                        with tab4:
                            st.subheader("📋 Complete Analysis Report")
                            
                            # Overall assessment
                            assessment = detailed_report['overall_assessment']
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Safety Status", assessment['safety_status'])
                            with col2:
                                st.metric("Confidence", f"{assessment['confidence']:.1%}")
                            with col3:
                                st.metric("Total Issues", detailed_report['issue_summary']['total_issues'])
                            
                            # Download report
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            report_text = f"""
Water Quality Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

OVERALL ASSESSMENT:
Status: {assessment['safety_status']}
Confidence: {assessment['confidence']:.1%}
Recommendation: {assessment['recommendation']}

PARAMETER VALUES:
"""
                            for param, analysis in detailed_report['parameter_analysis'].items():
                                report_text += f"{param}: {analysis['value']:.2f} ({analysis['status']})\n"
                            
                            report_text += f"\nISSUES FOUND: {detailed_report['issue_summary']['total_issues']}\n"
                            
                            for suggestion in detailed_report['suggestions']:
                                report_text += f"\n{suggestion['parameter']}: {suggestion['issue']}\n"
                                report_text += f"Priority: {suggestion['priority']}\n"
                                report_text += f"Treatments: {', '.join(suggestion['treatments'])}\n"
                            
                            st.download_button(
                                label="📄 Download Report",
                                data=report_text,
                                file_name=f"water_quality_report_{timestamp}.txt",
                                mime="text/plain"
                            )
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        
        # Debug test button for troubleshooting
        if st.button("🔬 Quick Debug Test"):
            st.write("Testing both sample types:")
            
            safe_sample = {
                'ph': 7.2, 'Hardness': 180.0, 'Solids': 350.0,
                'Chloramines': 2.5, 'Sulfate': 150.0, 'Conductivity': 400.0,
                'Organic_carbon': 5.0, 'Trihalomethanes': 45.0, 'Turbidity': 0.5
            }
            
            unsafe_sample = {
                'ph': 4.5, 'Hardness': 180.0, 'Solids': 1200.0,
                'Chloramines': 8.0, 'Sulfate': 450.0, 'Conductivity': 900.0,
                'Organic_carbon': 15.0, 'Trihalomethanes': 150.0, 'Turbidity': 5.0
            }
            
            try:
                safe_result = predict_water_quality(safe_sample, detailed_explanation=False)
                unsafe_result = predict_water_quality(unsafe_sample, detailed_explanation=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Safe Sample Result:**")
                    st.write(f"Prediction: {safe_result['safety_status']}")
                    st.write(f"Confidence: {safe_result['confidence']:.2%}")
                
                with col2:
                    st.write("**Unsafe Sample Result:**")
                    st.write(f"Prediction: {unsafe_result['safety_status']}")
                    st.write(f"Confidence: {unsafe_result['confidence']:.2%}")
                    
            except Exception as e:
                st.error(f"Debug test failed: {e}")
    
    elif app_mode == "📊 Model Performance":
        st.header("Model Performance Dashboard")
        
        display_model_info()
        
        # Load dataset for visualization
        df = load_model_performance()
        if df is not None:
            st.subheader("📈 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                safe_count = sum(df['Potability'])
                st.metric("Safe Samples", safe_count)
            with col3:
                unsafe_count = len(df) - safe_count
                st.metric("Unsafe Samples", unsafe_count)
            with col4:
                st.metric("Features", len(df.columns) - 1)
            
            # Distribution plots
            st.subheader("📊 Parameter Distributions")
            
            # Select parameter to visualize
            param_to_plot = st.selectbox(
                "Select Parameter",
                [col for col in df.columns if col != 'Potability']
            )
            
            fig = px.histogram(
                df, 
                x=param_to_plot, 
                color='Potability',
                color_discrete_map={0: 'red', 1: 'green'},
                title=f"{param_to_plot} Distribution by Water Safety",
                labels={'Potability': 'Water Safety', 0: 'Unsafe', 1: 'Safe'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "📚 About":
        st.header("About Water Quality Prediction System")
        
        st.markdown("""
        ## 🎯 Purpose
        This application uses machine learning to predict water quality safety and provide 
        actionable recommendations for water treatment.
        
        ## 🔬 How It Works
        1. **Input Parameters**: Enter 9 key water quality measurements
        2. **AI Analysis**: XGBoost model processes the data with 99%+ accuracy
        3. **Safety Prediction**: Get immediate safe/unsafe classification
        4. **Detailed Explanation**: Understand which parameters are problematic
        5. **Treatment Recommendations**: Receive specific suggestions for improvement
        
        ## 📊 Water Quality Parameters
        
        | Parameter | Safe Range | Description |
        |-----------|------------|-------------|
        | pH | 6.5 - 8.5 | Acidity/alkalinity level |
        | Hardness | 60 - 120 mg/L | Calcium and magnesium content |
        | Total Dissolved Solids | 100 - 500 ppm | Dissolved minerals and salts |
        | Chloramines | 0.2 - 4.0 ppm | Disinfectant residual |
        | Sulfate | 50 - 250 mg/L | Sulfur compound content |
        | Conductivity | 200 - 600 μS/cm | Electrical conductivity |
        | Organic Carbon | 2 - 8 ppm | Total organic matter |
        | Trihalomethanes | 5 - 80 μg/L | Disinfection byproducts |
        | Turbidity | 0.1 - 1.0 NTU | Water clarity measure |
        
        ## 🚀 Technology Stack
        - **Machine Learning**: XGBoost Classifier
        - **Web Framework**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly, Matplotlib
        - **Explainability**: SHAP values, Feature importance
        
        ## ⚠️ Disclaimer
        This tool is for educational and preliminary analysis purposes. 
        Always consult certified water quality professionals for official testing and treatment decisions.
        
        ## 👨‍💻 Developer
        Built with ❤️ using modern ML and web technologies.
        """)

if __name__ == "__main__":
    main()
