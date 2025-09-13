"""
Updated Streamlit Web Application for Water Quality Prediction
===============================================================

This app provides a user-friendly interface for:
1. User authentication (login/register)
2. Predicting water quality safety
3. Explaining predictions with detailed analysis
4. Providing treatment recommendations
5. Visualizing model performance
6. Tracking search history per user
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
import json
import struct  # For binary data unpacking
import io
from io import BytesIO  # For PDF output
# PDF generation imports will be handled dynamically in the generate_pdf_report function

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our modules
from data.data_loader import load_water_quality_data, prepare_features_target
from features.feature_engineering import engineer_features
from models.xgboost_model import WaterQualityXGBoostModel
from utils.explain import WaterQualityExplainer
from main import predict_water_quality
from database.session import (
    init_session_state, 
    render_auth_ui, 
    is_authenticated, 
    get_current_user,
    logout_user
)
from database.history import history

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
    
    .auth-form {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .history-item {
        background: white;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1e3c72;
        cursor: pointer;
    }
    
    .history-item:hover {
        background: #f8f9fa;
    }
    
    .safe-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
    }
    
    .unsafe-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
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
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="unsafe-water">
            <h2>⚠️ WATER IS NOT SAFE FOR CONSUMPTION</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Always show the unsafe probability (risk) on the gauge
    unsafe_probability = prediction_result['probabilities']['unsafe'] * 100  # This is the risk probability
    safe_probability = prediction_result['probabilities']['safe'] * 100      # This is the safe probability
    
    # Always use "Risk Probability (%)" for the gauge title for consistency
    gauge_title = "Risk Probability (%)"
    
    # Display the risk level (unsafe probability) on the gauge
    # Red represents higher risk
    gauge_steps = [
        {'range': [0, 50], 'color': "green"},
        {'range': [50, 80], 'color': "orange"},
        {'range': [80, 100], 'color': "red"}
    ]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = unsafe_probability,  # Always show unsafe probability (risk)
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': gauge_title},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': gauge_steps,
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display both probabilities below the gauge
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Safe Probability", value=f"{safe_probability:.1f}%", delta=None)
    with col2:
        st.metric(label="Risk Probability", value=f"{unsafe_probability:.1f}%", delta=None)
    
    # Add explanation for the gauge
    st.caption("The gauge above shows the risk level of your water. Higher values (red) indicate greater risk.")

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

def generate_pdf_report(detailed_report, assessment):
    """Generate a PDF report for water quality analysis using ReportLab
    
    Args:
        detailed_report: Detailed water quality report
        assessment: Overall assessment data
        
    Returns:
        PDF file bytes
    """
    # Use ReportLab which handles Unicode better
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    
    # Create a buffer to receive PDF data
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom styles
    red_style = ParagraphStyle(
        'RedStyle', 
        parent=normal_style, 
        textColor=colors.red
    )
    
    green_style = ParagraphStyle(
        'GreenStyle', 
        parent=normal_style, 
        textColor=colors.green
    )
    
    # Add title
    elements.append(Paragraph('Water Quality Analysis Report', title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add metadata
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Overall assessment
    elements.append(Paragraph('Overall Assessment', heading_style))
    
    # Safety status with color
    if assessment['safety_status'] == 'SAFE':
        status_style = green_style
    else:
        status_style = red_style
    
    elements.append(Paragraph(f"Status: {assessment['safety_status']}", status_style))
    
    # Get probabilities from the assessment
    safe_probability = assessment.get('probabilities', {}).get('safe', 0) * 100
    unsafe_probability = assessment.get('probabilities', {}).get('unsafe', 0) * 100
    
    # If probabilities are not in the assessment, use a default
    if safe_probability == 0 and unsafe_probability == 0:
        if assessment['safety_status'] == 'SAFE':
            safe_probability = assessment['confidence'] * 100
            unsafe_probability = 100 - safe_probability
        else:
            unsafe_probability = assessment['confidence'] * 100
            safe_probability = 100 - unsafe_probability
    
    elements.append(Paragraph(f"Safe Probability: {safe_probability:.1f}%", normal_style))
    elements.append(Paragraph(f"Risk Probability: {unsafe_probability:.1f}%", normal_style))
    elements.append(Paragraph(f"Recommendation: {assessment['recommendation']}", normal_style))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Parameter analysis
    elements.append(Paragraph('Parameter Analysis', heading_style))
    
    # Create parameter table
    param_data = [
        ['Parameter', 'Value', 'Safe Range', 'Status', 'Issue Type']
    ]
    
    # Add parameter data
    for param, analysis in detailed_report['parameter_analysis'].items():
        status = "Safe" if analysis['status'] == 'safe' else "Unsafe"
        issue = analysis['issue_type'].replace('_', ' ').title() if analysis['issue_type'] else 'None'
        param_data.append([
            param.replace('_', ' ').title(),
            f"{analysis['value']:.2f}",
            analysis['safe_range'],
            status,
            issue
        ])
    
    # Create table
    param_table = Table(param_data, colWidths=[1.5*inch, 0.8*inch, 1.2*inch, 0.8*inch, 1.5*inch])
    
    # Add table style
    param_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    
    # Add color to status cells
    for i in range(1, len(param_data)):
        if param_data[i][3] == "Safe":
            param_style.add('TEXTCOLOR', (3, i), (3, i), colors.green)
        else:
            param_style.add('TEXTCOLOR', (3, i), (3, i), colors.red)
    
    param_table.setStyle(param_style)
    elements.append(param_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Issues summary
    elements.append(Paragraph('Issues and Recommendations', heading_style))
    elements.append(Paragraph(f"Total Issues Found: {detailed_report['issue_summary']['total_issues']}", normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Group suggestions by priority
    high_priority = [s for s in detailed_report['suggestions'] if s['priority'] == 'high']
    medium_priority = [s for s in detailed_report['suggestions'] if s['priority'] == 'medium']
    low_priority = [s for s in detailed_report['suggestions'] if s['priority'] == 'low']
    
    # Add issues by priority
    if high_priority:
        high_style = ParagraphStyle('HighPriority', parent=subheading_style, textColor=colors.red)
        elements.append(Paragraph("HIGH PRIORITY ISSUES", high_style))
        
        for suggestion in high_priority:
            elements.append(Paragraph(f"{suggestion['parameter']} - {suggestion['severity'].title()} Issue", subheading_style))
            elements.append(Paragraph(f"<b>Problem:</b> {suggestion['issue']}", normal_style))
            elements.append(Paragraph(f"<b>Health Impact:</b> {suggestion['health_impact']}", normal_style))
            
            elements.append(Paragraph("<b>Recommended Treatments:</b>", normal_style))
            for i, treatment in enumerate(suggestion['treatments'], 1):
                elements.append(Paragraph(f"{i}. {treatment}", normal_style))
            
            elements.append(Spacer(1, 0.15*inch))
    
    # Next steps
    elements.append(Paragraph('Next Steps', heading_style))
    
    for i, step in enumerate(detailed_report['next_steps'], 1):
        if step.startswith("DO NOT"):
            elements.append(Paragraph(f"{i}. {step}", red_style))
        elif step.startswith("Address high-priority"):
            step_style = ParagraphStyle('OrangeStyle', parent=normal_style, textColor=colors.orangered)
            elements.append(Paragraph(f"{i}. {step}", step_style))
        else:
            elements.append(Paragraph(f"{i}. {step}", normal_style))
    
    elements.append(Spacer(1, 0.5*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle('Disclaimer', parent=normal_style, fontSize=8, fontName='Helvetica-Oblique')
    elements.append(Paragraph("Disclaimer: This report is for informational purposes only. Always consult with water quality professionals for official testing and treatment decisions.", disclaimer_style))
    
    # Build the PDF
    doc.build(elements)
    
    # Get PDF value from buffer
    buffer.seek(0)
    return buffer.getvalue()

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

def display_user_history():
    """Display search history for logged in user"""
    st.subheader("📜 Your Search History")
    
    user = get_current_user()
    
    if not user:
        st.warning("Please log in to view your search history")
        return
    
    try:
        # Get user's search history
        user_history = history.get_user_history(user['user_id'])
        
        if not user_history:
            st.info("You don't have any search history yet")
            return
        
        # Display search history
        for record in user_history:
            try:
                # Format timestamp
                timestamp = record['timestamp']
                if isinstance(timestamp, bytes):
                    timestamp = timestamp.decode('utf-8')
                timestamp = timestamp.split('.')[0] if '.' in timestamp else timestamp
                
                # Create expandable card for each search
                result_text = '✅ SAFE' if record['result'] == 'SAFE' else '⚠️ UNSAFE'
                confidence = record['confidence']
                if isinstance(confidence, bytes):
                    # Try to decode and convert to float
                    try:
                        confidence = struct.unpack('f', confidence)[0]
                    except:
                        confidence = 0.5  # Default if conversion fails
                elif not isinstance(confidence, (float, int)):
                    try:
                        confidence = float(confidence)
                    except:
                        confidence = 0.5  # Default if conversion fails
                
                with st.expander(f"**{timestamp}** - Result: {result_text} ({confidence:.1%})"):
                    # Display water parameters
                    st.write("**Water Parameters:**")
                    params_df = pd.DataFrame([record['water_params']])
                    st.dataframe(params_df.T.rename(columns={0: 'Value'}), use_container_width=True)
                    
                    # Add button to view full details
                    if st.button(f"View Full Report", key=f"view_{record['search_id']}"):
                        # Set session state to view this report
                        st.session_state.viewing_report = record['search_id']
                        st.rerun()
                    
                    # Add button to reuse these parameters
                    if st.button(f"Reuse These Parameters", key=f"reuse_{record['search_id']}"):
                        # Set session state to reuse these parameters
                        st.session_state.reuse_parameters = record['water_params']
                        st.rerun()
            except Exception as e:
                st.error(f"Error displaying record: {str(e)}")
                st.write(f"Record data: {record}")
                continue
    
    except Exception as e:
        st.error(f"Error retrieving search history: {str(e)}")
        st.error("Please try logging out and back in, or contact support if the issue persists.")

def display_history_details(search_id):
    """Display detailed view of a specific search from history"""
    try:
        # Get detailed search information
        search_details = history.get_search_details(search_id)
        
        if not search_details:
            st.error(f"Search details not found for ID: {search_id}")
            return
        
        st.subheader("🔍 Search History Details")
        
        # Display timestamp and result
        col1, col2 = st.columns(2)
        with col1:
            # Get timestamp from the appropriate field
            timestamp = search_details.get('timestamp') or search_details.get('search_timestamp')
            if isinstance(timestamp, bytes):
                timestamp = timestamp.decode('utf-8')
            timestamp = timestamp.split('.')[0] if '.' in timestamp else timestamp
            st.info(f"**Date & Time**: {timestamp}")
        
        with col2:
            confidence = search_details['confidence'] 
            if isinstance(confidence, bytes):
                # Try to decode and convert to float
                try:
                    confidence = struct.unpack('f', confidence)[0]
                except:
                    confidence = 0.5  # Default if conversion fails
            elif not isinstance(confidence, (float, int)):
                try:
                    confidence = float(confidence)
                except:
                    confidence = 0.5  # Default if conversion fails
                
            if search_details['result'] == 'SAFE':
                st.success(f"**Result**: SAFE (Confidence: {confidence:.1%})")
            else:
                st.error(f"**Result**: UNSAFE (Confidence: {confidence:.1%})")
        
        # Display water parameters
        st.write("**Water Parameters:**")
        params_df = pd.DataFrame([search_details['water_params']])
        st.dataframe(params_df, use_container_width=True)
        
        # If we have detailed report, display it
        if 'parameter_analysis' in search_details:
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Parameter Analysis", 
                "💡 Recommendations", 
                "🎯 Next Steps",
                "📋 Download Report"
            ])
            
            # Recreate detailed report structure for display functions
            detailed_report = {
                'parameter_analysis': search_details['parameter_analysis'],
                'suggestions': search_details['suggestions'],
                'next_steps': search_details['next_steps'],
                'issue_summary': {'total_issues': len(search_details['suggestions'])}
            }
            
            with tab1:
                display_parameter_analysis(detailed_report)
            
            with tab2:
                display_suggestions(detailed_report)
            
            with tab3:
                display_next_steps(detailed_report)
                
            with tab4:
                st.subheader("📋 Download Report Options")
                
                # Create overall assessment dict for PDF generation
                assessment = {
                    'safety_status': search_details['result'],
                    'confidence': search_details['confidence'],
                    'recommendation': "Please consult with water quality professionals for specific recommendations."
                }
                
                # Generate reports
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Text report
                report_text = f"""
Water Quality Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

OVERALL ASSESSMENT:
Status: {search_details['result']}
Confidence: {search_details['confidence']:.1%}

PARAMETER VALUES:
"""
                for param, analysis in detailed_report['parameter_analysis'].items():
                    report_text += f"{param}: {analysis['value']:.2f} ({analysis['status']})\n"
                
                report_text += f"\nISSUES FOUND: {len(search_details['suggestions'])}\n"
                
                for suggestion in detailed_report['suggestions']:
                    report_text += f"\n{suggestion['parameter']}: {suggestion['issue']}\n"
                    report_text += f"Priority: {suggestion['priority']}\n"
                    report_text += f"Treatments: {', '.join(suggestion['treatments'])}\n"
                
                # Download buttons
                try:
                    pdf_bytes = generate_pdf_report(detailed_report, assessment)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📄 Download Text Report",
                            data=report_text,
                            file_name=f"water_quality_report_{timestamp}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        st.download_button(
                            label="📊 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"water_quality_report_{timestamp}.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Error generating PDF report: {str(e)}")
                    st.download_button(
                        label="📄 Download Text Report",
                        data=report_text,
                        file_name=f"water_quality_report_{timestamp}.txt",
                        mime="text/plain"
                    )
        
        # Button to return to main view
        if st.button("Back to Main View"):
            st.session_state.viewing_report = None
            st.rerun()
        
    except Exception as e:
        st.error(f"Error displaying search details: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Check if we're viewing a specific report
    if is_authenticated() and 'viewing_report' in st.session_state and st.session_state.viewing_report:
        display_history_details(st.session_state.viewing_report)
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚰 Water Quality Prediction System</h1>
        <p>AI-powered water safety analysis with treatment recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    
    # Authentication UI - prominently displayed on main page if not logged in
    if not is_authenticated():
        st.subheader("👋 Welcome to Water Quality Prediction System!")
        st.write("Please login or create an account to access the water quality prediction features.")
        st.info("Creating an account allows you to save your water quality predictions and view your history.")
        render_auth_ui()
        
        # For unauthenticated users, only show About section
        app_mode = "📚 About"
        
        # Add some information about the system for unauthenticated users
        st.markdown("---")
        st.subheader("🚰 System Overview")
        st.write("""
        This AI-powered system analyzes water quality parameters to determine if water is safe for consumption.
        
        Create an account to:
        - Predict water quality safety with 99%+ accuracy
        - Get detailed parameter analysis and health impact information
        - Receive treatment recommendations for unsafe water
        - Track your water quality history over time
        """)
    else:
        # Show logged in status in sidebar
        st.sidebar.success(f"Logged in as: {st.session_state.user['username']}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()
        
        # Full navigation menu for authenticated users
        app_mode = st.sidebar.selectbox(
            "Choose App Mode",
            ["🧪 Predict Water Quality", "📊 Model Performance", "📜 Search History", "📚 About"]
        )
    
    # Search history mode requires authentication
    if app_mode == "📜 Search History" and not is_authenticated():
        st.warning("Please log in to view your search history")
        return
    
    if app_mode == "🧪 Predict Water Quality":
        st.header("Water Quality Prediction")
        
        # Verify authentication before allowing prediction
        if not is_authenticated():
            st.warning("Please login to access the water quality prediction features.")
            st.stop()
            
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Input form - check if we're reusing parameters
        if 'reuse_parameters' in st.session_state and st.session_state.reuse_parameters:
            water_sample = st.session_state.reuse_parameters
            st.info("Using parameters from your search history")
            
            # Display the parameters
            params_df = pd.DataFrame([water_sample])
            st.dataframe(params_df, use_container_width=True)
            
            # Add button to reset parameters
            if st.button("Reset Parameters"):
                st.session_state.reuse_parameters = None
                st.experimental_rerun()
        else:
            water_sample = create_parameter_input_form()
        
        # Get current user ID if authenticated
        user_id = None
        if is_authenticated():
            user = get_current_user()
            user_id = user['user_id']
        
        # Prediction button
        if st.button("🔍 Analyze Water Quality", type="primary"):
            with st.spinner("Analyzing water quality..."):
                try:
                    # Make prediction
                    prediction_result = predict_water_quality(
                        water_sample, 
                        detailed_explanation=True,
                        user_id=user_id,
                        save_to_history=is_authenticated()
                    )
                    
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
                            
                            # Calculate probabilities
                            safe_prob = prediction_result['probabilities']['safe'] * 100
                            unsafe_prob = prediction_result['probabilities']['unsafe'] * 100
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Safety Status", assessment['safety_status'])
                            with col2:
                                st.metric("Safe Probability", f"{safe_prob:.1f}%")
                            with col3:
                                st.metric("Risk Probability", f"{unsafe_prob:.1f}%")
                            with col4:
                                st.metric("Total Issues", detailed_report['issue_summary']['total_issues'])
                            
                            # Download report options
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Generate text report
                            report_text = f"""
Water Quality Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

OVERALL ASSESSMENT:
Status: {assessment['safety_status']}
Safe Probability: {prediction_result['probabilities']['safe']:.1%}
Risk Probability: {prediction_result['probabilities']['unsafe']:.1%}
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
                            
                            # Generate PDF report
                            try:
                                pdf_bytes = generate_pdf_report(detailed_report, assessment)
                                
                                # Create columns for download buttons
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Text report download button
                                    st.download_button(
                                        label="📄 Download Text Report",
                                        data=report_text,
                                        file_name=f"water_quality_report_{timestamp}.txt",
                                        mime="text/plain"
                                    )
                                
                                with col2:
                                    # PDF report download button
                                    st.download_button(
                                        label="� Download PDF Report",
                                        data=pdf_bytes,
                                        file_name=f"water_quality_report_{timestamp}.pdf",
                                        mime="application/pdf"
                                    )
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                                # Fall back to text report if PDF generation fails
                                st.download_button(
                                    label="�📄 Download Text Report",
                                    data=report_text,
                                    file_name=f"water_quality_report_{timestamp}.txt",
                                    mime="text/plain"
                                )
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        
        # Debug test button for troubleshooting
        with st.expander("🔬 Quick Debug Test"):
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
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Test Safe Sample"):
                    try:
                        result = predict_water_quality(
                            safe_sample, 
                            detailed_explanation=False,
                            user_id=user_id if is_authenticated() else None,
                            save_to_history=False
                        )
                        st.write(f"Prediction: {result['safety_status']}")
                        st.write(f"Confidence: {result['confidence']:.2%}")
                    except Exception as e:
                        st.error(f"Test failed: {e}")
            
            with col2:
                if st.button("Test Unsafe Sample"):
                    try:
                        result = predict_water_quality(
                            unsafe_sample, 
                            detailed_explanation=False,
                            user_id=user_id if is_authenticated() else None,
                            save_to_history=False
                        )
                        st.write(f"Prediction: {result['safety_status']}")
                        st.write(f"Confidence: {result['confidence']:.2%}")
                    except Exception as e:
                        st.error(f"Test failed: {e}")
    
    elif app_mode == "📊 Model Performance":
        st.header("Model Performance Dashboard")
        
        # Verify authentication before allowing access to model performance
        if not is_authenticated():
            st.warning("Please login to access the model performance dashboard.")
            st.stop()
            
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
    
    elif app_mode == "📜 Search History":
        st.header("Your Search History")
        
        if is_authenticated():
            display_user_history()
        else:
            st.warning("Please log in to view your search history")
    
    elif app_mode == "📚 About":
        st.header("About Water Quality Prediction System")
        
        # Show a call-to-action for non-authenticated users
        if not is_authenticated():
            st.info("👋 **Create an account or login above to access the full functionality of the system!**")
        
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
        
        ## 👤 User Accounts
        - **Create an account** to save your search history
        - **View past searches** to track water quality over time
        - **Reuse parameters** from previous searches
        
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
        - **Database**: SQLite
        - **Authentication**: JWT, Bcrypt
        
        ## ⚠️ Disclaimer
        This tool is for educational and preliminary analysis purposes. 
        Always consult certified water quality professionals for official testing and treatment decisions.
        
        ## 👨‍💻 Developer
        Built with ❤️ using modern ML and web technologies.
        """)
        
        # Repeat the call-to-action at the bottom for non-authenticated users
        if not is_authenticated():
            st.success("👆 **Sign up above to start using the Water Quality Prediction System!**")

if __name__ == "__main__":
    # Initialize session state with reuse parameters
    if 'reuse_parameters' not in st.session_state:
        st.session_state.reuse_parameters = None
    
    # Initialize viewing report state
    if 'viewing_report' not in st.session_state:
        st.session_state.viewing_report = None
    
    main()
