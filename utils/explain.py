"""
Explanation and suggestion logic for water quality predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import shap

class WaterQualityExplainer:
    """Provides explanations and suggestions for water quality predictions"""
    
    def __init__(self):
        """Initialize the explainer with water quality standards"""
        # WHO and EPA drinking water standards
        self.safe_ranges = {
            'ph': {'min': 6.5, 'max': 8.5, 'unit': '', 'name': 'pH'},
            'Hardness': {'min': 60, 'max': 120, 'unit': 'mg/L', 'name': 'Hardness'},
            'Solids': {'min': 100, 'max': 500, 'unit': 'ppm', 'name': 'Total Dissolved Solids'},
            'Chloramines': {'min': 0.2, 'max': 4.0, 'unit': 'ppm', 'name': 'Chloramines'},
            'Sulfate': {'min': 50, 'max': 250, 'unit': 'mg/L', 'name': 'Sulfate'},
            'Conductivity': {'min': 200, 'max': 600, 'unit': 'μS/cm', 'name': 'Conductivity'},
            'Organic_carbon': {'min': 2, 'max': 8, 'unit': 'ppm', 'name': 'Organic Carbon'},
            'Trihalomethanes': {'min': 5, 'max': 80, 'unit': 'μg/L', 'name': 'Trihalomethanes'},
            'Turbidity': {'min': 0.1, 'max': 1.0, 'unit': 'NTU', 'name': 'Turbidity'}
        }
        
        # Health impact descriptions
        self.health_impacts = {
            'ph': {
                'low': 'Acidic water can corrode pipes and leach metals, causing gastrointestinal issues',
                'high': 'Alkaline water can cause skin irritation and taste issues'
            },
            'Hardness': {
                'low': 'Very soft water may corrode pipes and lack essential minerals',
                'high': 'Very hard water can cause scale buildup and taste issues'
            },
            'Solids': {
                'low': 'Very low TDS may indicate lack of essential minerals',
                'high': 'High TDS can cause taste issues and potential health concerns'
            },
            'Chloramines': {
                'low': 'Insufficient disinfection may allow harmful bacteria',
                'high': 'Excessive chloramines can cause taste, odor, and respiratory issues'
            },
            'Sulfate': {
                'low': 'Low sulfate is generally not a concern',
                'high': 'High sulfate can cause gastrointestinal distress and laxative effects'
            },
            'Conductivity': {
                'low': 'Low conductivity may indicate very pure but mineral-deficient water',
                'high': 'High conductivity indicates high dissolved solids content'
            },
            'Organic_carbon': {
                'low': 'Low organic carbon is generally good',
                'high': 'High organic carbon can lead to disinfection byproducts and taste issues'
            },
            'Trihalomethanes': {
                'low': 'Low THMs is ideal for safety',
                'high': 'High THMs are linked to cancer risk and other health issues'
            },
            'Turbidity': {
                'low': 'Low turbidity indicates clear, well-filtered water',
                'high': 'High turbidity can harbor pathogens and reduce disinfection effectiveness'
            }
        }
        
        # Treatment suggestions
        self.treatment_suggestions = {
            'ph': {
                'low': ['Add pH adjustment chemicals (sodium hydroxide)', 'Install pH correction system', 'Use limestone filtration'],
                'high': ['Add acid injection system', 'Use acidification treatment', 'Install pH adjustment equipment']
            },
            'Hardness': {
                'low': ['Consider remineralization if needed', 'Add essential minerals'],
                'high': ['Install water softener', 'Use ion exchange treatment', 'Consider reverse osmosis']
            },
            'Solids': {
                'low': ['Consider remineralization for health benefits'],
                'high': ['Install reverse osmosis system', 'Use distillation', 'Apply advanced filtration']
            },
            'Chloramines': {
                'low': ['Increase disinfection treatment', 'Check disinfection system'],
                'high': ['Reduce chloramine dosing', 'Install activated carbon filter', 'Use alternative disinfection']
            },
            'Sulfate': {
                'low': ['No treatment typically needed'],
                'high': ['Install reverse osmosis', 'Use ion exchange', 'Consider distillation']
            },
            'Conductivity': {
                'low': ['Consider remineralization if water is too pure'],
                'high': ['Install reverse osmosis', 'Use deionization', 'Apply advanced filtration']
            },
            'Organic_carbon': {
                'low': ['Maintain current treatment'],
                'high': ['Install activated carbon filter', 'Use advanced oxidation', 'Improve coagulation/flocculation']
            },
            'Trihalomethanes': {
                'low': ['Maintain current disinfection practices'],
                'high': ['Install activated carbon filter', 'Use alternative disinfection', 'Reduce organic precursors']
            },
            'Turbidity': {
                'low': ['Maintain current filtration'],
                'high': ['Improve filtration system', 'Use coagulation/flocculation', 'Install better sediment filters']
            }
        }
    
    def analyze_parameters(self, water_sample: Dict) -> Dict:
        """
        Analyze individual water parameters against safety standards
        
        Args:
            water_sample: Dictionary with water quality parameters
            
        Returns:
            Dictionary with parameter analysis
        """
        analysis = {}
        
        for param, value in water_sample.items():
            if param in self.safe_ranges:
                ranges = self.safe_ranges[param]
                param_analysis = {
                    'value': value,
                    'safe_range': f"{ranges['min']}-{ranges['max']} {ranges['unit']}",
                    'status': 'safe',
                    'issue_type': None,
                    'health_impact': None,
                    'severity': 'none'
                }
                
                if value < ranges['min']:
                    param_analysis['status'] = 'unsafe'
                    param_analysis['issue_type'] = 'too_low'
                    param_analysis['health_impact'] = self.health_impacts[param]['low']
                    param_analysis['severity'] = self._calculate_severity(value, ranges['min'], 'low', param)
                    
                elif value > ranges['max']:
                    param_analysis['status'] = 'unsafe'
                    param_analysis['issue_type'] = 'too_high'
                    param_analysis['health_impact'] = self.health_impacts[param]['high']
                    param_analysis['severity'] = self._calculate_severity(value, ranges['max'], 'high', param)
                
                analysis[param] = param_analysis
        
        return analysis
    
    def _calculate_severity(self, value: float, threshold: float, direction: str, param: str) -> str:
        """Calculate severity of parameter violation"""
        if direction == 'low':
            ratio = threshold / value if value > 0 else float('inf')
        else:  # high
            ratio = value / threshold
        
        if ratio < 1.2:
            return 'mild'
        elif ratio < 2.0:
            return 'moderate'
        else:
            return 'severe'
    
    def get_suggestions(self, parameter_analysis: Dict) -> List[Dict]:
        """
        Generate treatment suggestions based on parameter analysis
        
        Args:
            parameter_analysis: Output from analyze_parameters
            
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        
        for param, analysis in parameter_analysis.items():
            if analysis['status'] == 'unsafe':
                issue_type = analysis['issue_type']
                severity = analysis['severity']
                
                # Map issue_type to treatment key
                treatment_key = 'low' if issue_type == 'too_low' else 'high'
                
                # Get treatments for this parameter and issue type
                if param in self.treatment_suggestions and treatment_key in self.treatment_suggestions[param]:
                    treatments = self.treatment_suggestions[param][treatment_key]
                else:
                    # Fallback treatments if parameter not found
                    treatments = ['Consult water quality specialist', 'Professional testing recommended']
                
                suggestion = {
                    'parameter': self.safe_ranges[param]['name'] if param in self.safe_ranges else param,
                    'issue': f"Value {analysis['value']:.2f} is {issue_type.replace('_', ' ')}",
                    'severity': severity,
                    'health_impact': analysis['health_impact'],
                    'treatments': treatments,
                    'priority': self._get_priority(param, severity)
                }
                
                suggestions.append(suggestion)
        
        # Sort by priority (high to low)
        suggestions.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
        
        return suggestions
    
    def _get_priority(self, param: str, severity: str) -> str:
        """Determine priority level based on parameter and severity"""
        high_priority_params = ['ph', 'Trihalomethanes', 'Turbidity']
        medium_priority_params = ['Chloramines', 'Organic_carbon']
        
        if param in high_priority_params:
            if severity in ['moderate', 'severe']:
                return 'high'
            else:
                return 'medium'
        elif param in medium_priority_params:
            if severity == 'severe':
                return 'high'
            else:
                return 'medium'
        else:
            if severity == 'severe':
                return 'medium'
            else:
                return 'low'
    
    def explain_prediction_with_shap(self, model, X_sample: pd.DataFrame, 
                                   X_background: pd.DataFrame) -> Dict:
        """
        Use SHAP to explain model predictions
        
        Args:
            model: Trained XGBoost model
            X_sample: Sample to explain
            X_background: Background dataset for SHAP
            
        Returns:
            Dictionary with SHAP explanation
        """
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Get feature contributions
            feature_contributions = {}
            for i, feature in enumerate(X_sample.columns):
                feature_contributions[feature] = shap_values[0][i]
            
            # Sort by absolute contribution
            sorted_contributions = sorted(
                feature_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            return {
                'shap_values': shap_values,
                'feature_contributions': feature_contributions,
                'top_contributors': sorted_contributions[:5],
                'explainer': explainer
            }
            
        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, water_sample: Dict, prediction: int, 
                                    prediction_proba: float, 
                                    feature_importance: Dict = None) -> Dict:
        """
        Generate a comprehensive water quality report
        
        Args:
            water_sample: Water quality parameters
            prediction: Model prediction (0=unsafe, 1=safe)
            prediction_proba: Prediction probability
            feature_importance: Feature importance from model
            
        Returns:
            Comprehensive report dictionary
        """
        # Analyze parameters
        parameter_analysis = self.analyze_parameters(water_sample)
        
        # Get suggestions
        suggestions = self.get_suggestions(parameter_analysis)
        
        # Overall assessment
        safety_status = "SAFE" if prediction == 1 else "UNSAFE"
        confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
        
        # Count issues by severity
        issue_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
        for analysis in parameter_analysis.values():
            if analysis['status'] == 'unsafe':
                issue_counts[analysis['severity']] += 1
        
        report = {
            'overall_assessment': {
                'safety_status': safety_status,
                'confidence': confidence,
                'recommendation': "Safe for consumption" if prediction == 1 else "NOT safe for consumption"
            },
            'parameter_analysis': parameter_analysis,
            'issue_summary': {
                'total_issues': len(suggestions),
                'by_severity': issue_counts,
                'critical_parameters': [s['parameter'] for s in suggestions if s['priority'] == 'high']
            },
            'suggestions': suggestions,
            'next_steps': self._get_next_steps(suggestions, prediction)
        }
        
        return report
    
    def _get_next_steps(self, suggestions: List[Dict], prediction: int) -> List[str]:
        """Generate next steps based on analysis"""
        if prediction == 1 and len(suggestions) == 0:
            return ["Water is safe for consumption", "Continue regular monitoring"]
        
        next_steps = []
        
        if len(suggestions) > 0:
            next_steps.append("DO NOT consume this water until issues are resolved")
            
            high_priority_issues = [s for s in suggestions if s['priority'] == 'high']
            if high_priority_issues:
                next_steps.append("Address high-priority issues immediately:")
                for issue in high_priority_issues[:3]:  # Top 3
                    next_steps.append(f"  - {issue['parameter']}: {issue['treatments'][0]}")
            
            next_steps.append("Consider professional water testing and treatment")
            next_steps.append("Consult with water quality specialists")
        
        return next_steps
