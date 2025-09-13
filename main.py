"""
Main pipeline script to orchestrate the water quality prediction workflow
"""

import pandas as pd
import numpy as np
from data.data_loader import load_and_prepare_data
from features.feature_engineering import engineer_features
from models.xgboost_model import WaterQualityXGBoostModel
from utils.explain import WaterQualityExplainer
import warnings
warnings.filterwarnings('ignore')

def main_training_pipeline(data_path: str = "data/synthetic_water_quality_dataset.csv",
                          model_path: str = "models/water_quality_xgboost.pkl",
                          perform_tuning: bool = False):
    """
    Complete training pipeline for water quality prediction
    
    Args:
        data_path: Path to the dataset
        model_path: Path to save the trained model
        perform_tuning: Whether to perform hyperparameter tuning
    """
    print("🚰 WATER QUALITY PREDICTION PIPELINE")
    print("=" * 50)
    
    # Step 1: Load and prepare data
    print("\n📊 STEP 1: Data Loading and Preparation")
    print("-" * 40)
    data = load_and_prepare_data(
        filepath=data_path,
        test_size=0.2,
        scale_features_flag=True,
        random_state=42
    )
    
    # Step 2: Feature Engineering
    print("\n🔧 STEP 2: Feature Engineering")
    print("-" * 40)
    X_train_engineered, feature_names = engineer_features(
        data['X_train'],
        create_ratios=True,
        create_interactions=True,
        create_polynomials=False,
        select_features=False
    )
    
    # Apply same engineering to test set
    X_test_engineered, _ = engineer_features(
        data['X_test'],
        create_ratios=True,
        create_interactions=True,
        create_polynomials=False,
        select_features=False
    )
    
    # Step 3: Model Training
    print("\n🤖 STEP 3: Model Training")
    print("-" * 40)
    model = WaterQualityXGBoostModel(random_state=42, verbose=True)
    
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        tuning_results = model.hyperparameter_tuning(X_train_engineered, data['y_train'])
        best_params = tuning_results['best_params']
        print(f"Best parameters: {best_params}")
    else:
        best_params = None
    
    # Train the model
    model.train(
        X_train_engineered, 
        data['y_train'],
        X_test_engineered,
        data['y_test'],
        params=best_params
    )
    
    # Step 4: Model Evaluation
    print("\n📈 STEP 4: Model Evaluation")
    print("-" * 40)
    evaluation_results = model.evaluate(X_test_engineered, data['y_test'])
    
    # Step 5: Feature Importance Analysis
    print("\n📊 STEP 5: Feature Importance Analysis")
    print("-" * 40)
    feature_importance = model.get_feature_importance(plot=True)
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Step 6: Cross-validation
    print("\n✅ STEP 6: Cross-validation")
    print("-" * 40)
    cv_results = model.cross_validate(X_train_engineered, data['y_train'])
    
    # Step 7: Save Model
    print("\n💾 STEP 7: Saving Model")
    print("-" * 40)
    model.save_model(model_path)
    
    print("\n🎉 TRAINING PIPELINE COMPLETED!")
    print("=" * 50)
    
    return {
        'model': model,
        'data': data,
        'feature_importance': feature_importance,
        'evaluation_results': evaluation_results,
        'cv_results': cv_results
    }

def predict_water_quality(water_sample: dict, 
                         model_path: str = "models/water_quality_xgboost.pkl",
                         detailed_explanation: bool = True) -> dict:
    """
    Predict water quality for a single sample and provide explanations
    
    Args:
        water_sample: Dictionary with water quality parameters
        model_path: Path to the trained model
        detailed_explanation: Whether to provide detailed explanations
        
    Returns:
        Dictionary with prediction results and explanations
    """
    print("🔍 WATER QUALITY PREDICTION")
    print("=" * 40)
    
    # Load the trained model
    model = WaterQualityXGBoostModel()
    model.load_model(model_path)
    
    # Load the original dataset to get scaling parameters
    try:
        from data.data_loader import load_and_prepare_data
        
        # Load and prepare data to get the scaler
        data = load_and_prepare_data(
            filepath="data/synthetic_water_quality_dataset.csv",
            test_size=0.2,
            scale_features_flag=True,
            random_state=42
        )
        scaler = data['scaler']
        
        # Prepare sample for prediction
        sample_df = pd.DataFrame([water_sample])
        
        # Apply same feature engineering as training
        sample_engineered, _ = engineer_features(
            sample_df,
            create_ratios=True,
            create_interactions=True,
            create_polynomials=False,
            select_features=False
        )
        
        # Scale the features using the same scaler from training
        if scaler is not None:
            # Get the original 9 features that were scaled during training
            original_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                               'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
            
            # Scale only the original features
            sample_original = sample_df[original_features]
            sample_scaled = pd.DataFrame(
                scaler.transform(sample_original),
                columns=original_features,
                index=sample_original.index
            )
            
            # Replace the original features in the engineered dataset with scaled ones
            for col in original_features:
                if col in sample_engineered.columns:
                    sample_engineered[col] = sample_scaled[col]
        
        # Make prediction
        prediction = model.predict(sample_engineered)[0]
        prediction_proba = model.predict_proba(sample_engineered)[0]
        
    except Exception as e:
        print(f"Warning: Could not load scaler, using unscaled prediction: {e}")
        
        # Fallback: predict without scaling (less accurate)
        sample_df = pd.DataFrame([water_sample])
        sample_engineered, _ = engineer_features(
            sample_df,
            create_ratios=True,
            create_interactions=True,
            create_polynomials=False,
            select_features=False
        )
        
        prediction = model.predict(sample_engineered)[0]
        prediction_proba = model.predict_proba(sample_engineered)[0]
    
    # Get prediction probability for the predicted class
    confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
    
    results = {
        'prediction': prediction,
        'safety_status': 'SAFE' if prediction == 1 else 'UNSAFE',
        'confidence': confidence,
        'probabilities': {
            'unsafe': prediction_proba[0],
            'safe': prediction_proba[1]
        }
    }
    
    if detailed_explanation:
        # Initialize explainer
        explainer = WaterQualityExplainer()
        
        # Generate comprehensive report
        report = explainer.generate_comprehensive_report(
            water_sample, prediction, confidence
        )
        
        results['detailed_report'] = report
        
        # Print summary
        print(f"🎯 Prediction: {results['safety_status']}")
        print(f"🎲 Confidence: {confidence:.2%}")
        
        if prediction == 0:  # Unsafe
            print(f"⚠️  Issues found: {report['issue_summary']['total_issues']}")
            print("🚨 Critical parameters:")
            for param in report['issue_summary']['critical_parameters']:
                print(f"   - {param}")
            
            print("\n💡 Immediate actions needed:")
            for step in report['next_steps'][:3]:
                print(f"   • {step}")
    
    return results

def demo_prediction():
    """Demo function showing how to use the prediction system"""
    print("🧪 DEMO: Water Quality Prediction")
    print("=" * 50)
    
    # Example water samples
    samples = {
        'safe_sample': {
            'ph': 7.2,
            'Hardness': 180.0,
            'Solids': 350.0,
            'Chloramines': 2.5,
            'Sulfate': 150.0,
            'Conductivity': 400.0,
            'Organic_carbon': 5.0,
            'Trihalomethanes': 45.0,
            'Turbidity': 0.5
        },
        'unsafe_sample': {
            'ph': 4.5,  # Too acidic
            'Hardness': 180.0,
            'Solids': 1200.0,  # Too high
            'Chloramines': 8.0,  # Too high
            'Sulfate': 450.0,  # Too high
            'Conductivity': 900.0,  # Too high
            'Organic_carbon': 15.0,  # Too high
            'Trihalomethanes': 150.0,  # Too high
            'Turbidity': 5.0  # Too high
        }
    }
    
    for sample_name, sample_data in samples.items():
        print(f"\n🧪 Testing {sample_name.replace('_', ' ').title()}:")
        print("-" * 30)
        
        try:
            result = predict_water_quality(sample_data)
            print(f"Result: {result['safety_status']} (Confidence: {result['confidence']:.2%})")
        except Exception as e:
            print(f"Error: {e}")
            print("Please ensure the model is trained first by running the training pipeline.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Run training pipeline
        results = main_training_pipeline(
            perform_tuning=False  # Set to True for hyperparameter tuning
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demo
        demo_prediction()
    else:
        print("🚰 Water Quality Prediction System")
        print("=" * 40)
        print("Usage:")
        print("  python main.py train    - Run training pipeline")
        print("  python main.py demo     - Run demo predictions")
        print("\nFor custom predictions, import and use predict_water_quality() function")
