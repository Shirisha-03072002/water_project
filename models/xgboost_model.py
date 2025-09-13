"""
XGBoost model training, evaluation, saving, and loading for water quality prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import os
from tqdm import tqdm
import time

class WaterQualityXGBoostModel:
    """XGBoost model for water quality prediction"""
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        """Initialize the XGBoost model"""
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.training_history = None
        
        # Default hyperparameters
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': random_state,
            'n_estimators': 200,  # Increased for better visualization
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 20
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              params: Optional[Dict] = None) -> None:
        """
        Train the XGBoost model with visual progress
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Custom hyperparameters (optional)
        """
        print("🔄 Training XGBoost model with visual progress...")
        
        # Use custom params or defaults
        model_params = params if params is not None else self.default_params.copy()
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Show training configuration
        if self.verbose:
            print(f"📊 Training Configuration:")
            print(f"   Training samples: {len(X_train):,}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Estimators: {model_params.get('n_estimators', 100)}")
            print(f"   Learning rate: {model_params.get('learning_rate', 0.1)}")
            print(f"   Max depth: {model_params.get('max_depth', 6)}")
            
        # Prepare evaluation set
        eval_set = []
        eval_names = []
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'validation']
            print(f"   Validation samples: {len(X_val):,}")
        else:
            eval_set = [(X_train, y_train)]
            eval_names = ['train']
        
        print("\n🚀 Starting training process...")
        
        # Initialize model
        self.model = xgb.XGBClassifier(**model_params)
        
        # Create a simple progress tracker using fit parameters
        if self.verbose:
            print("📊 Training in progress...")
            
            # Use verbose parameter for XGBoost built-in progress
            verbose_eval = max(1, model_params.get('n_estimators', 100) // 10)
            
            # Train the model with built-in verbose output
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=verbose_eval  # Show progress every 10% of iterations
            )
        else:
            # Train without verbose output
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        
        # Store training history
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        # Manual progress simulation for better user experience
        if self.verbose:
            print("\n⏱️  Simulating training progress visualization...")
            with tqdm(total=100, desc="Training Complete", unit="%") as pbar:
                for i in range(100):
                    time.sleep(0.01)  # Small delay for visual effect
                    pbar.update(1)
            print("✨ Training visualization complete!")
        
        self.is_trained = True
        print("✅ Model training completed!")
        
        # Print training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"   Training accuracy: {train_accuracy:.4f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            print(f"   Validation accuracy: {val_accuracy:.4f}")
        
        # Plot training curves if we have validation data
        if self.training_history and len(eval_set) > 1:
            self.plot_training_curves()
    
    def plot_training_curves(self) -> None:
        """Plot training and validation curves"""
        if not self.training_history:
            print("⚠️  No training history available for plotting")
            return
        
        print(f"📊 Available training history keys: {list(self.training_history.keys())}")
        
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Loss curves
        plt.subplot(1, 2, 1)
        
        # Handle different key formats in training history
        available_keys = list(self.training_history.keys())
        
        for eval_name in available_keys:
            metrics = self.training_history[eval_name]
            if 'logloss' in metrics:
                plt.plot(metrics['logloss'], label=f'{eval_name} loss')
            elif 'log_loss' in metrics:
                plt.plot(metrics['log_loss'], label=f'{eval_name} loss')
        
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('Training Progress - Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Learning progress visualization
        plt.subplot(1, 2, 2)
        
        # Get the first available dataset's loss values
        if available_keys:
            first_key = available_keys[0]
            metrics = self.training_history[first_key]
            
            # Try different metric names
            loss_values = None
            if 'logloss' in metrics:
                loss_values = metrics['logloss']
            elif 'log_loss' in metrics:
                loss_values = metrics['log_loss']
            elif 'error' in metrics:
                loss_values = metrics['error']
            
            if loss_values:
                iterations = range(len(loss_values))
                plt.plot(iterations, loss_values, 'b-', label=f'{first_key} Loss', linewidth=2)
                
                # If we have multiple datasets, plot validation too
                if len(available_keys) > 1:
                    second_key = available_keys[1]
                    val_metrics = self.training_history[second_key]
                    
                    val_loss = None
                    if 'logloss' in val_metrics:
                        val_loss = val_metrics['logloss']
                    elif 'log_loss' in val_metrics:
                        val_loss = val_metrics['log_loss']
                    elif 'error' in val_metrics:
                        val_loss = val_metrics['error']
                    
                    if val_loss:
                        plt.plot(iterations, val_loss, 'r-', label=f'{second_key} Loss', linewidth=2)
                        
                        # Highlight best iteration
                        best_iter = np.argmin(val_loss)
                        plt.axvline(x=best_iter, color='green', linestyle='--', alpha=0.7, 
                                   label=f'Best iteration ({best_iter})')
                        plt.scatter(best_iter, val_loss[best_iter], color='green', s=100, zorder=5)
                
                plt.xlabel('Iteration')
                plt.ylabel('Loss Value')
                plt.title('Model Learning Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No loss metrics found', 
                        transform=plt.gca().transAxes, ha='center', va='center')
                plt.title('No Training Metrics Available')
        else:
            plt.text(0.5, 0.5, 'No training history available', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('No Training History')
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Training curves saved as 'training_progress.png'")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("🔄 Evaluating model performance...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        print("✅ Model evaluation completed!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC Score: {auc_score:.4f}")
        print(f"   Precision (Safe): {class_report['1']['precision']:.4f}")
        print(f"   Recall (Safe): {class_report['1']['recall']:.4f}")
        print(f"   Precision (Unsafe): {class_report['0']['precision']:.4f}")
        print(f"   Recall (Unsafe): {class_report['0']['recall']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, plot: bool = True) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Args:
            plot: Whether to create a plot
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature importance
        importance_values = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(15)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title('Top 15 Feature Importance (XGBoost)')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig('feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return importance_df
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             cv_folds: int = 5) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and scores
        """
        print("🔄 Starting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Initialize base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state
        )
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print("✅ Hyperparameter tuning completed!")
        print(f"   Best accuracy: {grid_search.best_score_:.4f}")
        print(f"   Best parameters: {grid_search.best_params_}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'grid_search_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str = "models/water_quality_xgboost.pkl") -> None:
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath: str = "models/water_quality_xgboost.pkl") -> None:
        """Load a saved model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        print(f"✅ Model loaded from: {filepath}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        print(f"🔄 Performing {cv_folds}-fold cross-validation...")
        
        # Create model params without early stopping for cross-validation
        cv_params = self.default_params.copy()
        if 'early_stopping_rounds' in cv_params:
            del cv_params['early_stopping_rounds']
        
        # Initialize model
        model = xgb.XGBClassifier(**cv_params)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
        
        print("✅ Cross-validation completed!")
        print(f"   Mean accuracy: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        
        return results
