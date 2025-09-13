"""
Data loading and preprocessing module for water quality prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import os

def load_water_quality_data(filepath: str = "data/synthetic_water_quality_dataset.csv") -> pd.DataFrame:
    """
    Load water quality dataset from CSV file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame: Loaded water quality data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please run generate_synthetic_data.py first.")
    
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded successfully from {filepath}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Target: {df.columns[-1]}")
    
    return df

def prepare_features_target(df: pd.DataFrame, target_column: str = "Potability") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        
    Returns:
        Tuple of (features, target)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"✅ Features and target prepared")
    print(f"   Features shape: {X.shape}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data split completed")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    print(f"   Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"   Testing target distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   save_scaler: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
    """
    Scale features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Testing features
        save_scaler: Whether to return the fitted scaler
        
    Returns:
        Tuple of (scaled_X_train, scaled_X_test, scaler)
    """
    scaler = StandardScaler()
    
    # Fit on training data and transform both sets
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"✅ Features scaled using StandardScaler")
    print(f"   Feature means (after scaling): {X_train_scaled.mean().round(3).to_dict()}")
    print(f"   Feature stds (after scaling): {X_train_scaled.std().round(3).to_dict()}")
    
    return X_train_scaled, X_test_scaled, (scaler if save_scaler else None)

def load_and_prepare_data(filepath: str = "data/synthetic_water_quality_dataset.csv",
                         test_size: float = 0.2,
                         scale_features_flag: bool = True,
                         random_state: int = 42) -> dict:
    """
    Complete data loading and preparation pipeline
    
    Args:
        filepath: Path to the dataset
        test_size: Proportion for test split
        scale_features_flag: Whether to scale features
        random_state: Random seed
        
    Returns:
        Dictionary containing all prepared data components
    """
    print("🔄 Starting data loading and preparation pipeline...")
    
    # Load data
    df = load_water_quality_data(filepath)
    
    # Prepare features and target
    X, y = prepare_features_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Scale features if requested
    scaler = None
    if scale_features_flag:
        X_train, X_test, scaler = scale_features(X_train, X_test)
    
    print("✅ Data preparation pipeline completed!")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'raw_data': df
    }
