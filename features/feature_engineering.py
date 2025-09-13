"""
Feature engineering and selection functions for water quality prediction
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful ratio features for water quality prediction
    
    Args:
        df: Input DataFrame with water quality features
        
    Returns:
        DataFrame with additional ratio features
    """
    df_enhanced = df.copy()
    
    # Create meaningful ratios based on water chemistry
    if 'Solids' in df.columns and 'Conductivity' in df.columns:
        df_enhanced['Solids_Conductivity_Ratio'] = df['Solids'] / (df['Conductivity'] + 1e-6)
    
    if 'Organic_carbon' in df.columns and 'Trihalomethanes' in df.columns:
        df_enhanced['Organic_THM_Ratio'] = df['Organic_carbon'] / (df['Trihalomethanes'] + 1e-6)
    
    if 'Hardness' in df.columns and 'Solids' in df.columns:
        df_enhanced['Hardness_Solids_Ratio'] = df['Hardness'] / (df['Solids'] + 1e-6)
    
    if 'Sulfate' in df.columns and 'Conductivity' in df.columns:
        df_enhanced['Sulfate_Conductivity_Ratio'] = df['Sulfate'] / (df['Conductivity'] + 1e-6)
    
    print(f"✅ Created {len(df_enhanced.columns) - len(df.columns)} ratio features")
    return df_enhanced

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features that might be important for water quality
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features
    """
    df_enhanced = df.copy()
    
    # pH interactions (pH affects many chemical processes)
    if 'ph' in df.columns:
        for col in ['Chloramines', 'Trihalomethanes', 'Organic_carbon']:
            if col in df.columns:
                df_enhanced[f'ph_{col}_interaction'] = df['ph'] * df[col]
    
    # Turbidity interactions (affects treatment effectiveness)
    if 'Turbidity' in df.columns:
        for col in ['Organic_carbon', 'Trihalomethanes']:
            if col in df.columns:
                df_enhanced[f'Turbidity_{col}_interaction'] = df['Turbidity'] * df[col]
    
    print(f"✅ Created {len(df_enhanced.columns) - len(df.columns)} interaction features")
    return df_enhanced

def create_polynomial_features(df: pd.DataFrame, degree: int = 2, 
                             selected_features: List[str] = None) -> pd.DataFrame:
    """
    Create polynomial features for selected columns
    
    Args:
        df: Input DataFrame
        degree: Polynomial degree
        selected_features: Features to create polynomials for
        
    Returns:
        DataFrame with polynomial features
    """
    df_enhanced = df.copy()
    
    if selected_features is None:
        # Default to key features that might have non-linear relationships
        selected_features = ['ph', 'Turbidity', 'Organic_carbon', 'Trihalomethanes']
    
    for feature in selected_features:
        if feature in df.columns:
            for d in range(2, degree + 1):
                df_enhanced[f'{feature}_power_{d}'] = df[feature] ** d
    
    print(f"✅ Created polynomial features up to degree {degree}")
    return df_enhanced

def select_best_features(X: pd.DataFrame, y: pd.Series, 
                        k: int = 15, method: str = 'f_classif') -> Tuple[pd.DataFrame, List[str]]:
    """
    Select k best features using statistical tests
    
    Args:
        X: Features DataFrame
        y: Target Series
        k: Number of features to select
        method: Selection method ('f_classif' or 'mutual_info')
        
    Returns:
        Tuple of (selected_features_df, selected_feature_names)
    """
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("Method must be 'f_classif' or 'mutual_info'")
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Create DataFrame with selected features
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_,
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)
    
    print(f"✅ Selected {k} best features using {method}")
    print("Top selected features:")
    for _, row in feature_scores[feature_scores['selected']].head().iterrows():
        print(f"   {row['feature']}: {row['score']:.3f}")
    
    return X_selected_df, selected_features

def analyze_feature_importance(df: pd.DataFrame, target_col: str = 'Potability') -> pd.DataFrame:
    """
    Analyze feature correlations and importance
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        
    Returns:
        DataFrame with feature importance metrics
    """
    features = [col for col in df.columns if col != target_col]
    
    # Calculate correlations with target
    correlations = df[features].corrwith(df[target_col]).abs()
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation_with_target': correlations.values
    }).sort_values('correlation_with_target', ascending=False)
    
    print("✅ Feature importance analysis:")
    print(importance_df.head(10).to_string(index=False))
    
    return importance_df

def plot_feature_correlations(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot correlation heatmap of features
    
    Args:
        df: DataFrame with features
        figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)
    correlation_matrix = df.corr()
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8})
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()

def engineer_features(X: pd.DataFrame, create_ratios: bool = True,
                     create_interactions: bool = True,
                     create_polynomials: bool = False,
                     select_features: bool = True,
                     k_best: int = 15) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete feature engineering pipeline
    
    Args:
        X: Input features DataFrame
        create_ratios: Whether to create ratio features
        create_interactions: Whether to create interaction features
        create_polynomials: Whether to create polynomial features
        select_features: Whether to perform feature selection
        k_best: Number of best features to select
        
    Returns:
        Tuple of (engineered_features, feature_names)
    """
    print("🔄 Starting feature engineering pipeline...")
    
    X_engineered = X.copy()
    
    if create_ratios:
        X_engineered = create_ratio_features(X_engineered)
    
    if create_interactions:
        X_engineered = create_interaction_features(X_engineered)
    
    if create_polynomials:
        X_engineered = create_polynomial_features(X_engineered)
    
    print(f"✅ Feature engineering completed!")
    print(f"   Original features: {X.shape[1]}")
    print(f"   Engineered features: {X_engineered.shape[1]}")
    
    feature_names = list(X_engineered.columns)
    
    return X_engineered, feature_names
