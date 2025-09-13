"""
Synthetic Water Quality Data Generator
=====================================

Generates realistic synthetic water quality data with meaningful patterns
and correlations that make sense for water safety prediction.

This generator creates data based on real-world water quality relationships
and ensures the model can learn meaningful patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import os

class WaterQualityDataGenerator:
    """Sophisticated synthetic water quality data generator"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data generator
        
        Args:
            random_state: Random seed for reproducibility
        """
        np.random.seed(random_state)
        self.random_state = random_state
        
        # Define realistic parameter ranges and relationships
        self.parameter_ranges = {
            'ph': {'safe_min': 6.5, 'safe_max': 8.5, 'abs_min': 3.0, 'abs_max': 12.0},
            'Hardness': {'safe_min': 60, 'safe_max': 120, 'abs_min': 0, 'abs_max': 500},
            'Solids': {'safe_min': 100, 'safe_max': 500, 'abs_min': 0, 'abs_max': 2000},
            'Chloramines': {'safe_min': 0.2, 'safe_max': 4.0, 'abs_min': 0, 'abs_max': 12},
            'Sulfate': {'safe_min': 50, 'safe_max': 250, 'abs_min': 0, 'abs_max': 600},
            'Conductivity': {'safe_min': 200, 'safe_max': 600, 'abs_min': 50, 'abs_max': 1200},
            'Organic_carbon': {'safe_min': 2, 'safe_max': 8, 'abs_min': 0, 'abs_max': 25},
            'Trihalomethanes': {'safe_min': 5, 'safe_max': 80, 'abs_min': 0, 'abs_max': 200},
            'Turbidity': {'safe_min': 0.1, 'safe_max': 1.0, 'abs_min': 0, 'abs_max': 15}
        }
        
        # Define correlation patterns that make sense for water quality
        self.correlation_patterns = {
            'Solids_Conductivity': 0.85,  # Strong positive correlation
            'Hardness_Solids': 0.65,      # Moderate positive correlation
            'Organic_carbon_Trihalomethanes': 0.70,  # Organic matter increases THMs
            'Turbidity_Organic_carbon': 0.60,  # Turbidity often correlates with organics
            'Sulfate_Conductivity': 0.55,  # Sulfate contributes to conductivity
        }
    
    def generate_safe_water_sample(self) -> Dict:
        """Generate a sample that represents safe drinking water"""
        sample = {}
        
        # Generate safe parameters with some natural variation
        for param, ranges in self.parameter_ranges.items():
            # Generate within safe range with normal distribution
            safe_center = (ranges['safe_min'] + ranges['safe_max']) / 2
            safe_std = (ranges['safe_max'] - ranges['safe_min']) / 6  # 99.7% within range
            
            value = np.random.normal(safe_center, safe_std)
            
            # Ensure within safe bounds
            value = np.clip(value, ranges['safe_min'], ranges['safe_max'])
            sample[param] = value
        
        # Apply correlations to make it more realistic
        sample = self._apply_correlations(sample)
        
        return sample
    
    def generate_unsafe_water_sample(self) -> Dict:
        """Generate a sample that represents unsafe drinking water"""
        sample = {}
        
        # Choose which parameters will be problematic (1-3 parameters)
        num_problematic = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        problematic_params = np.random.choice(
            list(self.parameter_ranges.keys()), 
            size=num_problematic, 
            replace=False
        )
        
        for param, ranges in self.parameter_ranges.items():
            if param in problematic_params:
                # Generate unsafe values (outside safe range)
                if np.random.random() < 0.7:  # 70% chance of being too high
                    # Too high
                    unsafe_min = ranges['safe_max']
                    unsafe_max = ranges['abs_max']
                    value = np.random.uniform(unsafe_min, unsafe_max)
                else:
                    # Too low
                    unsafe_min = ranges['abs_min']
                    unsafe_max = ranges['safe_min']
                    value = np.random.uniform(unsafe_min, unsafe_max)
            else:
                # Generate mostly safe values for other parameters
                safe_center = (ranges['safe_min'] + ranges['safe_max']) / 2
                safe_std = (ranges['safe_max'] - ranges['safe_min']) / 4
                
                value = np.random.normal(safe_center, safe_std)
                
                # Sometimes make them slightly outside safe range
                if np.random.random() < 0.2:  # 20% chance
                    if np.random.random() < 0.5:
                        value = np.random.uniform(ranges['safe_max'], 
                                                 ranges['safe_max'] + (ranges['abs_max'] - ranges['safe_max']) * 0.3)
                    else:
                        value = np.random.uniform(ranges['safe_min'] - (ranges['safe_min'] - ranges['abs_min']) * 0.3,
                                                 ranges['safe_min'])
                
                # Ensure within absolute bounds
                value = np.clip(value, ranges['abs_min'], ranges['abs_max'])
            
            sample[param] = value
        
        # Apply correlations
        sample = self._apply_correlations(sample)
        
        return sample
    
    def _apply_correlations(self, sample: Dict) -> Dict:
        """Apply realistic correlations between parameters"""
        # Solids-Conductivity correlation
        if 'Solids' in sample and 'Conductivity' in sample:
            # Adjust conductivity based on solids
            target_conductivity = 100 + (sample['Solids'] * 0.4)
            current_conductivity = sample['Conductivity']
            adjusted = 0.7 * current_conductivity + 0.3 * target_conductivity
            sample['Conductivity'] = np.clip(adjusted, 
                                           self.parameter_ranges['Conductivity']['abs_min'],
                                           self.parameter_ranges['Conductivity']['abs_max'])
        
        # Hardness-Solids correlation
        if 'Hardness' in sample and 'Solids' in sample:
            target_hardness = 30 + (sample['Solids'] * 0.15)
            current_hardness = sample['Hardness']
            adjusted = 0.8 * current_hardness + 0.2 * target_hardness
            sample['Hardness'] = np.clip(adjusted,
                                       self.parameter_ranges['Hardness']['abs_min'],
                                       self.parameter_ranges['Hardness']['abs_max'])
        
        # Organic carbon - Trihalomethanes correlation
        if 'Organic_carbon' in sample and 'Trihalomethanes' in sample:
            target_thm = sample['Organic_carbon'] * 8 + np.random.normal(0, 10)
            current_thm = sample['Trihalomethanes']
            adjusted = 0.6 * current_thm + 0.4 * target_thm
            sample['Trihalomethanes'] = np.clip(adjusted,
                                              self.parameter_ranges['Trihalomethanes']['abs_min'],
                                              self.parameter_ranges['Trihalomethanes']['abs_max'])
        
        # Turbidity-Organic carbon correlation
        if 'Turbidity' in sample and 'Organic_carbon' in sample:
            if sample['Turbidity'] > 2:  # High turbidity can increase organic carbon
                sample['Organic_carbon'] = min(sample['Organic_carbon'] * 1.3,
                                             self.parameter_ranges['Organic_carbon']['abs_max'])
        
        return sample
    
    def generate_dataset(self, n_samples: int = 10000, 
                        safe_ratio: float = 0.6) -> pd.DataFrame:
        """
        Generate complete synthetic dataset
        
        Args:
            n_samples: Total number of samples to generate
            safe_ratio: Proportion of safe water samples (0.6 = 60% safe)
            
        Returns:
            DataFrame: Complete synthetic dataset
        """
        print(f"🔄 Generating {n_samples} synthetic water quality samples...")
        print(f"   Safe samples: {int(n_samples * safe_ratio)}")
        print(f"   Unsafe samples: {int(n_samples * (1 - safe_ratio))}")
        
        samples = []
        labels = []
        
        n_safe = int(n_samples * safe_ratio)
        n_unsafe = n_samples - n_safe
        
        # Generate safe samples
        for i in range(n_safe):
            if i % 1000 == 0:
                print(f"   Generated {i}/{n_safe} safe samples...")
            
            sample = self.generate_safe_water_sample()
            samples.append(sample)
            labels.append(1)  # Safe
        
        # Generate unsafe samples
        for i in range(n_unsafe):
            if i % 1000 == 0:
                print(f"   Generated {i}/{n_unsafe} unsafe samples...")
            
            sample = self.generate_unsafe_water_sample()
            samples.append(sample)
            labels.append(0)  # Unsafe
        
        # Create DataFrame
        df = pd.DataFrame(samples)
        df['Potability'] = labels
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"✅ Dataset generation completed!")
        print(f"   Final shape: {df.shape}")
        print(f"   Safe samples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"   Unsafe samples: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
        
        return df
    
    def analyze_generated_data(self, df: pd.DataFrame) -> None:
        """Analyze the quality of generated data"""
        print(f"\n📊 DATA QUALITY ANALYSIS")
        print("=" * 40)
        
        # Basic statistics
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Target distribution
        target_dist = df['Potability'].value_counts()
        print(f"Target distribution:")
        print(f"  Safe (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
        print(f"  Unsafe (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
        
        # Feature correlations
        print(f"\nFeature correlations with target:")
        correlations = df.corr()['Potability'].abs().sort_values(ascending=False)
        for feature, corr in correlations.items():
            if feature != 'Potability':
                print(f"  {feature}: {corr:.3f}")
        
        # Parameter range compliance
        print(f"\nParameter safety compliance:")
        for param in df.columns:
            if param != 'Potability' and param in self.parameter_ranges:
                ranges = self.parameter_ranges[param]
                safe_count = len(df[(df[param] >= ranges['safe_min']) & 
                                  (df[param] <= ranges['safe_max'])])
                print(f"  {param}: {safe_count}/{len(df)} ({safe_count/len(df)*100:.1f}%) within safe range")
    
    def plot_data_overview(self, df: pd.DataFrame) -> None:
        """Create overview plots of the generated data"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        feature_cols = [col for col in df.columns if col != 'Potability']
        
        for i, feature in enumerate(feature_cols):
            if i < len(axes):
                # Plot distribution by safety class
                for safety_class in [0, 1]:
                    data = df[df['Potability'] == safety_class][feature]
                    axes[i].hist(data, alpha=0.7, 
                               label=f"{'Safe' if safety_class else 'Unsafe'}", 
                               bins=30)
                
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{feature} Distribution')
                axes[i].legend()
                
                # Add safe range indicator
                if feature in self.parameter_ranges:
                    ranges = self.parameter_ranges[feature]
                    axes[i].axvline(ranges['safe_min'], color='green', linestyle='--', alpha=0.7)
                    axes[i].axvline(ranges['safe_max'], color='green', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('synthetic_data_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_dataset(self, df: pd.DataFrame, 
                    filepath: str = "data/water_quality_dataset.csv") -> None:
        """Save the generated dataset"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Dataset saved to: {filepath}")

def generate_water_quality_dataset(n_samples: int = 10000, 
                                 safe_ratio: float = 0.6,
                                 analyze: bool = True,
                                 plot: bool = True) -> pd.DataFrame:
    """
    Convenience function to generate synthetic water quality dataset
    
    Args:
        n_samples: Number of samples to generate
        safe_ratio: Proportion of safe samples
        analyze: Whether to run data analysis
        plot: Whether to create plots
        
    Returns:
        DataFrame: Generated dataset
    """
    generator = WaterQualityDataGenerator()
    df = generator.generate_dataset(n_samples, safe_ratio)
    
    if analyze:
        generator.analyze_generated_data(df)
    
    if plot:
        generator.plot_data_overview(df)
    
    generator.save_dataset(df)
    
    return df

if __name__ == "__main__":
    print("🚰 Synthetic Water Quality Data Generator")
    print("=" * 50)
    
    # Generate the dataset
    dataset = generate_water_quality_dataset(
        n_samples=10000,
        safe_ratio=0.65,  # 65% safe, 35% unsafe for slight imbalance
        analyze=True,
        plot=True
    )
    
    print(f"\n🎉 Synthetic dataset generation complete!")
    print(f"📁 Dataset saved as: data/water_quality_dataset.csv")
    print(f"📊 Ready for model training!")
