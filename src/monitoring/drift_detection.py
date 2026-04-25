import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta

class DataDriftDetector:
    """Detect data drift in retail demand data."""
    
    def __init__(self, output_dir: str = "monitoring"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = self._setup_logger()
        
        # Initialize drift detection parameters
        self.reference_stats = {}
        self.feature_stats = {}
        self.drift_threshold = 0.05  # p-value threshold for statistical tests
        self.kl_threshold = 0.1  # KL divergence threshold
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def compute_reference_statistics(self, reference_data: pd.DataFrame, 
                                   numeric_features: List[str]) -> Dict:
        """Compute statistics for reference data."""
        ref_stats = {}
        
        for feature in numeric_features:
            if feature in reference_data.columns:
                feature_data = reference_data[feature].dropna()
                
                ref_stats[feature] = {
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'median': float(feature_data.median()),
                    'q25': float(feature_data.quantile(0.25)),
                    'q75': float(feature_data.quantile(0.75)),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'skewness': float(stats.skew(feature_data)),
                    'kurtosis': float(stats.kurtosis(feature_data))
                }
        
        self.reference_stats = ref_stats
        self.logger.info(f"Computed reference statistics for {len(ref_stats)} features")
        
        return ref_stats
    
    def kolmogorov_smirnov_test(self, reference_data: np.ndarray, 
                                current_data: np.ndarray) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test for drift detection."""
        ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
        return ks_statistic, p_value
    
    def mann_whitney_u_test(self, reference_data: np.ndarray, 
                           current_data: np.ndarray) -> Tuple[float, float]:
        """Perform Mann-Whitney U test for drift detection."""
        u_statistic, p_value = stats.mannwhitneyu(reference_data, current_data, 
                                                 alternative='two-sided')
        return u_statistic, p_value
    
    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence."""
        # Create histograms
        hist_p, bin_edges = np.histogram(p, bins=50, density=True)
        hist_q, _ = np.histogram(q, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist_p = hist_p + epsilon
        hist_q = hist_q + epsilon
        
        # Normalize
        hist_p = hist_p / np.sum(hist_p)
        hist_q = hist_q / np.sum(hist_q)
        
        # Calculate KL divergence
        kl_div = np.sum(hist_p * np.log(hist_p / hist_q))
        
        return kl_div
    
    def detect_feature_drift(self, reference_data: pd.DataFrame, 
                           current_data: pd.DataFrame,
                           numeric_features: List[str]) -> Dict[str, Dict]:
        """Detect drift for individual features."""
        drift_results = {}
        
        for feature in numeric_features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            
            ref_data = reference_data[feature].dropna()
            curr_data = current_data[feature].dropna()
            
            if len(ref_data) == 0 or len(curr_data) == 0:
                continue
            
            # Statistical tests
            ks_stat, ks_p_value = self.kolmogorov_smirnov_test(ref_data, curr_data)
            mw_stat, mw_p_value = self.mann_whitney_u_test(ref_data, curr_data)
            
            # KL divergence
            kl_div = self.kl_divergence(ref_data.values, curr_data.values)
            
            # Determine drift
            is_drift_ks = ks_p_value < self.drift_threshold
            is_drift_mw = mw_p_value < self.drift_threshold
            is_drift_kl = kl_div > self.kl_threshold
            
            # Overall drift decision
            drift_count = sum([is_drift_ks, is_drift_mw, is_drift_kl])
            is_drift = drift_count >= 2  # At least 2 tests indicate drift
            
            drift_results[feature] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'mw_statistic': mw_stat,
                'mw_p_value': mw_p_value,
                'kl_divergence': kl_div,
                'is_drift_ks': is_drift_ks,
                'is_drift_mw': is_drift_mw,
                'is_drift_kl': is_drift_kl,
                'is_drift': is_drift,
                'drift_severity': 'high' if drift_count == 3 else 'medium' if drift_count == 2 else 'low'
            }
        
        return drift_results
    
    def detect_multivariate_drift(self, reference_data: pd.DataFrame, 
                                current_data: pd.DataFrame,
                                numeric_features: List[str]) -> Dict:
        """Detect multivariate drift using PCA."""
        # Select only numeric features
        ref_features = reference_data[numeric_features].select_dtypes(include=[np.number])
        curr_features = current_data[numeric_features].select_dtypes(include=[np.number])
        
        # Remove columns with all NaN
        ref_features = ref_features.dropna(axis=1, how='all')
        curr_features = curr_features.dropna(axis=1, how='all')
        
        # Use common columns
        common_features = list(set(ref_features.columns) & set(curr_features.columns))
        if len(common_features) == 0:
            return {'error': 'No common numeric features found'}
        
        ref_features = ref_features[common_features].fillna(ref_features.mean())
        curr_features = curr_features[common_features].fillna(curr_features.mean())
        
        # Standardize features
        scaler = StandardScaler()
        ref_scaled = scaler.fit_transform(ref_features)
        curr_scaled = scaler.transform(curr_features)
        
        # Apply PCA
        n_components = min(10, len(common_features))
        pca = PCA(n_components=n_components)
        
        ref_pca = pca.fit_transform(ref_scaled)
        curr_pca = pca.transform(curr_scaled)
        
        # Compare distributions of principal components
        multivariate_drift = {}
        explained_variance = pca.explained_variance_ratio_
        
        for i in range(n_components):
            comp_ref = ref_pca[:, i]
            comp_curr = curr_pca[:, i]
            
            ks_stat, ks_p_value = self.kolmogorov_smirnov_test(comp_ref, comp_curr)
            kl_div = self.kl_divergence(comp_ref, comp_curr)
            
            is_drift = ks_p_value < self.drift_threshold or kl_div > self.kl_threshold
            
            multivariate_drift[f'PC{i+1}'] = {
                'explained_variance': explained_variance[i],
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'kl_divergence': kl_div,
                'is_drift': is_drift
            }
        
        # Overall multivariate drift
        drifting_components = sum(1 for comp in multivariate_drift.values() if comp['is_drift'])
        overall_drift = drifting_components > n_components / 2
        
        return {
            'components': multivariate_drift,
            'overall_drift': overall_drift,
            'drifting_components': drifting_components,
            'total_components': n_components
        }
    
    def detect_seasonal_drift(self, data: pd.DataFrame, 
                           date_column: str = 'date',
                           target_column: str = 'demand') -> Dict:
        """Detect seasonal drift patterns."""
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Extract temporal features
        data['month'] = data[date_column].dt.month
        data['quarter'] = data[date_column].dt.quarter
        data['day_of_week'] = data[date_column].dt.dayofweek
        
        seasonal_drift = {}
        
        # Monthly drift
        monthly_stats = data.groupby('month')[target_column].agg(['mean', 'std', 'count'])
        
        # Compare recent months with historical patterns
        recent_months = data[data[date_column] >= data[date_column].max() - timedelta(days=90)]
        historical_months = data[data[date_column] < data[date_column].max() - timedelta(days=90)]
        
        if len(recent_months) > 0 and len(historical_months) > 0:
            recent_monthly = recent_months.groupby('month')[target_column].mean()
            historical_monthly = historical_months.groupby('month')[target_column].mean()
            
            for month in range(1, 13):
                if month in recent_monthly.index and month in historical_monthly.index:
                    recent_val = recent_monthly[month]
                    historical_val = historical_monthly[month]
                    
                    # Calculate percentage change
                    pct_change = ((recent_val - historical_val) / historical_val) * 100
                    
                    seasonal_drift[f'month_{month}'] = {
                        'recent_mean': float(recent_val),
                        'historical_mean': float(historical_val),
                        'percentage_change': float(pct_change),
                        'is_drift': abs(pct_change) > 20  # 20% threshold
                    }
        
        # Quarterly drift
        quarterly_stats = data.groupby('quarter')[target_column].agg(['mean', 'std', 'count'])
        
        # Day of week drift
        dow_stats = data.groupby('day_of_week')[target_column].agg(['mean', 'std', 'count'])
        
        return {
            'monthly_drift': seasonal_drift,
            'quarterly_stats': quarterly_stats.to_dict(),
            'dow_stats': dow_stats.to_dict(),
            'overall_seasonal_drift': any(
                info['is_drift'] for info in seasonal_drift.values()
            )
        }
    
    def generate_drift_report(self, reference_data: pd.DataFrame,
                            current_data: pd.DataFrame,
                            numeric_features: List[str],
                            save_report: bool = True) -> Dict:
        """Generate comprehensive drift report."""
        self.logger.info("Generating drift report...")
        
        # Compute reference statistics if not already done
        if not self.reference_stats:
            self.compute_reference_statistics(reference_data, numeric_features)
        
        # Feature-level drift
        feature_drift = self.detect_feature_drift(reference_data, current_data, numeric_features)
        
        # Multivariate drift
        multivariate_drift = self.detect_multivariate_drift(reference_data, current_data, numeric_features)
        
        # Seasonal drift
        seasonal_drift = self.detect_seasonal_drift(current_data)
        
        # Summary statistics
        total_features = len(feature_drift)
        drifting_features = sum(1 for f in feature_drift.values() if f['is_drift'])
        
        high_severity_drift = sum(
            1 for f in feature_drift.values() 
            if f['is_drift'] and f['drift_severity'] == 'high'
        )
        
        # Overall drift assessment
        overall_drift = (
            drifting_features > total_features * 0.3 or  # More than 30% features drifting
            multivariate_drift.get('overall_drift', False) or
            seasonal_drift.get('overall_seasonal_drift', False)
        )
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'reference_size': len(reference_data),
                'current_size': len(current_data),
                'features_analyzed': total_features
            },
            'feature_drift': feature_drift,
            'multivariate_drift': multivariate_drift,
            'seasonal_drift': seasonal_drift,
            'summary': {
                'total_features': total_features,
                'drifting_features': drifting_features,
                'high_severity_drift': high_severity_drift,
                'overall_drift': overall_drift,
                'drift_percentage': (drifting_features / total_features * 100) if total_features > 0 else 0
            }
        }
        
        if save_report:
            self.save_drift_report(drift_report)
        
        self.logger.info(f"Drift report generated. Overall drift: {overall_drift}")
        
        return drift_report
    
    def save_drift_report(self, drift_report: Dict, filename: str = None):
        """Save drift report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(drift_report, f, indent=2, default=str)
        
        self.logger.info(f"Drift report saved to {filepath}")
    
    def plot_drift_visualization(self, reference_data: pd.DataFrame,
                               current_data: pd.DataFrame,
                               feature: str,
                               save_plot: bool = True):
        """Create visualization for drift analysis."""
        plt.figure(figsize=(12, 8))
        
        # Reference data distribution
        plt.subplot(2, 2, 1)
        plt.hist(reference_data[feature].dropna(), bins=50, alpha=0.7, label='Reference', density=True)
        plt.hist(current_data[feature].dropna(), bins=50, alpha=0.7, label='Current', density=True)
        plt.title(f'Distribution Comparison: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        
        # Box plot comparison
        plt.subplot(2, 2, 2)
        data_to_plot = [
            reference_data[feature].dropna().values,
            current_data[feature].dropna().values
        ]
        plt.boxplot(data_to_plot, labels=['Reference', 'Current'])
        plt.title(f'Box Plot Comparison: {feature}')
        plt.ylabel(feature)
        
        # Q-Q plot
        plt.subplot(2, 2, 3)
        ref_data = reference_data[feature].dropna()
        curr_data = current_data[feature].dropna()
        
        # Create quantiles
        min_len = min(len(ref_data), len(curr_data))
        ref_quantiles = np.sort(ref_data)[:min_len]
        curr_quantiles = np.sort(curr_data)[:min_len]
        
        plt.scatter(ref_quantiles, curr_quantiles, alpha=0.5)
        plt.plot([ref_quantiles.min(), ref_quantiles.max()], 
                [ref_quantiles.min(), ref_quantiles.max()], 'r--')
        plt.title(f'Q-Q Plot: {feature}')
        plt.xlabel('Reference Quantiles')
        plt.ylabel('Current Quantiles')
        
        # Time series (if date column exists)
        plt.subplot(2, 2, 4)
        if 'date' in current_data.columns:
            current_data_sorted = current_data.sort_values('date')
            plt.plot(current_data_sorted['date'], current_data_sorted[feature], alpha=0.7)
            plt.title(f'Time Series: {feature}')
            plt.xlabel('Date')
            plt.ylabel(feature)
            plt.xticks(rotation=45)
        else:
            plt.plot(current_data[feature].values, alpha=0.7)
            plt.title(f'Sequence: {feature}')
            plt.xlabel('Index')
            plt.ylabel(feature)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_plot_{feature}_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Drift plot saved to {filepath}")
        
        plt.show()
    
    def load_reference_stats(self, filepath: str):
        """Load reference statistics from file."""
        with open(filepath, 'r') as f:
            self.reference_stats = json.load(f)
        self.logger.info(f"Reference statistics loaded from {filepath}")
    
    def save_reference_stats(self, filepath: str = None):
        """Save reference statistics to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"reference_stats_{timestamp}.json"
        
        full_path = self.output_dir / filepath
        
        with open(full_path, 'w') as f:
            json.dump(self.reference_stats, f, indent=2)
        
        self.logger.info(f"Reference statistics saved to {full_path}")

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.data_ingestion import DataIngestion
    
    # Load data
    ingestion = DataIngestion(data_dir="../../data/raw")
    train_df, test_df = ingestion.load_training_data()
    
    # Initialize drift detector
    drift_detector = DataDriftDetector()
    
    # Get numeric features
    numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Generate drift report
    drift_report = drift_detector.generate_drift_report(
        reference_data=train_df,
        current_data=test_df,
        numeric_features=numeric_features
    )
    
    print("=== Drift Detection Report ===")
    print(f"Overall Drift: {drift_report['summary']['overall_drift']}")
    print(f"Drifting Features: {drift_report['summary']['drifting_features']}/{drift_report['summary']['total_features']}")
    print(f"Drift Percentage: {drift_report['summary']['drift_percentage']:.2f}%")
    
    # Show top drifting features
    drifting_features = {
        k: v for k, v in drift_report['feature_drift'].items() 
        if v['is_drift']
    }
    
    if drifting_features:
        print(f"\n=== Top Drifting Features ===")
        for feature, info in list(drifting_features.items())[:5]:
            print(f"{feature}: KS p-value={info['ks_p_value']:.4f}, "
                  f"KL divergence={info['kl_divergence']:.4f}, "
                  f"Severity={info['drift_severity']}")
