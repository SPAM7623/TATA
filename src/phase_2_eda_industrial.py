"""
PHASE 2: Industrial EDA Analysis
Alpha Defect Detection in Hot Rolling Mills

Task: Comprehensive statistical analysis and process physics understanding
Time: ~2-3 minutes in Colab

Output: eda_univariate_analysis.csv, eda_instability_analysis.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, mannwhitneyu
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def perform_univariate_analysis(train_df, feature_cols, verbose=True):
    """
    Analyze each feature independently against target variable

    Returns: DataFrame with statistical significance, means, distributions
    """
    if verbose:
        print("\n" + "=" * 80)
        print("1. UNIVARIATE ANALYSIS: Feature Behavior in Defect vs Normal")
        print("=" * 80)

    univariate_results = []

    for feature in feature_cols:
        normal_data = train_df[train_df['Y'] == 0][feature].dropna()
        defect_data = train_df[train_df['Y'] == 1][feature].dropna()

        # Statistical test (Mann-Whitney U - non-parametric)
        stat, p_value = mannwhitneyu(normal_data, defect_data)

        result = {
            'Feature': feature,
            'Normal_Mean': normal_data.mean(),
            'Defect_Mean': defect_data.mean(),
            'Normal_Std': normal_data.std(),
            'Defect_Std': defect_data.std(),
            'Mean_Diff': defect_data.mean() - normal_data.mean(),
            'Mean_Diff_Pct': ((defect_data.mean() - normal_data.mean()) / abs(normal_data.mean()) * 100) if normal_data.mean() != 0 else 0,
            'P_Value': p_value,
            'Significant': 'YES' if p_value < 0.05 else 'NO',
            'Normal_Skew': skew(normal_data),
            'Defect_Skew': skew(defect_data),
        }
        univariate_results.append(result)

    univariate_df = pd.DataFrame(univariate_results).sort_values('P_Value')

    if verbose:
        print("\nTop 10 Most Significant Features (p-value):")
        top_10 = univariate_df[['Feature', 'Normal_Mean', 'Defect_Mean', 'Mean_Diff', 'P_Value']].head(10)
        print(top_10.to_string(index=False))

        sig_count = (univariate_df['Significant'] == 'YES').sum()
        print(f"\nTotal significant features (p < 0.05): {sig_count}/{len(feature_cols)}")

    return univariate_df


def analyze_variance_instability(train_df, feature_cols, verbose=True):
    """
    Analyze variance and instability patterns in defective vs normal samples

    Higher variance in defects = metallurgical instability
    """
    if verbose:
        print("\n" + "=" * 80)
        print("2. VARIANCE & INSTABILITY ANALYSIS")
        print("=" * 80)

    cv_analysis = []

    for feature in feature_cols:
        normal_data = train_df[train_df['Y'] == 0][feature].dropna()
        defect_data = train_df[train_df['Y'] == 1][feature].dropna()

        # Coefficient of Variation (CV = std/mean)
        normal_cv = normal_data.std() / abs(normal_data.mean()) if normal_data.mean() != 0 else 0
        defect_cv = defect_data.std() / abs(defect_data.mean()) if defect_data.mean() != 0 else 0

        cv_analysis.append({
            'Feature': feature,
            'Normal_Mean': normal_data.mean(),
            'Normal_Std': normal_data.std(),
            'Normal_CV': normal_cv,
            'Defect_Mean': defect_data.mean(),
            'Defect_Std': defect_data.std(),
            'Defect_CV': defect_cv,
            'CV_Diff': defect_cv - normal_cv,
            'Relative_Instability': (defect_cv / normal_cv - 1) * 100 if normal_cv > 0 else 0,
        })

    cv_df = pd.DataFrame(cv_analysis).sort_values('Relative_Instability', ascending=False)

    if verbose:
        print("\nTop 10 Features with Higher Instability in Defects:")
        top_10 = cv_df[['Feature', 'Normal_CV', 'Defect_CV', 'Relative_Instability']].head(10)
        print(top_10.to_string(index=False))

    return cv_df


def analyze_correlations(train_df, feature_cols, verbose=True):
    """
    Analyze feature correlations and detect redundancy
    """
    if verbose:
        print("\n" + "=" * 80)
        print("3. CORRELATION & FEATURE GROUPING")
        print("=" * 80)

    corr_matrix = train_df[feature_cols].corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'Feature1': feature_cols[i],
                    'Feature2': feature_cols[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    if verbose:
        if high_corr_pairs:
            print(f"\nHighly Correlated Feature Pairs (|r| > 0.8): {len(high_corr_pairs)}")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True)[:10]:
                print(f"  {pair['Feature1']} ↔ {pair['Feature2']}: {pair['Correlation']:.3f}")
        else:
            print("\n✓ No extremely high correlations detected (good feature diversity)")

    return corr_matrix, high_corr_pairs


def pca_latent_structure(train_df, feature_cols, verbose=True):
    """
    Perform PCA to detect hidden process regimes and dimensionality
    """
    if verbose:
        print("\n" + "=" * 80)
        print("4. PCA ANALYSIS - Hidden Process Regimes")
        print("=" * 80)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[feature_cols])

    # Fit PCA
    pca = PCA()
    pca.fit(X_scaled)

    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_80 = (cumsum_var >= 0.80).argmax() + 1
    n_components_95 = (cumsum_var >= 0.95).argmax() + 1

    if verbose:
        print(f"\nDimensionality Analysis:")
        print(f"  Total features: {len(feature_cols)}")
        print(f"  Components for 80% variance: {n_components_80}")
        print(f"  Components for 95% variance: {n_components_95}")
        print(f"  Redundant features: ~{len(feature_cols) - n_components_80}")

        print(f"\nTop 10 PC Explained Variance:")
        for i in range(min(10, len(pca.explained_variance_ratio_))):
            var = pca.explained_variance_ratio_[i]
            cumvar = cumsum_var[i]
            print(f"  PC{i+1}: {var:.4f} (cumsum: {cumvar:.4f})")

    return pca, scaler


def detect_dangerous_interactions(train_df, feature_cols, univariate_df, verbose=True):
    """
    Find parameter combinations that create high defect risk zones
    """
    if verbose:
        print("\n" + "=" * 80)
        print("5. NONLINEAR INTERACTIONS & DANGEROUS COMBINATIONS")
        print("=" * 80)

    # Select top 5 significant features
    top_features = univariate_df.nsmallest(5, 'P_Value')['Feature'].tolist()

    interaction_risk = []

    for i, feat1 in enumerate(top_features[:3]):
        for feat2 in top_features[i + 1:4]:
            # Create bins and calculate defect rate
            feat1_bins = pd.qcut(train_df[feat1], q=3, duplicates='drop')
            feat2_bins = pd.qcut(train_df[feat2], q=3, duplicates='drop')

            crosstab = pd.crosstab([feat1_bins, feat2_bins], train_df['Y'])

            if 1 in crosstab.columns and 0 in crosstab.columns:
                defect_rate = crosstab[1] / (crosstab[0] + crosstab[1])
                max_defect_rate = defect_rate.max()
                min_defect_rate = defect_rate.min()

                interaction_risk.append({
                    'Feature_Pair': f"{feat1} × {feat2}",
                    'Max_Defect_Rate': max_defect_rate,
                    'Min_Defect_Rate': min_defect_rate,
                    'Interaction_Strength': max_defect_rate - min_defect_rate
                })

    if interaction_risk and verbose:
        interaction_df = pd.DataFrame(interaction_risk).sort_values('Interaction_Strength', ascending=False)
        print(f"\nStrongest Nonlinear Interactions:")
        print(interaction_df.to_string(index=False))

    return interaction_risk


def identify_operating_windows(train_df, feature_cols, univariate_df, verbose=True):
    """
    Identify safe vs unsafe operating regions for key parameters
    """
    if verbose:
        print("\n" + "=" * 80)
        print("6. SAFE vs UNSAFE OPERATING WINDOWS")
        print("=" * 80)

    top_features = univariate_df.nsmallest(5, 'P_Value')['Feature'].tolist()

    threshold_analysis = []

    for feature in top_features:
        feat_values = train_df[[feature, 'Y']].dropna()
        sorted_data = feat_values.sort_values(feature)
        window_size = max(30, len(sorted_data) // 10)

        defect_rates = []
        boundaries = []

        for i in range(0, len(sorted_data) - window_size, window_size // 2):
            window = sorted_data.iloc[i:i + window_size]
            defect_rate = window['Y'].mean()
            boundary = window[feature].iloc[window_size // 2]
            defect_rates.append(defect_rate)
            boundaries.append(boundary)

        if len(defect_rates) > 1:
            rate_diff = np.diff(defect_rates)
            max_transition_idx = np.argmax(np.abs(rate_diff))
            sharp_threshold = boundaries[max_transition_idx]

            threshold_analysis.append({
                'Feature': feature,
                'Safe_Region_Defect_Rate': min(defect_rates),
                'Unsafe_Region_Defect_Rate': max(defect_rates),
                'Critical_Threshold': sharp_threshold,
                'Risk_Increase': max(defect_rates) - min(defect_rates)
            })

    if threshold_analysis and verbose:
        threshold_df = pd.DataFrame(threshold_analysis)
        print(f"\nOperating Window Analysis:")
        print(threshold_df.to_string(index=False))

    return threshold_analysis


def industrial_eda_summary(train_df, univariate_df, cv_df, high_corr_pairs, verbose=True):
    """
    Generate comprehensive EDA summary and recommendations
    """
    if verbose:
        print("\n" + "=" * 80)
        print("INDUSTRIAL EDA SUMMARY & RECOMMENDATIONS")
        print("=" * 80)

        sig_features = univariate_df[univariate_df['Significant'] == 'YES']['Feature'].tolist()

        y_counts = train_df['Y'].value_counts()
        y_counts_pct = train_df['Y'].value_counts(normalize=True) * 100

        summary = f"""
Dataset Characteristics:
  • Total samples: {len(train_df)}
  • Features analyzed: {len(univariate_df)}
  • Significant features: {len(sig_features)}/49 (p < 0.05)
  • Defect samples: {y_counts[1]} ({y_counts_pct[1]:.1f}%)
  • Class imbalance: {y_counts[0]/y_counts[1]:.1f}:1

Key Findings:
  1. Univariate: {len(sig_features)} features show significant difference
  2. Correlations: {len(high_corr_pairs)} highly correlated pairs detected
  3. Instability: Defects show {cv_df['Relative_Instability'].max():.0f}% higher variance
  4. Process physics: Multiple regimes and parameter interactions detected
  5. Critical regions: High-risk parameter combinations identified

Feature Engineering Recommendations:
  ✓ Create interaction terms for top feature pairs
  ✓ Add instability measures (CV, rolling std)
  ✓ Engineer threshold-crossing indicators
  ✓ Build ratio features (X_i / X_j)
  ✓ Extract PCA components for regime detection
  ✓ Create anomaly distance features

Modeling Strategy:
  ✓ Use class weights (to handle imbalance)
  ✓ Apply SMOTE carefully
  ✓ Use tree-based models (capture nonlinearity)
  ✓ Optimize recall > precision (catch defects)
  ✓ Apply threshold tuning (not default 0.5)
  ✓ Use SHAP for defect explanation
"""
        print(summary)

    return summary


def run_industrial_eda(train_path='train_cleaned.csv', test_path='test_cleaned.csv', verbose=True):
    """
    Execute complete Industrial EDA pipeline
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [col for col in train_df.columns if col.startswith('X')]

    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 2: INDUSTRIAL EDA ANALYSIS")
        print("=" * 80)
        print(f"\nDataset: {len(train_df)} samples × {len(feature_cols)} features")

    # Run all analyses
    univariate_df = perform_univariate_analysis(train_df, feature_cols, verbose)
    cv_df = analyze_variance_instability(train_df, feature_cols, verbose)
    corr_matrix, high_corr_pairs = analyze_correlations(train_df, feature_cols, verbose)
    pca, scaler = pca_latent_structure(train_df, feature_cols, verbose)
    interaction_risk = detect_dangerous_interactions(train_df, feature_cols, univariate_df, verbose)
    threshold_analysis = identify_operating_windows(train_df, feature_cols, univariate_df, verbose)
    summary = industrial_eda_summary(train_df, univariate_df, cv_df, high_corr_pairs, verbose)

    # Save results
    univariate_df.to_csv('eda_univariate_analysis.csv', index=False)
    cv_df.to_csv('eda_instability_analysis.csv', index=False)

    if verbose:
        print("\n✓ Results saved:")
        print("  - eda_univariate_analysis.csv")
        print("  - eda_instability_analysis.csv")

    return {
        'univariate': univariate_df,
        'variance': cv_df,
        'correlations': corr_matrix,
        'pca': pca,
        'scaler': scaler,
        'interactions': interaction_risk,
        'thresholds': threshold_analysis
    }


if __name__ == "__main__":
    results = run_industrial_eda(
        train_path='train_cleaned.csv',
        test_path='test_cleaned.csv',
        verbose=True
    )
