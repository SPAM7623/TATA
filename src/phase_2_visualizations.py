"""
PHASE 2 VISUALIZATIONS: Complete Plot Suite
Alpha Defect Detection in Hot Rolling Mills

Comprehensive visualization functions for all 20 EDA points
Creates publication-quality plots for industrial analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


# =============================================================================
# POINT 1-2: DATASET & CLASS IMBALANCE
# =============================================================================

def plot_class_distribution(y_train, output_file='viz_01_class_distribution.png'):
    """Plot class imbalance distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Class Imbalance Analysis', fontsize=14, fontweight='bold')

    # Bar plot
    counts = y_train.value_counts()
    axes[0].bar(['Normal (Y=0)', 'Defect (Y=1)'], counts.values, color=['green', 'red'], alpha=0.7)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Sample Distribution')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

    # Pie chart
    labels = [f'Normal\n{counts[0]} ({counts[0]/len(y_train)*100:.1f}%)',
              f'Defect\n{counts[1]} ({counts[1]/len(y_train)*100:.1f}%)']
    axes[1].pie(counts.values, labels=labels, colors=['green', 'red'], autopct='',
                startangle=90, explode=(0, 0.1))
    axes[1].set_title(f'Imbalance Ratio: {counts[0]/counts[1]:.1f}:1')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 4-5: DISTRIBUTIONS
# =============================================================================

def plot_distribution_comparison(train_df, feature_cols, top_features, output_file='viz_04_distributions.png'):
    """Plot defect vs normal distributions for top features"""
    n_features = min(12, len(top_features))
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    fig.suptitle('Feature Distributions: Normal vs Defective Samples', fontsize=14, fontweight='bold')

    for idx, feature in enumerate(top_features[:n_features]):
        ax = axes[idx // 3, idx % 3]

        normal = train_df[train_df['Y'] == 0][feature]
        defect = train_df[train_df['Y'] == 1][feature]

        ax.hist(normal, bins=20, alpha=0.6, label='Normal (Y=0)', color='green', density=True)
        ax.hist(defect, bins=20, alpha=0.6, label='Defect (Y=1)', color='red', density=True)

        ax.set_xlabel(feature, fontweight='bold')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{feature} Distribution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_box_plots(train_df, feature_cols, top_features, output_file='viz_05_boxplots.png'):
    """Plot box plots for variance comparison"""
    n_features = min(12, len(top_features))
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    fig.suptitle('Box Plots: Normal vs Defective Samples', fontsize=14, fontweight='bold')

    for idx, feature in enumerate(top_features[:n_features]):
        ax = axes[idx // 3, idx % 3]

        data_to_plot = [train_df[train_df['Y'] == 0][feature],
                        train_df[train_df['Y'] == 1][feature]]

        bp = ax.boxplot(data_to_plot, labels=['Normal', 'Defect'], patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.6)

        ax.set_ylabel(feature, fontweight='bold')
        ax.set_title(f'{feature} - Variance Comparison')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 9: CORRELATION HEATMAPS
# =============================================================================

def plot_correlation_heatmap(corr_matrix, title='Correlation Matrix', output_file='viz_09_correlation.png'):
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, ax=ax,
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1, linewidths=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_correlation_comparison(corr_normal, corr_defect, output_file='viz_10_corr_comparison.png'):
    """Plot normal vs defect correlation comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Correlation Comparison: Normal vs Defective', fontsize=14, fontweight='bold')

    sns.heatmap(corr_normal, cmap='coolwarm', center=0, square=True, ax=axes[0],
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1, linewidths=0.5)
    axes[0].set_title('Normal Samples', fontsize=12, fontweight='bold')

    sns.heatmap(corr_defect, cmap='coolwarm', center=0, square=True, ax=axes[1],
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1, linewidths=0.5)
    axes[1].set_title('Defective Samples', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 7-8: OUTLIERS & VARIANCE
# =============================================================================

def plot_outliers_variance(df_outliers, output_file='viz_07_outliers.png'):
    """Plot outlier analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Outlier Analysis', fontsize=14, fontweight='bold')

    # Top outlier features
    top_outliers = df_outliers.nlargest(10, 'Outliers_Defect')
    axes[0].barh(range(len(top_outliers)), top_outliers['Outliers_Defect'].values, color='red', alpha=0.7)
    axes[0].set_yticks(range(len(top_outliers)))
    axes[0].set_yticklabels(top_outliers['Feature'].values)
    axes[0].set_xlabel('Number of Outliers in Defects')
    axes[0].set_title('Features with Most Outliers (Defects)')
    axes[0].invert_yaxis()

    # Outlier ratio
    outlier_ratio = df_outliers[df_outliers['Outliers_Defect'] > 0].nlargest(10, 'Outlier_Ratio')
    axes[1].barh(range(len(outlier_ratio)), outlier_ratio['Outlier_Ratio'].values, color='orange', alpha=0.7)
    axes[1].set_yticks(range(len(outlier_ratio)))
    axes[1].set_yticklabels(outlier_ratio['Feature'].values)
    axes[1].set_xlabel('Outlier Ratio (Defects)')
    axes[1].set_title('Outlier Frequency in Defects')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_variance_instability(df_cv, output_file='viz_06_instability.png'):
    """Plot variance/instability analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Variance & Instability Analysis', fontsize=14, fontweight='bold')

    # Top unstable features
    top_unstable = df_cv.nlargest(10, 'Relative_Instability_%')
    axes[0].barh(range(len(top_unstable)), top_unstable['Relative_Instability_%'].values, color='darkred', alpha=0.7)
    axes[0].set_yticks(range(len(top_unstable)))
    axes[0].set_yticklabels(top_unstable['Feature'].values)
    axes[0].set_xlabel('Relative Instability (%)')
    axes[0].set_title('Features with Highest Instability in Defects')
    axes[0].invert_yaxis()

    # CV comparison
    cv_data = top_unstable[['Feature', 'Normal_CV', 'Defect_CV']].set_index('Feature')
    cv_data.plot(kind='barh', ax=axes[1], color=['green', 'red'], alpha=0.7)
    axes[1].set_xlabel('Coefficient of Variation')
    axes[1].set_title('CV Comparison: Normal vs Defect')
    axes[1].legend(['Normal', 'Defect'])
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 12-13: INTERACTIONS
# =============================================================================

def plot_interactions(df_interactions, output_file='viz_13_interactions.png'):
    """Plot dangerous parameter combinations"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Dangerous Parameter Combinations', fontsize=14, fontweight='bold')

    top_interactions = df_interactions.nlargest(10, 'Interaction_Strength')
    y_pos = np.arange(len(top_interactions))

    bars = ax.barh(y_pos, top_interactions['Interaction_Strength'].values, color='darkred', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_interactions['Feature_Pair'].values)
    ax.set_xlabel('Interaction Strength')
    ax.set_title('Top Parameter Combinations Causing Defects')

    # Add values on bars
    for i, (idx, row) in enumerate(top_interactions.iterrows()):
        ax.text(row['Interaction_Strength'], i, f" {row['Interaction_Strength']:.3f}",
                va='center', fontweight='bold')

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 14: OPERATING WINDOWS
# =============================================================================

def plot_operating_windows(df_thresholds, output_file='viz_14_operating_windows.png'):
    """Plot safe vs unsafe operating windows"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Safe vs Unsafe Operating Windows', fontsize=14, fontweight='bold')

    features = df_thresholds['Feature'].values
    safe_rates = df_thresholds['Safe_Rate'].values * 100
    unsafe_rates = df_thresholds['Unsafe_Rate'].values * 100

    x = np.arange(len(features))
    width = 0.35

    bars1 = ax.bar(x - width/2, safe_rates, width, label='Safe Region', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, unsafe_rates, width, label='Unsafe Region', color='red', alpha=0.7)

    ax.set_ylabel('Defect Rate (%)')
    ax.set_title('Defect Rates in Safe vs Unsafe Operating Regions')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 17: PCA VISUALIZATION
# =============================================================================

def plot_pca_analysis(pca, X_scaled, y_train, output_file='viz_17_pca.png'):
    """Plot PCA analysis"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)

    fig.suptitle('PCA Analysis - Latent Process Regimes', fontsize=14, fontweight='bold')

    # Variance explained
    ax1 = fig.add_subplot(gs[0, 0])
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax1.plot(range(1, min(21, len(cumsum)+1)), cumsum[:20], 'bo-', linewidth=2, markersize=6)
    ax1.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    ax1.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance')
    ax1.set_title('Variance Explained by Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scree plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(1, min(11, len(pca.explained_variance_ratio_)+1)),
           pca.explained_variance_ratio_[:10], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Scree Plot (Top 10 Components)')
    ax2.grid(True, alpha=0.3, axis='y')

    # PC1 vs PC2
    ax3 = fig.add_subplot(gs[1, 0])
    X_pca = pca.transform(X_scaled)
    normal_mask = y_train == 0
    defect_mask = y_train == 1
    ax3.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], alpha=0.5, c='green', label='Normal', s=20)
    ax3.scatter(X_pca[defect_mask, 0], X_pca[defect_mask, 1], alpha=0.5, c='red', label='Defect', s=20)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_title('PC1 vs PC2 - Process Regimes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # PC1 vs PC3
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 2], alpha=0.5, c='green', label='Normal', s=20)
    ax4.scatter(X_pca[defect_mask, 0], X_pca[defect_mask, 2], alpha=0.5, c='red', label='Defect', s=20)
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax4.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax4.set_title('PC1 vs PC3 - Alternative View')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    return X_pca


def plot_umap_visualization(X_umap, y_train, output_file='viz_17_umap.png'):
    """Plot UMAP 2D projection"""
    if X_umap is None:
        print("⚠️  UMAP not available (install: pip install umap-learn)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    normal_mask = y_train == 0
    defect_mask = y_train == 1

    ax.scatter(X_umap[normal_mask, 0], X_umap[normal_mask, 1],
              alpha=0.5, c='green', label='Normal', s=20)
    ax.scatter(X_umap[defect_mask, 0], X_umap[defect_mask, 1],
              alpha=0.5, c='red', label='Defect', s=20)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP Projection - Non-Linear Latent Structure', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_tsne_visualization(X_tsne, y_train, output_file='viz_17_tsne.png'):
    """Plot t-SNE 2D projection"""
    if X_tsne is None:
        print("⚠️  t-SNE not available")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Map y_train to X_tsne indices (t-SNE uses subset)
    sample_size = min(len(X_tsne), len(y_train))
    y_sample = y_train[:sample_size]

    normal_mask = y_sample == 0
    defect_mask = y_sample == 1

    ax.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1],
              alpha=0.5, c='green', label='Normal', s=20)
    ax.scatter(X_tsne[defect_mask, 0], X_tsne[defect_mask, 1],
              alpha=0.5, c='red', label='Defect', s=20)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Projection - Manifold Learning (Sampled)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 18: REGIME ANALYSIS
# =============================================================================

def plot_regime_analysis(regime_labels, y_train, X_pca, output_file='viz_18_regimes.png'):
    """Plot operating regimes"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hidden Operating Regimes', fontsize=14, fontweight='bold')

    # Regime sizes and defect rates
    unique_regimes = np.unique(regime_labels)
    regime_sizes = []
    defect_rates = []

    for regime in unique_regimes:
        mask = regime_labels == regime
        regime_sizes.append(np.sum(mask))
        defect_rates.append(np.sum(y_train[mask] == 1) / np.sum(mask) * 100)

    axes[0].bar(unique_regimes, regime_sizes, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Operating Regime')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Regime Distribution')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Defect rate per regime
    colors = ['green' if rate < 10 else 'orange' if rate < 20 else 'red' for rate in defect_rates]
    axes[1].bar(unique_regimes, defect_rates, color=colors, alpha=0.7)
    axes[1].set_xlabel('Operating Regime')
    axes[1].set_ylabel('Defect Rate (%)')
    axes[1].set_title('Defect Probability per Regime')
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, (regime, rate) in enumerate(zip(unique_regimes, defect_rates)):
        axes[1].text(regime, rate + 1, f'{rate:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_regime_scatter(X_pca, regime_labels, y_train, output_file='viz_18_regime_scatter.png'):
    """Plot regimes in PCA space"""
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_regimes = np.unique(regime_labels)
    colors_regime = plt.cm.Set3(np.linspace(0, 1, len(unique_regimes)))

    for regime in unique_regimes:
        mask = regime_labels == regime
        defect_mask = mask & (y_train == 1)
        normal_mask = mask & (y_train == 0)

        ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                  c=[colors_regime[regime]], marker='o', s=30, alpha=0.6,
                  label=f'Regime {regime} (Normal)', edgecolors='black', linewidth=0.5)
        ax.scatter(X_pca[defect_mask, 0], X_pca[defect_mask, 1],
                  c=[colors_regime[regime]], marker='X', s=100, alpha=0.8,
                  label=f'Regime {regime} (Defect)', edgecolors='black', linewidth=1)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Operating Regimes in PCA Space', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 16: ANOMALY BEHAVIOR
# =============================================================================

def plot_anomaly_analysis(anomaly_scores, y_train, output_file='viz_16_anomaly.png'):
    """Plot anomaly detection results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Anomaly Detection Analysis', fontsize=14, fontweight='bold')

    normal_anomaly = anomaly_scores[y_train == 0]
    defect_anomaly = anomaly_scores[y_train == 1]

    # Distribution of anomaly scores
    axes[0].hist(normal_anomaly, bins=30, alpha=0.6, label='Normal', color='green', density=True)
    axes[0].hist(defect_anomaly, bins=30, alpha=0.6, label='Defect', color='red', density=True)
    axes[0].axvline(x=-0.5, color='black', linestyle='--', linewidth=2, label='Anomaly Threshold')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Anomaly Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Anomaly classification
    normal_anomaly_count = np.sum(normal_anomaly < -0.5)
    normal_normal_count = np.sum(normal_anomaly >= -0.5)
    defect_anomaly_count = np.sum(defect_anomaly < -0.5)
    defect_normal_count = np.sum(defect_anomaly >= -0.5)

    categories = ['Normal\n(Not Anomaly)', 'Normal\n(Anomaly)', 'Defect\n(Not Anomaly)', 'Defect\n(Anomaly)']
    counts = [normal_normal_count, normal_anomaly_count, defect_normal_count, defect_anomaly_count]
    colors_bar = ['green', 'orange', 'red', 'darkred']

    axes[1].bar(categories, counts, color=colors_bar, alpha=0.7)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Anomaly Classification')
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, count in enumerate(counts):
        axes[1].text(i, count + max(counts)*0.02, str(count), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 4: UNIVARIATE SIGNIFICANCE
# =============================================================================

def plot_univariate_significance(df_univariate, output_file='viz_04_significance.png'):
    """Plot p-values for univariate tests"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort by p-value
    top_features = df_univariate.nsmallest(15, 'P_Value')

    colors = ['green' if p < 0.05 else 'gray' for p in top_features['P_Value'].values]
    ax.barh(range(len(top_features)), -np.log10(top_features['P_Value'].values), color=colors, alpha=0.7)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.set_xlabel('-log10(p-value)')
    ax.set_title('Univariate Significance Test Results (Mann-Whitney U)', fontsize=12, fontweight='bold')
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='Significance Level (p=0.05)')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# POINT 20: PROCESS FINGERPRINTS
# =============================================================================

def plot_process_fingerprints(train_df, feature_cols, output_file='viz_20_fingerprints.png'):
    """Plot process fingerprints"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Thermo-Mechanical Process Fingerprints', fontsize=14, fontweight='bold')

    normal_samples = train_df[train_df['Y'] == 0][feature_cols].iloc[:20]
    defect_samples = train_df[train_df['Y'] == 1][feature_cols].iloc[:20]

    # Normal fingerprints
    for idx, row in normal_samples.iterrows():
        axes[0, 0].plot(range(len(feature_cols)), row.values, alpha=0.3, color='green')
    axes[0, 0].plot(range(len(feature_cols)), normal_samples.mean().values,
                   color='green', linewidth=3, label='Mean')
    axes[0, 0].fill_between(range(len(feature_cols)),
                           normal_samples.mean() - normal_samples.std(),
                           normal_samples.mean() + normal_samples.std(),
                           alpha=0.2, color='green')
    axes[0, 0].set_title('Normal Coil Signatures')
    axes[0, 0].set_ylabel('Normalized Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Defect fingerprints
    for idx, row in defect_samples.iterrows():
        axes[0, 1].plot(range(len(feature_cols)), row.values, alpha=0.3, color='red')
    axes[0, 1].plot(range(len(feature_cols)), defect_samples.mean().values,
                   color='red', linewidth=3, label='Mean')
    axes[0, 1].fill_between(range(len(feature_cols)),
                           defect_samples.mean() - defect_samples.std(),
                           defect_samples.mean() + defect_samples.std(),
                           alpha=0.2, color='red')
    axes[0, 1].set_title('Defective Coil Signatures')
    axes[0, 1].set_ylabel('Normalized Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Mean comparison
    axes[1, 0].plot(range(len(feature_cols)), normal_samples.mean().values,
                   'g-o', linewidth=2, markersize=4, label='Normal', alpha=0.7)
    axes[1, 0].plot(range(len(feature_cols)), defect_samples.mean().values,
                   'r-s', linewidth=2, markersize=4, label='Defect', alpha=0.7)
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_title('Mean Fingerprint Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Std comparison
    axes[1, 1].plot(range(len(feature_cols)), normal_samples.std().values,
                   'g-o', linewidth=2, markersize=4, label='Normal', alpha=0.7)
    axes[1, 1].plot(range(len(feature_cols)), defect_samples.std().values,
                   'r-s', linewidth=2, markersize=4, label='Defect', alpha=0.7)
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Std Dev')
    axes[1, 1].set_title('Signature Variability Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# =============================================================================
# MASTER VISUALIZATION FUNCTION
# =============================================================================

def create_all_visualizations(train_df, feature_cols, df_univariate, df_cv, df_outliers,
                             df_interactions, df_thresholds, corr_overall, corr_normal, corr_defect,
                             pca, X_scaled, X_umap, X_tsne, regime_labels, anomaly_scores, y_train,
                             verbose=True):
    """Create all visualization plots"""

    if verbose:
        print("\n" + "="*80)
        print("CREATING COMPLETE VISUALIZATION SUITE")
        print("="*80)

    # Point 1-2
    plot_class_distribution(y_train)

    # Point 4-5
    top_features = df_univariate.nsmallest(12, 'P_Value')['Feature'].tolist()
    plot_univariate_significance(df_univariate)
    plot_distribution_comparison(train_df, feature_cols, top_features)
    plot_box_plots(train_df, feature_cols, top_features)

    # Point 6
    plot_variance_instability(df_cv)

    # Point 7-8
    plot_outliers_variance(df_outliers)

    # Point 9-10
    plot_correlation_heatmap(corr_overall, 'Overall Correlation Matrix', 'viz_09_corr_overall.png')
    plot_correlation_heatmap(corr_normal, 'Normal Samples Correlation', 'viz_10_corr_normal.png')
    plot_correlation_heatmap(corr_defect, 'Defect Samples Correlation', 'viz_10_corr_defect.png')
    plot_correlation_comparison(corr_normal, corr_defect)

    # Point 13
    plot_interactions(df_interactions)

    # Point 14
    plot_operating_windows(df_thresholds)

    # Point 16
    plot_anomaly_analysis(anomaly_scores, y_train)

    # Point 17
    X_pca = plot_pca_analysis(pca, X_scaled, y_train)
    plot_umap_visualization(X_umap, y_train)
    plot_tsne_visualization(X_tsne, y_train)

    # Point 18
    plot_regime_analysis(regime_labels, y_train, X_pca)
    plot_regime_scatter(X_pca, regime_labels, y_train)

    # Point 20
    plot_process_fingerprints(train_df, feature_cols)

    if verbose:
        print("\n" + "="*80)
        print("✓ ALL VISUALIZATIONS CREATED")
        print("="*80)
        print("\nGenerated Files:")
        print("  ✓ viz_01_class_distribution.png")
        print("  ✓ viz_04_significance.png")
        print("  ✓ viz_04_distributions.png")
        print("  ✓ viz_05_boxplots.png")
        print("  ✓ viz_06_instability.png")
        print("  ✓ viz_07_outliers.png")
        print("  ✓ viz_09_corr_overall.png")
        print("  ✓ viz_10_corr_normal.png")
        print("  ✓ viz_10_corr_defect.png")
        print("  ✓ viz_10_corr_comparison.png")
        print("  ✓ viz_13_interactions.png")
        print("  ✓ viz_14_operating_windows.png")
        print("  ✓ viz_16_anomaly.png")
        print("  ✓ viz_17_pca.png")
        print("  ✓ viz_17_umap.png")
        print("  ✓ viz_17_tsne.png")
        print("  ✓ viz_18_regimes.png")
        print("  ✓ viz_18_regime_scatter.png")
        print("  ✓ viz_20_fingerprints.png")


if __name__ == "__main__":
    print("Visualization module loaded. Use create_all_visualizations() to generate plots.")
