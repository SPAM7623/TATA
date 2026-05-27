"""
Google Colab Notebook - Alpha Defect Detection
Step-by-step guide for running in Google Colab

Copy each cell (between ### markers) into separate Colab cells
"""

# =============================================================================
# CELL 1: Mount Google Drive
# =============================================================================
### CELL 1 START ###
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted")
### CELL 1 END ###

# Expected output:
# Mounted at /content/drive


# =============================================================================
# CELL 2: Upload Files OR Copy from Drive
# =============================================================================
### CELL 2 START - OPTION A: Direct Upload ###
from google.colab import files

print("Upload train.csv and test.csv")
uploaded = files.upload()

# Check uploaded files
import os
print("\nFiles in /content:")
print(os.listdir('/content'))
### CELL 2 END ###

# Alternative OPTION B: Copy from Google Drive
### CELL 2B START - OPTION B: From Drive ###
import shutil

# Adjust paths to your Google Drive structure
shutil.copy('/content/drive/MyDrive/alpha_defect/train.csv', '/content/train.csv')
shutil.copy('/content/drive/MyDrive/alpha_defect/test.csv', '/content/test.csv')

import os
print("Files copied successfully")
print("Files in /content:", os.listdir('/content'))
### CELL 2B END ###


# =============================================================================
# CELL 3: Install Libraries
# =============================================================================
### CELL 3 START ###
!pip install -q pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn scipy

print("\n✓ All libraries installed successfully")
print("\nKey packages:")
print("  - pandas: data manipulation")
print("  - scikit-learn: ML algorithms & preprocessing")
print("  - xgboost, lightgbm: gradient boosting")
print("  - shap: model interpretability")
print("  - matplotlib, seaborn: visualization")
### CELL 3 END ###


# =============================================================================
# CELL 4: Phase 1 - Data Loading & Cleaning
# =============================================================================
### CELL 4 START ###
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 1: DATA LOADING & CLEANING")
print("="*80)

# Load datasets
print("\n[1/5] Loading datasets...")
train_df = pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test.csv')

print(f"  ✓ Train set: {train_df.shape[0]} rows × {train_df.shape[1]} cols")
print(f"  ✓ Test set:  {test_df.shape[0]} rows × {test_df.shape[1]} cols")

# Identify features
feature_cols = [col for col in train_df.columns if col.startswith('X')]
print(f"\n[2/5] Found {len(feature_cols)} features (X1-X49)")

# Convert to numeric
print("\n[3/5] Converting to numeric and detecting missing values...")
for col in feature_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

missing_after = train_df[feature_cols].isnull().sum()
cols_with_missing = missing_after[missing_after > 0]

if len(cols_with_missing) > 0:
    print(f"  ⚠ Missing values detected:")
    for col, count in cols_with_missing.items():
        pct = count / len(train_df) * 100
        print(f"    {col}: {count} ({pct:.2f}%)")
else:
    print(f"  ✓ No missing values detected")

# Handle missing values
print("\n[4/5] Handling missing values...")
filled_count = 0
for col in feature_cols:
    missing_count = train_df[col].isnull().sum()
    if missing_count > 0:
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        test_df[col].fillna(median_val, inplace=True)
        filled_count += missing_count
        print(f"  {col}: Filled {missing_count} values with median")

if filled_count == 0:
    print(f"  ✓ No missing values to fill")

# Validation
print("\n[5/5] Data quality validation...")

# Check for NaN remaining
remaining_nan = train_df[feature_cols].isnull().sum().sum()
assert remaining_nan == 0, f"Remaining NaN values: {remaining_nan}"
print(f"  ✓ All features numeric, no missing values")

# Target distribution
if 'Y' in train_df.columns:
    y_counts = train_df['Y'].value_counts().sort_index()
    y_pct = train_df['Y'].value_counts(normalize=True).sort_index() * 100

    print(f"\n  Target Variable Distribution:")
    for label in sorted(train_df['Y'].unique()):
        count = y_counts[label]
        pct = y_pct[label]
        label_name = 'NORMAL' if label == 0 else 'ALPHA DEFECT'
        print(f"    Y={label}: {count:4d} ({pct:5.2f}%) [{label_name}]")

    if len(y_counts) > 1:
        ratio = y_counts.iloc[0] / y_counts.iloc[1]
        print(f"  → Class imbalance ratio: {ratio:.1f}:1")

# Save cleaned data
print("\n" + "="*80)
print("✓ DATA LOADING COMPLETE")
print("="*80)

train_df.to_csv('/content/train_cleaned.csv', index=False)
test_df.to_csv('/content/test_cleaned.csv', index=False)

print(f"\nDatasets saved:")
print(f"  - train_cleaned.csv ({train_df.shape})")
print(f"  - test_cleaned.csv ({test_df.shape})")
print(f"\nReady for Phase 2: Industrial EDA")
### CELL 4 END ###


# =============================================================================
# CELL 5: Phase 2 - Industrial EDA
# =============================================================================
### CELL 5 START ###
from scipy.stats import skew, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load cleaned data
train_df = pd.read_csv('/content/train_cleaned.csv')
feature_cols = [col for col in train_df.columns if col.startswith('X')]

print("\n" + "="*80)
print("PHASE 2: INDUSTRIAL EDA ANALYSIS")
print("="*80)

# =========================================================================
# 1. UNIVARIATE ANALYSIS
# =========================================================================
print("\n[1/6] UNIVARIATE ANALYSIS")

univariate_results = []
for feature in feature_cols:
    normal_data = train_df[train_df['Y'] == 0][feature].dropna()
    defect_data = train_df[train_df['Y'] == 1][feature].dropna()

    stat, p_value = mannwhitneyu(normal_data, defect_data)

    result = {
        'Feature': feature,
        'Normal_Mean': normal_data.mean(),
        'Defect_Mean': defect_data.mean(),
        'Normal_Std': normal_data.std(),
        'Defect_Std': defect_data.std(),
        'Mean_Diff': defect_data.mean() - normal_data.mean(),
        'P_Value': p_value,
        'Significant': 'YES' if p_value < 0.05 else 'NO',
    }
    univariate_results.append(result)

univariate_df = pd.DataFrame(univariate_results).sort_values('P_Value')

print("\nTop 10 Most Significant Features:")
print(univariate_df[['Feature', 'Normal_Mean', 'Defect_Mean', 'P_Value']].head(10).to_string(index=False))

sig_count = (univariate_df['Significant'] == 'YES').sum()
print(f"\nTotal significant features (p < 0.05): {sig_count}/{len(feature_cols)}")

# =========================================================================
# 2. VARIANCE & INSTABILITY ANALYSIS
# =========================================================================
print("\n[2/6] VARIANCE & INSTABILITY ANALYSIS")

cv_analysis = []
for feature in feature_cols:
    normal_data = train_df[train_df['Y'] == 0][feature].dropna()
    defect_data = train_df[train_df['Y'] == 1][feature].dropna()

    normal_cv = normal_data.std() / abs(normal_data.mean()) if normal_data.mean() != 0 else 0
    defect_cv = defect_data.std() / abs(defect_data.mean()) if defect_data.mean() != 0 else 0

    cv_analysis.append({
        'Feature': feature,
        'Normal_CV': normal_cv,
        'Defect_CV': defect_cv,
        'Relative_Instability': (defect_cv / normal_cv - 1) * 100 if normal_cv > 0 else 0,
    })

cv_df = pd.DataFrame(cv_analysis).sort_values('Relative_Instability', ascending=False)

print("\nTop 10 Features with Higher Instability in Defects:")
print(cv_df[['Feature', 'Normal_CV', 'Defect_CV', 'Relative_Instability']].head(10).to_string(index=False))

# =========================================================================
# 3. CORRELATION ANALYSIS
# =========================================================================
print("\n[3/6] CORRELATION & FEATURE GROUPING")

corr_matrix = train_df[feature_cols].corr()

high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature1': feature_cols[i],
                'Feature2': feature_cols[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print(f"\nHighly Correlated Pairs (|r| > 0.8): {len(high_corr_pairs)}")
    for pair in sorted(high_corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True)[:5]:
        print(f"  {pair['Feature1']} ↔ {pair['Feature2']}: {pair['Correlation']:.3f}")
else:
    print("\n✓ No extremely high correlations (good feature diversity)")

# =========================================================================
# 4. PCA ANALYSIS
# =========================================================================
print("\n[4/6] PCA ANALYSIS - Hidden Process Regimes")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[feature_cols])

pca = PCA()
pca.fit(X_scaled)

cumsum_var = np.cumsum(pca.explained_variance_ratio_)
n_components_80 = (cumsum_var >= 0.80).argmax() + 1
n_components_95 = (cumsum_var >= 0.95).argmax() + 1

print(f"\nDimensionality Analysis:")
print(f"  Total features: {len(feature_cols)}")
print(f"  Components for 80% variance: {n_components_80}")
print(f"  Components for 95% variance: {n_components_95}")
print(f"  Redundant features: ~{len(feature_cols) - n_components_80}")

print(f"\nTop 5 PC Explained Variance:")
for i in range(5):
    var = pca.explained_variance_ratio_[i]
    cumvar = cumsum_var[i]
    print(f"  PC{i+1}: {var:.4f} (cumsum: {cumvar:.4f})")

# =========================================================================
# 5. DANGEROUS INTERACTIONS
# =========================================================================
print("\n[5/6] DANGEROUS PARAMETER COMBINATIONS")

top_features = univariate_df.nsmallest(5, 'P_Value')['Feature'].tolist()
interaction_risk = []

for i, feat1 in enumerate(top_features[:3]):
    for feat2 in top_features[i + 1:4]:
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

if interaction_risk:
    interaction_df = pd.DataFrame(interaction_risk).sort_values('Interaction_Strength', ascending=False)
    print(f"\nStrongest Interactions:")
    print(interaction_df.to_string(index=False))

# =========================================================================
# 6. SAVE RESULTS
# =========================================================================
print("\n[6/6] SAVING RESULTS")

univariate_df.to_csv('/content/eda_univariate_analysis.csv', index=False)
cv_df.to_csv('/content/eda_instability_analysis.csv', index=False)

print("\n" + "="*80)
print("✓ EDA ANALYSIS COMPLETE")
print("="*80)

print(f"\nKey Findings:")
print(f"  • {sig_count} significant features detected")
print(f"  • {len(high_corr_pairs)} highly correlated pairs")
print(f"  • {n_components_80} components for 80% variance")
print(f"  • {len(interaction_risk)} dangerous interactions identified")

print(f"\nResults saved:")
print(f"  - eda_univariate_analysis.csv")
print(f"  - eda_instability_analysis.csv")
### CELL 5 END ###


# =============================================================================
# CELL 6: Visualizations
# =============================================================================
### CELL 6 START ###
import matplotlib.pyplot as plt
import seaborn as sns

# Create distribution plots for top significant features
sig_features = univariate_df[univariate_df['Significant'] == 'YES']['Feature'].tolist()[:12]

fig, axes = plt.subplots(4, 3, figsize=(16, 12))
fig.suptitle('Feature Distributions: Normal vs Defective Samples', fontsize=14, fontweight='bold')

for idx, feature in enumerate(sig_features):
    ax = axes[idx // 3, idx % 3]

    normal = train_df[train_df['Y'] == 0][feature]
    defect = train_df[train_df['Y'] == 1][feature]

    ax.hist(normal, bins=20, alpha=0.6, label='Normal (Y=0)', color='green')
    ax.hist(defect, bins=20, alpha=0.6, label='Defect (Y=1)', color='red')

    ax.set_xlabel(feature, fontweight='bold')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title(f'{feature} Distribution')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/distributions_defect_vs_normal.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: distributions_defect_vs_normal.png")
### CELL 6 END ###


# =============================================================================
# CELL 7: Download Results
# =============================================================================
### CELL 7 START ###
from google.colab import files

print("Downloading EDA results...")
files.download('eda_univariate_analysis.csv')
files.download('eda_instability_analysis.csv')
files.download('distributions_defect_vs_normal.png')

print("✓ Downloaded successfully!")
print("\nYou can now proceed to Phase 3: Feature Engineering")
### CELL 7 END ###
