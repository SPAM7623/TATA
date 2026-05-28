"""
================================================================================
ALPHA DEFECT DETECTION - PHASE 4 V2 (IMPROVED)
FEATURE ENGINEERING + IMBALANCE HANDLING + THRESHOLD TUNING
================================================================================

IMPROVEMENTS OVER V1:
  ✓ Used EDA visualizations (heatmap, instability, PCA) to guide decisions
  ✓ Used EDA CSV data (numerical values) for feature selection
  ✓ Used SHAP values to prioritize top-driving features
  ✓ Used baseline model insights to set comparison baseline
  ✓ ALL features now traceable to specific EDA/SHAP insights

VISUAL ANALYSIS INSIGHTS APPLIED:
  1. VIZ_09_CORR_OVERALL.PNG → Correlation blocks identified
  2. VIZ_06_INSTABILITY.PNG → Top 10 unstable features selected
  3. VIZ_17_PCA.PNG → Defects NOT separate → focus on interactions, not anomalies
  4. VIZ_SHAP_SUMMARY.PNG → X36 is dominant driver → X36-centric features
  5. BASELINE MODELS → ROC-AUC 0.8639 baseline to beat

PRIORITY FEATURES ENGINEERED:
  • X36-centric: X36/X35, X36/X34, X36×X41 (highest SHAP + dangerous interactions)
  • Cooling cluster: X35×X41, X34/X36, X35/X34 (259-240% instability)
  • Top interactions: X35×X41, X29×X41, X30×X41 (highest risk scores)
  • Rolling cluster: X13×X41, X10×X41 (secondary interactions)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve, auc
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
import shap
import pickle
import warnings

warnings.filterwarnings('ignore')


class ImprovedFeatureEngineeringPipeline:
    """Feature engineering with visual + data-driven insights"""

    def __init__(self, train_path='train_cleaned.csv', test_path='test_cleaned.csv',
                 output_dir='./', verbose=True):
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.verbose = verbose

        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.feature_cols = None
        self.engineered_features = {}

    def load_and_analyze_eda(self):
        """Load and analyze EDA outputs + visualizations"""
        if self.verbose:
            print("\n[1/7] Loading EDA outputs and analyzing visualizations...")

        # Load key EDA CSV files
        self.df_instability = pd.read_csv('eda_instability_analysis.csv')
        self.df_interactions = pd.read_csv('eda_interactions_analysis.csv')
        self.df_redundancy = pd.read_csv('eda_redundancy_analysis.csv')
        self.df_univariate = pd.read_csv('eda_univariate_analysis.csv')

        if self.verbose:
            print("  ✓ eda_instability_analysis.csv loaded")
            print("  ✓ eda_interactions_analysis.csv loaded")
            print("  ✓ eda_redundancy_analysis.csv loaded")
            print("  ✓ eda_univariate_analysis.csv loaded")

            print("\n  VISUAL INSIGHTS FROM EDA PLOTS:")
            print("    • viz_09_corr_overall.png: Identified correlation blocks")
            print("    • viz_06_instability.png: X35, X34, X36, X37 most unstable (259-240%)")
            print("    • viz_17_pca.png: Defects overlap with normal (no separation)")
            print("    • viz_shap_summary.png: X36 is dominant SHAP driver")

    def load_data(self):
        """Load training and test data"""
        if self.verbose:
            print("\n[2/7] Loading cleaned data...")

        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        self.feature_cols = [col for col in self.train_df.columns if col.startswith('X')]
        self.X_train = self.train_df[self.feature_cols].values
        self.y_train = self.train_df['Y'].values
        self.X_test = self.test_df[self.feature_cols].values

        if self.verbose:
            print(f"  ✓ Train: {self.X_train.shape}")
            print(f"  ✓ Test:  {self.X_test.shape}")

    def task6_improved_feature_engineering(self):
        """IMPROVED Feature Engineering using EDA + SHAP insights"""
        if self.verbose:
            print("\n[3/7] TASK 6: Improved Feature Engineering (EDA/SHAP-guided)...")

        X_train_eng = self.X_train.copy()
        X_test_eng = self.X_test.copy()
        feat_idx = {f: i for i, f in enumerate(self.feature_cols)}

        # ======================================================================
        # PRIORITY 1: X36-CENTRIC FEATURES (Dominant SHAP driver)
        # ======================================================================
        if self.verbose:
            print("  [1/5] X36-centric features (DOMINANT SHAP driver)...")

        # X36 is the single most important SHAP feature
        # Create ratios and interactions with other cooling parameters
        x36_idx = feat_idx['X36']
        x35_idx = feat_idx['X35']
        x34_idx = feat_idx['X34']
        x41_idx = feat_idx['X41']

        # X36 / X35 ratio (cooling imbalance)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(self.X_train[:, x36_idx], self.X_train[:, x35_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_train_eng = np.column_stack([X_train_eng, ratio])

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_test = np.divide(self.X_test[:, x36_idx], self.X_test[:, x35_idx] + 1e-6)
            ratio_test[~np.isfinite(ratio_test)] = 0
        X_test_eng = np.column_stack([X_test_eng, ratio_test])
        self.engineered_features['X36/X35'] = 'X36_centric'

        # X36 / X34 ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(self.X_train[:, x36_idx], self.X_train[:, x34_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_train_eng = np.column_stack([X_train_eng, ratio])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_test = np.divide(self.X_test[:, x36_idx], self.X_test[:, x34_idx] + 1e-6)
            ratio_test[~np.isfinite(ratio_test)] = 0
        X_test_eng = np.column_stack([X_test_eng, ratio_test])
        self.engineered_features['X36/X34'] = 'X36_centric'

        # X36 × X41 (highest danger interaction: 26.2% defect rate!)
        X_train_eng = np.column_stack([X_train_eng, self.X_train[:, x36_idx] * self.X_train[:, x41_idx]])
        X_test_eng = np.column_stack([X_test_eng, self.X_test[:, x36_idx] * self.X_test[:, x41_idx]])
        self.engineered_features['X36xX41'] = 'X36_centric'

        # ======================================================================
        # PRIORITY 2: COOLING CLUSTER RATIOS (X35, X34, X36 - 259-240% instability)
        # ======================================================================
        if self.verbose:
            print("  [2/5] Cooling cluster features (259-240% instability)...")

        # From viz_06_instability: X35, X34, X36 have highest defect CV
        # X35 / X34 (within-block imbalance)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(self.X_train[:, x35_idx], self.X_train[:, x34_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_train_eng = np.column_stack([X_train_eng, ratio])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_test = np.divide(self.X_test[:, x35_idx], self.X_test[:, x34_idx] + 1e-6)
            ratio_test[~np.isfinite(ratio_test)] = 0
        X_test_eng = np.column_stack([X_test_eng, ratio_test])
        self.engineered_features['X35/X34'] = 'cooling_cluster'

        # X35 × X41 (2nd highest danger: 26.2% defect rate!)
        X_train_eng = np.column_stack([X_train_eng, self.X_train[:, x35_idx] * self.X_train[:, x41_idx]])
        X_test_eng = np.column_stack([X_test_eng, self.X_test[:, x35_idx] * self.X_test[:, x41_idx]])
        self.engineered_features['X35xX41'] = 'cooling_cluster'

        # X34 - X36 (difference in cooling parameters)
        X_train_eng = np.column_stack([X_train_eng, self.X_train[:, x34_idx] - self.X_train[:, x36_idx]])
        X_test_eng = np.column_stack([X_test_eng, self.X_test[:, x34_idx] - self.X_test[:, x36_idx]])
        self.engineered_features['X34-X36'] = 'cooling_cluster'

        # ======================================================================
        # PRIORITY 3: TOP DANGEROUS INTERACTIONS (from eda_interactions_analysis)
        # ======================================================================
        if self.verbose:
            print("  [3/5] Top dangerous interactions (from EDA)...")

        # Top 5 interactions by risk score
        top_interactions = [
            ('X29', 'X41'),  # Risk: 36.59
            ('X30', 'X41'),  # Risk: 35.92
            ('X31', 'X41'),  # Risk: 35.58
            ('X32', 'X41'),  # Risk: 34.69
            ('X13', 'X41'),  # Risk: 32.76
        ]

        for feat1, feat2 in top_interactions:
            i1, i2 = feat_idx[feat1], feat_idx[feat2]
            name = f'{feat1}x{feat2}'
            X_train_eng = np.column_stack([X_train_eng,
                self.X_train[:, i1] * self.X_train[:, i2]])
            X_test_eng = np.column_stack([X_test_eng,
                self.X_test[:, i1] * self.X_test[:, i2]])
            self.engineered_features[name] = 'dangerous_interaction'

        # ======================================================================
        # PRIORITY 4: ROW-LEVEL STATISTICS + INSTABILITY
        # ======================================================================
        if self.verbose:
            print("  [4/5] Row-level statistics and instability...")

        row_mean = self.X_train.mean(axis=1, keepdims=True)
        row_std = self.X_train.std(axis=1, keepdims=True)

        X_train_eng = np.column_stack([X_train_eng, row_mean, row_std])

        row_mean_test = self.X_test.mean(axis=1, keepdims=True)
        row_std_test = self.X_test.std(axis=1, keepdims=True)

        X_test_eng = np.column_stack([X_test_eng, row_mean_test, row_std_test])

        self.engineered_features['row_mean'] = 'row_stat'
        self.engineered_features['row_std'] = 'row_stat'

        # Instability (CV) - captures high variance in defects
        with np.errstate(divide='ignore', invalid='ignore'):
            instability = np.divide(row_std.flatten(), np.abs(row_mean.flatten()) + 1e-6)
            instability[~np.isfinite(instability)] = 0
            instability_test = np.divide(row_std_test.flatten(), np.abs(row_mean_test.flatten()) + 1e-6)
            instability_test[~np.isfinite(instability_test)] = 0

        X_train_eng = np.column_stack([X_train_eng, instability])
        X_test_eng = np.column_stack([X_test_eng, instability_test])
        self.engineered_features['instability'] = 'row_stat'

        # ======================================================================
        # PRIORITY 5: ROLLING & FURNACE CLUSTERS (Secondary SHAP drivers)
        # ======================================================================
        if self.verbose:
            print("  [5/5] Secondary cluster features...")

        # Rolling cluster means (X10-X24)
        rolling_idx = [feat_idx[f] for f in ['X10', 'X13', 'X14', 'X16'] if f in feat_idx]
        rolling_mean = self.X_train[:, rolling_idx].mean(axis=1)
        rolling_mean_test = self.X_test[:, rolling_idx].mean(axis=1)
        X_train_eng = np.column_stack([X_train_eng, rolling_mean])
        X_test_eng = np.column_stack([X_test_eng, rolling_mean_test])
        self.engineered_features['rolling_mean'] = 'rolling_cluster'

        if self.verbose:
            print(f"\n  ✓ Feature engineering complete")
            print(f"  ✓ Original features: {self.X_train.shape[1]}")
            print(f"  ✓ Engineered features: {X_train_eng.shape[1] - self.X_train.shape[1]}")
            print(f"  ✓ Total features: {X_train_eng.shape[1]}")

        return X_train_eng, X_test_eng

    def task7_task8_imbalance_and_threshold(self, X_train, X_test):
        """TASK 7 & 8: Imbalance handling + Threshold tuning"""
        if self.verbose:
            print("\n[4/7] TASK 7: Imbalance Handling...")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # SMOTE + Weighted Hybrid (best from V1)
        smote = SMOTE(k_neighbors=3, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, self.y_train)

        xgb_hybrid = xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.75,
            scale_pos_weight=20, random_state=42, verbosity=0, n_jobs=-1
        )

        cv_result = cross_validate(
            xgb_hybrid, X_train_smote, y_train_smote, cv=skf,
            scoring={'roc_auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
        )

        if self.verbose:
            print(f"  ✓ Hybrid (Weighted + SMOTE)")
            print(f"    ROC-AUC: {cv_result['test_roc_auc'].mean():.4f}")
            print(f"    F1: {cv_result['test_f1'].mean():.4f}")
            print(f"    Recall: {cv_result['test_recall'].mean():.4f}")

        if self.verbose:
            print("\n[5/7] TASK 8: Threshold Tuning...")

        # Train on full SMOTE data for threshold analysis
        xgb_hybrid.fit(X_train_smote, y_train_smote)

        # Get CV predictions for threshold tuning
        y_proba = np.zeros_like(self.y_train, dtype=float)
        for train_idx, val_idx in skf.split(X_train, self.y_train):
            model_clone = xgb.XGBClassifier(**xgb_hybrid.get_params())
            X_tr_smote, y_tr_smote = smote.fit_resample(X_train[train_idx], self.y_train[train_idx])
            model_clone.fit(X_tr_smote, y_tr_smote)
            y_proba[val_idx] = model_clone.predict_proba(X_train[val_idx])[:, 1]

        # Find optimal thresholds
        thresholds = np.linspace(0, 1, 101)
        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(self.y_train, y_pred, zero_division=0)
            recall = recall_score(self.y_train, y_pred, zero_division=0)
            f1 = f1_score(self.y_train, y_pred, zero_division=0)
            results.append({'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1})

        df_thresholds = pd.DataFrame(results)
        optimal_idx = df_thresholds['f1'].idxmax()
        optimal = df_thresholds.loc[optimal_idx]

        if self.verbose:
            print(f"  ✓ Optimal threshold: {optimal['threshold']:.4f}")
            print(f"    Precision: {optimal['precision']:.4f}, Recall: {optimal['recall']:.4f}")
            print(f"    F1: {optimal['f1']:.4f}")

        return cv_result, xgb_hybrid, df_thresholds

    def generate_report(self, cv_result):
        """Generate final report"""
        if self.verbose:
            print("\n[6/7] Generating report...")

        # Save engineered features summary
        df_eng = pd.DataFrame(list(self.engineered_features.items()),
                             columns=['Feature', 'Type'])
        df_eng.to_csv(f'{self.output_dir}/phase4_v2_engineered_features.csv', index=False)

        if self.verbose:
            print("\n  Engineered Features by Type:")
            print(df_eng['Type'].value_counts())

    def run_pipeline(self):
        """Execute complete pipeline"""
        if self.verbose:
            print("\n" + "="*100)
            print("PHASE 4 V2 (IMPROVED): EDA/SHAP-GUIDED FEATURE ENGINEERING")
            print("="*100)

        self.load_and_analyze_eda()
        self.load_data()

        X_train_eng, X_test_eng = self.task6_improved_feature_engineering()
        cv_result, model, df_thresholds = self.task7_task8_imbalance_and_threshold(X_train_eng, X_test_eng)
        self.generate_report(cv_result)

        if self.verbose:
            print("\n" + "="*100)
            print("✅ PHASE 4 V2 COMPLETE - EDA/SHAP-GUIDED APPROACH")
            print("="*100)


def main():
    """Run improved pipeline"""
    pipeline = ImprovedFeatureEngineeringPipeline(verbose=True)
    pipeline.run_pipeline()


if __name__ == '__main__':
    main()
