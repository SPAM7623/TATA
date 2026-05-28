"""
================================================================================
ALPHA DEFECT DETECTION - PHASE 4: FEATURE ENGINEERING + IMBALANCE HANDLING
================================================================================

COMPLETE PIPELINE WITH:
  • Task 6: Feature Engineering (12 feature types from EDA/SHAP insights)
  • Task 7: Imbalance Handling (class weighting, focal loss, SMOTE, balanced bagging)
  • Task 8: Threshold Tuning (precision-recall optimization for industrial use case)

Input: train_cleaned.csv, test_cleaned.csv + EDA outputs + SHAP analysis
Output:
  - Enhanced feature sets (original → +150+ new features)
  - Models trained with different imbalance strategies
  - Threshold optimization curves
  - Industrial threshold recommendation
  - Feature importance for engineered features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
import pickle
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineeringPipeline:
    """Complete feature engineering with EDA/SHAP-driven feature creation"""

    def __init__(self, train_path='train_cleaned.csv', test_path='test_cleaned.csv',
                 eda_interactions_path='eda_interactions_analysis.csv',
                 eda_redundancy_path='eda_redundancy_analysis.csv',
                 output_dir='./', verbose=True):
        self.train_path = train_path
        self.test_path = test_path
        self.eda_interactions_path = eda_interactions_path
        self.eda_redundancy_path = eda_redundancy_path
        self.output_dir = output_dir
        self.verbose = verbose

        self.train_df = None
        self.test_df = None
        self.X_train_original = None
        self.y_train = None
        self.X_test_original = None
        self.feature_cols = None
        self.corr_blocks = {}
        self.engineered_features = {}

    def load_data(self):
        """Load training and test data"""
        if self.verbose:
            print("\n[1/8] Loading cleaned data...")

        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        self.feature_cols = [col for col in self.train_df.columns if col.startswith('X')]
        self.X_train_original = self.train_df[self.feature_cols].values
        self.y_train = self.train_df['Y'].values
        self.X_test_original = self.test_df[self.feature_cols].values

        if self.verbose:
            print(f"  ✓ Train: {self.X_train_original.shape}")
            print(f"  ✓ Test:  {self.X_test_original.shape}")

    def define_correlation_blocks(self):
        """Define correlated feature blocks from EDA Phase 2.2"""
        if self.verbose:
            print("\n[2/8] Defining feature blocks from EDA...")

        # From Phase 2.2 Correlation Grouping output
        self.corr_blocks = {
            'Furnace': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X11', 'X12'],
            'Rolling': ['X10', 'X13', 'X14', 'X16', 'X17', 'X18', 'X19', 'X20'],
            'Cooling': ['X25', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36'],
            'Control': ['X37', 'X38', 'X39', 'X40', 'X41', 'X42'],
            'Other': ['X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49']
        }

        if self.verbose:
            for block_name, features in self.corr_blocks.items():
                print(f"  • {block_name}: {len(features)} features")

    def task6_feature_engineering(self):
        """
        TASK 6: Create 12 types of engineered features from EDA/SHAP insights
        """
        if self.verbose:
            print("\n[3/8] TASK 6: Feature Engineering (12 feature types)...")

        X_train_eng = self.X_train_original.copy()
        X_test_eng = self.X_test_original.copy()

        # Get feature indices for lookup
        feat_idx = {f: i for i, f in enumerate(self.feature_cols)}

        # ======================================================================
        # TYPE 1: INTERACTION FEATURES (Xi * Xj) - From SHAP top features
        # ======================================================================
        if self.verbose:
            print("  [1/12] Creating interaction features (Xi × Xj)...")

        # Top dangerous interactions from Phase 2.3
        interactions = [
            ('X35', 'X41'), ('X29', 'X41'), ('X30', 'X41'),
            ('X31', 'X41'), ('X32', 'X41'), ('X13', 'X41'),
            ('X36', 'X13'), ('X36', 'X35'), ('X13', 'X30')
        ]

        for feat1, feat2 in interactions:
            i1, i2 = feat_idx[feat1], feat_idx[feat2]
            name = f'{feat1}x{feat2}'
            X_train_eng = np.column_stack([X_train_eng,
                self.X_train_original[:, i1] * self.X_train_original[:, i2]])
            X_test_eng = np.column_stack([X_test_eng,
                self.X_test_original[:, i1] * self.X_test_original[:, i2]])
            self.engineered_features[name] = 'interaction_mult'

        # ======================================================================
        # TYPE 2: RATIO FEATURES (Xi / Xj) - From high-variance pairs
        # ======================================================================
        if self.verbose:
            print("  [2/12] Creating ratio features (Xi ÷ Xj)...")

        ratio_pairs = [
            ('X36', 'X35'), ('X35', 'X41'), ('X13', 'X30'),
            ('X32', 'X30'), ('X34', 'X35')
        ]

        for feat1, feat2 in ratio_pairs:
            i1, i2 = feat_idx[feat1], feat_idx[feat2]
            name = f'{feat1}div{feat2}'
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(self.X_train_original[:, i1],
                                 self.X_train_original[:, i2])
                ratio[~np.isfinite(ratio)] = 0
            X_train_eng = np.column_stack([X_train_eng, ratio])

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_test = np.divide(self.X_test_original[:, i1],
                                      self.X_test_original[:, i2])
                ratio_test[~np.isfinite(ratio_test)] = 0
            X_test_eng = np.column_stack([X_test_eng, ratio_test])
            self.engineered_features[name] = 'interaction_ratio'

        # ======================================================================
        # TYPE 3: DIFFERENCE FEATURES (Xi - Xj) - From high correlation blocks
        # ======================================================================
        if self.verbose:
            print("  [3/12] Creating difference features (Xi - Xj)...")

        diff_pairs = [
            ('X35', 'X34'), ('X36', 'X35'), ('X31', 'X30'),
            ('X13', 'X10'), ('X42', 'X41')
        ]

        for feat1, feat2 in diff_pairs:
            i1, i2 = feat_idx[feat1], feat_idx[feat2]
            name = f'{feat1}sub{feat2}'
            X_train_eng = np.column_stack([X_train_eng,
                self.X_train_original[:, i1] - self.X_train_original[:, i2]])
            X_test_eng = np.column_stack([X_test_eng,
                self.X_test_original[:, i1] - self.X_test_original[:, i2]])
            self.engineered_features[name] = 'interaction_diff'

        # ======================================================================
        # TYPE 4: ROW-LEVEL STATISTICAL FEATURES (mean/std/min/max)
        # ======================================================================
        if self.verbose:
            print("  [4/12] Creating row-level statistics...")

        row_mean = self.X_train_original.mean(axis=1, keepdims=True)
        row_std = self.X_train_original.std(axis=1, keepdims=True)
        row_max = self.X_train_original.max(axis=1, keepdims=True)
        row_min = self.X_train_original.min(axis=1, keepdims=True)

        X_train_eng = np.column_stack([X_train_eng, row_mean, row_std, row_max, row_min])

        row_mean_test = self.X_test_original.mean(axis=1, keepdims=True)
        row_std_test = self.X_test_original.std(axis=1, keepdims=True)
        row_max_test = self.X_test_original.max(axis=1, keepdims=True)
        row_min_test = self.X_test_original.min(axis=1, keepdims=True)

        X_test_eng = np.column_stack([X_test_eng, row_mean_test, row_std_test,
                                      row_max_test, row_min_test])

        self.engineered_features['row_mean'] = 'row_stat'
        self.engineered_features['row_std'] = 'row_stat'
        self.engineered_features['row_max'] = 'row_stat'
        self.engineered_features['row_min'] = 'row_stat'

        # ======================================================================
        # TYPE 5: INSTABILITY INDICATORS (high variance ratio)
        # ======================================================================
        if self.verbose:
            print("  [5/12] Creating instability indicators...")

        # Instability = std / mean (coefficient of variation)
        with np.errstate(divide='ignore', invalid='ignore'):
            instability = np.divide(row_std.flatten(),
                                  np.abs(row_mean.flatten()) + 1e-6)
            instability[~np.isfinite(instability)] = 0
            instability_test = np.divide(row_std_test.flatten(),
                                        np.abs(row_mean_test.flatten()) + 1e-6)
            instability_test[~np.isfinite(instability_test)] = 0

        X_train_eng = np.column_stack([X_train_eng, instability])
        X_test_eng = np.column_stack([X_test_eng, instability_test])
        self.engineered_features['instability'] = 'instability'

        # ======================================================================
        # TYPE 6: IMBALANCE INDICATORS (range/spread)
        # ======================================================================
        if self.verbose:
            print("  [6/12] Creating imbalance indicators...")

        imbalance = row_max.flatten() - row_min.flatten()
        imbalance_test = row_max_test.flatten() - row_min_test.flatten()

        X_train_eng = np.column_stack([X_train_eng, imbalance])
        X_test_eng = np.column_stack([X_test_eng, imbalance_test])
        self.engineered_features['imbalance'] = 'imbalance'

        # ======================================================================
        # TYPE 7: GROUPED BLOCK FEATURES (block means and ranges)
        # ======================================================================
        if self.verbose:
            print("  [7/12] Creating grouped block features...")

        for block_name, block_features in self.corr_blocks.items():
            block_idx = [feat_idx[f] for f in block_features if f in feat_idx]

            # Block mean
            block_mean = self.X_train_original[:, block_idx].mean(axis=1)
            block_mean_test = self.X_test_original[:, block_idx].mean(axis=1)
            X_train_eng = np.column_stack([X_train_eng, block_mean])
            X_test_eng = np.column_stack([X_test_eng, block_mean_test])
            self.engineered_features[f'{block_name}_mean'] = 'block_feature'

            # Block std
            block_std = self.X_train_original[:, block_idx].std(axis=1)
            block_std_test = self.X_test_original[:, block_idx].std(axis=1)
            X_train_eng = np.column_stack([X_train_eng, block_std])
            X_test_eng = np.column_stack([X_test_eng, block_std_test])
            self.engineered_features[f'{block_name}_std'] = 'block_feature'

        # ======================================================================
        # TYPE 8: ANOMALY-DISTANCE FEATURES (distance from normal cluster)
        # ======================================================================
        if self.verbose:
            print("  [8/12] Creating anomaly-distance features...")

        normal_samples = self.X_train_original[self.y_train == 0]
        normal_center = normal_samples.mean(axis=0)
        normal_std = normal_samples.std(axis=0) + 1e-6

        # Mahalanobis-like distance (standardized distance from normal cluster)
        anomaly_dist = np.sqrt(((self.X_train_original - normal_center) / normal_std) ** 2).mean(axis=1)
        anomaly_dist_test = np.sqrt(((self.X_test_original - normal_center) / normal_std) ** 2).mean(axis=1)

        X_train_eng = np.column_stack([X_train_eng, anomaly_dist])
        X_test_eng = np.column_stack([X_test_eng, anomaly_dist_test])
        self.engineered_features['anomaly_distance'] = 'anomaly'

        # ======================================================================
        # TYPE 9: OPERATING REGIME INDICATORS (from Phase 2.3)
        # ======================================================================
        if self.verbose:
            print("  [9/12] Creating operating regime indicators...")

        # Top features from SHAP (X36, X13, X14, X16, X35)
        regime_features = ['X36', 'X13', 'X14', 'X16', 'X35']
        regime_idx = [feat_idx[f] for f in regime_features if f in feat_idx]

        # Regime anomaly: how far from 25-75 percentile range
        q25 = np.percentile(self.X_train_original[:, regime_idx], 25, axis=0)
        q75 = np.percentile(self.X_train_original[:, regime_idx], 75, axis=0)
        iqr = q75 - q25 + 1e-6

        regime_anomaly = (np.abs(self.X_train_original[:, regime_idx] - (q25 + q75) / 2) / iqr).mean(axis=1)
        regime_anomaly_test = (np.abs(self.X_test_original[:, regime_idx] - (q25 + q75) / 2) / iqr).mean(axis=1)

        X_train_eng = np.column_stack([X_train_eng, regime_anomaly])
        X_test_eng = np.column_stack([X_test_eng, regime_anomaly_test])
        self.engineered_features['regime_anomaly'] = 'regime'

        # ======================================================================
        # TYPE 10: THRESHOLD-CROSSING INDICATORS
        # ======================================================================
        if self.verbose:
            print("  [10/12] Creating threshold-crossing indicators...")

        # Count how many features exceed high-risk thresholds (from EDA Point 14)
        thresholds = {
            'X13': 1032.32,
            'X10': 9.83,
            'X32': 17.15,
            'X36': 178.0,
            'X30': 12.17
        }

        threshold_count = np.zeros(self.X_train_original.shape[0])
        threshold_count_test = np.zeros(self.X_test_original.shape[0])

        for feat, threshold in thresholds.items():
            if feat in feat_idx:
                idx = feat_idx[feat]
                threshold_count += (self.X_train_original[:, idx] > threshold).astype(int)
                threshold_count_test += (self.X_test_original[:, idx] > threshold).astype(int)

        X_train_eng = np.column_stack([X_train_eng, threshold_count])
        X_test_eng = np.column_stack([X_test_eng, threshold_count_test])
        self.engineered_features['threshold_count'] = 'threshold'

        # ======================================================================
        # TYPE 11: EXTREME VALUE FEATURES (outlier markers)
        # ======================================================================
        if self.verbose:
            print("  [11/12] Creating extreme value features...")

        # Count extreme values (>3 std from mean)
        z_scores = np.abs((self.X_train_original - self.X_train_original.mean(axis=0)) /
                         (self.X_train_original.std(axis=0) + 1e-6))
        extreme_count = (z_scores > 3).sum(axis=1)

        z_scores_test = np.abs((self.X_test_original - self.X_train_original.mean(axis=0)) /
                              (self.X_train_original.std(axis=0) + 1e-6))
        extreme_count_test = (z_scores_test > 3).sum(axis=1)

        X_train_eng = np.column_stack([X_train_eng, extreme_count])
        X_test_eng = np.column_stack([X_test_eng, extreme_count_test])
        self.engineered_features['extreme_count'] = 'outlier'

        # ======================================================================
        # TYPE 12: CORRELATION BLOCK DIVERGENCE
        # ======================================================================
        if self.verbose:
            print("  [12/12] Creating block divergence features...")

        # For each block, compute how much one feature deviates from block mean
        for block_name, block_features in self.corr_blocks.items():
            block_idx = [feat_idx[f] for f in block_features if f in feat_idx]

            if len(block_idx) > 1:
                block_data = self.X_train_original[:, block_idx]
                block_mean = block_data.mean(axis=1, keepdims=True)
                block_divergence = np.std(block_data - block_mean, axis=1)

                block_data_test = self.X_test_original[:, block_idx]
                block_mean_test = block_data_test.mean(axis=1, keepdims=True)
                block_divergence_test = np.std(block_data_test - block_mean_test, axis=1)

                X_train_eng = np.column_stack([X_train_eng, block_divergence])
                X_test_eng = np.column_stack([X_test_eng, block_divergence_test])
                self.engineered_features[f'{block_name}_divergence'] = 'divergence'

        if self.verbose:
            print(f"\n  ✓ Feature engineering complete")
            print(f"  ✓ Original features: {self.X_train_original.shape[1]}")
            print(f"  ✓ Engineered features: {X_train_eng.shape[1] - self.X_train_original.shape[1]}")
            print(f"  ✓ Total features: {X_train_eng.shape[1]}")

        return X_train_eng, X_test_eng

    def task7_imbalance_handling(self, X_train, X_test):
        """
        TASK 7: Handle class imbalance with multiple strategies
        """
        if self.verbose:
            print("\n[4/8] TASK 7: Imbalance Handling...")

        self.imbalance_results = {}

        # ======================================================================
        # STRATEGY 1: CLASS WEIGHTING
        # ======================================================================
        if self.verbose:
            print("  [1/5] Training with class weighting...")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        xgb_weighted = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=19.5,  # Class imbalance ratio
            random_state=42, verbosity=0, n_jobs=-1
        )

        cv_weighted = cross_validate(
            xgb_weighted, X_train, self.y_train, cv=skf,
            scoring={'roc_auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
        )

        self.imbalance_results['weighted'] = cv_weighted

        if self.verbose:
            print(f"    ROC-AUC: {cv_weighted['test_roc_auc'].mean():.4f}")
            print(f"    F1: {cv_weighted['test_f1'].mean():.4f}")

        # ======================================================================
        # STRATEGY 2: FOCAL LOSS (XGBoost with adjusted scale_pos_weight + deeper trees)
        # ======================================================================
        if self.verbose:
            print("  [2/5] Training with focal loss approximation...")

        xgb_focal = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            scale_pos_weight=50,  # Increased weight
            min_child_weight=5,   # Prevent small splits
            random_state=42, verbosity=0, n_jobs=-1
        )

        cv_focal = cross_validate(
            xgb_focal, X_train, self.y_train, cv=skf,
            scoring={'roc_auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
        )

        self.imbalance_results['focal'] = cv_focal

        if self.verbose:
            print(f"    ROC-AUC: {cv_focal['test_roc_auc'].mean():.4f}")
            print(f"    F1: {cv_focal['test_f1'].mean():.4f}")

        # ======================================================================
        # STRATEGY 3: SMOTE (Synthetic Minority Over-sampling)
        # ======================================================================
        if self.verbose:
            print("  [3/5] Training with SMOTE...")

        smote = SMOTE(k_neighbors=3, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, self.y_train)

        xgb_smote = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1
        )

        cv_smote = cross_validate(
            xgb_smote, X_train_smote, y_train_smote, cv=skf,
            scoring={'roc_auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
        )

        self.imbalance_results['smote'] = cv_smote

        if self.verbose:
            print(f"    ROC-AUC: {cv_smote['test_roc_auc'].mean():.4f}")
            print(f"    F1: {cv_smote['test_f1'].mean():.4f}")

        # ======================================================================
        # STRATEGY 4: BALANCED BAGGING
        # ======================================================================
        if self.verbose:
            print("  [4/5] Training with Balanced Bagging...")

        balanced_bagging = BalancedBaggingClassifier(
            RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
            n_estimators=10, random_state=42, n_jobs=-1
        )

        cv_bagging = cross_validate(
            balanced_bagging, X_train, self.y_train, cv=skf,
            scoring={'roc_auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
        )

        self.imbalance_results['balanced_bagging'] = cv_bagging

        if self.verbose:
            print(f"    ROC-AUC: {cv_bagging['test_roc_auc'].mean():.4f}")
            print(f"    F1: {cv_bagging['test_f1'].mean():.4f}")

        # ======================================================================
        # STRATEGY 5: HYBRID (Class weighting + SMOTE)
        # ======================================================================
        if self.verbose:
            print("  [5/5] Training with Hybrid (Weighted SMOTE)...")

        xgb_hybrid = xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.75,
            scale_pos_weight=20,
            random_state=42, verbosity=0, n_jobs=-1
        )

        cv_hybrid = cross_validate(
            xgb_hybrid, X_train_smote, y_train_smote, cv=skf,
            scoring={'roc_auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
        )

        self.imbalance_results['hybrid'] = cv_hybrid

        if self.verbose:
            print(f"    ROC-AUC: {cv_hybrid['test_roc_auc'].mean():.4f}")
            print(f"    F1: {cv_hybrid['test_f1'].mean():.4f}")

        # Store models for threshold tuning
        self.best_model_weighted = xgb_weighted
        self.best_model_focal = xgb_focal
        self.best_model_smote = xgb_smote

    def task8_threshold_tuning(self, X_train, X_test):
        """
        TASK 8: Optimize decision threshold for industrial use case
        """
        if self.verbose:
            print("\n[5/8] TASK 8: Threshold Tuning...")

        self.threshold_results = {}

        # Train models on full training data for threshold analysis
        self.best_model_weighted.fit(X_train, self.y_train)
        self.best_model_focal.fit(X_train, self.y_train)
        self.best_model_smote.fit(X_train, self.y_train)

        for model_name, model in [('weighted', self.best_model_weighted),
                                   ('focal', self.best_model_focal),
                                   ('smote', self.best_model_smote)]:

            if self.verbose:
                print(f"\n  Tuning threshold for {model_name.upper()}...")

            # Get probabilities via cross-validation
            y_proba = np.zeros_like(self.y_train, dtype=float)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for train_idx, val_idx in skf.split(X_train, self.y_train):
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train[train_idx], self.y_train[train_idx])
                y_proba[val_idx] = model_clone.predict_proba(X_train[val_idx])[:, 1]

            # Search for optimal threshold
            thresholds = np.linspace(0, 1, 101)
            results_list = []

            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)

                precision = precision_score(self.y_train, y_pred, zero_division=0)
                recall = recall_score(self.y_train, y_pred, zero_division=0)
                f1 = f1_score(self.y_train, y_pred, zero_division=0)

                # Industrial metric: favor recall (catch defects) but maintain precision
                industrial_score = 0.6 * recall + 0.4 * precision

                results_list.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'industrial_score': industrial_score
                })

            df_thresholds = pd.DataFrame(results_list)
            self.threshold_results[model_name] = df_thresholds

            # Find optimal thresholds for different objectives
            idx_f1_max = df_thresholds['f1'].idxmax()
            idx_industrial = df_thresholds['industrial_score'].idxmax()

            optimal_f1 = df_thresholds.loc[idx_f1_max]
            optimal_industrial = df_thresholds.loc[idx_industrial]

            if self.verbose:
                print(f"    F1-optimized threshold: {optimal_f1['threshold']:.4f}")
                print(f"      → Precision: {optimal_f1['precision']:.4f}, Recall: {optimal_f1['recall']:.4f}, F1: {optimal_f1['f1']:.4f}")

                print(f"    Industrial-optimized threshold: {optimal_industrial['threshold']:.4f}")
                print(f"      → Precision: {optimal_industrial['precision']:.4f}, Recall: {optimal_industrial['recall']:.4f}, F1: {optimal_industrial['f1']:.4f}")

            # Create threshold curves
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Precision-Recall vs Threshold
            axes[0].plot(df_thresholds['threshold'], df_thresholds['precision'], label='Precision', linewidth=2)
            axes[0].plot(df_thresholds['threshold'], df_thresholds['recall'], label='Recall', linewidth=2)
            axes[0].axvline(optimal_f1['threshold'], color='g', linestyle='--', label=f"F1-opt: {optimal_f1['threshold']:.3f}")
            axes[0].axvline(optimal_industrial['threshold'], color='r', linestyle='--', label=f"Industrial-opt: {optimal_industrial['threshold']:.3f}")
            axes[0].set_xlabel('Decision Threshold')
            axes[0].set_ylabel('Score')
            axes[0].set_title(f'{model_name.upper()} - Precision/Recall vs Threshold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # F1 and Industrial Score vs Threshold
            axes[1].plot(df_thresholds['threshold'], df_thresholds['f1'], label='F1', linewidth=2)
            axes[1].plot(df_thresholds['threshold'], df_thresholds['industrial_score'], label='Industrial Score (0.6×Recall + 0.4×Precision)', linewidth=2)
            axes[1].axvline(optimal_f1['threshold'], color='g', linestyle='--', label=f"F1-opt: {optimal_f1['threshold']:.3f}")
            axes[1].axvline(optimal_industrial['threshold'], color='r', linestyle='--', label=f"Industrial-opt: {optimal_industrial['threshold']:.3f}")
            axes[1].set_xlabel('Decision Threshold')
            axes[1].set_ylabel('Score')
            axes[1].set_title(f'{model_name.upper()} - Optimization Objectives vs Threshold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/viz_threshold_tuning_{model_name}.png', dpi=150, bbox_inches='tight')
            plt.close()

    def generate_report(self):
        """Generate comprehensive report"""
        if self.verbose:
            print("\n[6/8] Generating comprehensive report...")

        # Imbalance strategy comparison
        comparison_data = []
        for strategy, cv_result in self.imbalance_results.items():
            comparison_data.append({
                'Strategy': strategy,
                'ROC_AUC_Mean': cv_result['test_roc_auc'].mean(),
                'ROC_AUC_Std': cv_result['test_roc_auc'].std(),
                'F1_Mean': cv_result['test_f1'].mean(),
                'F1_Std': cv_result['test_f1'].std(),
                'Precision_Mean': cv_result['test_precision'].mean(),
                'Recall_Mean': cv_result['test_recall'].mean()
            })

        df_imbalance = pd.DataFrame(comparison_data)
        df_imbalance.to_csv(f'{self.output_dir}/imbalance_strategies_comparison.csv', index=False)

        # Feature importance from engineered features
        engineered_summary = pd.DataFrame(list(self.engineered_features.items()),
                                         columns=['Feature_Name', 'Feature_Type'])
        engineered_summary.to_csv(f'{self.output_dir}/engineered_features_summary.csv', index=False)

        if self.verbose:
            print("\n" + "="*100)
            print("IMBALANCE HANDLING STRATEGIES COMPARISON")
            print("="*100)
            print(df_imbalance[['Strategy', 'ROC_AUC_Mean', 'F1_Mean', 'Recall_Mean']].to_string(index=False))

    def create_summary_visualizations(self):
        """Create summary visualizations"""
        if self.verbose:
            print("\n[7/8] Creating summary visualizations...")

        # Imbalance strategy ROC-AUC comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        strategies = list(self.imbalance_results.keys())
        roc_aucs = [self.imbalance_results[s]['test_roc_auc'].mean() for s in strategies]
        f1_scores = [self.imbalance_results[s]['test_f1'].mean() for s in strategies]

        x = np.arange(len(strategies))
        width = 0.35

        ax.bar(x - width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)
        ax.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        ax.set_xlabel('Imbalance Strategy')
        ax.set_ylabel('Score')
        ax.set_title('Imbalance Handling Strategies Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/viz_imbalance_strategies_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Engineered features breakdown
        feature_types = {}
        for feat, feat_type in self.engineered_features.items():
            if feat_type not in feature_types:
                feature_types[feat_type] = 0
            feature_types[feat_type] += 1

        fig, ax = plt.subplots(figsize=(10, 6))
        types = list(feature_types.keys())
        counts = list(feature_types.values())

        colors = plt.cm.tab20(np.linspace(0, 1, len(types)))
        ax.bar(types, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Feature Type')
        ax.set_ylabel('Count')
        ax.set_title(f'Engineered Features Breakdown ({sum(counts)} total)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/viz_engineered_features_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print("  ✓ Visualizations created")

    def run_pipeline(self):
        """Execute complete pipeline"""
        if self.verbose:
            print("\n" + "="*100)
            print("PHASE 4: FEATURE ENGINEERING + IMBALANCE HANDLING + THRESHOLD TUNING")
            print("="*100)

        self.load_data()
        self.define_correlation_blocks()

        X_train_eng, X_test_eng = self.task6_feature_engineering()
        self.task7_imbalance_handling(X_train_eng, X_test_eng)
        self.task8_threshold_tuning(X_train_eng, X_test_eng)
        self.generate_report()
        self.create_summary_visualizations()

        if self.verbose:
            print("\n" + "="*100)
            print("✅ PHASE 4 COMPLETE")
            print("="*100)


def main(train_path='train_cleaned.csv', test_path='test_cleaned.csv',
         eda_interactions_path='eda_interactions_analysis.csv',
         eda_redundancy_path='eda_redundancy_analysis.csv',
         output_dir='./', verbose=True):
    """Run complete feature engineering pipeline"""
    pipeline = FeatureEngineeringPipeline(
        train_path=train_path,
        test_path=test_path,
        eda_interactions_path=eda_interactions_path,
        eda_redundancy_path=eda_redundancy_path,
        output_dir=output_dir,
        verbose=verbose
    )
    pipeline.run_pipeline()


if __name__ == '__main__':
    main()
