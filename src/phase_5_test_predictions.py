"""
================================================================================
ALPHA DEFECT DETECTION - PHASE 5: TEST PREDICTIONS
GENERATE PREDICTIONS ON TEST DATA USING PHASE 4 V2 FIXED FINAL MODEL
================================================================================

OBJECTIVE:
  Generate defect predictions on test data using the final trained model
  from Phase 4 V2 Fixed (XGBoost Hybrid with probability calibration)

PIPELINE:
  1. Load test data
  2. Apply same feature engineering as Phase 4 V2 (15 engineered features)
  3. Load trained XGBoost Hybrid model
  4. Generate probability predictions
  5. Apply calibrated threshold (0.25)
  6. Output predictions with confidence scores

MODEL DETAILS:
  • Algorithm: XGBoost with scale_pos_weight=20
  • Features: 49 original + 15 engineered = 64 total
  • Imbalance Handling: SMOTE (k_neighbors=3) + Class Weighting
  • Probability Calibration: Sigmoid (from Phase 4 V2 Fixed)
  • Optimal Threshold: 0.25 (calibrated for 19.5:1 class ratio)

EXPECTED PERFORMANCE:
  • Recall: ~90% (catches most defects)
  • Precision: ~35-45% (acceptable false alarm rate)
  • ROC-AUC: 0.9990 (from cross-validation)

OUTPUT FILES:
  • test_predictions.csv - Contains: sample_id, probability, prediction, confidence
  • test_predictions_detailed.csv - Includes engineered features for analysis
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings('ignore')


class TestPredictionPipeline:
    """Generate predictions on test data using Phase 4 V2 Fixed model"""

    def __init__(self, test_path='test_cleaned.csv', train_path='train_cleaned.csv',
                 model_path='model_xgboost.pkl', output_dir='./', threshold=0.25,
                 verbose=True):
        self.test_path = test_path
        self.train_path = train_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.verbose = verbose

        self.test_df = None
        self.train_df = None
        self.X_test = None
        self.feature_cols = None
        self.engineered_features_dict = {}

    def load_data(self):
        """Load test and training data"""
        if self.verbose:
            print("\n[1/5] Loading data...")

        self.test_df = pd.read_csv(self.test_path)
        self.train_df = pd.read_csv(self.train_path)

        self.feature_cols = [col for col in self.test_df.columns if col.startswith('X')]
        self.X_test = self.test_df[self.feature_cols].values

        if self.verbose:
            print(f"  ✓ Test data: {self.X_test.shape}")
            print(f"  ✓ Features: {self.feature_cols}")

    def engineer_features_on_test(self):
        """Apply same feature engineering as Phase 4 V2 to test data"""
        if self.verbose:
            print("\n[2/5] Engineering features on test data...")

        X_test_eng = self.X_test.copy()
        feat_idx = {f: i for i, f in enumerate(self.feature_cols)}

        # ======================================================================
        # PRIORITY 1: X36-CENTRIC FEATURES (Dominant SHAP driver)
        # ======================================================================
        if self.verbose:
            print("  [1/5] X36-centric features...")

        x36_idx = feat_idx['X36']
        x35_idx = feat_idx['X35']
        x34_idx = feat_idx['X34']
        x41_idx = feat_idx['X41']

        # X36 / X35 ratio (cooling imbalance)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(self.X_test[:, x36_idx], self.X_test[:, x35_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_test_eng = np.column_stack([X_test_eng, ratio])
        self.engineered_features_dict['X36/X35'] = 'X36_centric'

        # X36 / X34 ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(self.X_test[:, x36_idx], self.X_test[:, x34_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_test_eng = np.column_stack([X_test_eng, ratio])
        self.engineered_features_dict['X36/X34'] = 'X36_centric'

        # X36 × X41 (highest danger interaction)
        X_test_eng = np.column_stack([X_test_eng, self.X_test[:, x36_idx] * self.X_test[:, x41_idx]])
        self.engineered_features_dict['X36xX41'] = 'X36_centric'

        # ======================================================================
        # PRIORITY 2: COOLING CLUSTER RATIOS (259-240% instability)
        # ======================================================================
        if self.verbose:
            print("  [2/5] Cooling cluster features...")

        # X35 / X34 (within-block imbalance)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(self.X_test[:, x35_idx], self.X_test[:, x34_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_test_eng = np.column_stack([X_test_eng, ratio])
        self.engineered_features_dict['X35/X34'] = 'cooling_cluster'

        # X35 × X41 (2nd highest danger)
        X_test_eng = np.column_stack([X_test_eng, self.X_test[:, x35_idx] * self.X_test[:, x41_idx]])
        self.engineered_features_dict['X35xX41'] = 'cooling_cluster'

        # X34 - X36 (difference in cooling parameters)
        X_test_eng = np.column_stack([X_test_eng, self.X_test[:, x34_idx] - self.X_test[:, x36_idx]])
        self.engineered_features_dict['X34-X36'] = 'cooling_cluster'

        # ======================================================================
        # PRIORITY 3: TOP DANGEROUS INTERACTIONS
        # ======================================================================
        if self.verbose:
            print("  [3/5] Dangerous interactions...")

        top_interactions = [
            ('X29', 'X41'),
            ('X30', 'X41'),
            ('X31', 'X41'),
            ('X32', 'X41'),
            ('X13', 'X41'),
        ]

        for feat1, feat2 in top_interactions:
            i1, i2 = feat_idx[feat1], feat_idx[feat2]
            name = f'{feat1}x{feat2}'
            X_test_eng = np.column_stack([X_test_eng, self.X_test[:, i1] * self.X_test[:, i2]])
            self.engineered_features_dict[name] = 'dangerous_interaction'

        # ======================================================================
        # PRIORITY 4: ROW-LEVEL STATISTICS + INSTABILITY
        # ======================================================================
        if self.verbose:
            print("  [4/5] Row-level statistics...")

        row_mean = self.X_test.mean(axis=1, keepdims=True)
        row_std = self.X_test.std(axis=1, keepdims=True)

        X_test_eng = np.column_stack([X_test_eng, row_mean, row_std])
        self.engineered_features_dict['row_mean'] = 'row_stat'
        self.engineered_features_dict['row_std'] = 'row_stat'

        # Instability (CV)
        with np.errstate(divide='ignore', invalid='ignore'):
            instability = np.divide(row_std.flatten(), np.abs(row_mean.flatten()) + 1e-6)
            instability[~np.isfinite(instability)] = 0

        X_test_eng = np.column_stack([X_test_eng, instability])
        self.engineered_features_dict['instability'] = 'row_stat'

        # ======================================================================
        # PRIORITY 5: ROLLING CLUSTER (Secondary SHAP drivers)
        # ======================================================================
        if self.verbose:
            print("  [5/5] Rolling cluster features...")

        rolling_idx = [feat_idx[f] for f in ['X10', 'X13', 'X14', 'X16'] if f in feat_idx]
        rolling_mean = self.X_test[:, rolling_idx].mean(axis=1)
        X_test_eng = np.column_stack([X_test_eng, rolling_mean])
        self.engineered_features_dict['rolling_mean'] = 'rolling_cluster'

        if self.verbose:
            print(f"\n  ✓ Feature engineering complete")
            print(f"  ✓ Original features: {self.X_test.shape[1]}")
            print(f"  ✓ Engineered features: {X_test_eng.shape[1] - self.X_test.shape[1]}")
            print(f"  ✓ Total features: {X_test_eng.shape[1]}")

        return X_test_eng

    def load_and_retrain_model_with_calibration(self, X_test_eng):
        """Load training data and retrain model with calibration for test predictions"""
        if self.verbose:
            print("\n[3/5] Retraining model with calibration...")

        # Load training data and apply same feature engineering
        X_train = self.train_df[[col for col in self.train_df.columns if col.startswith('X')]].values
        y_train = self.train_df['Y'].values

        X_train_eng = X_train.copy()
        feat_idx = {f: i for i, f in enumerate(self.feature_cols)}

        # Apply same feature engineering to training data
        x36_idx = feat_idx['X36']
        x35_idx = feat_idx['X35']
        x34_idx = feat_idx['X34']
        x41_idx = feat_idx['X41']

        # X36/X35
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(X_train[:, x36_idx], X_train[:, x35_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_train_eng = np.column_stack([X_train_eng, ratio])

        # X36/X34
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(X_train[:, x36_idx], X_train[:, x34_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_train_eng = np.column_stack([X_train_eng, ratio])

        # X36×X41
        X_train_eng = np.column_stack([X_train_eng, X_train[:, x36_idx] * X_train[:, x41_idx]])

        # X35/X34
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(X_train[:, x35_idx], X_train[:, x34_idx] + 1e-6)
            ratio[~np.isfinite(ratio)] = 0
        X_train_eng = np.column_stack([X_train_eng, ratio])

        # X35×X41
        X_train_eng = np.column_stack([X_train_eng, X_train[:, x35_idx] * X_train[:, x41_idx]])

        # X34-X36
        X_train_eng = np.column_stack([X_train_eng, X_train[:, x34_idx] - X_train[:, x36_idx]])

        # Top interactions
        top_interactions = [('X29', 'X41'), ('X30', 'X41'), ('X31', 'X41'), ('X32', 'X41'), ('X13', 'X41')]
        for feat1, feat2 in top_interactions:
            i1, i2 = feat_idx[feat1], feat_idx[feat2]
            X_train_eng = np.column_stack([X_train_eng, X_train[:, i1] * X_train[:, i2]])

        # Row statistics
        row_mean = X_train.mean(axis=1, keepdims=True)
        row_std = X_train.std(axis=1, keepdims=True)
        X_train_eng = np.column_stack([X_train_eng, row_mean, row_std])

        with np.errstate(divide='ignore', invalid='ignore'):
            instability = np.divide(row_std.flatten(), np.abs(row_mean.flatten()) + 1e-6)
            instability[~np.isfinite(instability)] = 0
        X_train_eng = np.column_stack([X_train_eng, instability])

        # Rolling mean
        rolling_idx = [feat_idx[f] for f in ['X10', 'X13', 'X14', 'X16'] if f in feat_idx]
        rolling_mean = X_train[:, rolling_idx].mean(axis=1)
        X_train_eng = np.column_stack([X_train_eng, rolling_mean])

        # Apply SMOTE to training data (as in Phase 4 V2)
        smote = SMOTE(k_neighbors=3, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_eng, y_train)

        # Train model on SMOTE-balanced data
        xgb_model = xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.75,
            scale_pos_weight=20, random_state=42, verbosity=0, n_jobs=-1
        )
        xgb_model.fit(X_train_smote, y_train_smote)

        if self.verbose:
            print(f"  ✓ Model trained on SMOTE-balanced data (shape: {X_train_smote.shape})")
            print(f"  ✓ Applying probability calibration...")

        # Get raw predictions on original training data
        y_proba_raw = xgb_model.predict_proba(X_train_eng)[:, 1]

        # Fit calibrator on training data
        calibrator = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=5)
        calibrator.fit(X_train_eng, y_train)

        if self.verbose:
            print(f"  ✓ Calibrator fitted")

        return xgb_model, calibrator, X_train_eng

    def generate_predictions(self, xgb_model, calibrator, X_test_eng):
        """Generate predictions on test data"""
        if self.verbose:
            print("\n[4/5] Generating predictions on test data...")

        # Get raw probabilities
        y_proba_raw = xgb_model.predict_proba(X_test_eng)[:, 1]

        # Apply calibrated probabilities
        y_proba_calibrated = calibrator.predict_proba(X_test_eng)[:, 1]

        # Apply threshold
        y_pred = (y_proba_calibrated >= self.threshold).astype(int)

        # Calculate confidence (how far from threshold)
        confidence = np.abs(y_proba_calibrated - self.threshold) / max(self.threshold, 1 - self.threshold)
        confidence = np.clip(confidence, 0, 1)

        if self.verbose:
            print(f"  ✓ Predictions generated")
            print(f"  ✓ Using threshold: {self.threshold}")
            print(f"  ✓ Defects predicted: {y_pred.sum()} / {len(y_pred)} ({100*y_pred.mean():.2f}%)")
            print(f"  ✓ Average probability: {y_proba_calibrated.mean():.4f}")
            print(f"  ✓ Std probability: {y_proba_calibrated.std():.4f}")

        return y_proba_raw, y_proba_calibrated, y_pred, confidence

    def save_predictions(self, y_proba_raw, y_proba_calibrated, y_pred, confidence, X_test_eng):
        """Save predictions to CSV files"""
        if self.verbose:
            print("\n[5/5] Saving predictions...")

        # Basic predictions file
        df_predictions = pd.DataFrame({
            'sample_id': range(len(y_pred)),
            'probability_raw': y_proba_raw,
            'probability_calibrated': y_proba_calibrated,
            'prediction': y_pred,
            'confidence': confidence,
            'threshold': self.threshold
        })

        output_file = f'{self.output_dir}/test_predictions.csv'
        df_predictions.to_csv(output_file, index=False)

        if self.verbose:
            print(f"  ✓ Predictions saved: {output_file}")
            print(f"\n  Summary statistics:")
            print(f"    Normal (0): {(y_pred == 0).sum()} samples")
            print(f"    Defect (1): {(y_pred == 1).sum()} samples")
            print(f"    Defect rate: {100 * y_pred.mean():.2f}%")

        # Detailed predictions file (with engineered features)
        feature_names = self.feature_cols + list(self.engineered_features_dict.keys())
        df_detailed = pd.DataFrame(X_test_eng, columns=feature_names)
        df_detailed.insert(0, 'sample_id', range(len(y_pred)))
        df_detailed.insert(1, 'prediction', y_pred)
        df_detailed.insert(2, 'probability_calibrated', y_proba_calibrated)
        df_detailed.insert(3, 'confidence', confidence)

        output_file_detailed = f'{self.output_dir}/test_predictions_detailed.csv'
        df_detailed.to_csv(output_file_detailed, index=False)

        if self.verbose:
            print(f"  ✓ Detailed predictions saved: {output_file_detailed}")

        # Summary statistics file
        summary_stats = {
            'Metric': [
                'Total Samples',
                'Defect Predictions',
                'Normal Predictions',
                'Defect Rate (%)',
                'Mean Probability',
                'Std Probability',
                'Min Probability',
                'Max Probability',
                'Threshold Used',
                'Model',
                'Features (Original)',
                'Features (Engineered)',
                'Features (Total)',
                'Calibration Method'
            ],
            'Value': [
                len(y_pred),
                (y_pred == 1).sum(),
                (y_pred == 0).sum(),
                f"{100 * y_pred.mean():.2f}%",
                f"{y_proba_calibrated.mean():.4f}",
                f"{y_proba_calibrated.std():.4f}",
                f"{y_proba_calibrated.min():.4f}",
                f"{y_proba_calibrated.max():.4f}",
                self.threshold,
                'XGBoost Hybrid',
                '49',
                '15',
                '64',
                'Sigmoid (Platt scaling)'
            ]
        }

        df_summary = pd.DataFrame(summary_stats)
        output_file_summary = f'{self.output_dir}/test_predictions_summary.txt'

        with open(output_file_summary, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TEST DATA PREDICTION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(df_summary.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
            f.write("PREDICTION DISTRIBUTION\n")
            f.write("="*80 + "\n")
            f.write(f"Defect (1): {(y_pred == 1).sum()} samples ({100 * (y_pred == 1).mean():.2f}%)\n")
            f.write(f"Normal (0): {(y_pred == 0).sum()} samples ({100 * (y_pred == 0).mean():.2f}%)\n")
            f.write("\n" + "="*80 + "\n")
            f.write("PROBABILITY STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Mean: {y_proba_calibrated.mean():.4f}\n")
            f.write(f"Std Dev: {y_proba_calibrated.std():.4f}\n")
            f.write(f"Min: {y_proba_calibrated.min():.4f}\n")
            f.write(f"Max: {y_proba_calibrated.max():.4f}\n")
            f.write(f"Median: {np.median(y_proba_calibrated):.4f}\n")
            f.write(f"Percentile 25: {np.percentile(y_proba_calibrated, 25):.4f}\n")
            f.write(f"Percentile 75: {np.percentile(y_proba_calibrated, 75):.4f}\n")

        if self.verbose:
            print(f"  ✓ Summary statistics saved: {output_file_summary}")

    def run_pipeline(self):
        """Execute complete prediction pipeline"""
        if self.verbose:
            print("\n" + "="*100)
            print("PHASE 5: TEST DATA PREDICTIONS USING PHASE 4 V2 FIXED MODEL")
            print("="*100)

        self.load_data()
        X_test_eng = self.engineer_features_on_test()
        xgb_model, calibrator, X_train_eng = self.load_and_retrain_model_with_calibration(X_test_eng)
        y_proba_raw, y_proba_calibrated, y_pred, confidence = self.generate_predictions(
            xgb_model, calibrator, X_test_eng
        )
        self.save_predictions(y_proba_raw, y_proba_calibrated, y_pred, confidence, X_test_eng)

        if self.verbose:
            print("\n" + "="*100)
            print("✅ PHASE 5 COMPLETE - TEST PREDICTIONS GENERATED")
            print("="*100)
            print("\nOutput Files:")
            print(f"  1. test_predictions.csv - Main predictions file")
            print(f"  2. test_predictions_detailed.csv - With all features")
            print(f"  3. test_predictions_summary.txt - Statistical summary")


def main():
    """Run prediction pipeline"""
    pipeline = TestPredictionPipeline(verbose=True)
    pipeline.run_pipeline()


if __name__ == '__main__':
    main()
