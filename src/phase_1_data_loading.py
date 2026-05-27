"""
PHASE 1: Data Loading & Cleaning
Alpha Defect Detection in Hot Rolling Mills

Task: Load datasets, handle missing values, validate data quality
Time: ~30 seconds in Colab

Output: train_cleaned.csv, test_cleaned.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

def load_and_clean_data(train_path='train.csv', test_path='test.csv', verbose=True):
    """
    Load and clean Alpha defect datasets

    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to test CSV file
    verbose : bool
        Print progress information

    Returns:
    --------
    train_df : pd.DataFrame
        Cleaned training data
    test_df : pd.DataFrame
        Cleaned test data
    feature_cols : list
        Feature column names (X1-X49)
    """

    if verbose:
        print("=" * 80)
        print("PHASE 1: DATA LOADING & CLEANING")
        print("=" * 80)

    # ========================================================================
    # STEP 1: LOAD DATASETS
    # ========================================================================
    if verbose:
        print("\n[1/5] Loading datasets...")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if verbose:
            print(f"  ✓ Train set: {train_df.shape[0]} rows × {train_df.shape[1]} cols")
            print(f"  ✓ Test set:  {test_df.shape[0]} rows × {test_df.shape[1]} cols")
    except FileNotFoundError as e:
        print(f"  ✗ Error: File not found - {e}")
        raise

    # ========================================================================
    # STEP 2: IDENTIFY FEATURE COLUMNS
    # ========================================================================
    if verbose:
        print("\n[2/5] Identifying features...")

    feature_cols = [col for col in train_df.columns if col.startswith('X')]

    if verbose:
        print(f"  ✓ Found {len(feature_cols)} features: X1-X{len(feature_cols)}")

    # ========================================================================
    # STEP 3: CONVERT TO NUMERIC & DETECT MISSING
    # ========================================================================
    if verbose:
        print("\n[3/5] Converting to numeric and detecting missing values...")

    # Count missing before
    missing_before = train_df[feature_cols].isnull().sum().sum()
    empty_before = (train_df[feature_cols].astype(str) == '').sum().sum()

    # Convert to numeric
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # Count missing after
    missing_after = train_df[feature_cols].isnull().sum()
    cols_with_missing = missing_after[missing_after > 0]

    if verbose:
        print(f"  ✓ Converted all features to numeric")
        if len(cols_with_missing) > 0:
            print(f"  ⚠ Missing values detected:")
            for col, count in cols_with_missing.items():
                pct = count / len(train_df) * 100
                print(f"    {col}: {count} ({pct:.2f}%)")
        else:
            print(f"  ✓ No missing values detected")

    # ========================================================================
    # STEP 4: HANDLE MISSING VALUES
    # ========================================================================
    if verbose:
        print("\n[4/5] Handling missing values...")

    filled_count = 0
    for col in feature_cols:
        missing_count = train_df[col].isnull().sum()
        if missing_count > 0:
            # Fill with median (robust to outliers)
            median_val = train_df[col].median()
            train_df[col].fillna(median_val, inplace=True)
            test_df[col].fillna(median_val, inplace=True)
            filled_count += missing_count
            if verbose:
                print(f"  {col}: Filled {missing_count} values with median ({median_val:.4f})")

    if verbose:
        if filled_count == 0:
            print(f"  ✓ No missing values to fill")
        else:
            print(f"  ✓ Total filled: {filled_count} missing values")

    # ========================================================================
    # STEP 5: DATA QUALITY VALIDATION
    # ========================================================================
    if verbose:
        print("\n[5/5] Data quality validation...")

    # Check for NaN remaining
    remaining_nan = train_df[feature_cols].isnull().sum().sum()
    assert remaining_nan == 0, f"Remaining NaN values: {remaining_nan}"

    # Check feature alignment
    train_features_set = set(train_df.columns)
    test_features_set = set(test_df.columns)

    if verbose:
        if 'Y' in train_df.columns:
            print(f"  ✓ Target variable (Y) present in train set")
        if 'Y' not in test_df.columns:
            print(f"  ✓ Target variable (Y) correctly absent in test set")

    # Check target distribution
    if 'Y' in train_df.columns:
        y_counts = train_df['Y'].value_counts().sort_index()
        y_pct = train_df['Y'].value_counts(normalize=True).sort_index() * 100

        if verbose:
            print(f"\n  Target Variable Distribution:")
            for label in sorted(train_df['Y'].unique()):
                count = y_counts[label]
                pct = y_pct[label]
                label_name = 'NORMAL' if label == 0 else 'ALPHA DEFECT'
                print(f"    Y={label}: {count:4d} ({pct:5.2f}%) [{label_name}]")

            if len(y_counts) > 1:
                ratio = y_counts.iloc[0] / y_counts.iloc[1]
                print(f"  → Class imbalance ratio: {ratio:.1f}:1")

    # Data type check
    if verbose:
        print(f"\n  Data Type Check:")
        print(f"    All features numeric: {train_df[feature_cols].dtypes.eq('float64').all() or train_df[feature_cols].dtypes.eq('int64').all()}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("✓ DATA LOADING & CLEANING COMPLETE")
        print("=" * 80)
        print(f"\nFinal Dataset Shapes:")
        print(f"  Train: {train_df.shape[0]} × {train_df.shape[1]}")
        print(f"  Test:  {test_df.shape[0]} × {test_df.shape[1]}")
        print(f"\nNext: Run Phase 2 - Industrial EDA")

    return train_df, test_df, feature_cols


def save_cleaned_data(train_df, test_df, output_dir='./'):
    """Save cleaned datasets to CSV"""
    train_path = f"{output_dir}/train_cleaned.csv"
    test_path = f"{output_dir}/test_cleaned.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✓ Saved: {train_path}")
    print(f"✓ Saved: {test_path}")

    return train_path, test_path


if __name__ == "__main__":
    # Load and clean data
    train_df, test_df, feature_cols = load_and_clean_data(
        train_path='train.csv',
        test_path='test.csv',
        verbose=True
    )

    # Save cleaned data
    save_cleaned_data(train_df, test_df, output_dir='./')

    # Display summary statistics
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY STATISTICS")
    print("=" * 80)
    print("\nFirst 10 features (Train set):")
    print(train_df[feature_cols[:10]].describe().to_string())
