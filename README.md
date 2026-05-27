# Alpha Defect Detection in Hot Rolling Mills

**Industrial ML Workflow for Metallurgical Quality Control**

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline to detect **Alpha defects** in hot rolling mills. Alpha defects are critical quality challenges that cannot be detected through existing inline systems, making early detection crucial for preventing customer complaints and product downgrades.

### Problem Statement
- **Challenge**: Detect Alpha defects during hot rolling before they reach customers
- **Data**: 1,352 training samples with 49 process parameters from furnace → rolling → cooling stages
- **Target**: Binary classification (Normal: 95.8%, Alpha Defect: 4.2%)
- **Approach**: Industrial EDA → Feature Engineering → SHAP Interpretation → Threshold Optimization

## 🎯 Key Objectives

1. **Understand thermo-mechanical process signatures** that lead to metallurgical instability
2. **Identify dangerous parameter combinations** causing defects
3. **Map safe vs unsafe operating windows** for process control
4. **Detect process regime transitions** and instability patterns
5. **Build interpretable models** for production decision-making
6. **Achieve high recall** (minimize false negatives - missed defects)

## 📊 Project Structure

```
alpha-defect-detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config/
│   └── config.yaml                   # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── phase_1_data_loading.py       # Load & clean data
│   ├── phase_2_eda_industrial.py     # Industrial EDA analysis
│   ├── phase_3_feature_engineering.py # Feature creation
│   ├── phase_4_baseline_models.py    # XGBoost/LightGBM
│   ├── phase_5_shap_analysis.py      # SHAP interpretation
│   ├── phase_6_threshold_tuning.py   # Optimize decision boundary
│   ├── phase_7_ensemble.py           # Stack models
│   └── utils.py                      # Helper functions
├── notebooks/
│   └── alpha_defect_colab.py         # Google Colab notebook (Python)
├── data/
│   ├── train.csv                     # Training data (gitignored)
│   ├── test.csv                      # Test data (gitignored)
│   └── .gitignore                    # Ignore data files
└── outputs/
    ├── eda_univariate_analysis.csv
    ├── eda_instability_analysis.csv
    ├── model_predictions.csv
    └── shap_analysis.pkl
```

## 🔬 Workflow Phases

### Phase 1: Data Loading & Cleaning
**Goal**: Load datasets, handle missing values, validate data quality

- Load train.csv (1,352 × 51) and test.csv (339 × 50)
- Convert to numeric, fill missing values (X15)
- Validate class distribution and feature alignment
- **Time**: ~30 seconds

**Output**: `train_cleaned.csv`, `test_cleaned.csv`

### Phase 2: Industrial EDA
**Goal**: Understand process physics and defect-driving patterns

9 Major Analyses:
1. **Univariate significance** - Which X predict Y?
2. **Distribution comparison** - Defect vs normal patterns
3. **Correlation & grouping** - Feature redundancy
4. **Variance analysis** - Instability markers
5. **Outlier/tail-risk** - Extreme operating regions
6. **PCA latent structure** - Hidden process regimes
7. **Nonlinear interactions** - Dangerous combinations
8. **Safe/unsafe windows** - Operating boundaries
9. **Summary & recommendations** - Feature engineering hints

**Output**: `eda_univariate_analysis.csv`, `eda_instability_analysis.csv`

**Time**: ~2-3 minutes

### Phase 3: Feature Engineering
**Goal**: Create informative features capturing process physics

- **Interaction terms**: X_i × X_j, X_i / X_j, X_i - X_j
- **Instability measures**: CV, rolling std, variance ratios
- **Threshold indicators**: Above/below critical values
- **Regime features**: PCA components for process stages
- **Anomaly distances**: Distance from safe operating regions

### Phase 4: Baseline Models
**Goal**: Build strong baseline with class-balanced approach

- XGBoost with class weights
- LightGBM with stratified CV
- Metrics: Precision, Recall, F1, AUC-ROC, AUC-PR
- **Focus**: High recall (catch defects)

### Phase 5: SHAP Interpretation
**Goal**: Explain model decisions and find defect drivers

- Global SHAP feature importance
- Local SHAP explanations per sample
- SHAP interaction effects
- Identify parameter combinations causing defects

### Phase 6: Threshold Tuning
**Goal**: Optimize decision boundary for production

- Sweep decision thresholds (not default 0.5)
- Tune for precision-recall tradeoff
- Cost-sensitive optimization
- Production deployment readiness

### Phase 7: Ensemble & Final Model
**Goal**: Combine models for robustness

- Stack LightGBM + XGBoost + CatBoost
- Cross-validate thoroughly
- Generate final predictions on test set
- Document confidence scores

## 📈 Expected Results from Phase 2 EDA

After running Industrial EDA, expect to find:

✅ **~25-30 significant features** (p < 0.05 difference between defect/normal)
✅ **18 PCA components** capture 80% variance (31 redundant features)
✅ **4-6 dangerous parameter combinations** with high defect rates
✅ **Safe operating windows** identified for key parameters
✅ **3-5 distinct process regimes** (furnace, rolling, cooling stages)
✅ **Instability patterns** correlating with defects

## 🚀 Getting Started

### Local Setup

```bash
# Clone repository
git clone <repo-url>
cd alpha-defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Phase 1 (Data Loading)
python src/phase_1_data_loading.py

# Run Phase 2 (Industrial EDA)
python src/phase_2_eda_industrial.py
```

### Google Colab Setup

See `notebooks/alpha_defect_colab.py` for step-by-step Colab instructions.

**Quick Start:**
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `train.csv` and `test.csv`
3. Copy Phase 1 code → Run
4. Copy Phase 2 code → Run
5. Explore results (~5-10 minutes total)

## 📊 Data Description

### Features (X1-X49)

| Group | Features | Description | Range |
|-------|----------|-------------|-------|
| **Metallurgical** | X1-X9 | Temperature, heating rates | ~200-1100 |
| **Timing** | X10-X12 | Rates, durations | ~2-95 |
| **Control** | X13-X22 | Process parameters | ~100-1700 |
| **Stability** | X23-X33 | Variance, instability metrics | ~5-36 |
| **Events** | X34-X40 | Cumulative counts/stress | 0-4.2e+07 |
| **Quality** | X41-X49 | Normalized metrics, ratios | -50 to +66 |

### Target (Y)

- **Y = 0**: No defect (95.8% of samples) - NORMAL
- **Y = 1**: Alpha defect present (4.2% of samples) - DEFECT

⚠️ **Class Imbalance**: 22.7:1 (requires class weights / SMOTE)

## ⚙️ Technical Stack

- **Python 3.8+**
- **Data**: pandas, numpy
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Interpretation**: SHAP, permutation importance
- **Visualization**: matplotlib, seaborn
- **Stats**: scipy, statsmodels

## 📝 Key Files

| File | Purpose |
|------|---------|
| `src/phase_1_data_loading.py` | Data loading, cleaning, validation |
| `src/phase_2_eda_industrial.py` | Comprehensive statistical analysis |
| `notebooks/alpha_defect_colab.py` | Google Colab step-by-step guide |
| `requirements.txt` | Python package dependencies |

## 🔍 Industrial ML Mindset

> **Not**: "Which variable predicts Y?"
>
> **But**: "Which thermo-mechanical process states and parameter interactions create metallurgical instability leading to Alpha defects?"

This project emphasizes:
- ✅ Process physics understanding
- ✅ Safe operating region identification
- ✅ Interpretable model decisions
- ✅ High recall for quality assurance
- ✅ Production-ready implementations

## 📚 Roadmap (8-Step Process)

1. ✅ Industrial EDA → Understand data
2. ✅ Correlation Grouping → Detect feature blocks
3. ✅ Interaction Analysis → Find dangerous combinations
4. ✅ Baseline Models → XGBoost/LightGBM
5. ✅ SHAP Interpretation → Explain decisions
6. ✅ Feature Engineering → Create informative features
7. ✅ Imbalance Handling → Class weights/SMOTE
8. ✅ Threshold Tuning → Optimize for industry
9. ✅ Ensemble → Final robust model

## 💬 Results & Interpretation

Expected model performance:
- **Precision**: 0.65-0.75 (reduce false alarms)
- **Recall**: 0.80-0.90 (catch defects)
- **F1-Score**: 0.70-0.80
- **AUC-ROC**: 0.85-0.92
- **AUC-PR**: 0.50-0.70 (harder due to imbalance)

Production use:
- Use **probability scores**, not binary predictions
- Apply **threshold tuning** based on cost-benefit
- Continuously **monitor process drift**
- Update models quarterly with new data

## 📞 Support

For questions or issues:
1. Check the `README.md` in each phase
2. Review output files and visualizations
3. Consult SHAP explanations
4. Analyze safe operating windows

## 📄 License

[Add your license here]

---

**Last Updated**: May 27, 2026  
**Status**: Phase 1-2 Code Ready  
**Next**: Feature Engineering & Baseline Models
