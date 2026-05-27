"""
Alpha Defect Detection in Hot Rolling Mills
Industrial ML Workflow Package
"""

__version__ = "0.1.0"
__author__ = "Data Science Team"
__description__ = "Comprehensive ML pipeline for Alpha defect detection in hot rolling mills"

# Import phase functions for easy access
try:
    from .phase_1_data_loading import load_and_clean_data, save_cleaned_data
    from .phase_2_eda_industrial import run_industrial_eda
except ImportError:
    pass

__all__ = [
    'load_and_clean_data',
    'save_cleaned_data',
    'run_industrial_eda',
]
