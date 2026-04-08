"""
niger-delta-containment-ml
============================
Paper 3: Quantifying Spill Containment Efficiency and Response Time
Disparities in Niger Delta Oil Infrastructure Using Machine Learning.

Target Journal: Stochastic Environmental Research and Risk Assessment (Springer)
"""
from .utils import load_data, add_temporal_features, add_outcome_variables, build_cer_features
from .models import train_classifiers, train_regressors
from .statistics import (mann_whitney_comparison, chow_test, compare_operators,
                          compare_facilities, full_comparison_report)
from .plots import plot_cer_rti
__version__ = '1.0.0'
