# Niger Delta Oil Spill - Containment Efficiency and Response Time ML Analysis

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Journal: SERRA](https://img.shields.io/badge/Journal-Stoch.%20Environ.%20Res.%20Risk%20Assess.%20(Springer)-green)](https://www.springer.com/journal/477)

---

## Paper

**Title:** Quantifying Spill Containment Efficiency and Response Time Disparities in Niger Delta Oil Infrastructure Using Machine Learning: A Comparative Analysis of Operators and Facility Types

**Target Journal:** Stochastic Environmental Research and Risk Assessment (Springer, IF approx. 4.2)

**Status:** Under Review

**Authors:** [Author 1](1), [Author 2](2), [Author 3](3)
> (1) University of Port Harcourt, Nigeria | (2) Covenant University, Ota, Nigeria | (3) FUTO, Owerri, Nigeria

---

## Overview

This repository contains the complete, reproducible analysis for Paper 3, which presents the **first ML-based quantitative analysis of spill containment efficiency and response time** for the Niger Delta — defining CER and RTI as explicit operational performance metrics and identifying the predictors of poor containment outcomes.

**Core innovation:** We transform mandatory NOSDRA incident reports into actionable operational performance metrics (CER and RTI), demonstrate that flowline facilities have a statistically significant containment efficiency deficit versus pipelines, and identify a structural response time plateau since 2020 using Chow structural break testing.

### Key Results

**Containment Efficiency Ratio (CER):**
- Mean CER = 41.2% (SD 38.4%), Median = 18.5%
- Most incidents achieve low-to-moderate recovery — this is the norm, not an exception

**Flowline Deficit (most actionable finding):**
- Pipeline CER = 45.1%
- Flowline CER = **28.3%**
- Mann-Whitney p = **0.002**, Cohen d = **0.41** (small-medium effect)
- This difference is independent of volume and surface type (confirmed by multivariate regression)

**Operator Comparison:**
- NAOC CER = 42.3% vs SPDC CER = 38.7%
- Mann-Whitney p = 0.14 — **no statistically significant difference** after controlling for confounders

**Response Time (RTI):**
- RTI improved **18%** from 2016 to 2020 (from ~11.2 days to ~6.8 days)
- Chow test confirms structural break at Q2 2020 (F=6.84, **p=0.009**)
- RTI has **plateaued** since 2020 — structural constraint identified

**ML Performance (CER classification, 5-fold CV):**

| Model | Accuracy | F1 |
|-------|----------|----|
| KNN (k=7) | 0.957 | 0.943 |
| Logistic Regression | 0.957 | 0.941 |
| Random Forest | 0.953 | 0.939 |
| Gradient Boosting | 0.934 | 0.921 |

**Top SHAP predictors (CER):**
1. Log-transformed spill volume — negative (larger spills harder to recover)
2. Ecosystem vulnerability — negative (swamp/water lower efficiency)
3. Incident-to-report lag — negative (delayed reporting = delayed response)
4. Is-flowline flag — negative (flowlines lack automated monitoring)
5. Dry season — positive (better site access in dry months)

---

## Repository Structure

```
niger-delta-containment-ml/
|
|-- data/
|   |-- sample_data.csv                    # 20-row synthetic test dataset
|   |-- data_dictionary.md                 # Variable definitions
|
|-- notebooks/
|   |-- 01_Containment_Efficiency.ipynb    # Complete CER and RTI analysis pipeline
|
|-- src/
|   |-- utils.py                           # Data loading, CER/RTI construction
|   |-- models.py                          # Classifier and regressor training and CV
|   |-- plots.py                           # All figure generation
|   |-- statistics.py                      # Mann-Whitney, Cohen d, Chow test
|   |-- __init__.py
|
|-- tests/
|   |-- test_cer.py                        # Unit tests for CER and RTI computation
|   |-- test_statistics.py                 # Unit tests for statistical comparisons
|   |-- __init__.py
|
|-- outputs/
|   |-- figures/                           # All figures (PNG, 150 DPI)
|   |-- results/                           # CER predictions, SHAP importance CSVs
|   |-- models/                            # Saved classifier and regressor (PKL)
|
|-- .github/
|   |-- workflows/
|       |-- tests.yml                      # CI pipeline
|
|-- requirements.txt
|-- environment.yml
|-- CITATION.cff
|-- LICENSE
|-- CONTRIBUTING.md
```

---

## Quick Start

### Install

```bash
git clone https://github.com/YOUR_USERNAME/niger-delta-containment-ml.git
cd niger-delta-containment-ml
pip install -r requirements.txt
```

### Run the analysis

```bash
jupyter lab
# Open notebooks/01_Containment_Efficiency.ipynb and run all cells
```

### Python API

```python
from src.utils import load_data, add_temporal_features, add_outcome_variables, build_cer_features
from src.models import train_classifiers, train_regressors, print_results_table
from scipy import stats
import numpy as np

# Load and prepare data
df = load_data('data/oils_data.csv')
df = add_temporal_features(df)
df = add_outcome_variables(df)

# CER subset
df_cer = df[df['CER'].notna() & (df['Estimated'] > 0)].copy()
df_cer['CER_class'] = (df_cer['CER']
    .apply(lambda x: 'Low' if x < 33 else ('High' if x >= 67 else 'Medium')))
df_cer['CER_label'] = df_cer['CER_class'].map({'Low': 0, 'Medium': 1, 'High': 2})

X_cer = build_cer_features(df_cer)
y_cer = df_cer['CER_label']

# Train CER classifiers
results = train_classifiers(X_cer, y_cer, paper=3, n_splits=5)
print_results_table(results, 'CER Classification - 5-Fold CV')

# Operator comparison
naoc_cer = df_cer[df_cer['Company'] == 'NAOC']['CER'].dropna()
spdc_cer = df_cer[df_cer['Company'] == 'SPDC']['CER'].dropna()
u_stat, p_val = stats.mannwhitneyu(naoc_cer, spdc_cer, alternative='two-sided')
d = (naoc_cer.mean() - spdc_cer.mean()) / np.sqrt(
    (naoc_cer.std()**2 + spdc_cer.std()**2) / 2)
print(f"NAOC mean CER: {naoc_cer.mean():.1f}%")
print(f"SPDC mean CER: {spdc_cer.mean():.1f}%")
print(f"Mann-Whitney p = {p_val:.4f}, Cohen d = {abs(d):.3f}")
```

---

## Methods

### Outcome Variables

**Containment Efficiency Ratio:**
```
CER = (Quantity_recovered / Estimated_volume) * 100%   where CER in [0%, 100%]
```
Available for n=278 incidents (estimated volume > 0). CER values above 100% capped.

**Response Time Index:**
```
RTI = (Date_Spill_Stop - Date_Incident) in days   where RTI >= 0
```
Available for n=238 incidents with non-missing Spill_stop date.

**CER classes:** Low (< 33%), Medium (33-67%), High (>= 67%)

### Statistical Tests

**Mann-Whitney U (non-parametric group comparison):**
```
U = n1 * n2 + n1*(n1+1)/2 - R1
```
Two-tailed, alpha = 0.05

**Cohen d (effect size):**
```
d = (mu1 - mu2) / sqrt((sigma1^2 + sigma2^2) / 2)
```
Thresholds: below 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, above 0.8 large

**Chow Structural Break Test:**
```
F = ((RSS_pooled - RSS1 - RSS2) / k) / ((RSS1 + RSS2) / (n1 + n2 - 2k))
```
Applied to quarterly RTI series with candidate breakpoint at Q2 2020

---

## Comparison Tables

### Operator Comparison (CER)

| Company | n | Mean CER | SD | Mann-Whitney p | Cohen d |
|---------|---|----------|----|----------------|---------|
| NAOC | 209 | 42.3% | 39.1% | 0.14 (not significant) | 0.09 (negligible) |
| SPDC | 69 | 38.7% | 37.2% | — | — |

### Facility Comparison (CER) — Key Finding

| Facility | n | Mean CER | SD | Mann-Whitney p | Cohen d |
|----------|---|----------|----|----------------|---------|
| Pipeline | 222 | 45.1% | 38.7% | **0.002** | **0.41 (small-medium)** |
| Flowline | 56 | 28.3% | 35.1% | — | — |

> The flowline deficit persists after multivariate adjustment for volume and surface type.

---

## Figures

| File | Description |
|------|-------------|
| `fig_study_area.png` | Study area map |
| `fig_eda.png` | Exploratory data analysis (shared with Paper 1) |
| `fig_cer_rti_p3.png` | 6-panel CER and RTI exploratory analysis |
| `fig_models_p3.png` | CER classification and RTI regression model comparison |
| `fig_feature_importance_p3.png` | Gini importance for CER and RTI models |
| `fig_rti_trend.png` | RTI structural break analysis and violin plots |

---

## Tests

```bash
pytest tests/ -v
```

---

## Citation

```bibtex
@article{author2024containment,
  title   = {Quantifying Spill Containment Efficiency and Response Time Disparities
             in Niger Delta Oil Infrastructure Using Machine Learning:
             A Comparative Analysis of Operators and Facility Types},
  author  = {[Author Names]},
  journal = {Stochastic Environmental Research and Risk Assessment},
  publisher = {Springer},
  year    = {2024},
  note    = {Under Review}
}
```

---

## Related Repositories

This is Paper 3 in a three-paper series:

- **Paper 1:** [niger-delta-phri-ml](https://github.com/YOUR_USERNAME/niger-delta-phri-ml) — Public health risk zone prediction
- **Paper 2:** [niger-delta-spatiotemporal-ml](https://github.com/YOUR_USERNAME/niger-delta-spatiotemporal-ml) — Spatiotemporal sabotage hotspot detection
- **Paper 3 (this repo):** Containment efficiency and response time analysis

---

## License

MIT License. The oil spill incident data is the property of NOSDRA, Nigeria, and is NOT included.
