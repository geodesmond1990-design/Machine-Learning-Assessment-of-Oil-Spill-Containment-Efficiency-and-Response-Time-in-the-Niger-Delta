"""
utils.py
--------
Data loading, cleaning, and preprocessing utilities for the
Niger Delta Oil Spill ML project.

Usage:
    from src.utils import load_data, preprocess

Authors: [Author Names]
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform minimal cleaning on the NOSDRA oil spill dataset.

    Parameters
    ----------
    filepath : str
        Path to oils_data.csv

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with parsed dates and standardised LGA names.
    """
    df = pd.read_csv(filepath, encoding='latin1')

    # Parse date columns
    for col in ['Incident_d', 'Report_dat', 'Spill_stop']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Standardise LGA names
    df['LGA'] = df['LGA'].str.strip().str.title()
    lga_map = {
        "Ahoada-West '": 'Ahoada-West',
        'Ahoada West':   'Ahoada-West',
        'Ahoada-West\' ': 'Ahoada-West',
    }
    df['LGA'] = df['LGA'].replace(lga_map)

    # Standardise contaminant
    df['Contaminan_raw'] = df['Contaminan'].copy()
    df['Contaminan'] = df['Contaminan'].str.strip().str.lower()
    df['Contaminan_clean'] = df['Contaminan'].apply(
        lambda x: x if x in ['cr', 'co', 'no'] else 'other'
    )

    # Standardise cause
    df['Cause_clean'] = df['Cause'].str.strip().str.lower()
    df['is_sabotage'] = (df['Cause_clean'] == 'sab').astype(int)

    print(f"Loaded {len(df)} records from {filepath}")
    print(f"Date range: {df['Incident_d'].min().date()} to {df['Incident_d'].max().date()}")
    print(f"Companies: {df['Company'].value_counts().to_dict()}")
    print(f"Missing Spill_stop: {df['Spill_stop'].isna().sum()} records")

    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, cyclic encoding, dry season indicator, and date lags."""
    df = df.copy()
    df['year']  = df['Incident_d'].dt.year
    df['month'] = df['Incident_d'].dt.month

    # Cyclic month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Dry season: November through February
    df['dry_season'] = df['month'].apply(lambda m: 1 if m in [11, 12, 1, 2] else 0)

    # Year normalised to [0, 1]
    y_min = df['year'].min()
    y_max = df['year'].max()
    df['year_norm'] = (df['year'] - y_min) / (y_max - y_min) if y_max > y_min else 0.0

    # Response time
    df['response_days'] = (
        df['Spill_stop'] - df['Incident_d']
    ).dt.days.clip(lower=0)

    # Report lag
    df['report_lag'] = (
        df['Report_dat'] - df['Incident_d']
    ).dt.days.clip(lower=0)

    # Impute missing response_days by facility-type-stratified median
    med_by_fa = df.groupby('Type_of_fa')['response_days'].median()
    overall_med = df['response_days'].median()
    df['response_days'] = df.apply(
        lambda r: med_by_fa.get(r['Type_of_fa'], overall_med)
        if pd.isna(r['response_days']) else r['response_days'],
        axis=1
    )

    # Impute missing report_lag by company median
    med_by_co = df.groupby('Company')['report_lag'].median()
    df['report_lag'] = df.apply(
        lambda r: med_by_co.get(r['Company'], 3.0)
        if pd.isna(r['report_lag']) else r['report_lag'],
        axis=1
    )

    return df


def add_outcome_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CER and RTI outcome variables.

    CER = (Quantity_recovered / Estimated_volume) * 100, capped at [0, 100].
    RTI = days from incident to containment.
    """
    df = df.copy()

    # CER
    df['CER'] = np.where(
        df['Estimated'] > 0,
        (df['Qauntity_r'] / df['Estimated']) * 100,
        np.nan
    )
    df['CER'] = df['CER'].clip(0, 100)

    # RTI (same as response_days but with explicit NaN for missing)
    df['RTI'] = (df['Spill_stop'] - df['Incident_d']).dt.days.clip(lower=0)

    # Log-transformed RTI for regression
    df['RTI_log'] = np.log1p(df['RTI'])

    return df


# ---------------------------------------------------------------------------
# PHRI Construction
# ---------------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def compute_phri(
    df: pd.DataFrame,
    w_svs: float = 0.30,
    w_rts: float = 0.25,
    w_css: float = 0.20,
    w_evs: float = 0.15,
    w_frs: float = 0.10
) -> pd.DataFrame:
    """
    Compute the Public Health Risk Index (PHRI).

    PHRI = w_svs*SVS + w_rts*RTS + w_css*CSS + w_evs*EVS + w_frs*FRS

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Estimated, response_days, Contaminan_clean,
        Spill_area, Type_of_fa
    w_* : float
        Weights for each sub-index (must sum to 1.0)

    Returns
    -------
    pd.DataFrame
        Input dataframe with SVS, RTS, CSS, EVS, FRS, PHRI, PHRI_class added.
    """
    assert abs(w_svs + w_rts + w_css + w_evs + w_frs - 1.0) < 1e-6, \
        "Weights must sum to 1.0"

    df = df.copy()

    # SVS - Spill Volume Score
    df['SVS'] = _minmax(np.log1p(df['Estimated']))

    # RTS - Response Time Score
    df['RTS'] = _minmax(df['response_days'])

    # CSS - Contamination Severity Score
    cont_map = {'cr': 1.00, 'co': 0.67, 'no': 0.33, 'other': 0.00}
    df['CSS'] = df['Contaminan_clean'].map(cont_map).fillna(0.0)

    # EVS - Ecosystem Vulnerability Score
    def _evs(spill_area):
        x = str(spill_area).lower().strip()
        if 'sw' in x or 'ss' in x or 'iw' in x:
            return 1.00
        if ',' in x:
            return 0.67
        if 'la' in x:
            return 0.33
        return 0.17

    df['EVS'] = df['Spill_area'].fillna('').apply(_evs)

    # FRS - Facility Risk Score
    frs_map = {'pl': 1.00, 'fl': 0.67, 'mf': 0.33, 'wh': 0.33}
    df['FRS'] = df['Type_of_fa'].str.strip().map(frs_map).fillna(0.33)

    # PHRI composite
    df['PHRI'] = (
        w_svs * df['SVS'] +
        w_rts * df['RTS'] +
        w_css * df['CSS'] +
        w_evs * df['EVS'] +
        w_frs * df['FRS']
    )

    # Discretise
    df['PHRI_class'] = pd.cut(
        df['PHRI'],
        bins=[-0.001, 0.33, 0.67, 1.01],
        labels=['Low', 'Medium', 'High']
    )

    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['PHRI_label'] = df['PHRI_class'].map(label_map).fillna(0).astype(int)

    print("PHRI computed.")
    print(df['PHRI_class'].value_counts().to_string())

    return df


# ---------------------------------------------------------------------------
# Feature Matrix Builder
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, lga_target_col: str = 'PHRI') -> pd.DataFrame:
    """
    Build the 16-dimensional feature matrix for ML models.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with all temporal and derived features.
    lga_target_col : str
        Column used for LGA target encoding.

    Returns
    -------
    pd.DataFrame
        Feature matrix X with named columns.
    """
    from sklearn.preprocessing import LabelEncoder

    X = pd.DataFrame(index=df.index)

    # Numerical
    X['log_volume']   = np.log1p(df['Estimated'])
    X['response_days'] = df['response_days']
    X['report_lag']   = df['report_lag']
    X['log_area']     = np.log1p(df['Estimate_1'].fillna(0))
    X['lat']          = df['Latitude']
    X['lon']          = df['Longitude']

    # Temporal
    X['month_sin']  = df['month_sin']
    X['month_cos']  = df['month_cos']
    X['year_norm']  = df['year_norm']
    X['dry_season'] = df['dry_season']

    # Categorical encodings
    X['company_enc']  = LabelEncoder().fit_transform(df['Company'])
    X['facility_enc'] = df['Type_of_fa'].str.strip().map(
        {'pl': 3, 'fl': 2, 'mf': 1, 'wh': 1}).fillna(1)
    X['contam_enc']   = df['Contaminan_clean'].map(
        {'cr': 3, 'co': 2, 'no': 1, 'other': 0}).fillna(0)
    X['eco_vuln']     = df.get('EVS', pd.Series(0.33, index=df.index))
    X['cause_sab']    = df.get('is_sabotage', pd.Series(0, index=df.index))

    # LGA target encoding
    if lga_target_col in df.columns:
        lga_mean = df.groupby('LGA')[lga_target_col].mean()
        X['lga_enc'] = df['LGA'].map(lga_mean).fillna(df[lga_target_col].mean())
    else:
        X['lga_enc'] = 0.0

    return X.fillna(X.median())


# ---------------------------------------------------------------------------
# CER / RTI Feature Matrix
# ---------------------------------------------------------------------------

def build_cer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for CER and RTI models (Paper 3)."""
    X = pd.DataFrame(index=df.index)

    X['log_vol']      = np.log1p(df['Estimated'].fillna(0))
    X['is_flowline']  = (df['Type_of_fa'].str.strip() == 'fl').astype(int)
    X['facility_enc'] = df['Type_of_fa'].str.strip().map(
        {'pl': 3, 'fl': 2, 'mf': 1, 'wh': 1}).fillna(1)
    X['is_naoc']      = (df['Company'] == 'NAOC').astype(int)
    X['contam_sev']   = df['Contaminan'].str.strip().map(
        {'cr': 3, 'co': 2, 'no': 1}).fillna(0)
    X['eco_vuln']     = df['Spill_area'].fillna('').apply(
        lambda x: 3 if ('sw' in str(x).lower() or 'ss' in str(x).lower()) else
                  (2 if 'la' in str(x).lower() else 1))
    X['is_crude']     = (df['Contaminan'].str.strip() == 'cr').astype(int)
    X['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
    X['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)
    X['dry_season']   = df['month'].apply(lambda m: 1 if m in [11, 12, 1, 2] else 0)
    X['year']         = df['year'].fillna(2020)
    X['report_lag']   = df['report_lag']
    X['log_report_lag'] = np.log1p(df['report_lag'])
    X['lat']          = df['Latitude'].fillna(df['Latitude'].mean())
    X['lon']          = df['Longitude'].fillna(df['Longitude'].mean())

    return X.fillna(X.median())


# ---------------------------------------------------------------------------
# Convenience: Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(filepath: str) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with:
        - df        : fully preprocessed dataframe
        - X_phri    : feature matrix for PHRI classification
        - y_phri    : PHRI class labels (0/1/2)
        - X_cer     : feature matrix for CER modelling
        - y_cer     : CER class labels (0/1/2)
        - y_cer_reg : CER continuous values
        - y_rti     : RTI log-transformed values
        - df_cer    : CER subset (non-missing)
        - df_rti    : RTI subset (non-missing)
    """
    df = load_data(filepath)
    df = add_temporal_features(df)
    df = add_outcome_variables(df)
    df = compute_phri(df)

    X_phri = build_feature_matrix(df, lga_target_col='PHRI')
    y_phri = df['PHRI_label']

    df_cer = df[df['CER'].notna() & (df['Estimated'] > 0)].copy()
    df_cer['CER_class'] = pd.cut(
        df_cer['CER'], bins=[-0.1, 33, 67, 100.1], labels=['Low', 'Medium', 'High']
    )
    df_cer['CER_label'] = df_cer['CER_class'].map(
        {'Low': 0, 'Medium': 1, 'High': 2}).fillna(0).astype(int)
    X_cer = build_cer_features(df_cer)
    y_cer     = df_cer['CER_label']
    y_cer_reg = df_cer['CER']

    df_rti = df[df['RTI'].notna()].copy()
    X_rti  = build_cer_features(df_rti)
    y_rti  = df_rti['RTI_log']

    print(f"\nPreprocessing complete.")
    print(f"  PHRI X shape: {X_phri.shape}")
    print(f"  CER X shape:  {X_cer.shape}")
    print(f"  RTI X shape:  {X_rti.shape}")

    return {
        'df': df,
        'X_phri': X_phri, 'y_phri': y_phri,
        'X_cer': X_cer, 'y_cer': y_cer, 'y_cer_reg': y_cer_reg,
        'X_rti': X_rti, 'y_rti': y_rti,
        'df_cer': df_cer, 'df_rti': df_rti,
    }
