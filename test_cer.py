"""
test_cer.py
Unit tests for CER, RTI construction and statistical comparison methods.
Run: pytest tests/
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import add_temporal_features, add_outcome_variables
from src.statistics import (
    mann_whitney_comparison, chow_test, linear_trend,
    find_best_breakpoint, compare_operators, compare_facilities
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def cer_df():
    """Sample dataframe for CER/RTI testing."""
    return pd.DataFrame({
        'Incident_d':  pd.to_datetime(['2020-01-10', '2021-06-15', '2022-11-20', '2023-03-05']),
        'Report_dat':  pd.to_datetime(['2020-01-11', '2021-06-18', '2022-11-22', '2023-03-07']),
        'Spill_stop':  pd.to_datetime(['2020-01-18', '2021-07-01', '2022-12-10', '2023-03-12']),
        'Company':     ['NAOC', 'SPDC', 'NAOC', 'SPDC'],
        'Type_of_fa':  ['pl', 'fl', 'pl', 'fl'],
        'Estimated':   [100.0, 10.0, 500.0, 5.0],
        'Qauntity_r':  [40.0, 8.0, 100.0, 4.5],
        'Contaminan':  ['cr', 'no', 'co', 'cr'],
        'Spill_area':  ['la', 'ss', 'sw', 'la'],
        'Latitude':    [5.1, 4.9, 5.2, 5.0],
        'Longitude':   [6.5, 6.3, 6.6, 6.4],
        'LGA':         ['Ahoada-West', 'Yenagoa', 'Ahoada-West', 'Abua-Odual'],
        'Estimate_1':  [5.0, 0.5, 10.0, 0.2],
        'Contaminan_clean': ['cr', 'no', 'co', 'cr'],
        'Cause':       ['sab', 'sab', 'sab', 'sab'],
    })


# ============================================================
# CER / RTI Construction Tests
# ============================================================

class TestCERConstruction:
    def test_cer_formula(self, cer_df):
        df = add_temporal_features(cer_df)
        df = add_outcome_variables(df)
        # Row 0: 40/100*100 = 40%
        assert abs(df.loc[0, 'CER'] - 40.0) < 0.01

    def test_cer_upper_bound(self, cer_df):
        """CER must never exceed 100%."""
        df = add_temporal_features(cer_df)
        df = add_outcome_variables(df)
        assert (df['CER'].dropna() <= 100.0).all()

    def test_cer_lower_bound(self, cer_df):
        """CER must be non-negative."""
        df = add_temporal_features(cer_df)
        df = add_outcome_variables(df)
        assert (df['CER'].dropna() >= 0.0).all()

    def test_cer_nan_when_zero_volume(self):
        """CER should be NaN when estimated volume is 0 or negative."""
        df = pd.DataFrame({
            'Incident_d': [pd.Timestamp('2021-01-01')],
            'Report_dat': [pd.Timestamp('2021-01-02')],
            'Spill_stop': [pd.Timestamp('2021-01-05')],
            'Estimated':  [0.0],
            'Qauntity_r': [0.0],
        })
        df = add_temporal_features(df)
        df = add_outcome_variables(df)
        assert pd.isna(df.loc[0, 'CER'])

    def test_rti_non_negative(self, cer_df):
        """RTI must always be >= 0."""
        df = add_temporal_features(cer_df)
        df = add_outcome_variables(df)
        assert (df['RTI'].dropna() >= 0).all()

    def test_rti_matches_date_difference(self, cer_df):
        """RTI for row 0 should be Jan 18 minus Jan 10 = 8 days."""
        df = add_temporal_features(cer_df)
        df = add_outcome_variables(df)
        assert df.loc[0, 'RTI'] == 8

    def test_rti_log_transform(self, cer_df):
        df = add_temporal_features(cer_df)
        df = add_outcome_variables(df)
        assert (df['RTI_log'].dropna() >= 0).all()
        # RTI_log = log(RTI + 1)
        expected = np.log1p(df.loc[0, 'RTI'])
        assert abs(df.loc[0, 'RTI_log'] - expected) < 0.001


# ============================================================
# Statistical Comparison Tests
# ============================================================

class TestMannWhitney:
    def test_identical_groups_not_significant(self):
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = mann_whitney_comparison(a, b)
        assert not result['significant']

    def test_clearly_different_groups_significant(self):
        np.random.seed(42)
        a = np.random.normal(80, 5, 50)
        b = np.random.normal(20, 5, 50)
        result = mann_whitney_comparison(a, b)
        assert result['significant']
        assert result['p_value'] < 0.001

    def test_large_d_for_very_different_groups(self):
        a = np.ones(30) * 100
        b = np.zeros(30)
        result = mann_whitney_comparison(a, b)
        assert abs(result['cohens_d']) > 1.0
        assert result['effect_label'] == 'large'

    def test_negligible_effect_near_zero_d(self):
        np.random.seed(0)
        a = np.random.normal(50, 10, 100)
        b = np.random.normal(50.1, 10, 100)
        result = mann_whitney_comparison(a, b)
        assert result['abs_cohens_d'] < 0.2

    def test_returns_all_required_keys(self):
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([2, 3, 4, 5, 6], dtype=float)
        result = mann_whitney_comparison(a, b)
        required = ['label_a', 'label_b', 'n_a', 'n_b', 'mean_a', 'mean_b',
                    'U_stat', 'p_value', 'cohens_d', 'significant']
        for key in required:
            assert key in result

    def test_handles_nan_values(self):
        a = np.array([10.0, np.nan, 30.0, 40.0])
        b = np.array([15.0, 25.0, np.nan, 35.0])
        # Should not raise
        result = mann_whitney_comparison(a, b)
        assert isinstance(result['p_value'], float)


# ============================================================
# Temporal / Chow Test
# ============================================================

class TestLinearTrend:
    def test_declining_series_has_negative_slope(self):
        series = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        result = linear_trend(series)
        assert result['slope'] < 0

    def test_flat_series_slope_near_zero(self):
        series = np.ones(20) * 5.0
        result = linear_trend(series)
        assert abs(result['slope']) < 1e-10

    def test_perfect_trend_r_squared_one(self):
        series = np.arange(20, dtype=float)
        result = linear_trend(series)
        assert result['r_squared'] > 0.999


class TestChowTest:
    def test_obvious_break_detected(self):
        """Two segments with clearly different slopes should give significant F."""
        np.random.seed(42)
        seg1 = np.linspace(10, 7, 15) + np.random.normal(0, 0.1, 15)
        seg2 = np.linspace(7, 7.2, 15) + np.random.normal(0, 0.1, 15)
        series = np.concatenate([seg1, seg2])
        result = chow_test(series, breakpoint=15)
        assert 'F_stat' in result
        assert result['F_stat'] > 0

    def test_no_break_in_homogeneous_series(self):
        """Single linear trend should give non-significant F."""
        np.random.seed(0)
        series = np.linspace(10, 5, 30) + np.random.normal(0, 0.05, 30)
        result = chow_test(series, breakpoint=15)
        assert result['p_value'] > 0.05

    def test_boundary_breakpoint_returns_error(self):
        series = np.arange(10, dtype=float)
        result = chow_test(series, breakpoint=1)
        assert 'error' in result

    def test_find_best_breakpoint_returns_valid(self):
        np.random.seed(42)
        seg1 = np.linspace(10, 7, 20) + np.random.normal(0, 0.1, 20)
        seg2 = np.linspace(7, 7.1, 10) + np.random.normal(0, 0.1, 10)
        series = np.concatenate([seg1, seg2])
        result = find_best_breakpoint(series)
        assert 'best_breakpoint' in result
        assert 5 <= result['best_breakpoint'] <= 25
