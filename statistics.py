"""
statistics.py
-------------
Statistical comparison utilities for Paper 3: non-parametric group comparisons,
effect size calculation, temporal trend testing, and Chow structural break test.

Authors: [Author Names]
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Group Comparisons
# ---------------------------------------------------------------------------

def mann_whitney_comparison(group_a: np.ndarray, group_b: np.ndarray,
                             label_a: str = 'Group A', label_b: str = 'Group B',
                             alternative: str = 'two-sided') -> dict:
    """
    Perform Mann-Whitney U test and compute Cohen d effect size.

    Parameters
    ----------
    group_a, group_b : array-like
        Observation vectors for the two groups.
    label_a, label_b : str
        Labels for reporting.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    dict with keys: label_a, label_b, mean_a, mean_b, sd_a, sd_b,
                    n_a, n_b, U_stat, p_value, cohens_d, effect_label, significant
    """
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    u_stat, p_val = stats.mannwhitneyu(a, b, alternative=alternative)

    # Cohen d
    pooled_sd = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    # Effect size label (Cohen 1988)
    abs_d = abs(d)
    if abs_d < 0.2:
        effect_label = 'negligible'
    elif abs_d < 0.5:
        effect_label = 'small'
    elif abs_d < 0.8:
        effect_label = 'medium'
    else:
        effect_label = 'large'

    return {
        'label_a':      label_a,
        'label_b':      label_b,
        'n_a':          len(a),
        'n_b':          len(b),
        'mean_a':       round(a.mean(), 2),
        'mean_b':       round(b.mean(), 2),
        'median_a':     round(np.median(a), 2),
        'median_b':     round(np.median(b), 2),
        'sd_a':         round(a.std(ddof=1), 2),
        'sd_b':         round(b.std(ddof=1), 2),
        'U_stat':       round(float(u_stat), 1),
        'p_value':      round(float(p_val), 4),
        'cohens_d':     round(float(d), 3),
        'abs_cohens_d': round(abs_d, 3),
        'effect_label': effect_label,
        'significant':  p_val < 0.05,
    }


def print_comparison(result: dict):
    """Pretty-print a Mann-Whitney comparison result."""
    sig = 'SIGNIFICANT' if result['significant'] else 'not significant'
    print(f"\n{result['label_a']} vs {result['label_b']}")
    print(f"  {result['label_a']}: n={result['n_a']}, mean={result['mean_a']:.1f}, median={result['median_a']:.1f}, SD={result['sd_a']:.1f}")
    print(f"  {result['label_b']}: n={result['n_b']}, mean={result['mean_b']:.1f}, median={result['median_b']:.1f}, SD={result['sd_b']:.1f}")
    print(f"  Mann-Whitney U={result['U_stat']:.0f}, p={result['p_value']:.4f} ({sig})")
    print(f"  Cohen d={result['cohens_d']:.3f} ({result['effect_label']} effect)")


def compare_operators(df_cer: pd.DataFrame, metric: str = 'CER') -> dict:
    """Compare CER or RTI between NAOC and SPDC."""
    naoc = df_cer[df_cer['Company'] == 'NAOC'][metric].dropna().values
    spdc = df_cer[df_cer['Company'] == 'SPDC'][metric].dropna().values
    return mann_whitney_comparison(naoc, spdc, label_a='NAOC', label_b='SPDC')


def compare_facilities(df_cer: pd.DataFrame, metric: str = 'CER') -> dict:
    """Compare CER or RTI between pipeline (pl) and flowline (fl) facilities."""
    pl = df_cer[df_cer['Type_of_fa'].str.strip() == 'pl'][metric].dropna().values
    fl = df_cer[df_cer['Type_of_fa'].str.strip() == 'fl'][metric].dropna().values
    return mann_whitney_comparison(pl, fl, label_a='Pipeline', label_b='Flowline')


def compare_surface(df_cer: pd.DataFrame, metric: str = 'CER') -> dict:
    """Compare CER or RTI between land and swamp/water surface types."""
    land  = df_cer[df_cer['Spill_area'].str.strip() == 'la'][metric].dropna().values
    water = df_cer[df_cer['Spill_area'].str.strip().isin(['ss', 'sw'])][metric].dropna().values
    return mann_whitney_comparison(land, water, label_a='Land', label_b='Swamp/Water')


# ---------------------------------------------------------------------------
# Temporal Trend Analysis
# ---------------------------------------------------------------------------

def linear_trend(series: np.ndarray) -> dict:
    """
    Fit a linear trend to a time series using OLS.

    Parameters
    ----------
    series : np.ndarray
        Time series values.

    Returns
    -------
    dict with keys: slope, intercept, r_squared, p_value, std_err, significant
    """
    t = np.arange(len(series), dtype=float)
    slope, intercept, r_val, p_val, std_err = stats.linregress(t, series)
    return {
        'slope':      round(float(slope), 4),
        'intercept':  round(float(intercept), 4),
        'r_squared':  round(float(r_val**2), 4),
        'p_value':    round(float(p_val), 4),
        'std_err':    round(float(std_err), 4),
        'significant': p_val < 0.05,
    }


def chow_test(series: np.ndarray, breakpoint: int) -> dict:
    """
    Perform a Chow structural break F-test.

    Tests whether the linear relationship in segment 1 (indices 0 to breakpoint-1)
    differs from segment 2 (indices breakpoint to n-1).

    H0: No structural break (same coefficients in both segments).

    Parameters
    ----------
    series : np.ndarray
        Time series values.
    breakpoint : int
        Index at which to split the series (first index of second segment).

    Returns
    -------
    dict with keys: F_stat, p_value, breakpoint, n1, n2, k, significant
    """
    if breakpoint < 3 or breakpoint > len(series) - 3:
        return {'error': 'Breakpoint too close to series boundary for valid Chow test'}

    t = np.arange(len(series), dtype=float)

    def ols_rss(x, y):
        slope, intercept, _, _, _ = stats.linregress(x, y)
        y_pred = slope * x + intercept
        return np.sum((y - y_pred) ** 2)

    # Full model RSS (pooled)
    rss_pooled = ols_rss(t, series)

    # Segment RSSs
    rss1 = ols_rss(t[:breakpoint], series[:breakpoint])
    rss2 = ols_rss(t[breakpoint:], series[breakpoint:])

    n1 = breakpoint
    n2 = len(series) - breakpoint
    k  = 2  # intercept + slope

    numerator   = (rss_pooled - rss1 - rss2) / k
    denominator = (rss1 + rss2) / (n1 + n2 - 2 * k)

    if denominator <= 0:
        return {'error': 'Denominator <= 0, insufficient degrees of freedom'}

    F_stat = numerator / denominator
    p_val  = 1 - stats.f.cdf(F_stat, dfn=k, dfd=n1 + n2 - 2 * k)

    return {
        'F_stat':     round(float(F_stat), 3),
        'p_value':    round(float(p_val), 4),
        'breakpoint': int(breakpoint),
        'n1':         int(n1),
        'n2':         int(n2),
        'k':          int(k),
        'rss_pooled': round(float(rss_pooled), 4),
        'rss1':       round(float(rss1), 4),
        'rss2':       round(float(rss2), 4),
        'significant': p_val < 0.05,
    }


def find_best_breakpoint(series: np.ndarray, min_seg_len: int = 5) -> dict:
    """
    Search for the breakpoint that maximises the Chow F-statistic.

    Parameters
    ----------
    series : np.ndarray
        Time series values.
    min_seg_len : int
        Minimum observations required in each segment.

    Returns
    -------
    dict with keys: best_breakpoint, best_F, best_p, all_results
    """
    n = len(series)
    candidates = range(min_seg_len, n - min_seg_len)
    results = []
    for bp in candidates:
        r = chow_test(series, bp)
        if 'error' not in r:
            results.append((bp, r['F_stat'], r['p_value']))

    if not results:
        return {'error': 'No valid breakpoints found'}

    best = max(results, key=lambda x: x[1])
    return {
        'best_breakpoint': best[0],
        'best_F':          round(best[1], 3),
        'best_p':          round(best[2], 4),
        'all_results':     results,
    }


# ---------------------------------------------------------------------------
# Full Comparison Report
# ---------------------------------------------------------------------------

def full_comparison_report(df_cer: pd.DataFrame, df_rti: pd.DataFrame) -> pd.DataFrame:
    """
    Run all group comparisons (operator and facility) for both CER and RTI
    and return results as a formatted DataFrame.
    """
    rows = []
    for metric, data in [('CER', df_cer), ('RTI', df_rti)]:
        r_op  = compare_operators(data, metric)
        r_fac = compare_facilities(data, metric)
        for r, comparison in [(r_op, 'Company'), (r_fac, 'Facility')]:
            rows.append({
                'Metric':      metric,
                'Comparison':  comparison,
                'Group A':     f"{r['label_a']}: {r['mean_a']:.1f}",
                'Group B':     f"{r['label_b']}: {r['mean_b']:.1f}",
                'U statistic': r['U_stat'],
                'p-value':     r['p_value'],
                "Cohen's d":   r['abs_cohens_d'],
                'Effect':      r['effect_label'],
                'Significant': 'Yes' if r['significant'] else 'No',
            })
    return pd.DataFrame(rows)
