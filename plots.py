"""
plots.py
--------
All figure generation functions for the Niger Delta Oil Spill ML project.
Generates all 16 publication-quality figures used in the three Springer papers.

Usage:
    from src.plots import plot_study_area, plot_eda, plot_model_comparison, ...

Authors: [Author Names]
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

BLUE   = '#1F3864'
RED    = '#C0392B'
GREEN  = '#1E8449'
ORANGE = '#D68910'
TEAL   = '#117A65'
GREY   = '#7F8C8D'
NAVY   = '#154360'
PURPLE = '#6C3483'

plt.rcParams.update({
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _save(fig, path, dpi=150):
    """Save figure and close."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: Study Area Map
# ---------------------------------------------------------------------------

def plot_study_area(df: pd.DataFrame, save_path: str = 'outputs/figures/fig1_study_area.png'):
    """
    Plot study area map with incident locations coloured by PHRI class.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_facecolor('#D6EAF8')
    fig.patch.set_facecolor('#FDFEFE')

    # Niger Delta rough outline
    delta_lon = [5.8, 6.0, 6.3, 6.6, 6.9, 7.1, 7.0, 6.8, 6.5, 6.2, 5.9, 5.8]
    delta_lat = [4.6, 4.4, 4.3, 4.3, 4.4, 4.6, 5.0, 5.3, 5.4, 5.3, 5.0, 4.6]
    ax.fill(delta_lon, delta_lat, color='#A9DFBF', alpha=0.45, zorder=1)
    ax.plot(delta_lon, delta_lat, color='#1E8449', lw=1.5, zorder=2)

    # River schematic
    rivers = [
        ([6.1, 6.25, 6.4, 6.55, 6.7], [4.65, 4.7, 4.75, 4.72, 4.68]),
        ([6.2, 6.35, 6.5, 6.65], [5.0, 4.9, 4.85, 4.8]),
        ([6.4, 6.5, 6.6, 6.7, 6.8], [5.1, 5.0, 4.9, 4.8, 4.75]),
    ]
    for rx, ry in rivers:
        ax.plot(rx, ry, color='#2980B9', lw=1.2, alpha=0.7, zorder=3)

    # LGA centroids
    lga_info = {
        'Ahoada-West': (6.47, 5.10, 273, RED),
        'Yenagoa':     (6.25, 4.93, 45,  BLUE),
        'Abua-Odual':  (6.72, 4.95, 14,  GREEN),
        'Ogba/E/N':    (6.60, 5.25, 1,   ORANGE),
    }
    for lga, (lon, lat, n, col) in lga_info.items():
        ax.scatter(lon, lat, s=max(80, n * 0.5), c=col,
                   zorder=6, edgecolors='white', linewidths=1.5, alpha=0.9)
        ax.annotate(f'{lga}\n(n={n})', (lon, lat),
                    xytext=(lon + 0.06, lat + 0.06), fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='none'))

    # Incident scatter by PHRI class
    color_map = {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E74C3C'}
    for cls, col in color_map.items():
        if 'PHRI_class' in df.columns:
            mask = df['PHRI_class'] == cls
        else:
            mask = pd.Series([True] * len(df), index=df.index)
        ax.scatter(df.loc[mask, 'Longitude'], df.loc[mask, 'Latitude'],
                   s=15, c=col, alpha=0.6, zorder=5, edgecolors='none',
                   label=f'PHRI {cls}')

    # Nigeria inset
    axins = ax.inset_axes([0.72, 0.72, 0.26, 0.26])
    axins.set_facecolor('#ECF0F1')
    axins.set_xlim(2.5, 15); axins.set_ylim(4, 14)
    ng_lon = [2.7, 4, 5, 6, 8, 10, 12, 14, 14, 12, 10, 8, 6, 4, 3, 2.7]
    ng_lat = [6,  4.5,4, 4, 4.5,4,  4,  5,  8,  12, 13, 14,13,12, 9, 6]
    axins.fill(ng_lon, ng_lat, color='#BDC3C7', alpha=0.8)
    axins.plot(ng_lon, ng_lat, color='#7F8C8D', lw=0.8)
    axins.scatter([6.4], [5.0], s=50, c=RED, zorder=5)
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title('Nigeria', fontsize=7, pad=2)

    # Legend
    leg_patches = [mpatches.Patch(color=c, label=f'PHRI {k}') for k, c in color_map.items()]
    leg_patches.append(mpatches.Patch(color='#2980B9', label='Rivers'))
    ax.legend(handles=leg_patches, loc='lower left', fontsize=9, framealpha=0.9)

    ax.set_xlim(6.0, 7.0); ax.set_ylim(4.6, 5.5)
    ax.set_xlabel('Longitude (E)', fontsize=11)
    ax.set_ylabel('Latitude (N)', fontsize=11)
    ax.set_title('Study Area: Niger Delta, Nigeria\nOil Spill Incidents 2016-2024 (n=335)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.text(0.01, 0.01, 'Source: NOSDRA Incident Database 2016-2024',
            transform=ax.transAxes, fontsize=7, color='grey')

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 2: Exploratory Data Analysis
# ---------------------------------------------------------------------------

def plot_eda(df: pd.DataFrame, save_path: str = 'outputs/figures/fig2_eda.png'):
    """6-panel EDA figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Exploratory Data Analysis - Niger Delta Oil Spill Dataset (n=335, 2016-2024)',
                 fontsize=12, fontweight='bold')

    # (a) Cause
    cause_map = {'sab': 'Sabotage', 'cor': 'Corrosion', 'eqf': 'Equip. Failure'}
    cc = df['Cause'].str.strip().map(cause_map).fillna('Other').value_counts().head(5)
    bars = axes[0, 0].bar(cc.index, cc.values, color=[RED, BLUE, GREEN, ORANGE, GREY][:len(cc)])
    axes[0, 0].bar_label(bars, fontsize=9)
    axes[0, 0].set_title('(a) Cause Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # (b) Facility type pie
    fac_map = {'pl': 'Pipeline\n(80.6%)', 'fl': 'Flowline\n(18.2%)',
               'mf': 'Manifold\n(0.6%)', 'wh': 'Wellhead\n(0.6%)'}
    fc = df['Type_of_fa'].str.strip().map(fac_map).fillna('Other').value_counts()
    axes[0, 1].pie(fc.values, labels=fc.index, autopct='',
                   colors=[BLUE, ORANGE, GREEN, RED][:len(fc)],
                   startangle=90, wedgeprops=dict(edgecolor='white', linewidth=1.5))
    axes[0, 1].set_title('(b) Facility Type', fontweight='bold')

    # (c) Company
    comp = df['Company'].value_counts()
    b = axes[0, 2].bar(comp.index, comp.values, color=[BLUE, RED])
    axes[0, 2].bar_label(b)
    axes[0, 2].set_title('(c) Incidents by Company', fontweight='bold')

    # (d) Volume distribution
    vol = df['Estimated'][df['Estimated'] > 0]
    axes[1, 0].hist(np.log1p(vol), bins=25, color=TEAL, edgecolor='white')
    axes[1, 0].axvline(np.log1p(vol.mean()), color=RED, linestyle='--', lw=2,
                       label=f'Mean={vol.mean():.1f} bbl')
    axes[1, 0].axvline(np.log1p(vol.median()), color=ORANGE, linestyle=':', lw=2,
                       label=f'Median={vol.median():.1f} bbl')
    axes[1, 0].set_title('(d) log(Volume+1) Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('log(barrels+1)')
    axes[1, 0].legend(fontsize=8)

    # (e) LGA
    lga_c = df['LGA'].str.strip().str.title().value_counts().head(5)
    axes[1, 1].barh(lga_c.index[::-1], lga_c.values[::-1], color=BLUE)
    axes[1, 1].set_title('(e) Incidents by LGA (Top 5)', fontweight='bold')
    axes[1, 1].set_xlabel('Count')

    # (f) PHRI class
    if 'PHRI_class' in df.columns:
        pc = df['PHRI_class'].value_counts().reindex(['Low', 'Medium', 'High'])
        bars2 = axes[1, 2].bar(pc.index, pc.values, color=[GREEN, ORANGE, RED])
        axes[1, 2].bar_label(bars2, fontsize=10)
        for bar, val in zip(bars2, pc.values):
            axes[1, 2].text(bar.get_x() + bar.get_width() / 2., val + 1,
                            f'{val / len(df) * 100:.1f}%',
                            ha='center', va='bottom', fontsize=9, color='#555')
    axes[1, 2].set_title('(f) PHRI Class Distribution', fontweight='bold')

    plt.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: Model Comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: dict,
    title: str = 'Model Performance Comparison',
    save_path: str = 'outputs/figures/fig_model_comparison.png'
):
    """Bar chart comparing model AUC with error bars and multi-metric panel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    names = list(results.keys())
    auc_vals = [results[k]['AUC'] for k in names]
    auc_std  = [results[k].get('AUC_std', 0) for k in names]
    bar_cols = [RED if v == max(auc_vals) else BLUE for v in auc_vals]

    bars = axes[0].bar(names, auc_vals, color=bar_cols, yerr=auc_std,
                       capsize=5, edgecolor='white', linewidth=1.5)
    axes[0].bar_label(bars, labels=[f'{v:.3f}' for v in auc_vals], padding=3, fontsize=10)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_title('ROC-AUC (Cross-Validation)', fontweight='bold')
    axes[0].set_ylabel('AUC')
    axes[0].tick_params(axis='x', rotation=20)

    metrics = ['AUC', 'Accuracy', 'F1']
    x = np.arange(len(names)); w = 0.25
    mc = [BLUE, GREEN, ORANGE]
    for i, (metric, col) in enumerate(zip(metrics, mc)):
        vals = [results[k].get(metric, 0) for k in names]
        axes[1].bar(x + i * w, vals, w, label=metric, color=col, alpha=0.85, edgecolor='white')
    axes[1].set_xticks(x + w)
    axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_ylim(0.5, 1.05)
    axes[1].set_title('Multi-Metric Comparison', fontweight='bold')
    axes[1].legend()

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: Temporal Analysis (Paper 2)
# ---------------------------------------------------------------------------

def plot_temporal(df: pd.DataFrame, save_path: str = 'outputs/figures/fig_temporal.png'):
    """4-panel temporal analysis figure for Paper 2."""
    df_sab = df[df.get('is_sabotage', df['Cause'].str.strip() == 'sab') == 1].copy()
    df_sab['Incident_d'] = pd.to_datetime(df_sab['Incident_d'])
    monthly = df_sab.set_index('Incident_d').resample('ME')['Spill_ID'].count()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Temporal Analysis of Sabotage-Induced Oil Spills (2016-2024)',
                 fontsize=12, fontweight='bold')

    # Monthly series
    axes[0, 0].plot(monthly.index, monthly.values, color=RED, lw=2)
    axes[0, 0].fill_between(monthly.index, monthly.values, alpha=0.2, color=RED)
    axes[0, 0].set_title('(a) Monthly Incident Count', fontweight='bold')
    axes[0, 0].set_ylabel('Incidents/Month')

    # Season box plot
    monthly_df = pd.DataFrame({'month': monthly.index.month, 'count': monthly.values})
    mnames = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    present = [mnames[i] for i in range(12) if i + 1 in monthly_df['month'].values]
    data_by_month = [monthly_df[monthly_df['month'] == m + 1]['count'].values
                     for m in range(12) if m + 1 in monthly_df['month'].values]
    axes[0, 1].boxplot(data_by_month, labels=present, patch_artist=True,
                       boxprops=dict(facecolor='#AED6F1', color=BLUE))
    axes[0, 1].axvspan(10.5, 12.5, alpha=0.15, color=ORANGE, label='Dry season (Nov-Dec)')
    axes[0, 1].axvspan(0.5, 2.5, alpha=0.15, color=ORANGE)
    axes[0, 1].set_title('(b) Seasonal Pattern', fontweight='bold')
    axes[0, 1].legend(fontsize=8)

    # Annual trend
    yearly = df_sab.groupby(df_sab['Incident_d'].dt.year).size().reindex(range(2016, 2025), fill_value=0)
    b = axes[1, 0].bar(yearly.index, yearly.values,
                       color=[BLUE if y < 2020 else RED for y in yearly.index])
    axes[1, 0].bar_label(b, fontsize=9)
    axes[1, 0].set_title('(c) Annual Incident Count', fontweight='bold')
    axes[1, 0].set_ylabel('Incidents/Year')

    # Dry vs Wet pie
    if 'season' not in df_sab.columns:
        df_sab['season'] = df_sab['Incident_d'].dt.month.apply(
            lambda m: 'Dry' if m in [11, 12, 1, 2, 3] else 'Wet')
    sv = df_sab['season'].value_counts()
    axes[1, 1].pie(sv.values, labels=sv.index, colors=[ORANGE, '#2980B9'],
                   autopct='%1.1f%%', startangle=90,
                   wedgeprops=dict(edgecolor='white', linewidth=2))
    axes[1, 1].set_title('(d) Dry vs Wet Season', fontweight='bold')

    plt.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: CER / RTI EDA (Paper 3)
# ---------------------------------------------------------------------------

def plot_cer_rti(df: pd.DataFrame, save_path: str = 'outputs/figures/fig_cer_rti.png'):
    """6-panel CER and RTI exploratory analysis figure for Paper 3."""
    df_cer = df[df['CER'].notna() & (df['Estimated'] > 0)].copy()
    df_rti = df[df['RTI'].notna()].copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Containment Efficiency (CER) and Response Time (RTI) - Exploratory Analysis',
                 fontsize=12, fontweight='bold')

    # (a) CER distribution
    axes[0, 0].hist(df_cer['CER'], bins=25, color=GREEN, edgecolor='white')
    axes[0, 0].axvline(df_cer['CER'].mean(), color=RED, lw=2, linestyle='--',
                       label=f'Mean={df_cer["CER"].mean():.1f}%')
    axes[0, 0].axvline(df_cer['CER'].median(), color=ORANGE, lw=2, linestyle=':',
                       label=f'Median={df_cer["CER"].median():.1f}%')
    axes[0, 0].set_title('(a) CER Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('CER (%)')
    axes[0, 0].legend(fontsize=8)

    # (b) RTI distribution
    axes[0, 1].hist(df_rti['RTI'].clip(0, 60), bins=25, color=ORANGE, edgecolor='white')
    axes[0, 1].axvline(df_rti['RTI'].mean(), color=RED, lw=2, linestyle='--',
                       label=f'Mean={df_rti["RTI"].mean():.1f}d')
    axes[0, 1].set_title('(b) RTI Distribution (clipped 0-60d)', fontweight='bold')
    axes[0, 1].set_xlabel('RTI (days)')
    axes[0, 1].legend(fontsize=8)

    # (c) CER by company
    naoc = df_cer[df_cer['Company'] == 'NAOC']['CER'].dropna()
    spdc = df_cer[df_cer['Company'] == 'SPDC']['CER'].dropna()
    axes[0, 2].boxplot([naoc, spdc], labels=['NAOC', 'SPDC'], patch_artist=True,
                       boxprops=dict(facecolor='#AED6F1'))
    _, p_co = stats.mannwhitneyu(naoc, spdc, alternative='two-sided')
    axes[0, 2].set_title(f'(c) CER by Company (p={p_co:.3f})', fontweight='bold')
    axes[0, 2].set_ylabel('CER (%)')

    # (d) CER by facility
    pl = df_cer[df_cer['Type_of_fa'] == 'pl']['CER'].dropna()
    fl = df_cer[df_cer['Type_of_fa'] == 'fl']['CER'].dropna()
    axes[1, 0].boxplot([pl, fl], labels=['Pipeline', 'Flowline'], patch_artist=True,
                       boxprops=dict(facecolor='#FDEBD0'))
    _, p_fa = stats.mannwhitneyu(pl, fl, alternative='two-sided')
    axes[1, 0].set_title(f'(d) CER by Facility (p={p_fa:.3f})', fontweight='bold')
    axes[1, 0].set_ylabel('CER (%)')

    # (e) CER vs Volume
    axes[1, 1].scatter(np.log1p(df_cer['Estimated']), df_cer['CER'],
                       alpha=0.4, s=18, color=TEAL)
    r, p_r = stats.pearsonr(np.log1p(df_cer['Estimated']), df_cer['CER'].fillna(0))
    axes[1, 1].set_title(f'(e) CER vs log(Volume) r={r:.3f}', fontweight='bold')
    axes[1, 1].set_xlabel('log(Volume+1)')
    axes[1, 1].set_ylabel('CER (%)')

    # (f) Quarterly CER trend
    df_cer['quarter'] = pd.to_datetime(df_cer['Incident_d']).dt.to_period('Q')
    qt = df_cer.groupby('quarter')['CER'].mean().reset_index()
    axes[1, 2].plot(range(len(qt)), qt['CER'], 'o-', color=GREEN, lw=2, markersize=5)
    axes[1, 2].set_title('(f) Quarterly Mean CER Trend', fontweight='bold')
    axes[1, 2].set_ylabel('Mean CER (%)')
    axes[1, 2].set_xlabel('Quarter')
    axes[1, 2].set_xticks(range(0, len(qt), 4))
    axes[1, 2].set_xticklabels(
        [str(qt['quarter'].iloc[i]) for i in range(0, len(qt), 4)], rotation=30, fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Convenience: Generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(df: pd.DataFrame, output_dir: str = 'outputs/figures'):
    """Generate all 6 core figures from the dataset."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating figures...")
    plot_study_area(df, f'{output_dir}/fig1_study_area.png')
    plot_eda(df, f'{output_dir}/fig2_eda.png')
    plot_temporal(df, f'{output_dir}/fig7_temporal.png')
    plot_cer_rti(df, f'{output_dir}/fig9_cer_rti.png')
    print("Core figures complete. Run notebooks for ML-dependent figures.")
