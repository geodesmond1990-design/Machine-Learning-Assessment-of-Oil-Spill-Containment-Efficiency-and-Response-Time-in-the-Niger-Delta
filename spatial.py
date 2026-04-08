"""
spatial.py
----------
Spatial analysis functions: KDE, DBSCAN clustering, Getis-Ord Gi-star,
and Moran I spatial autocorrelation for the Niger Delta Oil Spill ML project.

Authors: [Author Names]
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Coordinate Utilities
# ---------------------------------------------------------------------------

def coords_to_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Convert WGS84 decimal degree coordinates to approximate kilometres.

    Parameters
    ----------
    lat, lon : np.ndarray
        Latitude and longitude arrays.

    Returns
    -------
    np.ndarray of shape (n, 2)
        Coordinates in approximate km.
    """
    mean_lat = np.mean(lat)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(mean_lat))
    return np.column_stack([lat * km_per_deg_lat, lon * km_per_deg_lon])


# ---------------------------------------------------------------------------
# Kernel Density Estimation
# ---------------------------------------------------------------------------

def compute_kde(
    lat: np.ndarray,
    lon: np.ndarray,
    bandwidth: float = 0.04,
    grid_size: int = 100,
    margin: float = 0.05
) -> dict:
    """
    Compute Gaussian KDE over a regular grid.

    Parameters
    ----------
    lat, lon : np.ndarray
        Incident coordinates in decimal degrees.
    bandwidth : float
        KDE bandwidth in degrees (default 0.04 ~= 4 km).
    grid_size : int
        Number of grid points in each dimension.
    margin : float
        Degrees margin around bounding box.

    Returns
    -------
    dict with keys: LAT_grid, LON_grid, density
    """
    coords = np.column_stack([lat, lon])
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(np.radians(coords))

    lat_grid = np.linspace(lat.min() - margin, lat.max() + margin, grid_size)
    lon_grid = np.linspace(lon.min() - margin, lon.max() + margin, grid_size)
    LAT, LON = np.meshgrid(lat_grid, lon_grid)

    grid_pts = np.column_stack([LAT.ravel(), LON.ravel()])
    log_density = kde.score_samples(np.radians(grid_pts))
    density = np.exp(log_density).reshape(LAT.shape)

    return {'LAT_grid': LAT, 'LON_grid': LON, 'density': density}


# ---------------------------------------------------------------------------
# DBSCAN Clustering
# ---------------------------------------------------------------------------

def run_dbscan(
    lat: np.ndarray,
    lon: np.ndarray,
    eps_km: float = 0.5,
    min_samples: int = 5
) -> dict:
    """
    Run DBSCAN on incident coordinates.

    Parameters
    ----------
    lat, lon : np.ndarray
        Incident coordinates.
    eps_km : float
        Neighbourhood radius in km.
    min_samples : int
        Minimum points to form a core point.

    Returns
    -------
    dict with keys: labels, n_clusters, noise_count, clustered_pct, silhouette
    """
    coords_km = coords_to_km(lat, lon)

    db = DBSCAN(eps=eps_km, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(coords_km)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = (labels == -1).sum()
    clustered_pct = (labels != -1).mean() * 100

    # Silhouette score (requires > 1 cluster and > noise)
    sil = None
    if n_clusters >= 2:
        try:
            from sklearn.metrics import silhouette_score
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = silhouette_score(coords_km[mask], labels[mask])
        except Exception:
            pass

    return {
        'labels':         labels,
        'n_clusters':     n_clusters,
        'noise_count':    noise_count,
        'clustered_pct':  clustered_pct,
        'silhouette':     sil,
    }


def characterise_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    volume_col: str = 'Estimated',
    company_col: str = 'Company',
    facility_col: str = 'Type_of_fa'
) -> pd.DataFrame:
    """
    Return a summary table characterising each DBSCAN cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Original incident dataframe.
    labels : np.ndarray
        DBSCAN cluster labels.

    Returns
    -------
    pd.DataFrame
        Cluster summary table.
    """
    df = df.copy()
    df['cluster'] = labels
    clustered = df[df['cluster'] >= 0]

    summary = clustered.groupby('cluster').agg(
        n_incidents=(volume_col, 'count'),
        mean_vol=(volume_col, 'mean'),
        max_vol=(volume_col, 'max'),
        centroid_lat=('Latitude', 'mean'),
        centroid_lon=('Longitude', 'mean'),
        dominant_company=(company_col, lambda x: x.mode()[0]),
        dominant_facility=(facility_col, lambda x: x.mode()[0])
    ).sort_values('n_incidents', ascending=False).reset_index()

    summary['pct_of_total'] = (summary['n_incidents'] / len(df) * 100).round(1)
    return summary


def kdistance_plot(lat: np.ndarray, lon: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Compute the k-distance array for DBSCAN epsilon selection.

    Returns the sorted (descending) distances to the k-th nearest neighbour
    for each point, which can be plotted to identify the elbow.
    """
    coords_km = coords_to_km(lat, lon)
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords_km)
    distances, _ = nbrs.kneighbors(coords_km)
    k_dists = np.sort(distances[:, k - 1])[::-1]
    return k_dists


# ---------------------------------------------------------------------------
# Getis-Ord Gi-star
# ---------------------------------------------------------------------------

def getis_ord_gistar(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    bandwidth_km: float = 3.0
) -> np.ndarray:
    """
    Compute Getis-Ord Gi-star statistic for each point.

    Parameters
    ----------
    lat, lon : np.ndarray
        Coordinates of observations.
    values : np.ndarray
        Attribute values (e.g. log-transformed spill volume).
    bandwidth_km : float
        Search radius in km for spatial weight construction.

    Returns
    -------
    np.ndarray
        Gi-star z-score for each point.
        z > 2.58  => hotspot  (p < 0.01)
        z < -2.58 => coldspot (p < 0.01)
    """
    coords_km = coords_to_km(lat, lon)
    n = len(values)
    x_mean = values.mean()
    x_std  = values.std()
    S = np.sqrt(np.mean(values**2) - x_mean**2)

    # Euclidean distance matrix
    dist_matrix = cdist(coords_km, coords_km, metric='euclidean')
    W = (dist_matrix <= bandwidth_km).astype(float)

    gi_stars = np.zeros(n)
    for i in range(n):
        w_sum  = W[i].sum()
        w2_sum = (W[i]**2).sum()
        num = np.dot(W[i], values) - x_mean * w_sum
        denom_sq = (n * w2_sum - w_sum**2) / (n - 1)
        denom = S * np.sqrt(denom_sq) if denom_sq > 0 else 1e-10
        gi_stars[i] = num / denom

    return gi_stars


def classify_hotspots(gi_stars: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Classify Gi-star z-scores into hotspot / not significant / coldspot.

    Parameters
    ----------
    gi_stars : np.ndarray
        Gi-star z-scores.
    alpha : float
        Significance level. 0.01 => z_crit = 2.576; 0.05 => z_crit = 1.96.

    Returns
    -------
    np.ndarray of str
        Array of labels: 'Hot Spot', 'Not Significant', 'Cold Spot'.
    """
    z_crit = 2.576 if alpha <= 0.01 else 1.96
    labels = np.where(
        gi_stars > z_crit, 'Hot Spot',
        np.where(gi_stars < -z_crit, 'Cold Spot', 'Not Significant')
    )
    return labels


# ---------------------------------------------------------------------------
# Moran I Spatial Autocorrelation
# ---------------------------------------------------------------------------

def moran_i(
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    k_neighbours: int = 5,
    n_permutations: int = 999
) -> dict:
    """
    Compute Moran I statistic with permutation-based significance testing.

    Parameters
    ----------
    values : np.ndarray
        Attribute values (e.g. PHRI scores).
    lat, lon : np.ndarray
        Spatial coordinates.
    k_neighbours : int
        Number of nearest neighbours for weight matrix construction.
    n_permutations : int
        Number of random permutations for p-value estimation.

    Returns
    -------
    dict with keys: I, E_I, Var_I, z_score, p_value_norm, p_value_perm
    """
    coords_km = coords_to_km(lat, lon)
    n = len(values)

    # Build k-nearest-neighbour weight matrix
    nbrs = NearestNeighbors(n_neighbors=k_neighbours + 1).fit(coords_km)
    _, indices = nbrs.kneighbors(coords_km)
    W = np.zeros((n, n))
    for i, row in enumerate(indices):
        for j in row[1:]:  # Exclude self
            W[i, j] = 1
            W[j, i] = 1

    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W_std = W / row_sums

    # Moran I
    z = values - values.mean()
    S0 = W_std.sum()
    I = (n / S0) * (z @ W_std @ z) / (z @ z)

    # Expected value and variance under normality
    E_I = -1 / (n - 1)

    # Permutation p-value
    perm_I = np.array([
        (n / S0) * (np.random.permutation(z) @ W_std @ np.random.permutation(z)) / (z @ z)
        for _ in range(n_permutations)
    ])
    p_perm = np.mean(np.abs(perm_I) >= np.abs(I))

    return {
        'I':             round(float(I), 4),
        'E_I':           round(float(E_I), 4),
        'p_value_perm':  round(float(p_perm), 4),
        'n_permutations': n_permutations,
    }
