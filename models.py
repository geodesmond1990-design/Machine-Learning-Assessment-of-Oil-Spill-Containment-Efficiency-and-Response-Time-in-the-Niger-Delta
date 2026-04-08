"""
models.py
---------
Model training, cross-validation, evaluation, and persistence
for the Niger Delta Oil Spill ML project.

Usage:
    from src.models import train_classifiers, train_regressors, evaluate_clf

Authors: [Author Names]
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                               RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not installed. SMOTE will not be applied.")

import joblib


# ---------------------------------------------------------------------------
# Classifier definitions
# ---------------------------------------------------------------------------

def get_classifiers(paper: int = 1) -> dict:
    """
    Return a dictionary of classifiers for the specified paper.

    Parameters
    ----------
    paper : int
        1 = PHRI classification (Papers 1)
        2 = Sabotage risk classification (Paper 2)
        3 = CER classification (Paper 3)
    """
    if paper in [1, 3]:
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=500, random_state=42, C=0.5),
            'Random Forest': RandomForestClassifier(
                n_estimators=300, max_depth=7, min_samples_leaf=5,
                random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=42),
            'SVM': SVC(
                kernel='rbf', C=5, gamma='scale',
                probability=True, random_state=42),
            'ANN': MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu',
                max_iter=300, early_stopping=True, random_state=42),
        }
    elif paper == 2:
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=500, random_state=42, C=1.0),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                random_state=42),
        }
    else:
        return get_classifiers(paper=1)


def get_regressors() -> dict:
    """Return regression models for RTI prediction (Paper 3)."""
    return {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    }


# ---------------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------------

def train_classifiers(
    X: np.ndarray,
    y: pd.Series,
    paper: int = 1,
    n_splits: int = 5,
    use_smote: bool = True,
    verbose: bool = True
) -> dict:
    """
    Train and evaluate classifiers using stratified K-fold cross-validation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix (pre-scaled or will be scaled inside each fold).
    y : array-like of shape (n_samples,)
        Target class labels.
    paper : int
        Which paper's classifier set to use.
    n_splits : int
        Number of CV folds.
    use_smote : bool
        Whether to apply SMOTE within each training fold.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Per-model performance dictionary with keys:
        AUC, AUC_std, Accuracy, Precision, Recall, F1.
    """
    clfs = get_classifiers(paper)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = RobustScaler()

    results = {}
    for name, clf in clfs.items():
        if verbose:
            print(f"  Training {name}...", end=' ')
        aucs, accs, precs, recs, f1s = [], [], [], [], []

        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        for fold_idx, (tr, te) in enumerate(cv.split(X_arr, y_arr)):
            X_tr, X_te = X_arr[tr], X_arr[te]
            y_tr, y_te = y_arr[tr], y_arr[te]

            # Scale within fold (prevent leakage)
            sc = RobustScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            # SMOTE on training fold only
            if use_smote and HAS_SMOTE:
                try:
                    sm = SMOTE(random_state=42, k_neighbors=3)
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                except Exception:
                    pass  # Skip SMOTE if too few samples per class

            clf.fit(X_tr, y_tr)
            y_pred  = clf.predict(X_te)
            y_proba = clf.predict_proba(X_te)

            n_classes = len(np.unique(y_arr))
            if n_classes == 2:
                auc = roc_auc_score(y_te, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_te, y_proba, multi_class='ovr',
                                    labels=sorted(np.unique(y_arr)))

            aucs.append(auc)
            accs.append(accuracy_score(y_te, y_pred))
            precs.append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
            recs.append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
            f1s.append(f1_score(y_te, y_pred, average='weighted', zero_division=0))

        results[name] = {
            'AUC':       np.mean(aucs),
            'AUC_std':   np.std(aucs),
            'Accuracy':  np.mean(accs),
            'Precision': np.mean(precs),
            'Recall':    np.mean(recs),
            'F1':        np.mean(f1s),
        }
        if verbose:
            print(f"AUC={results[name]['AUC']:.4f} +/- {results[name]['AUC_std']:.4f}  "
                  f"Acc={results[name]['Accuracy']:.4f}  F1={results[name]['F1']:.4f}")

    return results


def train_regressors(
    X: np.ndarray,
    y: pd.Series,
    n_splits: int = 5,
    verbose: bool = True
) -> dict:
    """
    Train and evaluate regression models using K-fold cross-validation.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Continuous target (RTI_log).
    n_splits : int
        Number of folds.

    Returns
    -------
    dict
        Per-model dictionary with RMSE, MAE, R2.
    """
    regs = get_regressors()
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for name, reg in regs.items():
        if verbose:
            print(f"  Training {name}...", end=' ')
        rmses, maes, r2s = [], [], []

        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        for tr, te in cv.split(X_arr):
            X_tr, X_te = X_arr[tr], X_arr[te]
            y_tr, y_te = y_arr[tr], y_arr[te]

            sc = RobustScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            reg.fit(X_tr, y_tr)
            y_pred = reg.predict(X_te)

            rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
            maes.append(mean_absolute_error(y_te, y_pred))
            r2s.append(r2_score(y_te, y_pred))

        results[name] = {
            'RMSE': np.mean(rmses),
            'MAE':  np.mean(maes),
            'R2':   np.mean(r2s),
        }
        if verbose:
            print(f"RMSE={results[name]['RMSE']:.4f}  R2={results[name]['R2']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Final model training and evaluation on held-out test set
# ---------------------------------------------------------------------------

def train_final_model(
    X: np.ndarray,
    y: pd.Series,
    model_name: str = 'Gradient Boosting',
    paper: int = 1,
    test_size: float = 0.2,
    use_smote: bool = True,
    save_path: str = None
):
    """
    Train the best model on a train/test split and return all evaluation artefacts.

    Returns
    -------
    dict with keys: model, scaler, X_test, y_test, y_pred, y_proba, report
    """
    X_arr = X.values if hasattr(X, 'values') else np.array(X)
    y_arr = y.values if hasattr(y, 'values') else np.array(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y_arr, test_size=test_size,
        stratify=y_arr, random_state=42
    )

    sc = RobustScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    if use_smote and HAS_SMOTE:
        try:
            sm = SMOTE(random_state=42, k_neighbors=3)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except Exception:
            pass

    clfs = get_classifiers(paper)
    clf = clfs[model_name]
    clf.fit(X_tr, y_tr)

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)

    report = classification_report(y_te, y_pred, output_dict=True)
    print(f"\nFinal model: {model_name}")
    print(classification_report(y_te, y_pred))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({'model': clf, 'scaler': sc}, save_path)
        print(f"Model saved to {save_path}")

    return {
        'model':   clf,
        'scaler':  sc,
        'X_test':  X_te,
        'y_test':  y_te,
        'y_pred':  y_pred,
        'y_proba': y_proba,
        'report':  report,
    }


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert the results dictionary from train_classifiers into a DataFrame."""
    df = pd.DataFrame(results).T
    df = df.sort_values('AUC', ascending=False)
    return df.round(4)


def print_results_table(results: dict, title: str = 'Model Comparison'):
    """Print a formatted results table."""
    df = results_to_dataframe(results)
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(df.to_string())
    print('='*65)
