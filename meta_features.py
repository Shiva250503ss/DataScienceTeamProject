"""
Meta-Feature Extraction — Single source of truth for 40 dataset meta-features.

MUST produce IDENTICAL output to the extract_meta_features() in:
    RL_MODEL_PPO_CORRECT/DataScienceTeamProject/train_rl_model_selector.py

Feature layout (40 total):
    [0-5]    Basic (6)       : shape, type ratios, dimensionality
    [6-8]    Missing (3)     : patterns of missing data
    [9-18]   Statistical (10): distribution properties of numeric features
    [19-21]  Categorical (3) : cardinality of categorical columns
    [22-24]  Target (3)      : properties of the target variable
    [25-27]  PCA (3)         : intrinsic dimensionality via PCA
    [28-31]  Landmarks (4)   : quick 3-fold CV scores with simple models
    [32-39]  Signal (8)      : sparsity, linear signal, nonlinearity, class count
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


N_META_FEATURES = 40


def extract_meta_features(X, y, task_type='classification'):
    """
    Compute all 40 meta-features from the dataset.
    All values are normalised to [0, 1] to match the observation space.
    """
    features = []

    # -- Prepare data ----------------------------------------------------------
    if isinstance(X, pd.DataFrame):
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        miss_col = X.isnull().mean()
        total_missing = float(X.isnull().mean().mean())
        cols_with_miss = float((miss_col > 0).mean())
        max_missing = float(miss_col.max())
        X_num = X[num_cols].values if num_cols else np.zeros((len(X), 1))
    else:
        X = np.asarray(X, dtype=float)
        X_num = X
        cat_cols = []
        num_cols = list(range(X.shape[1]))
        nan_mask = np.isnan(X_num)
        total_missing = float(nan_mask.mean())
        cols_with_miss = float(np.any(nan_mask, axis=0).mean())
        max_missing = float(nan_mask.mean(axis=0).max())

    n_samples, n_features_total = X_num.shape
    n_numeric = len(num_cols) if num_cols else X_num.shape[1]
    n_categorical = len(cat_cols)
    X_num = np.nan_to_num(X_num.astype(float), nan=0.0)

    # -- BASIC (6) [0-5] ------------------------------------------------------
    features.append(float(min(n_samples / 100_000, 1.0)))            # 0
    features.append(float(min(n_features_total / 100, 1.0)))         # 1
    features.append(float(n_numeric / max(n_features_total, 1)))     # 2
    features.append(float(n_categorical / max(n_features_total, 1))) # 3
    features.append(float(min(len(np.unique(y)) / max(n_samples, 1), 1.0)))  # 4
    features.append(float(min(n_features_total / max(n_samples, 1), 1.0)))   # 5

    # -- MISSING (3) [6-8] ----------------------------------------------------
    features.append(float(min(total_missing, 1.0)))    # 6
    features.append(float(min(cols_with_miss, 1.0)))   # 7
    features.append(float(min(max_missing, 1.0)))      # 8

    # -- STATISTICAL (10) [9-18] -----------------------------------------------
    n_cols = min(n_numeric, 50)
    Xs = X_num[:, :n_cols]
    col_std = np.std(Xs, axis=0) + 1e-10
    col_mean = np.mean(Xs, axis=0)

    col_skew = np.array([stats.skew(Xs[:, i]) for i in range(n_cols)])
    col_kurt = np.array([stats.kurtosis(Xs[:, i]) for i in range(n_cols)])

    # 9. Mean |skewness| (normalise by 10)
    features.append(float(min(np.mean(np.abs(col_skew)) / 10.0, 1.0)))
    # 10. Mean |kurtosis| (normalise by 50)
    features.append(float(min(np.mean(np.abs(col_kurt)) / 50.0, 1.0)))
    # 11. Outlier ratio: fraction of rows with any |z-score| > 3
    z_scores = np.abs((Xs - col_mean) / col_std)
    features.append(float(np.mean(np.any(z_scores > 3, axis=1))))
    # 12. Mean absolute pairwise correlation
    if n_cols > 1:
        corr = np.corrcoef(Xs.T)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[~np.isnan(upper)]
        features.append(float(np.mean(np.abs(upper))) if len(upper) else 0.0)
    else:
        features.append(0.0)
    # 13. Mean coefficient of variation
    cv = col_std / (np.abs(col_mean) + 1e-10)
    features.append(float(min(np.mean(cv), 1.0)))
    # 14. Std of skewness values (normalise by 10)
    features.append(float(min(np.std(col_skew) / 10.0, 1.0)))
    # 15. Std of kurtosis values (normalise by 50)
    features.append(float(min(np.std(col_kurt) / 50.0, 1.0)))
    # 16. Range ratio: mean (max-min) / std per column (normalise by 20)
    col_range = np.max(Xs, axis=0) - np.min(Xs, axis=0)
    features.append(float(min(np.mean(col_range / col_std) / 20.0, 1.0)))
    # 17. Zero ratio
    features.append(float(np.mean(Xs == 0)))
    # 18. Class balance (classification) or target CV (regression)
    y_arr = np.asarray(y, dtype=float)
    if task_type == 'classification':
        _, counts = np.unique(y_arr, return_counts=True)
        features.append(float(counts.min() / counts.max()))
    else:
        y_cv = np.std(y_arr) / (np.abs(np.mean(y_arr)) + 1e-10)
        features.append(float(min(y_cv / 10.0, 1.0)))

    # -- CATEGORICAL (3) [19-21] -----------------------------------------------
    if cat_cols and isinstance(X, pd.DataFrame):
        cards = [X[c].nunique() / n_samples for c in cat_cols]
        features.append(float(min(np.mean(cards), 1.0)))  # 19
        features.append(float(min(np.max(cards), 1.0)))   # 20
        features.append(float(np.mean([c > 0.05 for c in cards])))  # 21
    else:
        features.extend([0.0, 0.0, 0.0])

    # -- TARGET (3) [22-24] ----------------------------------------------------
    # 22. Target |skewness| (normalise by 10)
    features.append(float(min(abs(stats.skew(y_arr)) / 10.0, 1.0)))
    # 23. Target |kurtosis| (normalise by 50)
    features.append(float(min(abs(stats.kurtosis(y_arr)) / 50.0, 1.0)))
    # 24. Target entropy (classification) or target CV (regression)
    if task_type == 'classification':
        _, counts = np.unique(y_arr, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_ent = np.log(max(len(counts), 2))
        features.append(float(entropy / max_ent))
    else:
        tgt_cv = np.std(y_arr) / (np.abs(np.mean(y_arr)) + 1e-10)
        features.append(float(min(tgt_cv / 10.0, 1.0)))

    # -- PCA (3) [25-27] -------------------------------------------------------
    try:
        Xs_scaled = StandardScaler().fit_transform(Xs)
        n_pca = min(n_cols, n_samples - 1, 50)
        pca = PCA(n_components=n_pca, random_state=42)
        pca.fit(Xs_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        # 25. Normalised # components for 95% variance
        n95 = int(np.searchsorted(cumvar, 0.95)) + 1
        features.append(float(n95 / n_pca))
        # 26. Variance explained by the first 50% of components
        half = max(1, n_pca // 2)
        features.append(float(cumvar[half - 1]))
        # 27. Intrinsic dim: components for 50% variance (normalised)
        n50 = int(np.searchsorted(cumvar, 0.50)) + 1
        features.append(float(n50 / n_pca))
    except Exception:
        features.extend([0.5, 0.5, 0.5])

    # -- LANDMARKS (4) [28-31] ------------------------------------------------
    Xl = StandardScaler().fit_transform(Xs)

    def _lm(estimator, scoring):
        try:
            cv_obj = KFold(n_splits=3, shuffle=True, random_state=0)
            sc = cross_val_score(estimator, Xl, y, cv=cv_obj,
                                 scoring=scoring, error_score=0.0)
            return float(np.clip(np.mean(sc), 0.0, 1.0))
        except Exception:
            return 0.5

    if task_type == 'classification':
        sc = 'accuracy'
        lm_dt = _lm(DecisionTreeClassifier(max_depth=3, random_state=42), sc)   # 28
        lm_nb = _lm(GaussianNB(), sc)                                            # 29
        lm_lr = _lm(LogisticRegression(max_iter=1000, random_state=42), sc)      # 30
        lm_knn = _lm(KNeighborsClassifier(n_neighbors=3), sc)                    # 31
        features.extend([lm_dt, lm_nb, lm_lr, lm_knn])
    else:
        sc = 'r2'
        lm_dt = _lm(DecisionTreeRegressor(max_depth=3, random_state=42), sc)     # 28
        lm_lr = _lm(Ridge(alpha=1.0), sc)                                        # 29
        lm_knn = _lm(KNeighborsRegressor(n_neighbors=3), sc)                     # 30
        lm_dt2 = _lm(DecisionTreeRegressor(max_depth=5, random_state=42), sc)    # 31
        lm_nb = lm_dt2
        features.extend([lm_dt, lm_lr, lm_knn, lm_dt2])

    # -- SIGNAL (8) [32-39] ---------------------------------------------------

    # 32. Sparsity ratio
    col_ranges = np.ptp(Xs, axis=0) + 1e-10
    near_zero = np.abs(Xs) < (0.05 * col_ranges)
    features.append(float(np.mean(near_zero)))

    # 33. Mean |feature-target correlation|
    y_arr2 = np.asarray(y, dtype=float)
    y_std = float(np.std(y_arr2)) + 1e-10
    Xs_std = np.std(Xs, axis=0) + 1e-10
    ftcorr = np.abs(np.dot((Xs - Xs.mean(0)).T, y_arr2 - y_arr2.mean()) /
                    (n_samples * Xs_std * y_std))
    features.append(float(np.clip(np.mean(ftcorr), 0.0, 1.0)))

    # 34. Max |feature-target correlation|
    features.append(float(np.clip(np.max(ftcorr), 0.0, 1.0)))

    # 35. Std of feature-target correlations
    features.append(float(np.clip(np.std(ftcorr) * 5.0, 0.0, 1.0)))

    # 36. Nonlinearity gap: DT_landmark - LR_landmark
    nl_gap = float(np.clip((lm_dt - lm_lr + 1.0) / 2.0, 0.0, 1.0))
    features.append(nl_gap)

    # 37. Explicit class count (normalised by 20)
    n_classes_raw = len(np.unique(y)) if task_type == 'classification' else 1
    features.append(float(min(n_classes_raw / 20.0, 1.0)))

    # 38. KNN consistency score
    knn_advantage = float(np.clip((lm_knn - lm_lr + 1.0) / 2.0, 0.0, 1.0))
    features.append(knn_advantage)

    # 39. Feature density
    max_std = float(np.max(Xs_std)) + 1e-10
    active_feats = float(np.mean(Xs_std > 0.1 * max_std))
    features.append(active_feats)

    assert len(features) == 40, f"Feature count error: got {len(features)}, expected 40"
    arr = np.array(features, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr
