"""
Profiler Agent - Dataset Analysis and Meta-feature Extraction
Full implementation matching DataPilot specification (Section 5.1)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
import re
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect column types using rule-based approach.
    Returns dict with keys: numeric, categorical, datetime, text, id
    """
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'id': []
    }

    for col in df.columns:
        col_data = df[col]

        # 1. Numeric Detection: dtype in [int, float] and >95% valid numbers
        if pd.api.types.is_numeric_dtype(col_data):
            valid_ratio = col_data.notna().sum() / len(col_data)
            if valid_ratio > 0.95:
                # Check if it's actually an ID column
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio == 1.0:
                    # All unique - likely ID
                    column_types['id'].append(col)
                elif col_data.nunique() <= 20 and unique_ratio < 0.05:
                    # Low cardinality numeric - treat as categorical
                    column_types['categorical'].append(col)
                else:
                    column_types['numeric'].append(col)
                continue

        # 2. DateTime Detection: Attempt pd.to_datetime(); if >80% success, mark as datetime
        if col_data.dtype == 'object':
            try:
                converted = pd.to_datetime(col_data, errors='coerce')
                success_ratio = converted.notna().sum() / len(col_data)
                if success_ratio > 0.8:
                    column_types['datetime'].append(col)
                    continue
            except:
                pass

        # 3. ID Detection: 100% unique values and sequential/UUID patterns
        unique_ratio = col_data.nunique() / len(col_data)
        if unique_ratio == 1.0:
            # Check for sequential or UUID patterns
            sample = col_data.dropna().head(10).astype(str)
            # UUID pattern
            uuid_pattern = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', re.I)
            if any(uuid_pattern.match(str(x)) for x in sample):
                column_types['id'].append(col)
                continue
            # Sequential pattern (e.g., ID_001, ID_002)
            if all(str(x).isdigit() or re.match(r'^\w+_?\d+$', str(x)) for x in sample):
                column_types['id'].append(col)
                continue

        # 4. Text Detection: String columns with mean word count >5 (free-form text)
        if col_data.dtype == 'object':
            try:
                word_counts = col_data.dropna().astype(str).apply(lambda x: len(str(x).split()))
                mean_words = word_counts.mean()
                if mean_words > 5:
                    column_types['text'].append(col)
                    continue
            except:
                pass

        # 5. Categorical Detection: Object/string columns OR numeric with <=20 unique and <5% unique ratio
        if col_data.dtype in ['object', 'category']:
            column_types['categorical'].append(col)
        elif unique_ratio < 0.05 and col_data.nunique() <= 20:
            column_types['categorical'].append(col)

    return column_types


def detect_target_column(df: pd.DataFrame, target_col: str = None) -> str:
    """
    Detect target column using heuristics.
    1. Use provided target_col if specified
    2. Look for columns named 'target', 'label', 'y'
    3. Use last column
    4. For classification: column with lowest cardinality
    """
    if target_col and target_col in df.columns:
        return target_col

    # Check for common target names
    common_names = ['target', 'label', 'y', 'class', 'output']
    for name in common_names:
        if name in df.columns:
            return name

    # Use last column as default
    return df.columns[-1]


def detect_task_type(y: pd.Series) -> str:
    """Detect if task is classification or regression."""
    if pd.api.types.is_numeric_dtype(y):
        unique_count = y.nunique()
        # Heuristic: if unique values <= sqrt(n) or < 20, it's classification
        if unique_count <= np.sqrt(len(y)) or unique_count < 20:
            return 'classification'
        return 'regression'
    return 'classification'


def calculate_quality_score(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> float:
    """
    Calculate overall data quality score (0-100).
    Factors: missing values, duplicates, outliers, type consistency
    """
    score = 100.0

    # Penalize for missing values (up to -30 points)
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    score -= min(missing_ratio * 100, 30)

    # Penalize for duplicates (up to -20 points)
    dup_ratio = df.duplicated().sum() / len(df)
    score -= min(dup_ratio * 100, 20)

    # Penalize for too many ID columns (up to -10 points)
    id_ratio = len(column_types['id']) / max(df.shape[1], 1)
    score -= min(id_ratio * 50, 10)

    # Penalize for outliers in numeric columns (up to -20 points)
    numeric_cols = column_types['numeric']
    if numeric_cols:
        outlier_ratios = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            outlier_ratios.append(outliers / len(df))
        avg_outlier_ratio = np.mean(outlier_ratios)
        score -= min(avg_outlier_ratio * 100, 20)

    # Bonus for having datetime columns (data richness)
    if column_types['datetime']:
        score += 5

    return max(0, min(100, score))


def detect_warnings(df: pd.DataFrame, column_types: Dict[str, List[str]], meta_features: np.ndarray) -> List[str]:
    """
    Detect potential data issues and return warnings.
    """
    warnings = []

    # High missing values
    if meta_features[6] > 0.3:  # missing_ratio
        warnings.append(f"High missing values: {meta_features[6]*100:.1f}% of data is missing")

    # Many duplicates
    dup_count = df.duplicated().sum()
    if dup_count > len(df) * 0.1:
        warnings.append(f"High duplicate ratio: {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)")

    # High class imbalance
    if meta_features[18] > 0.8:  # imbalance_ratio
        warnings.append(f"Severe class imbalance detected: {meta_features[18]*100:.1f}%")

    # Too many ID columns
    if len(column_types['id']) > df.shape[1] * 0.2:
        warnings.append(f"Many ID columns detected: {len(column_types['id'])} columns appear to be IDs")

    # High dimensionality
    if meta_features[5] > 0.5:  # dimensionality
        warnings.append(f"High dimensionality: {df.shape[1]} features for {df.shape[0]} samples")

    # Too many categorical features
    if len(column_types['categorical']) > len(column_types['numeric']):
        warnings.append(f"More categorical ({len(column_types['categorical'])}) than numeric ({len(column_types['numeric'])}) features")

    # High cardinality categorical features
    for col in column_types['categorical']:
        cardinality = df[col].nunique()
        if cardinality > 50:
            warnings.append(f"High cardinality in '{col}': {cardinality} unique values")

    # Outliers
    if meta_features[11] > 0.15:  # outlier_ratio
        warnings.append(f"High outlier ratio: {meta_features[11]*100:.1f}% of values are outliers")

    return warnings


class ProfilerAgent:
    """Analyzes dataset and extracts meta-features with full specification compliance."""

    def analyze(self, df: pd.DataFrame, target_col: str = None) -> Tuple[Dict, np.ndarray, str, str, float, List[str]]:
        """
        Analyze dataset and extract 32 meta-features.

        Args:
            df: Input DataFrame
            target_col: Target column name (optional, will be detected if not provided)

        Returns:
            (profile_report, meta_features, detected_target, task_type, quality_score, warnings)
        """
        # Detect target column
        detected_target = detect_target_column(df, target_col)

        if detected_target not in df.columns:
            raise ValueError(f"Target column '{detected_target}' not found in dataframe")

        X = df.drop(columns=[detected_target])
        y = df[detected_target]

        # Detect task type
        task_type = detect_task_type(y)

        # Detect column types
        column_types = detect_column_types(X)

        # Extract meta-features
        meta = self._extract_meta_features(X, y, task_type, column_types)

        # Calculate quality score
        quality_score = calculate_quality_score(df, column_types)

        # Detect warnings
        warnings = detect_warnings(df, column_types, meta)

        # Build profile report
        report = {
            'dataset_shape': df.shape,
            'task_type': task_type,
            'numeric_cols': len(column_types['numeric']),
            'categorical_cols': len(column_types['categorical']),
            'datetime_cols': len(column_types['datetime']),
            'text_cols': len(column_types['text']),
            'id_cols': len(column_types['id']),
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'duplicates': df.duplicated().sum(),
            'quality_score': quality_score,
            'column_types': column_types,
        }

        return report, meta, detected_target, task_type, quality_score, warnings

    def _extract_meta_features(self, X: pd.DataFrame, y: pd.Series, task_type: str, column_types: Dict) -> np.ndarray:
        """Extract all 32 meta-features as per report specification."""
        features = []

        n_samples, n_features = X.shape
        numeric_cols = column_types['numeric']
        categorical_cols = column_types['categorical']

        # ==================== BASIC FEATURES (6) ====================
        features.append(min(n_samples / 100000, 1.0))  # n_samples (normalized)
        features.append(min(n_features / 100, 1.0))   # n_features (normalized)
        features.append(len(numeric_cols) / max(n_features, 1))  # n_numeric (ratio)
        features.append(len(categorical_cols) / max(n_features, 1))  # n_categorical (ratio)
        features.append(min(y.nunique() / n_samples, 1.0))  # n_classes/target_unique
        features.append(min(n_features / n_samples, 1.0))  # dimensionality

        # ==================== MISSING VALUE FEATURES (3) ====================
        missing_counts = X.isnull().sum()
        missing_percentages = (missing_counts / n_samples)
        missing_ratio = (missing_counts.sum() / (n_samples * n_features))
        features.append(np.clip(missing_ratio, 0, 1))  # missing_ratio
        cols_with_missing = (missing_counts > 0).sum()
        features.append(min(cols_with_missing / n_features, 1.0))  # cols_with_missing (ratio)
        max_missing = missing_percentages.max() if len(missing_percentages) > 0 else 0
        features.append(np.clip(max_missing, 0, 1))  # max_missing_percent

        # ==================== STATISTICAL FEATURES (10) ====================
        if len(numeric_cols) > 0:
            X_num = X[numeric_cols].fillna(X[numeric_cols].mean())

            # Skewness metrics
            skewness_vals = [abs(X_num[col].skew()) for col in numeric_cols]
            features.append(np.clip(np.mean(skewness_vals) if skewness_vals else 0.0, 0, 1))  # mean_skewness

            # Kurtosis metrics
            kurtosis_vals = [X_num[col].kurtosis() for col in numeric_cols]
            features.append(np.clip(np.mean(kurtosis_vals) if kurtosis_vals else 0.0, 0, 1))  # mean_kurtosis

            # Outlier ratio (IQR method)
            outlier_ratios = []
            for col in numeric_cols:
                Q1 = X_num[col].quantile(0.25)
                Q3 = X_num[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((X_num[col] < Q1 - 1.5*IQR) | (X_num[col] > Q3 + 1.5*IQR)).sum()
                outlier_ratios.append(outliers / len(X_num))
            features.append(np.clip(np.mean(outlier_ratios) if outlier_ratios else 0.0, 0, 1))  # outlier_ratio

            # Mean correlation
            try:
                corr_matrix = X_num.corr().values
                upper_corr = np.abs(np.triu(corr_matrix, k=1))
                valid_corr = upper_corr[upper_corr > 0]
                features.append(np.clip(np.mean(valid_corr) if len(valid_corr) > 0 else 0.0, 0, 1))  # mean_correlation
            except:
                features.append(0.0)

            # Coefficient of variation (CV) metrics
            cv_vals = [X_num[col].std() / (abs(X_num[col].mean()) + 1e-10) for col in numeric_cols]
            features.append(np.clip(np.mean(cv_vals) if cv_vals else 0.0, 0, 1))  # cv_mean
            features.append(np.clip(np.std(cv_vals) if len(cv_vals) > 1 else 0.0, 0, 1))  # cv_std

            # Skewness and Kurtosis standard deviations
            features.append(np.clip(np.std(skewness_vals) if len(skewness_vals) > 1 else 0.0, 0, 1))  # skew_std
            features.append(np.clip(np.std(kurtosis_vals) if len(kurtosis_vals) > 1 else 0.0, 0, 1))  # kurt_std

            # Range ratio (normalized range)
            range_ratios = []
            for col in numeric_cols:
                col_range = X_num[col].max() - X_num[col].min()
                col_mean = abs(X_num[col].mean())
                range_ratios.append(col_range / (col_mean + 1e-10))
            features.append(np.clip(np.mean(range_ratios) if range_ratios else 0.0, 0, 1))  # range_ratio

            # Zero ratio
            zero_ratios = [(X_num[col] == 0).sum() / len(X_num) for col in numeric_cols]
            features.append(np.clip(np.mean(zero_ratios) if zero_ratios else 0.0, 0, 1))  # zero_ratio
        else:
            features.extend([0.0] * 10)  # All 10 statistical features

        # ==================== CATEGORICAL FEATURES (3) ====================
        if len(categorical_cols) > 0:
            cardinalities = [X[col].nunique() for col in categorical_cols]
            features.append(min(np.mean(cardinalities) / 100, 1.0))  # mean_cardinality
            features.append(min(np.max(cardinalities) / 100, 1.0))  # max_cardinality
            high_card = sum(1 for c in cardinalities if c > 50)
            features.append(min(high_card / len(categorical_cols), 1.0))  # high_cardinality_ratio
        else:
            features.extend([0.0, 0.0, 0.0])

        # ==================== TARGET FEATURES (3) ====================
        target_numeric = y.copy()
        if not pd.api.types.is_numeric_dtype(y):
            target_numeric = pd.Series(pd.Categorical(y).codes.astype(float), index=y.index)

        # Target imbalance
        value_counts = y.value_counts()
        if len(value_counts) > 1:
            imbalance = 1 - (value_counts.min() / value_counts.max())
            features.append(np.clip(imbalance, 0, 1))  # target_imbalance
        else:
            features.append(0.0)

        features.append(np.clip(abs(target_numeric.skew()), 0, 1))  # target_skewness
        features.append(np.clip(target_numeric.kurtosis(), 0, 1))  # target_kurtosis

        # ==================== PCA FEATURES (3) ====================
        if len(numeric_cols) >= 2:
            try:
                X_num = X[numeric_cols].fillna(X[numeric_cols].mean())
                X_scaled = (X_num - X_num.mean()) / (X_num.std() + 1e-10)
                from sklearn.decomposition import PCA
                pca = PCA()
                pca.fit(X_scaled)
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_comp_95 = np.argmax(cumsum >= 0.95) + 1 if np.any(cumsum >= 0.95) else len(numeric_cols)
                features.append(min(n_comp_95 / len(numeric_cols), 1.0))  # pca_95_components
                n_comp_50 = np.argmax(cumsum >= 0.50) + 1 if np.any(cumsum >= 0.50) else len(numeric_cols)
                features.append(min(n_comp_50 / len(numeric_cols), 1.0))  # pca_50_variance
                intrinsic = np.argmax(cumsum >= 0.90) + 1 if np.any(cumsum >= 0.90) else len(numeric_cols)
                features.append(min(intrinsic / len(numeric_cols), 1.0))  # intrinsic_dim
            except:
                features.extend([0.5, 0.5, 0.5])
        else:
            features.extend([0.5, 0.5, 0.5])

        # ==================== LANDMARK FEATURES (4) ====================
        # Compute quick baseline model scores
        landmark_scores = self._compute_landmark_features(X, y, task_type, numeric_cols, categorical_cols)
        features.extend(landmark_scores)

        # Ensure exactly 32 features
        return np.array(features[:32], dtype=np.float32)

    def _compute_landmark_features(self, X: pd.DataFrame, y: pd.Series, task_type: str,
                                   numeric_cols: List[str], categorical_cols: List[str]) -> List[float]:
        """
        Compute quick baseline model scores (landmark features).
        Returns: [dt_score, nb_score, lr_score, knn_score]
        """
        try:
            # Prepare data: use only numeric features for quick evaluation
            if not numeric_cols:
                return [0.5, 0.5, 0.5, 0.5]

            X_num = X[numeric_cols].fillna(X[numeric_cols].median()).values

            # Limit to 500 samples for speed
            if len(X_num) > 500:
                indices = np.random.choice(len(X_num), 500, replace=False)
                X_num = X_num[indices]
                y_sample = y.iloc[indices]
            else:
                y_sample = y

            # Encode target if needed
            if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y_sample):
                le = LabelEncoder()
                y_sample = le.fit_transform(y_sample)

            scores = []

            # Decision Tree
            try:
                if task_type == 'classification':
                    model = DecisionTreeClassifier(max_depth=5, random_state=42)
                else:
                    model = DecisionTreeRegressor(max_depth=5, random_state=42)
                score = cross_val_score(model, X_num, y_sample, cv=min(3, len(X_num)//10), scoring='accuracy' if task_type == 'classification' else 'r2').mean()
                scores.append(np.clip(score, 0, 1))
            except:
                scores.append(0.5)

            # Naive Bayes (classification only)
            try:
                if task_type == 'classification':
                    model = GaussianNB()
                    score = cross_val_score(model, X_num, y_sample, cv=min(3, len(X_num)//10), scoring='accuracy').mean()
                    scores.append(np.clip(score, 0, 1))
                else:
                    scores.append(0.5)  # N/A for regression
            except:
                scores.append(0.5)

            # Logistic/Linear Regression
            try:
                if task_type == 'classification':
                    model = LogisticRegression(max_iter=100, random_state=42)
                    score = cross_val_score(model, X_num, y_sample, cv=min(3, len(X_num)//10), scoring='accuracy').mean()
                else:
                    model = LinearRegression()
                    score = cross_val_score(model, X_num, y_sample, cv=min(3, len(X_num)//10), scoring='r2').mean()
                scores.append(np.clip(score, 0, 1))
            except:
                scores.append(0.5)

            # KNN
            try:
                if task_type == 'classification':
                    model = KNeighborsClassifier(n_neighbors=5)
                    score = cross_val_score(model, X_num, y_sample, cv=min(3, len(X_num)//10), scoring='accuracy').mean()
                else:
                    model = KNeighborsRegressor(n_neighbors=5)
                    score = cross_val_score(model, X_num, y_sample, cv=min(3, len(X_num)//10), scoring='r2').mean()
                scores.append(np.clip(score, 0, 1))
            except:
                scores.append(0.5)

            return scores
        except Exception as e:
            # If anything fails, return default scores
            return [0.5, 0.5, 0.5, 0.5]


# Helper functions for backward compatibility
def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Get numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    """Get categorical column names."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()
