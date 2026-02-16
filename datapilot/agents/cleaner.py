"""
Cleaner Agent - Data Cleaning and Preprocessing
Full implementation matching DataPilot specification (Section 5.2)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.impute import KNNImputer
from scipy import stats


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Get numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    """Get categorical column names."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_datetime_cols(df: pd.DataFrame) -> List[str]:
    """Get datetime column names."""
    return df.select_dtypes(include=['datetime64']).columns.tolist()


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that appear to be IDs (100% unique values)."""
    id_cols = []
    for col in df.columns:
        if df[col].nunique() == len(df):
            id_cols.append(col)
    return id_cols


class CleanerAgent:
    """Cleans data by handling missing values, duplicates, and outliers with full spec compliance."""

    def clean(self, df: pd.DataFrame, target_col: str, task_type: str,
              meta_features: np.ndarray = None, profile_report: Dict = None) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
        """
        Clean dataset intelligently based on meta-features.

        Args:
            df: Input DataFrame
            target_col: Target column name
            task_type: 'classification' or 'regression'
            meta_features: 32 meta-features to guide cleaning strategy
            profile_report: Optional profile report with column types

        Returns:
            (cleaned_dataframe, cleaning_report, imputation_map, outlier_bounds)
        """
        df_clean = df.copy()

        # Initialize output dictionaries
        imputation_map = {}
        outlier_bounds = {}
        transformations = []

        # Determine cleaning strategy based on meta-features
        if meta_features is not None:
            missing_ratio = meta_features[6]  # Index 6 is missing_ratio
            outlier_ratio = meta_features[11]  # Index 11 is outlier_ratio
            imbalance_ratio = meta_features[22]  # Index 22 is target_imbalance
        else:
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            outlier_ratio = 0.05
            imbalance_ratio = 0.0

        # Get column types
        numeric_cols = get_numeric_cols(df_clean)
        categorical_cols = get_categorical_cols(df_clean)
        datetime_cols = get_datetime_cols(df_clean)
        id_cols = detect_id_columns(df_clean)

        # Remove target from processing lists
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        if target_col in datetime_cols:
            datetime_cols.remove(target_col)

        # ==================== 1. REMOVE ID COLUMNS ====================
        if id_cols:
            for col in id_cols:
                if col != target_col:  # Don't drop target
                    df_clean = df_clean.drop(columns=[col])
                    transformations.append(f"Dropped ID column: {col}")
                    # Update column lists
                    if col in numeric_cols:
                        numeric_cols.remove(col)
                    if col in categorical_cols:
                        categorical_cols.remove(col)

        # ==================== 2. REMOVE DUPLICATES ====================
        dup_count = df_clean.duplicated().sum()
        if dup_count > 0:
            df_clean = df_clean.drop_duplicates()
            transformations.append(f"Removed {dup_count} duplicate rows")

        # ==================== 3. HANDLE MISSING VALUES ====================
        missing_handling = self._handle_missing_values(
            df_clean, numeric_cols, categorical_cols, datetime_cols,
            target_col, imputation_map, transformations
        )
        df_clean = missing_handling['df']
        imputation_map = missing_handling['imputation_map']

        # Update column lists after potential column drops
        numeric_cols = [c for c in numeric_cols if c in df_clean.columns]
        categorical_cols = [c for c in categorical_cols if c in df_clean.columns]

        # ==================== 4. HANDLE OUTLIERS ====================
        outlier_handling = self._handle_outliers(
            df_clean, numeric_cols, target_col, outlier_ratio,
            outlier_bounds, transformations
        )
        df_clean = outlier_handling['df']
        outlier_bounds = outlier_handling['outlier_bounds']

        # ==================== 5. HANDLE CLASS IMBALANCE (LOG ONLY) ====================
        if imbalance_ratio > 0.7 and task_type == 'classification':
            transformations.append(f"High class imbalance detected ({imbalance_ratio:.2%}). Consider using SMOTE or class weights.")

        # Build cleaning report
        report = {
            'original_shape': df.shape,
            'final_shape': df_clean.shape,
            'rows_removed': df.shape[0] - df_clean.shape[0],
            'columns_removed': df.shape[1] - df_clean.shape[1],
            'duplicates_removed': dup_count,
            'id_columns_dropped': [col for col in id_cols if col not in df_clean.columns],
            'missing_ratio': missing_ratio,
            'outlier_ratio': outlier_ratio,
            'imbalance_ratio': imbalance_ratio,
            'transformations': transformations,
        }

        return df_clean, report, imputation_map, outlier_bounds

    def _handle_missing_values(self, df: pd.DataFrame, numeric_cols: List[str],
                               categorical_cols: List[str], datetime_cols: List[str],
                               target_col: str, imputation_map: Dict,
                               transformations: List[str]) -> Dict:
        """
        Handle missing values according to specification:
        - Numeric <5%: Mean or Median (based on skewness)
        - Numeric 5-30%: KNN Imputation
        - Numeric >30%: Create indicator + median
        - Categorical <10%: Mode
        - Categorical 10-30%: Mode or "Unknown"
        - Categorical >30%: Separate "Missing" category
        - Datetime: Forward/backward fill
        """
        df_clean = df.copy()

        # ==================== NUMERIC COLUMNS ====================
        for col in numeric_cols:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)

            if missing_pct == 0:
                continue

            if missing_pct < 0.05:
                # <5%: Use mean or median based on skewness
                skewness = abs(df_clean[col].skew())
                if skewness > 1:
                    # Skewed distribution: use median
                    fill_value = df_clean[col].median()
                    method = 'median'
                else:
                    # Normal distribution: use mean
                    fill_value = df_clean[col].mean()
                    method = 'mean'

                df_clean[col] = df_clean[col].fillna(fill_value)
                imputation_map[col] = {'method': method, 'value': fill_value}
                transformations.append(f"Filled {col} ({missing_pct:.1%} missing) with {method}: {fill_value:.4f}")

            elif missing_pct < 0.30:
                # 5-30%: KNN Imputation
                try:
                    # Get all numeric columns for KNN
                    knn_cols = [c for c in numeric_cols if c in df_clean.columns and df_clean[c].notna().sum() > 0]
                    if len(knn_cols) >= 2:
                        imputer = KNNImputer(n_neighbors=5)
                        df_clean[knn_cols] = imputer.fit_transform(df_clean[knn_cols])
                        imputation_map[col] = {'method': 'knn', 'n_neighbors': 5}
                        transformations.append(f"Filled {col} ({missing_pct:.1%} missing) with KNN imputation")
                    else:
                        # Fallback to median if not enough columns for KNN
                        fill_value = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        imputation_map[col] = {'method': 'median', 'value': fill_value}
                        transformations.append(f"Filled {col} ({missing_pct:.1%} missing) with median (KNN fallback): {fill_value:.4f}")
                except Exception as e:
                    # Fallback to median on error
                    fill_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    imputation_map[col] = {'method': 'median', 'value': fill_value}
                    transformations.append(f"Filled {col} ({missing_pct:.1%} missing) with median (KNN failed): {fill_value:.4f}")

            else:
                # >30%: Create indicator column + fill with median
                indicator_col = f"{col}_missing_indicator"
                df_clean[indicator_col] = df_clean[col].isnull().astype(int)
                fill_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_value)
                imputation_map[col] = {'method': 'indicator+median', 'value': fill_value, 'indicator_col': indicator_col}
                transformations.append(f"Created indicator for {col} ({missing_pct:.1%} missing) and filled with median: {fill_value:.4f}")

        # ==================== CATEGORICAL COLUMNS ====================
        for col in categorical_cols:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)

            if missing_pct == 0:
                continue

            if missing_pct < 0.10:
                # <10%: Mode
                mode_value = df_clean[col].mode()
                if len(mode_value) > 0:
                    fill_value = mode_value[0]
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    imputation_map[col] = {'method': 'mode', 'value': fill_value}
                    transformations.append(f"Filled {col} ({missing_pct:.1%} missing) with mode: {fill_value}")

            elif missing_pct < 0.30:
                # 10-30%: Mode or "Unknown"
                # Use "Unknown" if mode is not dominant
                mode_value = df_clean[col].mode()
                if len(mode_value) > 0 and df_clean[col].value_counts(normalize=True).iloc[0] > 0.5:
                    # Mode is dominant (>50%), use it
                    fill_value = mode_value[0]
                    method = 'mode'
                else:
                    # Mode is not dominant, use "Unknown"
                    fill_value = 'Unknown'
                    method = 'unknown'

                df_clean[col] = df_clean[col].fillna(fill_value)
                imputation_map[col] = {'method': method, 'value': fill_value}
                transformations.append(f"Filled {col} ({missing_pct:.1%} missing) with {method}: {fill_value}")

            else:
                # >30%: Separate "Missing" category
                df_clean[col] = df_clean[col].fillna('Missing')
                imputation_map[col] = {'method': 'missing_category', 'value': 'Missing'}
                transformations.append(f"Filled {col} ({missing_pct:.1%} missing) with explicit 'Missing' category")

        # ==================== DATETIME COLUMNS ====================
        for col in datetime_cols:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)

            if missing_pct == 0:
                continue

            # Forward fill, then backward fill
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            imputation_map[col] = {'method': 'forward_backward_fill'}
            transformations.append(f"Filled {col} ({missing_pct:.1%} missing) with forward/backward fill")

        # Drop columns with >50% missing (too much to impute)
        cols_to_drop = []
        for col in df_clean.columns:
            if col != target_col:
                missing_pct = df_clean[col].isnull().sum() / len(df_clean)
                if missing_pct > 0.5:
                    cols_to_drop.append(col)

        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            transformations.append(f"Dropped {len(cols_to_drop)} columns with >50% missing: {cols_to_drop}")

        return {'df': df_clean, 'imputation_map': imputation_map}

    def _handle_outliers(self, df: pd.DataFrame, numeric_cols: List[str],
                        target_col: str, outlier_ratio: float,
                        outlier_bounds: Dict, transformations: List[str]) -> Dict:
        """
        Handle outliers according to specification:
        - Detection: Both IQR and Z-score methods
        - <1% outliers: Remove rows
        - 1-5% Normal distribution: Cap (Winsorization)
        - 1-5% Skewed distribution: Keep
        - >5%: Keep + flag for review
        """
        df_clean = df.copy()

        for col in numeric_cols:
            # Skip if column has too few values
            if df_clean[col].notna().sum() < 10:
                continue

            # ==================== OUTLIER DETECTION ====================
            # Method 1: IQR
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_iqr = Q1 - 1.5 * IQR
            upper_iqr = Q3 + 1.5 * IQR
            outliers_iqr = (df_clean[col] < lower_iqr) | (df_clean[col] > upper_iqr)

            # Method 2: Z-score
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            # Map z-scores back to original indices
            z_score_mask = pd.Series(False, index=df_clean.index)
            z_score_mask.loc[df_clean[col].notna()] = z_scores > 3

            # High-confidence outliers: flagged by BOTH methods
            high_confidence_outliers = outliers_iqr & z_score_mask
            outlier_pct = high_confidence_outliers.sum() / len(df_clean)

            if outlier_pct == 0:
                continue

            # Check distribution (Normal vs Skewed)
            skewness = abs(df_clean[col].skew())
            is_skewed = skewness > 1

            # ==================== OUTLIER HANDLING ====================
            if outlier_pct < 0.01:
                # <1%: Remove rows (likely errors)
                rows_before = len(df_clean)
                df_clean = df_clean[~high_confidence_outliers]
                rows_removed = rows_before - len(df_clean)
                outlier_bounds[col] = {'method': 'remove', 'removed_count': rows_removed}
                transformations.append(f"Removed {rows_removed} rows with outliers in {col} ({outlier_pct:.2%})")

            elif outlier_pct < 0.05:
                if not is_skewed:
                    # 1-5% Normal: Cap at boundaries (Winsorization)
                    df_clean[col] = df_clean[col].clip(lower_iqr, upper_iqr)
                    outlier_bounds[col] = {'method': 'winsorization', 'lower': lower_iqr, 'upper': upper_iqr}
                    transformations.append(f"Capped outliers in {col} ({outlier_pct:.2%}) at [{lower_iqr:.4f}, {upper_iqr:.4f}]")
                else:
                    # 1-5% Skewed: Keep (may be valid extreme values)
                    outlier_bounds[col] = {'method': 'keep', 'reason': 'skewed_distribution'}
                    transformations.append(f"Kept outliers in {col} ({outlier_pct:.2%}) - skewed distribution")

            else:
                # >5%: Keep + flag for review
                outlier_bounds[col] = {'method': 'keep', 'reason': 'too_many_outliers', 'outlier_pct': outlier_pct}
                transformations.append(f"Flagged {col} for review: {outlier_pct:.2%} outliers (too many to remove)")

        return {'df': df_clean, 'outlier_bounds': outlier_bounds}
