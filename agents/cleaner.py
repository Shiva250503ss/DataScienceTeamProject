# agents/cleaner.py

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from sklearn.impute import KNNImputer, SimpleImputer
from agents.base import BaseAgent


class CleanerAgent(BaseAgent):
    """
    Agent responsible for data cleaning.
    
    This is the SECOND agent in the pipeline. It takes the profiled data and:
      1. Removes duplicate rows
      2. Handles missing values (smart strategy per column)
      3. Detects and handles outliers (IQR-based Winsorization)
      4. Fixes data types based on profiler's column type detection
    
    Cleaning strategies are chosen automatically based on:
      - Column type (numeric vs categorical vs datetime)
      - Missing percentage (low → simple impute, medium → KNN, high → indicator column)
      - Data distribution (skewed → median, normal → mean)
    
    Owner: Bhavana
    """
    
    def __init__(self):
        super().__init__("CleanerAgent")
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Clean the dataset"""
        self.log("Starting data cleaning...")

        df = state['current_data'].copy()
        profile = state['profile_report']
        column_types = profile['column_types']

        # Log LLM context hints so the team can see what the AI noticed
        data_context = state.get('data_context', {})
        if data_context:
            self.log(f"Domain context: {data_context.get('domain', 'unknown')}")
            cleaning_hints = data_context.get('cleaning_hints', {})
            for col, hint in cleaning_hints.items():
                if col in df.columns:
                    self.log(f"  LLM hint for '{col}': {hint}")
        
        cleaning_report = {
            'missing_value_handling': {},
            'outlier_handling': {},
            'duplicate_removal': {},
            'transformations': []
        }
        
        # =====================================================================
        # Step 0: Safety-net coercion of "disguised numeric" string columns.
        # The profiler already runs this, but the cleaner re-checks because:
        #   a) Profiler's category standardization may have merged variants
        #      that now look numeric (e.g. "$100" + "100" → all "100").
        #   b) Edge cases the profiler missed (mixed delimiters, new data).
        # When a column is coerced, update column_types so downstream steps
        # (missing-value handling, outlier detection, feature engineering)
        # treat it as numeric instead of categorical.
        # =====================================================================
        df, coerced_cols = self._coerce_numeric_strings(df)
        if coerced_cols:
            self.log(f"Coerced {len(coerced_cols)} string columns to numeric: {coerced_cols}")
            for col in coerced_cols:
                column_types[col] = 'numeric'
            # Push the updated column_types back into the profile so that
            # the feature agent and modeler also see the corrected types.
            state['profile_report']['column_types'] = column_types
            cleaning_report['transformations'].append(
                f"Coerced disguised-numeric columns to float64: {coerced_cols}"
            )

        # =====================================================================
        # Step 1: Remove duplicate rows
        # =====================================================================
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        cleaning_report['duplicate_removal'] = {
            'rows_removed': n_removed,
            'rows_remaining': len(df)
        }
        self.log(f"Removed {n_removed} duplicate rows")
        
        # =====================================================================
        # Step 2: Handle missing values (smart strategy per column)
        # =====================================================================
        df, imputation_map = self._handle_missing_values(df, column_types, profile['statistics'])
        cleaning_report['missing_value_handling'] = imputation_map
        
        # =====================================================================
        # Step 3: Handle outliers (IQR-based detection + Winsorization)
        # =====================================================================
        df, outlier_bounds = self._handle_outliers(df, column_types)
        cleaning_report['outlier_handling'] = outlier_bounds
        
        # =====================================================================
        # Step 4: Fix data types
        # =====================================================================
        df = self._fix_data_types(df, column_types)
        cleaning_report['transformations'].append('Fixed data types')
        
        # =====================================================================
        # Step 5: Standardize non-uniform categories
        # =====================================================================
        uniformity_issues = profile.get('uniformity_issues', {})
        if uniformity_issues:
            df, standardization_map = self._standardize_categories(df, column_types, uniformity_issues)
            cleaning_report['category_standardization'] = standardization_map
            if standardization_map:
                self.log(f"Standardized categories in {len(standardization_map)} columns")
        else:
            cleaning_report['category_standardization'] = {}
        
        # Update pipeline state
        state['current_data'] = df
        state['cleaning_report'] = cleaning_report
        state['imputation_map'] = imputation_map
        state['outlier_bounds'] = outlier_bounds
        state['stage'] = 'cleaned'
        
        self.log(f"Cleaning complete. {len(df)} rows remaining.")
        return state
    
    # =========================================================================
    # STEP 2: Missing Value Handling
    # =========================================================================
    
    def _handle_missing_values(self, df: pd.DataFrame, column_types: Dict[str, str],
                               statistics: Dict) -> tuple:
        """
        Handle missing values with smart strategy selection per column.
        
        Strategy selection logic:
          NUMERIC columns:
            - <5% missing + skewed  → median
            - <5% missing + normal  → mean
            - 5-30% missing         → KNN imputation (uses neighbor columns)
            - >30% missing          → median + binary indicator column
          
          CATEGORICAL columns:
            - <10% missing → mode (most frequent value)
            - ≥10% missing → new 'Missing' category
          
          DATETIME columns:
            - Forward-fill then backward-fill
          
          ID columns:
            - Skipped (not imputed)
        """
        imputation_map = {}
        
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            
            col_type = column_types.get(col, 'unknown')
            missing_pct = df[col].isna().mean() * 100
            
            if col_type == 'numeric':
                imputation_map[col] = self._impute_numeric(df, col, missing_pct, statistics.get(col, {}))
            elif col_type == 'categorical':
                imputation_map[col] = self._impute_categorical(df, col, missing_pct)
            elif col_type == 'datetime':
                imputation_map[col] = self._impute_datetime(df, col)
            elif col_type == 'id':
                imputation_map[col] = {'strategy': 'skip', 'reason': 'ID column'}
            
            if col in imputation_map:
                self.log(f"  {col}: {imputation_map[col].get('strategy', 'unknown')} "
                        f"({missing_pct:.1f}% missing)")
        
        return df, imputation_map
    
    def _impute_numeric(self, df: pd.DataFrame, col: str,
                       missing_pct: float, stats: Dict) -> Dict:
        """
        Impute a numeric column based on missing percentage and distribution.
        
        - <5% missing:  Simple imputation (median if skewed, mean if normal)
        - 5-30% missing: KNN imputation using correlated numeric columns
        - >30% missing:  Median + create binary indicator column '{col}_missing'
        """
        skewness = stats.get('skewness', 0)
        
        if missing_pct < 5:
            # Low missing — use median for skewed data, mean for normal
            if abs(skewness) > 1:
                value = df[col].median()
                strategy = 'median'
            else:
                value = df[col].mean()
                strategy = 'mean'
            df[col].fillna(value, inplace=True)
            return {'strategy': strategy, 'value': round(float(value), 4)}
        
        elif missing_pct < 30:
            # Medium missing — use KNN imputation (leverages correlations)
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    imputer = KNNImputer(n_neighbors=5)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    return {'strategy': 'knn', 'n_neighbors': 5}
            except:
                pass
            # Fallback to median if KNN fails
            value = df[col].median()
            df[col].fillna(value, inplace=True)
            return {'strategy': 'median (knn_fallback)', 'value': round(float(value), 4)}
        
        else:
            # High missing — median + binary indicator column
            df[f'{col}_missing'] = df[col].isna().astype(int)
            value = df[col].median()
            df[col].fillna(value, inplace=True)
            return {
                'strategy': 'median_with_indicator',
                'value': round(float(value), 4),
                'indicator_col': f'{col}_missing'
            }
    
    def _impute_categorical(self, df: pd.DataFrame, col: str,
                           missing_pct: float) -> Dict:
        """
        Impute a categorical column.
        
        - <10% missing: Fill with mode (most frequent value)
        - ≥10% missing: Create a new 'Missing' category
        """
        if missing_pct < 10:
            value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(value, inplace=True)
            return {'strategy': 'mode', 'value': str(value)}
        else:
            df[col].fillna('Missing', inplace=True)
            return {'strategy': 'missing_category', 'value': 'Missing'}
    
    def _impute_datetime(self, df: pd.DataFrame, col: str) -> Dict:
        """
        Impute datetime column using forward-fill then backward-fill.
        This preserves temporal ordering.
        """
        df[col] = df[col].ffill().bfill()
        return {'strategy': 'forward_backward_fill'}
    
    # =========================================================================
    # STEP 3: Outlier Handling
    # =========================================================================
    
    def _handle_outliers(self, df: pd.DataFrame,
                        column_types: Dict[str, str]) -> tuple:
        """
        Detect and handle outliers in numeric columns using the IQR method.
        
        Detection: Any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        
        Action:
          - If outliers < 5% of data → Winsorize (clip to bounds)
          - If outliers ≥ 5% of data → Keep as-is (likely natural variation)
        """
        outlier_bounds = {}
        
        for col in df.columns:
            if column_types.get(col) != 'numeric':
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_pct = n_outliers / len(df) * 100
            
            if n_outliers > 0:
                outlier_bounds[col] = {
                    'lower_bound': round(float(lower), 4),
                    'upper_bound': round(float(upper), 4),
                    'n_outliers': int(n_outliers),
                    'outlier_pct': round(outlier_pct, 2)
                }
                
                # Winsorize (clip) only if outliers are a small fraction
                if outlier_pct < 5:
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    outlier_bounds[col]['action'] = 'clipped (winsorized)'
                    self.log(f"  {col}: Clipped {n_outliers} outliers ({outlier_pct:.1f}%)")
                else:
                    outlier_bounds[col]['action'] = 'kept (natural variation)'
                    self.log(f"  {col}: Kept {n_outliers} outliers ({outlier_pct:.1f}% — likely natural)")
        
        return df, outlier_bounds
    
    # =========================================================================
    # STEP 4: Fix Data Types
    # =========================================================================
    
    def _fix_data_types(self, df: pd.DataFrame,
                       column_types: Dict[str, str]) -> pd.DataFrame:
        """
        Ensure columns have correct pandas dtypes based on the profiler's detection.
        
        - numeric    → pd.to_numeric (coerce errors)
        - categorical → str
        - datetime   → pd.to_datetime (coerce errors)
        """
        for col, ctype in column_types.items():
            if col not in df.columns:
                continue
            
            try:
                if ctype == 'numeric':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif ctype == 'categorical':
                    df[col] = df[col].astype(str)
                elif ctype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                self.log(f"  Warning: Could not convert {col} to {ctype}: {e}")
        
        return df
    
    # =========================================================================
    # STEP 5: Standardize Non-Uniform Categories
    # =========================================================================
    
    def _standardize_categories(self, df: pd.DataFrame, column_types: Dict[str, str],
                                 uniformity_issues: Dict) -> tuple:
        """
        Standardize non-uniform category values detected by the Profiler.
        
        For example:
          'Male', 'male', 'MALE'  →  'Male' (most frequent variant wins)
          ' Yes ', 'Yes', 'yes'   →  'Yes'
        
        Args:
            df: DataFrame to clean
            column_types: column type mapping from profiler
            uniformity_issues: {col: {canonical_lower: [variant1, variant2, ...]}}
        
        Returns:
            Tuple of (cleaned DataFrame, standardization mapping applied)
        """
        standardization_map = {}
        
        for col, issues in uniformity_issues.items():
            if col not in df.columns:
                continue
            
            col_map = {}
            for canonical, variants in issues.items():
                # Pick the most common variant as the standard form
                counts = {}
                for v in variants:
                    counts[v] = (df[col].astype(str) == v).sum()
                standard = max(counts, key=counts.get)
                
                for v in variants:
                    if v != standard:
                        col_map[v] = standard
            
            if col_map:
                df[col] = df[col].astype(str).replace(col_map)
                standardization_map[col] = col_map
                self.log(f"  {col}: Standardized {len(col_map)} variants -> {list(set(col_map.values()))}")
        
        return df, standardization_map

    # =========================================================================
    # STEP 0: Coerce "Disguised Numeric" String Columns (safety net)
    # =========================================================================

    # Regex to strip common non-numeric formatting characters:
    #   $  €  £  ¥  ₹  ₩  ₫  ¢  — currency symbols
    #   ,  %                     — thousand separators, percent signs
    #   whitespace               — leading/trailing/embedded spaces
    _NUMERIC_NOISE_RE = re.compile(r'[$€£¥₹₩₫¢,%\s]')

    # Pattern for parenthesised negatives, e.g.  "(1,234.56)" → "-1234.56"
    _PAREN_NEGATIVE_RE = re.compile(r'^\s*\((.+)\)\s*$')

    def _coerce_numeric_strings(self, df: pd.DataFrame) -> tuple:
        """
        Detect object-dtype columns whose values are really numbers hidden
        behind formatting characters and convert them to float64.

        Common cases handled:
          • Currency symbols:    "$1,234.56"  → 1234.56
          • Percent signs:       "50%"        → 50.0
          • Parenthesised negs:  "(500.00)"   → -500.0
          • Thousand separators: "1,000,000"  → 1000000.0
          • Mixed whitespace:    " 42 "       → 42.0
          • Unit suffixes:       "10 kg"      → 10.0

        Heuristic: if ≥80% of non-null values become valid floats after
        stripping, the whole column is converted.

        Returns:
            (df, list_of_coerced_column_names)
        """
        df = df.copy()
        coerced_cols = []

        for col in df.columns:
            if df[col].dtype != 'object':
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            # Quick reject: if <50% of values contain a digit, skip.
            sample = series.head(200).astype(str)
            has_digit = sample.str.contains(r'\d', regex=True)
            if has_digit.sum() < len(sample) * 0.5:
                continue

            cleaned = self._clean_numeric_series(series.astype(str))
            numeric_parsed = pd.to_numeric(cleaned, errors='coerce')
            n_valid = numeric_parsed.notna().sum()
            ratio = n_valid / len(series)

            if ratio >= 0.80:
                full_cleaned = self._clean_numeric_series(df[col].astype(str))
                df[col] = pd.to_numeric(full_cleaned, errors='coerce')
                coerced_cols.append(col)

        return df, coerced_cols

    def _clean_numeric_series(self, s: pd.Series) -> pd.Series:
        """Strip formatting noise from a string Series so it can be parsed."""
        # 1. Handle trailing percent sign
        s = s.str.replace('%', '', regex=False)
        # 2. Handle parenthesised negatives  ("(500)" → "-500")
        mask = s.str.match(self._PAREN_NEGATIVE_RE, na=False)
        s = s.where(~mask, '-' + s.str.replace(r'[()\s]', '', regex=True))
        # 3. Strip currency symbols, commas, whitespace
        s = s.str.replace(self._NUMERIC_NOISE_RE, '', regex=True)
        # 4. Strip trailing alphabetic unit suffixes (e.g. "10kg" → "10")
        s = s.str.replace(r'[a-zA-Z]+$', '', regex=True)
        # 5. Strip leading alphabetic prefixes (e.g. "USD100" → "100")
        s = s.str.replace(r'^[a-zA-Z]+', '', regex=True)
        return s
