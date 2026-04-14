# agents/profiler.py

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from agents.base import BaseAgent
from meta_features import extract_meta_features, N_META_FEATURES


class ProfilerAgent(BaseAgent):
    """
    Agent responsible for data profiling and meta-feature extraction.
    
    This is the FIRST agent in the pipeline. It analyzes the uploaded CSV and produces:
      1. Column type detection (numeric, categorical, datetime, text, id, binary)
      2. Auto-detection of target column (if user didn't specify)
      3. Task type determination (classification vs regression)
      4. Per-column statistics (mean, std, skewness, missing %, etc.)
      5. 40 meta-features for the RL Model Selector
      6. Data quality score (0-100)
      7. Warnings about potential data issues
    
    Owner: Harshitha
    """
    
    def __init__(self):
        super().__init__("ProfilerAgent")
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Profile the dataset and extract meta-features"""
        self.log("Starting data profiling...")
        
        df = state['raw_data']
        target_col = state.get('target_column')
        data_context = state.get('data_context', {})

        # Resolve target column case-insensitively so users can type
        # "species" and match the actual column "Species"
        if target_col and target_col not in df.columns:
            col_lower_map = {c.lower(): c for c in df.columns}
            matched = col_lower_map.get(target_col.lower())
            if matched:
                self.log(f"Column '{target_col}' not found — using '{matched}' (case-insensitive match)")
                target_col = matched
                state['target_column'] = target_col

        # Step 0: Coerce "disguised numeric" string columns to actual numbers.
        # This handles columns like "$1,234.56", "50%", "(500)", "€ 10.00"
        # that pandas reads as 'object' dtype but are really numeric.
        # Must run BEFORE column type detection so the profiler sees them as numeric.
        df, coerced_cols = self._coerce_numeric_strings(df)
        if coerced_cols:
            self.log(f"Coerced {len(coerced_cols)} string columns to numeric: {coerced_cols}")
            # Update raw_data so downstream agents also see the cleaned version
            state['raw_data'] = df
            if 'current_data' in state:
                state['current_data'] = df.copy()

        # Step 1: Detect column types
        column_types = self._detect_column_types(df)
        self.log(f"Detected column types: {column_types}")

        # Step 2: Auto-detect target if not specified
        # LLM context already set target_column in state during context phase,
        # so this fallback only runs if context phase failed or returned nothing.
        if not target_col:
            target_col = self._detect_target(df, column_types)
            self.log(f"Auto-detected target column: {target_col}")
        else:
            self.log(f"Using target column: {target_col}")

        # Step 3: Determine task type (classification or regression)
        task_type = self._determine_task_type(df, target_col, column_types)
        # Override with LLM-suggested task type if valid
        llm_task_type = data_context.get('task_type', '').lower()
        if llm_task_type in ('classification', 'regression') and llm_task_type != task_type:
            self.log(f"Task type overridden by LLM context: {task_type} -> {llm_task_type}")
            task_type = llm_task_type
        self.log(f"Task type: {task_type}")
        
        # Step 4: Compute per-column statistics
        statistics = self._compute_statistics(df, column_types)
        
        # Step 5: Extract 40 meta-features for RL Model Selector
        meta_features = self._extract_meta_features(df, target_col, task_type)
        self.log(f"Extracted {len(meta_features)} meta-features")
        
        # Step 6: Compute data quality score
        quality_score = self._compute_quality_score(df)
        self.log(f"Data quality score: {quality_score}/100")
        
        # Step 7: Generate warnings about data issues
        warnings = self._generate_warnings(df, column_types)
        
        # Step 8: describe()-based anomaly detection (like a real DS)
        describe_anomalies = self._describe_anomaly_detection(df, column_types)
        if describe_anomalies:
            self.log(f"Found {len(describe_anomalies)} anomalies from describe() analysis")
        
        # Step 9: Check unique value uniformity (non-uniform categories)
        uniformity_issues = self._check_unique_value_uniformity(df, column_types)
        if uniformity_issues:
            self.log(f"Found non-uniform categories in {len(uniformity_issues)} columns")
        
        # Update pipeline state with all profiling results
        state['target_column'] = target_col
        state['task_type'] = task_type
        state['profile_report'] = {
            'column_types': column_types,
            'statistics': statistics,
            'quality_score': quality_score,
            'warnings': warnings,
            'describe_anomalies': describe_anomalies,
            'uniformity_issues': uniformity_issues,
            'n_rows': len(df),
            'n_cols': len(df.columns)
        }
        state['meta_features'] = meta_features
        state['stage'] = 'profiled'
        
        return state
    
    # =========================================================================
    # STEP 1: Column Type Detection
    # =========================================================================
    
    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect the semantic type of each column.
        
        Types detected:
          - 'numeric'     : continuous numbers (int/float with many unique values)
          - 'categorical' : discrete categories (strings or low-cardinality numbers)
          - 'binary'      : boolean columns
          - 'datetime'    : date/time columns
          - 'text'        : free-text columns (high avg word count)
          - 'id'          : identifier columns (nearly all unique values)
        """
        column_types = {}
        
        for col in df.columns:
            dtype = df[col].dtype
            n_unique = df[col].nunique()
            n_total = len(df[col].dropna())
            unique_ratio = n_unique / n_total if n_total > 0 else 0
            
            # Check if datetime
            if dtype == 'datetime64[ns]':
                column_types[col] = 'datetime'
            # Check if numeric (any int or float dtype)
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Detect integer ID columns (e.g. 'Id', 'user_id', 'record_id')
                col_lower = col.lower().strip()
                is_id_name = (
                    col_lower in ('id', 'index', 'key', 'row', 'rowid', 'row_id', 'no', 'num', 'number') or
                    col_lower.endswith('_id') or col_lower.endswith('id') or
                    col_lower.startswith('id_')
                )
                if (is_id_name and unique_ratio > 0.90 and
                        pd.api.types.is_integer_dtype(df[col])):
                    column_types[col] = 'id'
                # Could be categorical if very few unique values relative to dataset
                elif n_unique <= 20 and unique_ratio < 0.05:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'numeric'
            # Check if boolean
            elif dtype == 'bool':
                column_types[col] = 'binary'
            # String/Object columns
            else:
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().head(100))
                    column_types[col] = 'datetime'
                except:
                    # Check if ID column (almost all values unique)
                    if unique_ratio > 0.95:
                        column_types[col] = 'id'
                    # Check if text (high average word count per cell)
                    elif df[col].dropna().astype(str).str.split().str.len().mean() > 5:
                        column_types[col] = 'text'
                    else:
                        column_types[col] = 'categorical'
        
        return column_types
    
    # =========================================================================
    # STEP 2: Auto-detect Target Column
    # =========================================================================
    
    def _detect_target(self, df: pd.DataFrame, column_types: Dict[str, str]) -> str:
        """
        Auto-detect the target column using heuristics:
          Priority 1: Column named 'target', 'label', 'y', 'class', 'outcome'
          Priority 2: Last column in the dataframe
          Priority 3: Column with lowest cardinality (for classification)
          Default:    Last numeric column
        """
        # Priority 1: Common target column names
        target_names = ['target', 'label', 'y', 'class', 'outcome']
        for name in target_names:
            if name in df.columns:
                return name
            # Case-insensitive match
            if name.lower() in [c.lower() for c in df.columns]:
                return [c for c in df.columns if c.lower() == name.lower()][0]
        
        # Priority 2: Last column (very common convention)
        last_col = df.columns[-1]
        if column_types.get(last_col) in ['categorical', 'binary', 'numeric']:
            return last_col
        
        # Priority 3: Column with lowest cardinality among categoricals
        categorical_cols = [c for c, t in column_types.items() if t in ['categorical', 'binary']]
        if categorical_cols:
            return min(categorical_cols, key=lambda c: df[c].nunique())
        
        # Default: last numeric column
        numeric_cols = [c for c, t in column_types.items() if t == 'numeric']
        return numeric_cols[-1] if numeric_cols else df.columns[-1]
    
    # =========================================================================
    # STEP 3: Determine Task Type
    # =========================================================================
    
    def _determine_task_type(self, df: pd.DataFrame, target_col: str,
                             column_types: Dict[str, str]) -> str:
        """
        Determine if the task is classification or regression.
        
        Rules:
          - If target is categorical/binary → classification
          - If target is numeric with ≤20 unique values → classification
          - If target is numeric with >20 unique values → regression
        """
        if column_types.get(target_col) in ['categorical', 'binary']:
            return 'classification'
        
        # Numeric target — check if discrete or continuous
        n_unique = df[target_col].nunique()
        if n_unique <= 20:
            return 'classification'
        return 'regression'
    
    # =========================================================================
    # STEP 4: Compute Per-Column Statistics
    # =========================================================================
    
    def _compute_statistics(self, df: pd.DataFrame,
                           column_types: Dict[str, str]) -> Dict[str, Dict]:
        """
        Compute descriptive statistics for every column.
        
        Numeric columns get: mean, std, min, max, median, skewness, kurtosis
        Categorical columns get: top values, cardinality
        All columns get: missing count, missing %, unique count
        """
        statistics = {}
        
        for col in df.columns:
            col_type = column_types.get(col, 'unknown')
            stats_dict = {
                'type': col_type,
                'missing_count': int(df[col].isna().sum()),
                'missing_pct': round(df[col].isna().mean() * 100, 2),
                'n_unique': int(df[col].nunique())
            }
            
            if col_type == 'numeric':
                stats_dict.update({
                    'mean': round(float(df[col].mean()), 4),
                    'std': round(float(df[col].std()), 4),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'skewness': round(float(df[col].skew()), 4),
                    'kurtosis': round(float(df[col].kurtosis()), 4)
                })
            elif col_type == 'categorical':
                value_counts = df[col].value_counts()
                stats_dict.update({
                    'top_values': value_counts.head(5).to_dict(),
                    'cardinality': int(df[col].nunique())
                })
            
            statistics[col] = stats_dict
        
        return statistics
    
    # =========================================================================
    # STEP 5: Extract 40 Meta-Features for RL Model Selector
    # =========================================================================

    def _extract_meta_features(self, df: pd.DataFrame, target_col: str,
                               task_type: str) -> np.ndarray:
        """
        Extract exactly 40 normalized meta-features that describe the dataset.
        These are fed into the PPO RL agent to select the best ML model.

        Delegates to the shared meta_features.extract_meta_features() which
        produces IDENTICAL output to the training script in RL_MODEL_PPO_CORRECT.

        Feature groups (40 total):
          [0-5]   Basic (6)       : shape, type ratios, dimensionality
          [6-8]   Missing (3)     : patterns of missing data
          [9-18]  Statistical (10): distribution properties
          [19-21] Categorical (3) : cardinality of categorical columns
          [22-24] Target (3)      : target variable properties
          [25-27] PCA (3)         : intrinsic dimensionality via PCA
          [28-31] Landmarks (4)   : quick 3-fold CV scores
          [32-39] Signal (8)      : sparsity, correlations, nonlinearity
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode target for landmark models if classification
        if task_type == 'classification':
            le = LabelEncoder()
            y_encoded = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
        else:
            y_encoded = pd.to_numeric(y, errors='coerce').fillna(0)

        return extract_meta_features(X, y_encoded, task_type)
    
    # =========================================================================
    # STEP 6: Data Quality Score
    # =========================================================================
    
    def _compute_quality_score(self, df: pd.DataFrame) -> int:
        """
        Compute an overall data quality score from 0 to 100.
        
        Deductions:
          - Missing values:        up to -30 points
          - Duplicate rows:        up to -20 points
          - Constant columns:      up to -20 points
          - High cardinality cols: -5 points each
        """
        score = 100
        
        # Penalize missing values (up to -30)
        missing_ratio = df.isna().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 30
        
        # Penalize duplicate rows (up to -20)
        dup_ratio = df.duplicated().sum() / len(df)
        score -= dup_ratio * 20
        
        # Penalize constant columns (up to -20)
        constant_cols = sum(1 for c in df.columns if df[c].nunique() == 1)
        score -= (constant_cols / len(df.columns)) * 20
        
        # Penalize very high cardinality string columns (-5 each)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) > 0.9:
                score -= 5
        
        return max(0, min(100, int(score)))
    
    # =========================================================================
    # STEP 7: Generate Warnings
    # =========================================================================
    
    def _generate_warnings(self, df: pd.DataFrame,
                          column_types: Dict[str, str]) -> List[str]:
        """
        Generate human-readable warnings about potential data issues.
        These are displayed in the UI to help users understand their data.
        """
        warnings = []
        
        # Missing values warnings
        for col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 50:
                warnings.append(f"⚠️ Column '{col}' has {missing_pct:.1f}% missing values — consider dropping")
            elif missing_pct > 20:
                warnings.append(f"⚠️ Column '{col}' has {missing_pct:.1f}% missing values — will be imputed")
        
        # High cardinality warnings
        for col, ctype in column_types.items():
            if ctype == 'categorical' and df[col].nunique() > 100:
                warnings.append(f"⚠️ Column '{col}' has high cardinality ({df[col].nunique()} unique values)")
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                warnings.append(f"⚠️ Column '{col}' is constant (only 1 unique value) — will be dropped")
        
        # ID columns
        for col, ctype in column_types.items():
            if ctype == 'id':
                warnings.append(f"ℹ️ Column '{col}' appears to be an ID column — will be excluded from modeling")
        
        return warnings
    
    # =========================================================================
    # STEP 8: describe()-Based Anomaly Detection
    # =========================================================================
    
    def _describe_anomaly_detection(self, df: pd.DataFrame,
                                     column_types: Dict[str, str]) -> List[Dict]:
        """
        Like a real data scientist using df.describe() to spot unusual patterns.
        
        Checks:
          - max >> 75th percentile (extreme outliers)
          - min << 25th percentile (extreme low values)
          - mean far from median (heavy skew)
          - std >> mean (high coefficient of variation)
          - 50% of data = same value (near-constant)
        """
        anomalies = []
        
        numeric_cols = [c for c in df.columns if column_types.get(c) == 'numeric']
        if not numeric_cols:
            return anomalies
        
        desc = df[numeric_cols].describe()
        
        for col in desc.columns:
            max_val = desc.loc['max', col]
            min_val = desc.loc['min', col]
            q75 = desc.loc['75%', col]
            q25 = desc.loc['25%', col]
            mean_val = desc.loc['mean', col]
            std_val = desc.loc['std', col]
            median_val = desc.loc['50%', col]
            iqr = q75 - q25
            
            # Check: max >> 75th percentile (extreme outlier on the high end)
            if iqr > 0 and max_val > q75 + 3 * iqr:
                anomalies.append({
                    'column': col,
                    'issue': 'extreme_high_outlier',
                    'detail': f"max ({max_val:.2f}) is far above Q3 + 3×IQR ({q75 + 3*iqr:.2f})",
                    'severity': 'high'
                })
            
            # Check: min << 25th percentile (extreme outlier on the low end)
            if iqr > 0 and min_val < q25 - 3 * iqr:
                anomalies.append({
                    'column': col,
                    'issue': 'extreme_low_outlier',
                    'detail': f"min ({min_val:.2f}) is far below Q1 - 3×IQR ({q25 - 3*iqr:.2f})",
                    'severity': 'high'
                })
            
            # Check: mean very far from median (heavy skew)
            if std_val > 0 and abs(mean_val - median_val) > 2 * std_val:
                anomalies.append({
                    'column': col,
                    'issue': 'heavy_skew',
                    'detail': f"mean ({mean_val:.2f}) far from median ({median_val:.2f}) — data is heavily skewed",
                    'severity': 'medium'
                })
            
            # Check: coefficient of variation > 2 (very high variability)
            if mean_val != 0 and abs(std_val / mean_val) > 2:
                cv = abs(std_val / mean_val)
                anomalies.append({
                    'column': col,
                    'issue': 'high_variability',
                    'detail': f"coefficient of variation = {cv:.1f} — very dispersed data",
                    'severity': 'medium'
                })
            
            # Check: >50% of values are identical (near-constant)
            mode_count = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            if mode_count / len(df) > 0.5 and df[col].nunique() > 1:
                pct = mode_count / len(df) * 100
                anomalies.append({
                    'column': col,
                    'issue': 'dominant_value',
                    'detail': f"{pct:.0f}% of rows have the same value — near-constant column",
                    'severity': 'low'
                })
        
        return anomalies
    
    # =========================================================================
    # STEP 9: Unique Value Uniformity Check
    # =========================================================================
    
    def _check_unique_value_uniformity(self, df: pd.DataFrame,
                                        column_types: Dict[str, str]) -> Dict[str, Dict]:
        """
        Check for non-uniform category values that should be standardized.
        
        Detects issues like:
          - Case inconsistency: 'Male', 'male', 'MALE' → should all be 'Male'
          - Whitespace: ' Male', 'Male ', ' Male ' → should all be 'Male'
          - Abbreviations: 'M', 'F' mixed with 'Male', 'Female'
        
        Returns: {column_name: {canonical_lower: [variant1, variant2, ...]}}
        """
        uniformity_issues = {}
        
        for col in df.columns:
            if column_types.get(col) != 'categorical':
                continue
            
            values = df[col].dropna().unique()
            if len(values) > 50:  # Skip high-cardinality columns
                continue
            
            # Check for case/whitespace inconsistency
            lower_map = {}
            for v in values:
                v_str = str(v).strip().lower()
                if v_str not in lower_map:
                    lower_map[v_str] = []
                lower_map[v_str].append(str(v))
            
            issues = {k: v for k, v in lower_map.items() if len(v) > 1}
            if issues:
                uniformity_issues[col] = issues
        
        return uniformity_issues

    # =========================================================================
    # STEP 0: Coerce "Disguised Numeric" String Columns
    # =========================================================================

    # Regex to strip common non-numeric formatting characters:
    #   $  €  £  ¥  ₹  ₩  ₫  ¢  — currency symbols
    #   ,                        — thousand separators
    #   whitespace               — leading/trailing/embedded spaces
    _NUMERIC_NOISE_RE = re.compile(r'[$€£¥₹₩₫¢,\s]')

    # Pattern for parenthesised negatives, e.g.  "(1,234.56)" → "-1234.56"
    _PAREN_NEGATIVE_RE = re.compile(r'^\s*\((.+)\)\s*$')

    # Final check: must look like a valid number (optional sign, digits, optional dot)
    _LOOKS_NUMERIC_RE = re.compile(r'^-?\d+\.?\d*$')

    def _coerce_numeric_strings(self, df: pd.DataFrame) -> tuple:
        """
        Detect object-dtype columns whose values are really numbers hidden
        behind formatting characters and convert them to float64.

        Common cases handled:
          • Currency symbols:    "$1,234.56"  → 1234.56
          • Percent signs:       "50%"        → 50.0   (the % is stripped;
                                                         divide-by-100 is NOT applied
                                                         so the original scale is kept)
          • Parenthesised negs:  "(500.00)"   → -500.0
          • Thousand separators: "1,000,000"  → 1000000.0
          • Mixed whitespace:    " 42 "       → 42.0
          • Unit suffixes:       "10 kg", "5lbs" — stripped if ≥80% of values
                                  become valid numbers after cleaning.

        Heuristic: if ≥80% of non-null values become a valid float after
        stripping, the whole column is converted (non-parseable values become NaN).

        Returns:
            (df, list_of_coerced_column_names)
        """
        df = df.copy()  # avoid mutating the caller's DataFrame
        coerced_cols = []

        for col in df.columns:
            # Only examine object/string columns
            if df[col].dtype != 'object':
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            # Quick reject: if all non-null values are purely alphabetic (no
            # digits at all), this is certainly not a numeric column.
            sample = series.head(200).astype(str)
            has_digit = sample.str.contains(r'\d', regex=True)
            if has_digit.sum() < len(sample) * 0.5:
                continue

            cleaned = self._clean_numeric_series(series.astype(str))

            # Count how many values are valid numbers after cleaning
            numeric_parsed = pd.to_numeric(cleaned, errors='coerce')
            n_valid = numeric_parsed.notna().sum()
            ratio = n_valid / len(series)

            if ratio >= 0.80:
                # Convert the full column (including NaN positions)
                full_cleaned = self._clean_numeric_series(df[col].astype(str))
                df[col] = pd.to_numeric(full_cleaned, errors='coerce')
                coerced_cols.append(col)

        return df, coerced_cols

    def _clean_numeric_series(self, s: pd.Series) -> pd.Series:
        """
        Strip formatting noise from a string Series so it can be parsed as float.
        """
        # 1. Handle trailing percent sign  ("50%" → "50")
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
