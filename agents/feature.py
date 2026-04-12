# agents/feature.py

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler,
    MinMaxScaler, RobustScaler
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import TargetEncoder
from agents.base import BaseAgent


class FeatureAgent(BaseAgent):
    """
    Agent responsible for feature engineering.
    
    This is the THIRD agent in the pipeline. It takes cleaned data and:
      1. Removes ID columns (not useful for modeling)
      2. Encodes categorical columns (binary→label, low-card→one-hot, high-card→target)
      3. Scales numerical columns (skewed→RobustScaler, normal→StandardScaler)
      4. Selects top features if dimensionality is too high (>50 features)
      5. Encodes the target variable (LabelEncoder for classification)
    
    Stores all encoders/scalers so they can be reused for new predictions.
    
    Owner: Bhavana
    """
    
    def __init__(self):
        super().__init__("FeatureAgent")
        self.encoders = {}   # Stores fitted encoders for each column
        self.scalers = {}    # Stores fitted scalers
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features for modeling"""
        self.log("Starting feature engineering...")
        
        df = state['current_data'].copy()
        target_col = state['target_column']
        task_type = state['task_type']
        column_types = state['profile_report']['column_types']
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        feature_report = {
            'encoding': {},
            'scaling': {},
            'feature_selection': {},
            'new_features': [],
            'dropped_columns': []
        }
        
        # =====================================================================
        # Step 1: Remove non-modelable columns (ID, datetime, text)
        # Bool columns are converted to int (0/1).
        # Datetime columns cannot be fed to sklearn — drop them.
        # =====================================================================
        drop_types = {'id', 'datetime', 'text'}
        drop_cols = [c for c, t in column_types.items() if t in drop_types and c in X.columns]
        X = X.drop(columns=drop_cols)
        feature_report['dropped_columns'] = drop_cols
        self.log(f"Removed non-modelable columns ({', '.join(drop_types)}): {drop_cols}")

        # Snapshot of column names BEFORE encoding — used by the Predict UI
        # so users see original column names (e.g. "m_dep") not OHE-expanded
        # names (e.g. "m_dep_0.2", "m_dep_0.3").
        original_input_columns = X.columns.tolist()

        # Convert bool columns to int
        bool_cols = [c for c in X.columns if X[c].dtype == bool or str(X[c].dtype) == 'bool']
        for col in bool_cols:
            X[col] = X[col].astype(int)
        if bool_cols:
            self.log(f"Converted bool columns to int: {bool_cols}")

        # Drop any remaining non-numeric, non-object columns sklearn can't handle
        # (e.g. datetime64 that wasn't caught by column_types)
        bad_cols = [c for c in X.columns
                    if not pd.api.types.is_numeric_dtype(X[c])
                    and not pd.api.types.is_object_dtype(X[c])]
        if bad_cols:
            X = X.drop(columns=bad_cols)
            self.log(f"Dropped unsupported dtype columns: {bad_cols}")
        
        # =====================================================================
        # Step 2: Encode categorical columns
        # =====================================================================
        X, encoding_info = self._encode_categoricals(X, y, column_types, task_type)
        feature_report['encoding'] = encoding_info
        
        # =====================================================================
        # Step 3: Feature selection (only if >50 features)
        # =====================================================================
        if len(X.columns) > 50:
            X, selection_info = self._select_features(X, y, task_type)
            feature_report['feature_selection'] = selection_info
        else:
            feature_report['feature_selection'] = {
                'method': 'none',
                'reason': f'Only {len(X.columns)} features — no selection needed'
            }

        # =====================================================================
        # Step 4: VIF-based multicollinearity removal
        # =====================================================================
        X, vif_info = self._remove_high_vif_features(X, y=y)
        feature_report['vif_analysis'] = vif_info
        if vif_info.get('removed_features'):
            self.log(f"VIF: Removed {len(vif_info['removed_features'])} multicollinear features")

        # =====================================================================
        # Step 5: Scale numerical columns (after VIF so scaler matches final
        # feature set — avoids sklearn feature-name mismatch at predict time)
        # =====================================================================
        X, scaling_info = self._scale_numericals(X, column_types)
        feature_report['scaling'] = scaling_info
        
        # =====================================================================
        # Step 5: Encode target variable (classification only)
        # =====================================================================
        if task_type == 'classification':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
            self.encoders['target'] = le
            self.log(f"Encoded target: {list(le.classes_)}")
        
        # =====================================================================
        # Final: enforce all columns are numeric float64 for sklearn.
        # Use explicit dtype checks — more reliable than try/except astype
        # across pandas/numpy versions.
        # =====================================================================
        cols_to_drop = []
        for col in list(X.columns):
            if (pd.api.types.is_datetime64_any_dtype(X[col]) or
                    pd.api.types.is_timedelta64_dtype(X[col])):
                cols_to_drop.append(col)
            elif pd.api.types.is_bool_dtype(X[col]):
                X[col] = X[col].astype(np.float64)
            elif pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(np.float64)
            else:
                cols_to_drop.append(col)
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            self.log(f"Dropped non-numeric columns before modeling: {cols_to_drop}")

        # Fill any remaining NaN/inf so sklearn doesn't error
        X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

        self.log(f"Final X shape: {X.shape}, dtypes: {X.dtypes.value_counts().to_dict()}")

        # Update pipeline state
        state['X'] = X
        state['y'] = y
        state['feature_names'] = X.columns.tolist()
        state['original_input_columns'] = original_input_columns
        state['feature_report'] = feature_report
        state['encoders'] = self.encoders
        state['scalers'] = self.scalers
        state['stage'] = 'featured'
        
        self.log(f"Feature engineering complete. {len(X.columns)} features ready for modeling.")
        return state
    
    # =========================================================================
    # STEP 2: Categorical Encoding
    # =========================================================================
    
    def _encode_categoricals(self, X: pd.DataFrame, y: pd.Series,
                            column_types: Dict[str, str],
                            task_type: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical columns using the best strategy per column.
        
        Strategy selection:
          - 2 unique values (binary)  → LabelEncoder (0/1)
          - ≤10 unique values         → One-Hot Encoding (drop_first to avoid multicollinearity)
          - >10 unique values         → Target Encoding (uses target mean per category)
        
        Target encoding is preferred for high-cardinality because:
          - One-hot would create too many columns
          - It captures the relationship between category and target
          - Smoothing prevents overfitting on rare categories
        """
        encoding_info = {}
        
        categorical_cols = [c for c in X.columns if column_types.get(c) == 'categorical']
        
        for col in categorical_cols:
            n_unique = X[col].nunique()
            
            if n_unique == 2:
                # Binary → Label encoding (0/1)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                encoding_info[col] = {
                    'method': 'label',
                    'n_unique': n_unique,
                    'mapping': dict(zip(le.classes_, le.transform(le.classes_).tolist())),
                    'categories': le.classes_.tolist(),  # for Predict UI selectbox
                }
                self.log(f"  {col}: Label encoded (binary, {n_unique} values)")

            elif n_unique <= 10:
                # Low cardinality → One-Hot encoding (dtype=int avoids bool columns)
                # Capture all categories BEFORE get_dummies (drop_first removes the first)
                all_categories = sorted(X[col].dropna().astype(str).unique().tolist())
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True).astype(int)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                encoding_info[col] = {
                    'method': 'onehot',
                    'n_unique': n_unique,
                    'new_cols': dummies.columns.tolist(),
                    'categories': all_categories,  # for Predict UI selectbox (all values)
                }
                self.log(f"  {col}: One-hot encoded ({n_unique} values -> {len(dummies.columns)} columns)")
            
            else:
                # High cardinality → Target encoding
                te = TargetEncoder(cols=[col], smoothing=1.0)
                X[col] = te.fit_transform(X[col], y)
                self.encoders[col] = te
                encoding_info[col] = {
                    'method': 'target',
                    'n_unique': n_unique,
                    'reason': 'High cardinality — target encoding preserves info without explosion'
                }
                self.log(f"  {col}: Target encoded ({n_unique} values)")
        
        return X, encoding_info
    
    # =========================================================================
    # STEP 3: Numerical Scaling
    # =========================================================================
    
    def _scale_numericals(self, X: pd.DataFrame,
                         column_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale numerical columns to normalize their ranges.
        
        Strategy selection (based on average skewness):
          - Mean skewness > 1   → RobustScaler (resistant to outliers, uses median/IQR)
          - Mean skewness ≤ 1   → StandardScaler (zero mean, unit variance)
        
        Why scale?
          - Many ML algorithms (SVM, KNN, LogReg) are sensitive to feature magnitude
          - Tree-based models don't need it, but it doesn't hurt
          - Ensures fair comparison between features
        """
        scaling_info = {}
        
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        
        if len(numeric_cols) == 0:
            return X, scaling_info
        
        # Check distribution to pick scaler
        skewness = X[numeric_cols].skew().abs().mean()
        
        if skewness > 1:
            scaler = RobustScaler()
            method = 'robust'
            reason = f'Mean absolute skewness = {skewness:.2f} (>1) -> using RobustScaler'
        else:
            scaler = StandardScaler()
            method = 'standard'
            reason = f'Mean absolute skewness = {skewness:.2f} (<=1) -> using StandardScaler'
        
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers['numeric'] = scaler
        
        scaling_info = {
            'method': method,
            'reason': reason,
            'columns_scaled': numeric_cols,
            'n_columns': len(numeric_cols)
        }
        
        self.log(f"Scaled {len(numeric_cols)} numeric columns with {method} scaler")
        return X, scaling_info
    
    # =========================================================================
    # STEP 4: Feature Selection (for high-dimensional data)
    # =========================================================================
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series,
                        task_type: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Select the most important features when dimensionality is too high (>50).
        
        Two-stage process:
          Stage 1: Remove highly correlated features (|correlation| > 0.95)
                   These are redundant — keeping both hurts more than helps
          
          Stage 2: Rank remaining by Mutual Information and keep top 50
                   MI measures general dependency (not just linear), works for both
                   classification and regression
        """
        self.log("Performing feature selection (>50 features detected)...")
        
        # Stage 1: Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X = X.drop(columns=to_drop)
        self.log(f"  Removed {len(to_drop)} highly correlated features")
        
        # Stage 2: Mutual Information ranking
        if task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        
        # Keep top 50 features
        top_features = mi_df.head(50)['feature'].tolist()
        X = X[top_features]
        
        selection_info = {
            'method': 'correlation_filter + mutual_information',
            'dropped_correlated': to_drop,
            'n_dropped_correlated': len(to_drop),
            'selected_features': top_features,
            'n_selected': len(top_features),
            'top_10_mi_scores': mi_df.head(10).to_dict('records')
        }
        
        self.log(f"  Selected top {len(top_features)} features by mutual information")
        return X, selection_info
    
    # =========================================================================
    # STEP 4b: VIF-Based Multicollinearity Removal
    # =========================================================================
    
    def _compute_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Variance Inflation Factor for each numeric feature.
        
        VIF measures how much a feature is explained by other features:
          VIF = 1 / (1 - R²)  where R² is from regressing feature j on all others
          
          VIF = 1    → no multicollinearity
          VIF = 1-5  → moderate (acceptable)
          VIF = 5-10 → high (concerning)
          VIF > 10   → severe (should remove one of the correlated features)
        
        Uses sklearn's LinearRegression instead of statsmodels to avoid
        an extra dependency.
        """
        from sklearn.linear_model import LinearRegression
        
        vif_data = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return vif_data
        
        X_numeric = X[numeric_cols].fillna(0)
        
        for col in numeric_cols:
            y_vif = X_numeric[col].values
            X_vif = X_numeric.drop(columns=[col]).values
            
            if X_vif.shape[1] == 0:
                continue
            
            try:
                lr = LinearRegression()
                lr.fit(X_vif, y_vif)
                r2 = lr.score(X_vif, y_vif)
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
                vif_data[col] = round(vif, 2)
            except Exception:
                vif_data[col] = 1.0
        
        return vif_data
    
    def _remove_high_vif_features(self, X: pd.DataFrame,
                                    threshold: float = 10.0,
                                    y: pd.Series = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Iteratively remove multicollinear features until all VIF scores are
        below threshold.

        When multiple features exceed the threshold, instead of blindly dropping
        the highest-VIF one, we drop the feature with the LOWEST absolute
        correlation to the target (y).  This preserves predictive power — e.g.
        petal_length and petal_width are both collinear, but whichever is more
        correlated with the target is kept.

        Args:
            X: Feature DataFrame (numeric columns only are examined)
            threshold: VIF threshold (default 10)
            y: Target series used to break ties in favour of predictive features

        Returns:
            Tuple of (cleaned DataFrame, VIF analysis report)
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Skip VIF for small feature sets: tree models handle collinearity fine
        # and removing features from a small set hurts more than it helps.
        if len(numeric_cols) < 10:
            return X, {
                'method': 'vif',
                'threshold': threshold,
                'removed_features': [],
                'final_vif_scores': self._compute_vif(X),
                'reason': f'Only {len(numeric_cols)} numeric features — VIF removal skipped to preserve all features'
            }

        # Pre-compute target correlations once (used to pick which to drop)
        target_corr = {}
        if y is not None:
            try:
                y_numeric = pd.to_numeric(y, errors='coerce').fillna(0)
                for col in numeric_cols:
                    target_corr[col] = abs(float(X[col].corr(y_numeric)))
            except Exception:
                pass

        removed = []
        max_iterations = min(len(numeric_cols), 20)

        for _ in range(max_iterations):
            vif_scores = self._compute_vif(X)
            if not vif_scores:
                break

            max_vif_val = max(vif_scores.values())
            if max_vif_val <= threshold:
                break

            # Candidates: all features above the threshold
            candidates = [col for col, v in vif_scores.items() if v > threshold]

            if target_corr:
                # Drop the candidate least correlated with the target
                col_to_drop = min(candidates, key=lambda c: target_corr.get(c, 0.0))
            else:
                # Fallback: drop the one with the highest VIF
                col_to_drop = max(candidates, key=lambda c: vif_scores[c])

            X = X.drop(columns=[col_to_drop])
            removed.append({'feature': col_to_drop, 'vif': vif_scores[col_to_drop]})
            self.log(
                f"  VIF: Removed '{col_to_drop}' "
                f"(VIF={vif_scores[col_to_drop]:.1f}, "
                f"target_corr={target_corr.get(col_to_drop, float('nan')):.3f})"
            )

        final_vif = self._compute_vif(X)

        return X, {
            'method': 'vif',
            'threshold': threshold,
            'removed_features': removed,
            'n_removed': len(removed),
            'final_vif_scores': final_vif
        }
