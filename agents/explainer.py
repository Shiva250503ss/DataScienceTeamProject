# agents/explainer.py

"""
Explainer Agent — Model Explainability with SHAP, LIME, and LLM Narratives.

This is the SIXTH agent in the ML pipeline. After models are trained, it
provides explanations for HOW and WHY the model makes predictions.

Three levels of explainability:

  1. GLOBAL EXPLANATIONS (what features matter overall?)
     - SHAP summary plot (beeswarm)
     - SHAP feature importance (bar)
     - Feature interaction effects

  2. LOCAL EXPLANATIONS (why did the model predict X for this row?)
     - SHAP waterfall plot for individual predictions
     - LIME explanation for individual predictions
     - Counterfactual: "If feature X was Y, the prediction would change"

  3. LLM NARRATIVES (plain-English explanations)
     - LLM reads the SHAP values and writes a human-readable explanation
     - "The model predicted 'high risk' mainly because the customer's
        income ($23K) is below average and their debt ratio (0.85) is
        very high. Reducing the debt ratio below 0.5 would likely
        change the prediction."

Owner: Explainability Team
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import shap
import lime
import lime.lime_tabular

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agents.base import BaseAgent

warnings.filterwarnings('ignore')


# =========================================================================
# LLM PROMPT TEMPLATES
# =========================================================================

PIPELINE_NARRATIVE_PROMPT = """You are a senior data scientist presenting a complete end-to-end ML pipeline analysis to a business audience. Write a structured report covering the entire pipeline — from raw data through to business recommendations.

━━━ PIPELINE SUMMARY ━━━
Dataset   : {dataset_name}
Target    : {target_col} ({task_type})
Rows/Cols : {n_rows} rows × {n_cols} columns
Quality   : {quality_score}/100

━━━ DATA QUALITY FINDINGS ━━━
{data_issues}

━━━ CLEANING STEPS APPLIED ━━━
{cleaning_summary}

━━━ FEATURE ENGINEERING ━━━
{feature_summary}

━━━ VARIABLE CORRELATIONS ━━━
{correlations}

━━━ MODEL SELECTION & PERFORMANCE ━━━
Models evaluated (CV scores):
{model_scores}
Final model chosen: {best_model} ({metric_name} = {best_score:.4f})

━━━ KEY PREDICTORS (SHAP) ━━━
{top_features}

Write your report in exactly these 6 sections using bold headers. Each section should be 2–4 clear sentences. Use plain English — no jargon. Be specific about the numbers where it adds value.

**1. Dataset Overview & Data Vulnerabilities**
Describe the dataset size, target variable, quality score, and the main data issues found (missing values, anomalies, outliers).

**2. Data Cleaning Applied**
Summarise what cleaning was done: duplicate removal, imputation strategies, outlier treatment. Explain WHY it was needed.

**3. Feature Engineering**
Describe what transformations were applied: encoding, scaling, multicollinearity removal (VIF). Explain the impact on model readiness.

**4. Key Variable Relationships**
Which variables are highly correlated or collinear? What does this mean for interpretation? Highlight any surprising patterns.

**5. Model Selection & Performance**
Which model won and why. Compare it against alternatives. Explain what the {metric_name} score of {best_score:.4f} means in practical terms.

**6. Business Insights & Recommendations**
Give 3–4 actionable business recommendations based on the top predictors. Frame them as decisions the business can take, not technical observations.
"""

LOCAL_EXPLANATION_PROMPT = """You are explaining a SINGLE prediction to a non-technical user.

The model predicted: {prediction}
{confidence_text}

The key factors behind this prediction (SHAP values):
{shap_explanation}

The actual feature values for this data point:
{feature_values}

Write a 2-3 sentence explanation in PLAIN ENGLISH:
- Why did the model make this prediction?
- Which features pushed the prediction in this direction?
- What would need to change for a different prediction?
"""


class ExplainerAgent(BaseAgent):
    """
    Agent responsible for model explainability.

    Takes the trained models from ModelerAgent and generates:
      - SHAP global explanations (feature importance across all data)
      - SHAP local explanations (individual prediction breakdowns)
      - LIME local explanations (alternative local method)
      - LLM-generated narratives in plain English

    Outputs:
      state['explanations'] = {
          'shap_values': np.array,
          'shap_importance': pd.DataFrame,
          'global_narrative': str,
          'local_explanations': [...],
          'charts': {...}
      }
    """

    def __init__(self):
        super().__init__("ExplainerAgent")
        self.colors = {
            'positive': '#DC2626',   # Red = pushes prediction UP
            'negative': '#2563EB',   # Blue = pushes prediction DOWN
            'neutral':  '#9CA3AF',
        }
        self.template = 'plotly_dark'

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive model explanations."""
        self.log("Starting model explainability analysis...")

        X = state['X']
        y = state['y']
        task_type = state['task_type']
        target_col = state['target_column']
        feature_names = state.get('feature_names', X.columns.tolist() if isinstance(X, pd.DataFrame) else None)
        best_model_name = state.get('best_model_name', 'Ensemble')
        trained_models = state.get('trained_models', {})
        ensemble_model = state.get('ensemble_model')
        cv_scores = state.get('cv_scores', {})

        # Pick the best single model for SHAP (ensembles can be tricky)
        if best_model_name in trained_models:
            explain_model = trained_models[best_model_name]
            model_name = best_model_name
        elif trained_models:
            model_name = list(trained_models.keys())[0]
            explain_model = trained_models[model_name]
        elif ensemble_model is not None:
            explain_model = ensemble_model
            model_name = 'Ensemble'
        else:
            self.log("No trained model available — skipping explainability.")
            state['explanation_report'] = {'error': 'No model available to explain'}
            state['stage'] = 'explained'
            return state

        self.log(f"Explaining model: {model_name}")

        # Create output directory
        output_dir = state.get('output_dir', './output')
        explain_dir = os.path.join(output_dir, 'explanations')
        os.makedirs(explain_dir, exist_ok=True)

        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        # Fix NumPy boolean subtract error: SHAP internally does
        # bool_array - bool_array which newer NumPy forbids.
        # Convert boolean columns to int BEFORE passing to SHAP/LIME.
        bool_cols = X.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            X = X.copy()
            X[bool_cols] = X[bool_cols].astype(int)
            self.log(f"  Converted {len(bool_cols)} boolean columns to int for SHAP compatibility")

        explanations = {
            'model_name': model_name,
            'charts': {}
        }

        # =====================================================================
        # Step 1: SHAP Global Explanations
        # =====================================================================
        self.log("Step 1: Computing SHAP values (global explanations)...")
        shap_results = self._compute_shap_values(explain_model, X, task_type)

        if shap_results:
            shap_values, shap_importance = shap_results
            explanations['shap_values'] = shap_values
            explanations['shap_importance'] = shap_importance

            # Generate SHAP charts
            self.log("  Generating SHAP plots...")
            shap_charts = self._create_shap_charts(
                shap_values, X, shap_importance, task_type, explain_dir
            )
            explanations['charts'].update(shap_charts)

        # =====================================================================
        # Step 2: LIME Local Explanations (for sample predictions)
        # =====================================================================
        self.log("Step 2: Computing LIME explanations (local)...")
        lime_results = self._compute_lime_explanations(
            explain_model, X, y, task_type, n_samples=5
        )
        if lime_results:
            explanations['lime_explanations'] = lime_results
            lime_charts = self._create_lime_charts(lime_results, explain_dir)
            explanations['charts'].update(lime_charts)

        # =====================================================================
        # Step 3: Feature Interaction Chart
        # =====================================================================
        if shap_results:
            self.log("Step 3: Feature interaction analysis...")
            interaction_chart = self._create_interaction_chart(
                shap_values, X, shap_importance, explain_dir
            )
            if interaction_chart:
                explanations['charts'].update(interaction_chart)

        # =====================================================================
        # Step 4: LLM Global Narrative
        # =====================================================================
        self.log("Step 4: Generating LLM explanation narrative...")
        metric_name = 'Accuracy' if task_type == 'classification' else 'R²'
        metric_value = cv_scores.get(model_name, state.get('ensemble_score', 0))

        global_narrative = self._generate_global_narrative(
            model_name, task_type, target_col, metric_name, metric_value,
            shap_importance if shap_results else None,
            state=state,
        )
        explanations['global_narrative'] = global_narrative

        # =====================================================================
        # Step 5: Local LLM Narratives (for sample predictions)
        # =====================================================================
        self.log("Step 5: Generating local prediction narratives...")
        local_narratives = self._generate_local_narratives(
            explain_model, X, y, task_type,
            shap_values if shap_results else None,
            n_samples=3
        )
        explanations['local_narratives'] = local_narratives

        # =====================================================================
        # Step 6: Summary Explainability Dashboard
        # =====================================================================
        self.log("Step 6: Building explainability dashboard...")
        summary_chart = self._create_summary_dashboard(
            explanations, model_name, task_type, metric_name, metric_value,
            explain_dir
        )
        explanations['charts']['summary_dashboard'] = summary_chart

        # Update pipeline state
        state['explanations'] = explanations
        state['explain_dir'] = explain_dir
        state['stage'] = 'explained'

        self.log(f"Explainability complete! {len(explanations['charts'])} charts in {explain_dir}")
        return state

    # =========================================================================
    # STEP 1: SHAP VALUES
    # =========================================================================

    def _compute_shap_values(self, model, X: pd.DataFrame,
                              task_type: str) -> Optional[Tuple]:
        """
        Compute SHAP values for the model.

        SHAP (SHapley Additive exPlanations) assigns each feature a score
        for each prediction, showing how much that feature pushed the
        prediction up or down from the baseline.

        Returns:
            Tuple of (shap_values array, importance DataFrame) or None
        """
        try:
            # Choose the right SHAP explainer
            model_type = type(model).__name__

            # Slow models (O(n²+) per predict call) need aggressive sample limits
            slow_model_types = ('SVC', 'SVR', 'KNeighborsClassifier', 'KNeighborsRegressor')
            is_slow_model = model_type in slow_model_types

            tree_model_types = (
                'XGBClassifier', 'XGBRegressor',
                'LGBMClassifier', 'LGBMRegressor',
                'RandomForestClassifier', 'RandomForestRegressor',
                'ExtraTreesClassifier', 'ExtraTreesRegressor',
                'GradientBoostingClassifier', 'GradientBoostingRegressor',
                'CatBoostClassifier', 'CatBoostRegressor',
            )
            if model_type in tree_model_types:
                # Tree SHAP — try with check_additivity=False first for robustness
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X, check_additivity=False)
                except Exception as tree_err:
                    self.log(f"  TreeExplainer failed ({tree_err}), falling back to KernelExplainer")
                    n_explain = min(50, len(X))
                    X_explain = X.sample(n=n_explain, random_state=42)
                    explainer = shap.KernelExplainer(
                        model.predict_proba if hasattr(model, 'predict_proba') and task_type == 'classification'
                        else model.predict,
                        background.iloc[:min(50, len(background))]
                    )
                    shap_values = explainer.shap_values(X_explain, nsamples=100)
                    X = X_explain
            else:
                # Kernel SHAP — model-agnostic but slower
                # For slow models (SVC/SVR/KNN): use far fewer samples
                if is_slow_model:
                    n_background = min(30, len(X))
                    n_explain = min(30, len(X))
                    nsamples = 50       # perturbations per sample — fast enough for SVR
                    self.log(f"  Using fast SHAP settings for {model_type} "
                             f"(bg={n_background}, explain={n_explain}, nsamples={nsamples})")
                else:
                    n_background = min(100, len(X))
                    n_explain = min(100, len(X))
                    nsamples = 'auto'   # SHAP default

                background = shap.kmeans(X, min(n_background, len(X)))
                X_explain = X.sample(n=n_explain, random_state=42)

                explainer = shap.KernelExplainer(
                    model.predict_proba if hasattr(model, 'predict_proba') and task_type == 'classification'
                    else model.predict,
                    background
                )
                shap_values = explainer.shap_values(X_explain, nsamples=nsamples)
                X = X_explain  # Use the smaller sample for charts

            # Handle multi-class SHAP values
            # Newer SHAP may return 3D ndarray (n_samples, n_features, n_classes)
            # Older SHAP returns a list of 2D arrays, one per class
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # (n_samples, n_features, n_classes) — average abs across classes
                shap_values = np.mean(np.abs(shap_values), axis=2)
            elif isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]  # Binary: positive class
                else:
                    # Multi-class: mean absolute across classes
                    shap_values = np.mean(
                        np.abs(np.array(shap_values)), axis=0
                    )

            # Ensure 2D float array
            shap_values = np.array(shap_values, dtype=float)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            # Compute feature importance (mean absolute SHAP)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            self.log(f"  SHAP computed for {len(X)} samples, {len(X.columns)} features")
            self.log(f"  Top 3 features: {importance_df.head(3)['feature'].tolist()}")

            return shap_values, importance_df

        except Exception as e:
            self.log(f"  SHAP computation failed: {e}")
            return None

    # =========================================================================
    # STEP 2: LIME EXPLANATIONS
    # =========================================================================

    def _compute_lime_explanations(self, model, X: pd.DataFrame,
                                    y: pd.Series, task_type: str,
                                    n_samples: int = 5) -> Optional[List[Dict]]:
        """
        Compute LIME explanations for a few sample predictions.

        LIME creates a simple local approximation around each prediction,
        showing which features were most important FOR THAT SPECIFIC PREDICTION.
        """
        try:
            feature_names = X.columns.tolist()

            if task_type == 'classification':
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X.values,
                    feature_names=feature_names,
                    class_names=[str(c) for c in sorted(y.unique())],
                    mode='classification',
                    random_state=42
                )
            else:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X.values,
                    feature_names=feature_names,
                    mode='regression',
                    random_state=42
                )

            # Pick diverse samples (different predictions)
            predictions = model.predict(X)
            unique_preds = pd.Series(predictions).unique()

            sample_indices = []
            for pred_val in unique_preds[:n_samples]:
                idx = np.where(predictions == pred_val)[0]
                if len(idx) > 0:
                    sample_indices.append(idx[0])
                if len(sample_indices) >= n_samples:
                    break

            # Fill remaining if needed
            while len(sample_indices) < n_samples and len(sample_indices) < len(X):
                idx = np.random.randint(0, len(X))
                if idx not in sample_indices:
                    sample_indices.append(idx)

            results = []
            for idx in sample_indices:
                try:
                    exp = explainer.explain_instance(
                        X.iloc[idx].values,
                        model.predict_proba if hasattr(model, 'predict_proba') and task_type == 'classification'
                        else model.predict,
                        num_features=10,
                        num_samples=1000
                    )

                    feature_weights = exp.as_list()
                    prediction = predictions[idx]
                    actual = y.iloc[idx] if idx < len(y) else None

                    results.append({
                        'index': int(idx),
                        'prediction': str(prediction),
                        'actual': str(actual) if actual is not None else None,
                        'feature_weights': feature_weights,
                        'feature_values': {col: float(X.iloc[idx][col])
                                          for col in feature_names[:15]}
                    })
                except Exception as e:
                    self.log(f"  LIME failed for sample {idx}: {e}")

            self.log(f"  LIME computed for {len(results)} samples")
            return results if results else None

        except Exception as e:
            self.log(f"  LIME computation failed: {e}")
            return None

    # =========================================================================
    # SHAP CHARTS
    # =========================================================================

    def _create_shap_charts(self, shap_values: np.ndarray, X: pd.DataFrame,
                             importance: pd.DataFrame, task_type: str,
                             output_dir: str) -> Dict:
        """Generate SHAP visualization charts."""
        charts = {}

        # --- 1. Feature Importance Bar Chart ---
        top_n = min(20, len(importance))
        imp = importance.head(top_n).sort_values('importance', ascending=True)

        fig = go.Figure(data=[go.Bar(
            x=imp['importance'],
            y=imp['feature'],
            orientation='h',
            marker=dict(
                color=imp['importance'],
                colorscale='Reds',
                colorbar=dict(title='Mean |SHAP|')
            ),
            text=[f'{v:.4f}' for v in imp['importance']],
            textposition='auto'
        )])
        fig.update_layout(
            title=dict(text=f'SHAP Feature Importance (Top {top_n})', font=dict(size=20)),
            xaxis_title='Mean |SHAP Value|',
            template=self.template,
            height=max(400, top_n * 28)
        )
        charts['shap_importance'] = fig
        self._save_figure(fig, output_dir, 'shap_importance')

        # --- 2. SHAP Beeswarm Plot (using Plotly scatter) ---
        top_features = importance.head(15)['feature'].tolist()
        top_indices = [X.columns.get_loc(f) for f in top_features if f in X.columns]

        if len(top_indices) > 0:
            fig = go.Figure()
            for i, feat_idx in enumerate(top_indices):
                feat_name = X.columns[feat_idx]
                feat_shap = shap_values[:, feat_idx]
                feat_vals = X.iloc[:, feat_idx].values

                # Normalize feature values for coloring
                fmin, fmax = np.nanmin(feat_vals), np.nanmax(feat_vals)
                if fmax > fmin:
                    norm_vals = (feat_vals - fmin) / (fmax - fmin)
                else:
                    norm_vals = np.zeros_like(feat_vals)

                # Add jitter for visibility
                jitter = np.random.normal(0, 0.1, size=len(feat_shap))

                # Subsample for performance
                n_plot = min(500, len(feat_shap))
                sample_idx = np.random.choice(len(feat_shap), n_plot, replace=False)

                fig.add_trace(go.Scatter(
                    x=feat_shap[sample_idx],
                    y=[i + j for j in jitter[sample_idx]],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=norm_vals[sample_idx],
                        colorscale='RdBu_r',
                        colorbar=dict(title='Feature Value') if i == 0 else None,
                        showscale=(i == 0),
                        opacity=0.6
                    ),
                    name=feat_name,
                    showlegend=False,
                    hovertemplate=f'{feat_name}<br>SHAP: %{{x:.4f}}<br>Value: %{{text}}<extra></extra>',
                    text=[f'{v:.2f}' for v in feat_vals[sample_idx]]
                ))

            fig.update_layout(
                title=dict(text='SHAP Beeswarm Plot (Feature Impact)', font=dict(size=20)),
                xaxis_title='SHAP Value (impact on prediction)',
                yaxis=dict(
                    tickvals=list(range(len(top_features))),
                    ticktext=top_features
                ),
                template=self.template,
                height=max(500, len(top_features) * 35),
                shapes=[dict(
                    type='line', x0=0, x1=0, y0=-0.5,
                    y1=len(top_features) - 0.5,
                    line=dict(color='gray', width=1, dash='dash')
                )]
            )
            charts['shap_beeswarm'] = fig
            self._save_figure(fig, output_dir, 'shap_beeswarm')

        # --- 3. SHAP Waterfall (for first prediction) ---
        if len(shap_values) > 0:
            sample_idx = 0
            sample_shap = shap_values[sample_idx]

            # Sort by absolute SHAP value
            sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:12]
            feat_names = [X.columns[i] for i in sorted_idx]
            feat_shap_vals = [sample_shap[i] for i in sorted_idx]

            colors = [self.colors['positive'] if v > 0 else self.colors['negative']
                      for v in feat_shap_vals]

            fig = go.Figure(data=[go.Bar(
                x=feat_shap_vals,
                y=feat_names,
                orientation='h',
                marker_color=colors,
                text=[f'{v:+.4f}' for v in feat_shap_vals],
                textposition='auto'
            )])
            fig.update_layout(
                title=dict(text='SHAP Waterfall — Sample Prediction Breakdown', font=dict(size=20)),
                xaxis_title='SHAP Value (Red = increases, Blue = decreases prediction)',
                template=self.template,
                height=max(400, len(feat_names) * 30),
                yaxis=dict(autorange='reversed'),
                shapes=[dict(
                    type='line', x0=0, x1=0, y0=-0.5, y1=len(feat_names) - 0.5,
                    line=dict(color='gray', width=1, dash='dash')
                )]
            )
            charts['shap_waterfall'] = fig
            self._save_figure(fig, output_dir, 'shap_waterfall')

        return charts

    # =========================================================================
    # LIME CHARTS
    # =========================================================================

    def _create_lime_charts(self, lime_results: List[Dict],
                             output_dir: str) -> Dict:
        """Generate LIME explanation charts."""
        charts = {}

        for i, result in enumerate(lime_results[:3]):
            weights = result['feature_weights']
            if not weights:
                continue

            features = [w[0] for w in weights[:10]]
            values = [w[1] for w in weights[:10]]

            colors = [self.colors['positive'] if v > 0 else self.colors['negative']
                      for v in values]

            fig = go.Figure(data=[go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker_color=colors,
                text=[f'{v:+.4f}' for v in values],
                textposition='auto'
            )])

            pred = result['prediction']
            actual = result.get('actual', '?')
            fig.update_layout(
                title=dict(
                    text=f'LIME Explanation — Sample #{result["index"]} (Pred: {pred}, Actual: {actual})',
                    font=dict(size=16)
                ),
                xaxis_title='Feature Weight',
                template=self.template,
                height=max(350, len(features) * 30),
                yaxis=dict(autorange='reversed'),
                shapes=[dict(
                    type='line', x0=0, x1=0, y0=-0.5, y1=len(features) - 0.5,
                    line=dict(color='gray', width=1, dash='dash')
                )]
            )

            chart_name = f'lime_sample_{i+1}'
            charts[chart_name] = fig
            self._save_figure(fig, output_dir, chart_name)

        return charts

    # =========================================================================
    # FEATURE INTERACTION CHART
    # =========================================================================

    def _create_interaction_chart(self, shap_values: np.ndarray,
                                   X: pd.DataFrame,
                                   importance: pd.DataFrame,
                                   output_dir: str) -> Optional[Dict]:
        """Create a SHAP dependence/interaction plot for the top 2 features."""
        try:
            top2 = importance.head(2)['feature'].tolist()
            if len(top2) < 2:
                return None

            feat1, feat2 = top2
            idx1 = X.columns.get_loc(feat1)
            idx2 = X.columns.get_loc(feat2)

            fig = go.Figure(data=go.Scatter(
                x=X[feat1],
                y=shap_values[:, idx1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=X[feat2],
                    colorscale='Viridis',
                    colorbar=dict(title=feat2),
                    opacity=0.6
                ),
                hovertemplate=f'{feat1}: %{{x:.2f}}<br>SHAP: %{{y:.4f}}<br>{feat2}: %{{marker.color:.2f}}<extra></extra>'
            ))
            fig.update_layout(
                title=dict(
                    text=f'SHAP Dependence: {feat1} (colored by {feat2})',
                    font=dict(size=18)
                ),
                xaxis_title=feat1,
                yaxis_title=f'SHAP value for {feat1}',
                template=self.template,
                height=500
            )

            self._save_figure(fig, output_dir, 'shap_interaction')
            return {'shap_interaction': fig}

        except Exception as e:
            self.log(f"  Interaction chart failed: {e}")
            return None

    # =========================================================================
    # LLM NARRATIVES
    # =========================================================================

    def _generate_global_narrative(self, model_name: str, task_type: str,
                                    target_col: str, metric_name: str,
                                    metric_value: float,
                                    importance: Optional[pd.DataFrame],
                                    state: Optional[Dict] = None) -> str:
        """Generate a structured pipeline narrative using the LLM."""

        # ── Extract pipeline context from state ───────────────────────────────
        state = state or {}
        profile  = state.get('profile_report', {})
        cleaning = state.get('cleaning_report', {})
        feat_rep = state.get('feature_report', {})
        cv_scores = state.get('cv_scores', {})
        ensemble_score = state.get('ensemble_score', 0.0)
        dataset_name   = state.get('dataset_name', 'Dataset')

        n_rows         = profile.get('n_rows', '?')
        n_cols         = profile.get('n_cols', '?')
        quality_score  = profile.get('quality_score', '?')

        # ── 1. Data quality issues ─────────────────────────────────────────────
        warnings_list  = profile.get('warnings', [])
        anomalies_list = profile.get('describe_anomalies', [])
        missing_raw    = profile.get('missing_summary', {})
        # missing_summary values may be counts or pcts — keep whichever is non-zero
        missing_cols   = {k: v for k, v in missing_raw.items() if v and v > 0}

        issue_lines = []
        if quality_score != '?' and int(quality_score) < 80:
            issue_lines.append(f"Data quality score is {quality_score}/100 — improvement was needed.")
        if missing_cols:
            sample = list(missing_cols.items())[:5]
            issue_lines.append(
                "Missing values in: " + ", ".join(f"{c} ({v})" for c, v in sample)
                + ("…" if len(missing_cols) > 5 else "")
            )
        if anomalies_list:
            issue_lines.append(f"Statistical anomalies detected in {len(anomalies_list)} column(s).")
        if warnings_list:
            issue_lines += [str(w) for w in warnings_list[:3]]
        data_issues = "\n".join(issue_lines) if issue_lines else "No major data quality issues detected."

        # ── 2. Cleaning summary ────────────────────────────────────────────────
        dup     = cleaning.get('duplicate_removal', {})
        mv      = cleaning.get('missing_value_handling', {})
        outliers = cleaning.get('outlier_handling', {})
        cat_std  = cleaning.get('category_standardization', {})

        clean_lines = []
        rows_removed = dup.get('rows_removed', 0)
        if rows_removed:
            clean_lines.append(f"Removed {rows_removed:,} duplicate rows.")
        if mv:
            strategies: Dict[str, int] = {}
            for info in mv.values():
                s = info.get('strategy', 'unknown')
                strategies[s] = strategies.get(s, 0) + 1
            clean_lines.append(
                "Missing value imputation: "
                + ", ".join(f"{cnt} col(s) → {strat}" for strat, cnt in strategies.items())
            )
        if outliers:
            clean_lines.append(
                f"Outlier treatment (IQR method) applied to {len(outliers)} column(s)."
            )
        if cat_std:
            clean_lines.append(
                f"Category standardisation applied to {len(cat_std)} column(s)."
            )
        cleaning_summary = "\n".join(clean_lines) if clean_lines else "No significant cleaning required."

        # ── 3. Feature engineering summary ────────────────────────────────────
        encoding = feat_rep.get('encoding', {})
        scaling  = feat_rep.get('scaling', {})
        vif_info = feat_rep.get('vif_analysis', {})
        dropped  = feat_rep.get('dropped_columns', [])

        feat_lines = []
        if encoding:
            enc_types: Dict[str, int] = {}
            for info in encoding.values():
                m = info.get('method', 'unknown')
                enc_types[m] = enc_types.get(m, 0) + 1
            feat_lines.append(
                "Encoding: " + ", ".join(f"{cnt} col(s) via {m}" for m, cnt in enc_types.items())
            )
        if scaling:
            feat_lines.append(
                f"Scaling: {scaling.get('method', 'Standard')} applied to "
                f"{scaling.get('n_columns', '?')} column(s)."
            )
        removed_vif = vif_info.get('removed_features', [])
        if removed_vif:
            feat_lines.append(
                f"VIF multicollinearity check removed {len(removed_vif)} highly correlated feature(s)."
            )
        if dropped:
            feat_lines.append(
                f"Dropped {len(dropped)} ID-like / non-informative column(s): "
                + ", ".join(str(d) for d in dropped[:5])
                + ("…" if len(dropped) > 5 else "")
            )
        feature_summary = "\n".join(feat_lines) if feat_lines else "Standard encoding and scaling applied."

        # ── 4. Correlations / multicollinearity ────────────────────────────────
        final_vif = vif_info.get('final_vif_scores', {})
        if final_vif:
            high = [(k, v) for k, v in final_vif.items() if v > 5]
            high.sort(key=lambda x: x[1], reverse=True)
            if high:
                corr_str = (
                    "High VIF features (>5, indicating multicollinearity): "
                    + ", ".join(f"{k} (VIF={v:.1f})" for k, v in high[:6])
                )
            else:
                corr_str = "All VIF scores ≤ 5 — no significant multicollinearity detected."
        else:
            corr_str = "VIF analysis not available; standard correlation checks were performed."

        # ── 5. Model scores ────────────────────────────────────────────────────
        all_model_scores = {**cv_scores, 'Ensemble': ensemble_score}
        model_scores_str = "\n".join(
            f"  {'→' if n == model_name else ' '} {n}: {s:.4f}"
            for n, s in sorted(all_model_scores.items(), key=lambda x: x[1], reverse=True)
        )
        best_score = all_model_scores.get(model_name, metric_value)

        # ── 6. SHAP top features ───────────────────────────────────────────────
        if importance is not None and len(importance) > 0:
            top_features_str = "\n".join(
                f"  {i+1}. {row['feature']} (SHAP={row['importance']:.4f})"
                for i, (_, row) in enumerate(importance.head(8).iterrows())
            )
        else:
            top_features_str = "  SHAP values not available for this model type."

        # ── Build and call prompt ──────────────────────────────────────────────
        prompt = PIPELINE_NARRATIVE_PROMPT.format(
            dataset_name=dataset_name,
            target_col=target_col,
            task_type=task_type,
            n_rows=n_rows,
            n_cols=n_cols,
            quality_score=quality_score,
            data_issues=data_issues,
            cleaning_summary=cleaning_summary,
            feature_summary=feature_summary,
            correlations=corr_str,
            model_scores=model_scores_str,
            best_model=model_name,
            metric_name=metric_name,
            best_score=best_score,
            top_features=top_features_str,
        )

        try:
            narrative = self.ask_llm(prompt)
            return narrative.strip()
        except Exception as e:
            self.log(f"  LLM global narrative failed: {e}")
            return (
                f"**1. Dataset Overview**\n"
                f"Dataset '{dataset_name}' has {n_rows} rows and {n_cols} columns. "
                f"Target: '{target_col}' ({task_type}). Quality score: {quality_score}/100.\n\n"
                f"**5. Model Performance**\n"
                f"Best model: {model_name} with {metric_name} = {best_score:.4f}. "
                f"See SHAP charts above for feature importance details."
            )

    def _generate_local_narratives(self, model, X: pd.DataFrame,
                                    y: pd.Series, task_type: str,
                                    shap_values: Optional[np.ndarray],
                                    n_samples: int = 3) -> List[Dict]:
        """Generate plain-English explanations for individual predictions."""
        narratives = []

        sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)

        for idx in sample_indices:
            row = X.iloc[idx]
            prediction = model.predict(row.values.reshape(1, -1))[0]

            confidence_text = ""
            if hasattr(model, 'predict_proba') and task_type == 'classification':
                try:
                    proba = model.predict_proba(row.values.reshape(1, -1))[0]
                    confidence = max(proba)
                    confidence_text = f"Confidence: {confidence:.1%}"
                except:
                    pass

            # Build SHAP explanation string
            shap_explanation = ""
            if shap_values is not None and idx < len(shap_values):
                sample_shap = shap_values[idx]
                sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:5]
                for si in sorted_idx:
                    feat = X.columns[si]
                    sv = sample_shap[si]
                    direction = "increases" if sv > 0 else "decreases"
                    shap_explanation += f"  - {feat} = {row.iloc[si]:.2f} -> {direction} prediction (SHAP: {sv:+.4f})\n"
            else:
                shap_explanation = "  (SHAP not available for this model)"

            # Feature values
            feature_values = '\n'.join([
                f"  - {col}: {row[col]:.4f}" if isinstance(row[col], float)
                else f"  - {col}: {row[col]}"
                for col in X.columns[:10]
            ])

            prompt = LOCAL_EXPLANATION_PROMPT.format(
                prediction=prediction,
                confidence_text=confidence_text,
                shap_explanation=shap_explanation,
                feature_values=feature_values
            )

            try:
                narrative = self.ask_llm(prompt).strip()
            except:
                narrative = f"The model predicted '{prediction}' for this data point."

            narratives.append({
                'index': int(idx),
                'prediction': str(prediction),
                'actual': str(y.iloc[idx]),
                'confidence': confidence_text,
                'narrative': narrative
            })

        return narratives

    # =========================================================================
    # SUMMARY DASHBOARD
    # =========================================================================

    def _create_summary_dashboard(self, explanations: Dict, model_name: str,
                                   task_type: str, metric_name: str,
                                   metric_value: float,
                                   output_dir: str) -> go.Figure:
        """Create a combined explainability summary."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Top Feature Importance (SHAP)',
                'Prediction Breakdown (Sample)',
                'Pipeline Summary',
                'Model Info'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Panel 1: Top 10 importance
        importance = explanations.get('shap_importance')
        if importance is not None and len(importance) > 0:
            top10 = importance.head(10).sort_values('importance', ascending=True)
            fig.add_trace(go.Bar(
                x=top10['importance'], y=top10['feature'],
                orientation='h', marker_color='#DC2626',
                showlegend=False
            ), row=1, col=1)
        else:
            fig.add_trace(go.Bar(
                x=[0], y=['SHAP data unavailable'],
                orientation='h', marker_color='#4B5563',
                showlegend=False
            ), row=1, col=1)

        # Panel 2: Sample waterfall
        shap_values = explanations.get('shap_values')
        if shap_values is not None and len(shap_values) > 0:
            sv = shap_values[0]
            sorted_idx = np.argsort(np.abs(sv))[::-1][:8]
            if importance is not None:
                all_features = importance['feature'].tolist()
                feat_names = []
                for si in sorted_idx:
                    if si < len(all_features):
                        feat_names.append(all_features[si])
                    else:
                        feat_names.append(f'Feature {si}')
            else:
                feat_names = [f'Feature {si}' for si in sorted_idx]

            fig.add_trace(go.Bar(
                x=[sv[i] for i in sorted_idx],
                y=feat_names,
                orientation='h',
                marker_color=['#DC2626' if sv[i] > 0 else '#2563EB' for i in sorted_idx],
                showlegend=False
            ), row=1, col=2)
        else:
            fig.add_trace(go.Bar(
                x=[0], y=['SHAP data unavailable'],
                orientation='h', marker_color='#4B5563',
                showlegend=False
            ), row=1, col=2)

        # Panel 3: Narrative summary (strip markdown symbols for clean table display)
        narrative = explanations.get('global_narrative', 'No narrative generated.')
        import re as _re
        # Remove markdown bold (**text**) and heading markers for clean table text
        clean_narrative = _re.sub(r'\*\*(.+?)\*\*', r'\1', narrative)
        narrative_lines = clean_narrative.split('.')[:6]
        narrative_display = '.<br>'.join(narrative_lines) + '.'

        fig.add_trace(go.Table(
            header=dict(values=['<b>Pipeline Narrative (Summary)</b>'],
                       fill_color='#2563EB', font=dict(color='white', size=13)),
            cells=dict(values=[[narrative_display]],
                      fill_color='#1e293b', font=dict(color='#e2e8f0', size=11),
                      align='left', height=30)
        ), row=2, col=1)

        # Panel 4: Model info
        fig.add_trace(go.Table(
            header=dict(values=['<b>Metric</b>', '<b>Value</b>'],
                       fill_color='#7C3AED', font=dict(color='white', size=13)),
            cells=dict(
                values=[
                    ['Model', 'Task', metric_name, 'SHAP Features', 'LIME Samples'],
                    [model_name, task_type.title(), f'{metric_value:.4f}',
                     str(len(importance)) if importance is not None else 'N/A',
                     str(len(explanations.get('lime_explanations', [])))]
                ],
                fill_color='#1e293b', font=dict(color='#e2e8f0', size=12), align='left'
            )
        ), row=2, col=2)

        fig.update_layout(
            title=dict(text='🔍 Model Explainability Summary', font=dict(size=22)),
            template=self.template, height=900, showlegend=False
        )

        self._save_figure(fig, output_dir, 'explainability_dashboard')
        return fig

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _save_figure(self, fig: go.Figure, output_dir: str, name: str):
        """Save a Plotly figure as HTML and optionally PNG."""
        try:
            html_path = os.path.join(output_dir, f'{name}.html')
            fig.write_html(html_path, include_plotlyjs='cdn', full_html=True,
                           config={'responsive': True, 'displayModeBar': True})
        except Exception as e:
            self.log(f"  Warning: Could not save HTML for {name}: {e}")
        try:
            fig.write_image(os.path.join(output_dir, f'{name}.png'), scale=2)
        except:
            pass
