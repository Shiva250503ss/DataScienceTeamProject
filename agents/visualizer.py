# agents/visualizer.py

"""
ML Pipeline Visualizer Agent — Generates technical visualizations for the
automated ML pipeline stages (profiling → cleaning → features → models).

NOTE: This is the ML-FOCUSED visualizer. For business intelligence / data
analysis dashboards, see agents/data_analyzer.py (DataAnalyzerAgent).

TWO VISUALIZATION FEATURES IN DATAPILOT:
  1. ML Visualizer (THIS FILE) → Pipeline technical charts for data scientists
  2. Data Analyzer (data_analyzer.py) → LLM-powered smart dashboards for managers

This agent creates interactive Plotly charts covering:

  1. DATA OVERVIEW (from Profiler)
     - Data types distribution (pie chart)
     - Missing values heatmap
     - Quality score gauge
     - Dataset shape summary card

  2. CLEANING IMPACT (from Cleaner)
     - Before vs after row counts (bar chart)
     - Missing values: before vs after per column (grouped bar)
     - Outlier distribution (box plots before/after)

  3. FEATURE ANALYSIS (from Feature Agent)
     - Feature correlation heatmap
     - Feature importance bar chart (from mutual information)
     - Distribution of top features (violin/histogram)
     - Target variable distribution

  4. MODEL PERFORMANCE (from Modeler)
     - Model comparison bar chart (CV scores with error bars)
     - RL recommendation confidence radar chart
     - Ensemble vs individual models comparison
     - Confusion matrix (classification) or Residual plot (regression)
     - Learning insights summary

All charts are saved as interactive HTML files (Plotly) and static PNGs.

Owner: Visualizer Team
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from agents.base import BaseAgent

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_predict


class VisualizerAgent(BaseAgent):
    """
    Agent responsible for generating all visualizations.

    Produces 4 visualization groups, one per pipeline stage:
      - profiling_visuals  : data overview charts
      - cleaning_visuals   : cleaning impact charts
      - feature_visuals    : feature analysis charts
      - model_visuals      : model performance charts

    Each group is a dict of {chart_name: plotly Figure or file path}.
    All charts are also saved to disk under output_dir/visualizations/.
    """

    def __init__(self):
        super().__init__("VisualizerAgent")
        # Default color palette — consistent across all charts
        self.colors = {
            'primary':   '#2563EB',  # Blue
            'secondary': '#7C3AED',  # Purple
            'success':   '#059669',  # Green
            'warning':   '#D97706',  # Amber
            'danger':    '#DC2626',  # Red
            'info':      '#0891B2',  # Cyan
            'light':     '#F3F4F6',  # Light gray
            'dark':      '#1F2937',  # Dark gray
        }
        self.color_sequence = [
            '#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626',
            '#0891B2', '#EC4899', '#F59E0B', '#10B981', '#6366F1'
        ]
        # Plotly template
        self.template = 'plotly_dark'

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all visualizations from the pipeline state."""
        self.log("Starting visualization generation...")

        # Create output directory
        output_dir = state.get('output_dir', './output')
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        all_visuals = {}

        # =====================================================================
        # Group 1: DATA OVERVIEW (from Profiler)
        # =====================================================================
        if state.get('profile_report'):
            self.log("Generating data overview charts...")
            profiling_visuals = self._create_profiling_visuals(state, viz_dir)
            all_visuals['profiling'] = profiling_visuals
            self.log(f"  Created {len(profiling_visuals)} profiling charts")

        # =====================================================================
        # Group 2: CLEANING IMPACT (from Cleaner)
        # =====================================================================
        if state.get('cleaning_report'):
            self.log("Generating cleaning impact charts...")
            cleaning_visuals = self._create_cleaning_visuals(state, viz_dir)
            all_visuals['cleaning'] = cleaning_visuals
            self.log(f"  Created {len(cleaning_visuals)} cleaning charts")

        # =====================================================================
        # Group 3: FEATURE ANALYSIS (from Feature Agent)
        # =====================================================================
        if state.get('X') is not None:
            self.log("Generating feature analysis charts...")
            feature_visuals = self._create_feature_visuals(state, viz_dir)
            all_visuals['features'] = feature_visuals
            self.log(f"  Created {len(feature_visuals)} feature charts")

        # =====================================================================
        # Group 4: MODEL PERFORMANCE (from Modeler)
        # =====================================================================
        if state.get('cv_scores'):
            self.log("Generating model performance charts...")
            model_visuals = self._create_model_visuals(state, viz_dir)
            all_visuals['models'] = model_visuals
            self.log(f"  Created {len(model_visuals)} model charts")

        # Generate summary dashboard (combines key charts)
        self.log("Generating summary dashboard...")
        dashboard = self._create_dashboard(state, all_visuals, viz_dir)
        all_visuals['dashboard'] = dashboard

        # Update pipeline state
        state['visualizations'] = all_visuals
        state['viz_dir'] = viz_dir
        state['stage'] = 'visualized'

        total = sum(len(v) if isinstance(v, dict) else 1 for v in all_visuals.values())
        self.log(f"Visualization complete! Generated {total} charts in {viz_dir}")

        return state

    # =========================================================================
    # GROUP 1: DATA OVERVIEW (Profiling)
    # =========================================================================

    def _create_profiling_visuals(self, state: Dict, viz_dir: str) -> Dict:
        """Create data overview visualizations from the profiler output."""
        visuals = {}
        profile = state['profile_report']
        df = state.get('raw_data', state.get('current_data'))

        # --- 1.1 Data Types Distribution (Donut Chart) ---
        column_types = profile['column_types']
        type_counts = pd.Series(column_types).value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.45,
            marker_colors=self.color_sequence[:len(type_counts)],
            textinfo='label+percent+value',
            textposition='outside',
            pull=[0.05] * len(type_counts)
        )])
        fig.update_layout(
            title=dict(text='Column Types Distribution', font=dict(size=20)),
            template=self.template,
            showlegend=True,
            height=500,
            annotations=[dict(
                text=f"{len(column_types)}<br>Columns",
                x=0.5, y=0.5, font_size=18, showarrow=False
            )]
        )
        visuals['column_types'] = fig
        self._save_figure(fig, viz_dir, 'column_types')

        # --- 1.2 Missing Values Heatmap ---
        if df is not None:
            missing = df.isnull()
            if missing.any().any():
                missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
                cols_with_missing = missing_pct[missing_pct > 0]

                if len(cols_with_missing) > 0:
                    # Bar chart of missing percentages
                    top_n = min(20, len(cols_with_missing))
                    top_missing = cols_with_missing.head(top_n)

                    fig = go.Figure(data=[go.Bar(
                        x=top_missing.values,
                        y=top_missing.index,
                        orientation='h',
                        marker_color=[
                            self.colors['danger'] if v > 30
                            else self.colors['warning'] if v > 10
                            else self.colors['info']
                            for v in top_missing.values
                        ],
                        text=[f'{v:.1f}%' for v in top_missing.values],
                        textposition='auto'
                    )])
                    fig.update_layout(
                        title=dict(text='Missing Values by Column (Top 20)', font=dict(size=20)),
                        xaxis_title='Missing %',
                        yaxis_title='Column',
                        template=self.template,
                        height=max(400, top_n * 30),
                        yaxis=dict(autorange='reversed')
                    )
                    visuals['missing_values'] = fig
                    self._save_figure(fig, viz_dir, 'missing_values')

        # --- 1.3 Quality Score Gauge ---
        quality_score = profile.get('quality_score', 0)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            title={'text': "Data Quality Score", 'font': {'size': 24}},
            delta={'reference': 70, 'increasing': {'color': self.colors['success']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': [0, 40], 'color': '#FEE2E2'},    # Red zone
                    {'range': [40, 70], 'color': '#FEF3C7'},   # Yellow zone
                    {'range': [70, 100], 'color': '#D1FAE5'},  # Green zone
                ],
                'threshold': {
                    'line': {'color': self.colors['danger'], 'width': 4},
                    'thickness': 0.75,
                    'value': quality_score
                }
            }
        ))
        fig.update_layout(template=self.template, height=400)
        visuals['quality_score'] = fig
        self._save_figure(fig, viz_dir, 'quality_score')

        # --- 1.4 Dataset Summary Card ---
        fig = go.Figure()
        summary_text = (
            f"<b>Dataset Summary</b><br><br>"
            f"Rows: <b>{profile['n_rows']:,}</b><br>"
            f"Columns: <b>{profile['n_cols']}</b><br>"
            f"Numeric: <b>{sum(1 for t in column_types.values() if t == 'numeric')}</b><br>"
            f"Categorical: <b>{sum(1 for t in column_types.values() if t == 'categorical')}</b><br>"
            f"Binary: <b>{sum(1 for t in column_types.values() if t == 'binary')}</b><br>"
            f"Target: <b>{state.get('target_column', 'N/A')}</b><br>"
            f"Task: <b>{state.get('task_type', 'N/A').title()}</b><br>"
            f"Quality: <b>{quality_score}/100</b>"
        )
        fig.add_annotation(
            text=summary_text, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#e2e8f0'), align="left",
            bordercolor=self.colors['primary'],
            borderwidth=2, borderpad=20,
            bgcolor='rgba(30,41,59,0.95)'
        )
        fig.update_layout(
            template=self.template, height=400,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            title=dict(text='Dataset Overview', font=dict(size=20))
        )
        visuals['dataset_summary'] = fig
        self._save_figure(fig, viz_dir, 'dataset_summary')

        # --- 1.5 Numeric Distributions (Histograms) ---
        if df is not None:
            numeric_cols = [c for c, t in column_types.items()
                          if t == 'numeric' and c in df.columns
                          and c != state.get('target_column')]
            if numeric_cols:
                n_cols_plot = min(12, len(numeric_cols))
                selected_cols = numeric_cols[:n_cols_plot]
                n_rows_grid = (n_cols_plot + 2) // 3
                n_cols_grid = min(3, n_cols_plot)

                fig = make_subplots(
                    rows=n_rows_grid, cols=n_cols_grid,
                    subplot_titles=selected_cols
                )
                for idx, col in enumerate(selected_cols):
                    row = idx // n_cols_grid + 1
                    col_pos = idx % n_cols_grid + 1
                    fig.add_trace(
                        go.Histogram(
                            x=df[col].dropna(),
                            nbinsx=30,
                            marker_color=self.color_sequence[idx % len(self.color_sequence)],
                            opacity=0.8,
                            name=col,
                            showlegend=False
                        ),
                        row=row, col=col_pos
                    )
                fig.update_layout(
                    title=dict(text='Numeric Feature Distributions', font=dict(size=20)),
                    template=self.template,
                    height=300 * n_rows_grid,
                    showlegend=False
                )
                visuals['numeric_distributions'] = fig
                self._save_figure(fig, viz_dir, 'numeric_distributions')

        # --- 1.6 Target Variable Distribution ---
        if df is not None and state.get('target_column') in df.columns:
            target_col = state['target_column']
            task_type = state.get('task_type', 'classification')

            if task_type == 'classification':
                vc = df[target_col].value_counts()
                fig = go.Figure(data=[go.Bar(
                    x=vc.index.astype(str),
                    y=vc.values,
                    marker_color=self.color_sequence[:len(vc)],
                    text=vc.values,
                    textposition='auto'
                )])
                fig.update_layout(
                    title=dict(text=f'Target Distribution: {target_col}', font=dict(size=20)),
                    xaxis_title='Class',
                    yaxis_title='Count',
                    template=self.template, height=450
                )
            else:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pd.to_numeric(df[target_col], errors='coerce').dropna(),
                    nbinsx=50,
                    marker_color=self.colors['primary'],
                    opacity=0.8,
                    name='Distribution'
                ))
                fig.add_trace(go.Violin(
                    y=pd.to_numeric(df[target_col], errors='coerce').dropna(),
                    side='positive', line_color=self.colors['secondary'],
                    name='Density', visible='legendonly'
                ))
                fig.update_layout(
                    title=dict(text=f'Target Distribution: {target_col}', font=dict(size=20)),
                    xaxis_title=target_col,
                    yaxis_title='Frequency',
                    template=self.template, height=450
                )

            visuals['target_distribution'] = fig
            self._save_figure(fig, viz_dir, 'target_distribution')

        return visuals

    # =========================================================================
    # GROUP 2: CLEANING IMPACT
    # =========================================================================

    def _create_cleaning_visuals(self, state: Dict, viz_dir: str) -> Dict:
        """Create cleaning impact visualizations."""
        visuals = {}
        cleaning_report = state['cleaning_report']
        raw_data = state.get('raw_data')
        cleaned_data = state.get('current_data')

        # --- 2.1 Cleaning Summary (Before vs After) ---
        if raw_data is not None and cleaned_data is not None:
            categories = ['Rows', 'Duplicates Removed', 'Columns']
            before_vals = [len(raw_data), 0, len(raw_data.columns)]
            after_vals = [len(cleaned_data),
                         cleaning_report.get('duplicate_removal', {}).get('rows_removed', 0),
                         len(cleaned_data.columns)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Before Cleaning',
                x=categories, y=before_vals,
                marker_color=self.colors['warning'],
                text=before_vals, textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='After Cleaning',
                x=categories, y=after_vals,
                marker_color=self.colors['success'],
                text=after_vals, textposition='auto'
            ))
            fig.update_layout(
                title=dict(text='Cleaning Impact: Before vs After', font=dict(size=20)),
                barmode='group',
                template=self.template, height=450
            )
            visuals['cleaning_summary'] = fig
            self._save_figure(fig, viz_dir, 'cleaning_summary')

        # --- 2.2 Missing Values: Before vs After ---
        if raw_data is not None and cleaned_data is not None:
            before_missing = (raw_data.isnull().sum() / len(raw_data) * 100)
            after_missing = (cleaned_data.isnull().sum() / len(cleaned_data) * 100)

            # Only show columns that had missing values
            had_missing = before_missing[before_missing > 0].sort_values(ascending=False)
            if len(had_missing) > 0:
                top_n = min(15, len(had_missing))
                cols_to_show = had_missing.head(top_n).index

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Before',
                    y=cols_to_show,
                    x=[before_missing[c] for c in cols_to_show],
                    orientation='h',
                    marker_color=self.colors['danger'],
                    opacity=0.7
                ))
                fig.add_trace(go.Bar(
                    name='After',
                    y=cols_to_show,
                    x=[after_missing.get(c, 0) for c in cols_to_show],
                    orientation='h',
                    marker_color=self.colors['success'],
                    opacity=0.7
                ))
                fig.update_layout(
                    title=dict(text='Missing Values: Before vs After Cleaning', font=dict(size=20)),
                    xaxis_title='Missing %',
                    barmode='group',
                    template=self.template,
                    height=max(400, top_n * 35),
                    yaxis=dict(autorange='reversed')
                )
                visuals['missing_before_after'] = fig
                self._save_figure(fig, viz_dir, 'missing_before_after')

        # --- 2.3 Outlier Detection Visualization ---
        outlier_bounds = state.get('outlier_bounds', {})
        if outlier_bounds and cleaned_data is not None:
            cols_with_outliers = list(outlier_bounds.keys())[:8]  # Top 8

            if cols_with_outliers:
                fig = make_subplots(
                    rows=2, cols=min(4, len(cols_with_outliers)),
                    subplot_titles=[f'{c}' for c in cols_with_outliers[:8]]
                )
                for idx, col in enumerate(cols_with_outliers):
                    row = idx // 4 + 1
                    col_pos = idx % 4 + 1
                    if col in cleaned_data.columns:
                        fig.add_trace(
                            go.Box(
                                y=cleaned_data[col].dropna(),
                                marker_color=self.color_sequence[idx % len(self.color_sequence)],
                                name=col, showlegend=False
                            ),
                            row=row, col=col_pos
                        )
                fig.update_layout(
                    title=dict(text='Feature Distributions After Outlier Handling', font=dict(size=20)),
                    template=self.template,
                    height=600, showlegend=False
                )
                visuals['outlier_boxplots'] = fig
                self._save_figure(fig, viz_dir, 'outlier_boxplots')

        return visuals

    # =========================================================================
    # GROUP 3: FEATURE ANALYSIS
    # =========================================================================

    def _create_feature_visuals(self, state: Dict, viz_dir: str) -> Dict:
        """Create feature analysis visualizations."""
        visuals = {}
        X = state['X']
        y = state['y']
        task_type = state.get('task_type', 'classification')
        feature_report = state.get('feature_report', {})
        target_col = state.get('target_column', '')

        # Use cleaned (pre-encoding) data for distribution/correlation charts
        # so we show meaningful original-scale values, not scaled/encoded values
        current_df = state.get('current_data')

        # Guard: keep only numeric columns to avoid datetime/bool errors
        if isinstance(X, pd.DataFrame):
            numeric_X_cols = [
                col for col in X.columns
                if not pd.api.types.is_datetime64_any_dtype(X[col])
                and not pd.api.types.is_timedelta64_dtype(X[col])
                and (pd.api.types.is_numeric_dtype(X[col])
                     or pd.api.types.is_bool_dtype(X[col]))
            ]
            X = X[numeric_X_cols]

        # --- 3.1 Feature Correlation Heatmap ---
        # Prefer using cleaned pre-encoding data for meaningful correlations
        corr_source = None
        if current_df is not None:
            num_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in num_cols:
                num_cols.remove(target_col)
            if len(num_cols) > 1:
                corr_source = current_df[num_cols]
        if corr_source is None and isinstance(X, pd.DataFrame) and len(X.columns) > 1:
            corr_source = X.select_dtypes(include=[np.number])

        if corr_source is not None and len(corr_source.columns) > 1:
            n_features_plot = min(20, len(corr_source.columns))
            variances = corr_source.var().sort_values(ascending=False)
            top_features = variances.head(n_features_plot).index.tolist()

            corr_matrix = corr_source[top_features].corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 9},
                hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
            ))
            fig.update_layout(
                title=dict(text=f'Feature Correlation Heatmap (Top {n_features_plot})',
                          font=dict(size=20)),
                template=self.template,
                height=600, width=700,
                xaxis=dict(tickangle=45)
            )
            visuals['correlation_heatmap'] = fig
            self._save_figure(fig, viz_dir, 'correlation_heatmap')

        # --- 3.2 Feature Importance (Mutual Information) ---
        if isinstance(X, pd.DataFrame) and len(X.columns) > 0:
            try:
                from sklearn.feature_selection import (
                    mutual_info_classif, mutual_info_regression
                )
                if task_type == 'classification':
                    mi_scores = mutual_info_classif(
                        X.fillna(0), y, random_state=42
                    )
                else:
                    mi_scores = mutual_info_regression(
                        X.fillna(0), y, random_state=42
                    )

                mi_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': mi_scores
                }).sort_values('importance', ascending=True)

                # Show top 20
                top_n = min(20, len(mi_df))
                mi_top = mi_df.tail(top_n)

                fig = go.Figure(data=[go.Bar(
                    x=mi_top['importance'],
                    y=mi_top['feature'],
                    orientation='h',
                    marker=dict(
                        color=mi_top['importance'],
                        colorscale='Viridis',
                        colorbar=dict(title='MI Score')
                    ),
                    text=[f'{v:.3f}' for v in mi_top['importance']],
                    textposition='auto'
                )])
                fig.update_layout(
                    title=dict(
                        text=f'Feature Importance (Mutual Information, Top {top_n})',
                        font=dict(size=20)
                    ),
                    xaxis_title='Mutual Information Score',
                    template=self.template,
                    height=max(400, top_n * 28)
                )
                visuals['feature_importance'] = fig
                self._save_figure(fig, viz_dir, 'feature_importance')
            except Exception as e:
                self.log(f"  Warning: Feature importance chart failed: {e}")

        # --- 3.3 Top Feature Distributions (Violin Plots) ---
        # Determine candidate features — prefer pre-encoding cleaned data
        dist_source = None
        if current_df is not None:
            dist_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in dist_cols:
                dist_cols.remove(target_col)
            if dist_cols:
                dist_source = current_df[dist_cols]
        if dist_source is None and isinstance(X, pd.DataFrame) and len(X.columns) > 0:
            dist_source = X

        if dist_source is not None and len(dist_source.columns) > 0:
            # Use MI scores if available, else pick by variance from dist_source
            try:
                candidate_feats = mi_df.tail(6)['feature'].tolist()
                # Keep only features present in dist_source
                top_feats = [f for f in candidate_feats if f in dist_source.columns]
                if not top_feats:
                    top_feats = dist_source.var().sort_values(ascending=False).head(6).index.tolist()
            except Exception:
                top_feats = dist_source.var().sort_values(ascending=False).head(6).index.tolist()

            n_top = min(6, len(top_feats))
            n_cols_v = min(3, n_top)
            n_rows_v = (n_top + n_cols_v - 1) // n_cols_v

            fig = make_subplots(
                rows=n_rows_v, cols=n_cols_v,
                subplot_titles=top_feats[:n_top]
            )
            for idx, feat in enumerate(top_feats[:n_top]):
                row = idx // n_cols_v + 1
                col_pos = idx % n_cols_v + 1
                feat_data = dist_source[feat].dropna()
                if len(feat_data) > 1:
                    fig.add_trace(
                        go.Violin(
                            y=feat_data,
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=self.color_sequence[idx % len(self.color_sequence)],
                            line_color='rgba(255,255,255,0.8)',
                            opacity=0.75,
                            name=feat, showlegend=False
                        ),
                        row=row, col=col_pos
                    )
            fig.update_layout(
                title=dict(text='Top Feature Distributions', font=dict(size=20)),
                template=self.template,
                height=300 * n_rows_v,
                showlegend=False
            )
            visuals['top_feature_distributions'] = fig
            self._save_figure(fig, viz_dir, 'top_feature_distributions')

        return visuals

    # =========================================================================
    # GROUP 4: MODEL PERFORMANCE
    # =========================================================================

    def _create_model_visuals(self, state: Dict, viz_dir: str) -> Dict:
        """Create model performance visualizations."""
        visuals = {}
        cv_scores = state['cv_scores']
        task_type = state.get('task_type', 'classification')
        ensemble_score = state.get('ensemble_score', 0)
        best_model_name = state.get('best_model_name', '')
        recommendations = state.get('model_recommendations', [])

        # --- 4.1 Model Comparison Bar Chart ---
        # Include all 4 models (3 RL-selected + ensemble) and rank by actual CV score
        all_model_scores = {**cv_scores, '🏆 Ensemble': ensemble_score}
        sorted_pairs = sorted(all_model_scores.items(), key=lambda x: x[1], reverse=True)
        model_names = [p[0] for p in sorted_pairs]
        scores = [p[1] for p in sorted_pairs]

        # Color: rank-1 = gold, rank-2 = silver, rank-3 = bronze, rank-4 = grey
        rank_colors = [
            self.colors['warning'],   # 1st — gold/amber
            self.colors['info'],      # 2nd — cyan
            self.colors['secondary'], # 3rd — purple
            self.colors['dark'],      # 4th
        ]
        # Ensure ensemble is visually distinct regardless of rank
        bar_colors = []
        for i, name in enumerate(model_names):
            if name == best_model_name or (name == '🏆 Ensemble' and best_model_name == 'Ensemble'):
                bar_colors.append(self.colors['success'])   # winner = green
            elif '🏆' in name:
                bar_colors.append(self.colors['primary'])   # ensemble (not winner) = blue
            else:
                bar_colors.append(rank_colors[min(i, len(rank_colors) - 1)])

        metric_name = 'Accuracy' if task_type == 'classification' else 'R² Score'

        fig = go.Figure(data=[go.Bar(
            x=model_names, y=scores,
            marker_color=bar_colors,
            text=[f'{s:.4f}' for s in scores],
            textposition='auto',
            textfont=dict(size=13, color='white')
        )])
        fig.update_layout(
            title=dict(text=f'Model Performance Comparison ({metric_name})',
                      font=dict(size=20)),
            xaxis_title='Model',
            yaxis_title=metric_name,
            template=self.template,
            height=500,
            yaxis=dict(range=[
                max(0, min(scores) - 0.1),
                min(1.0, max(scores) + 0.05)
            ]),
            xaxis=dict(tickangle=30)
        )
        visuals['model_comparison'] = fig
        self._save_figure(fig, viz_dir, 'model_comparison')

        # --- 4.2 RL Recommendation Confidence (Radar Chart) ---
        if recommendations:
            rec_names = [r[0] for r in recommendations]
            rec_conf = [r[1] for r in recommendations]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=rec_conf + [rec_conf[0]],  # Close the polygon
                theta=rec_names + [rec_names[0]],
                fill='toself',
                fillcolor='rgba(37, 99, 235, 0.2)',
                line=dict(color=self.colors['primary'], width=2),
                name='RL Confidence'
            ))
            # Add actual scores if available
            actual_scores_for_rec = [cv_scores.get(n, 0) for n in rec_names]
            fig.add_trace(go.Scatterpolar(
                r=actual_scores_for_rec + [actual_scores_for_rec[0]],
                theta=rec_names + [rec_names[0]],
                fill='toself',
                fillcolor='rgba(5, 150, 105, 0.2)',
                line=dict(color=self.colors['success'], width=2),
                name='Actual CV Score'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=dict(text='RL Recommendations vs Actual Performance',
                          font=dict(size=20)),
                template=self.template, height=500,
                showlegend=True
            )
            visuals['rl_recommendations'] = fig
            self._save_figure(fig, viz_dir, 'rl_recommendations')

        # --- 4.3 Ensemble vs Best Single Model ---
        if best_model_name and ensemble_score:
            best_single = cv_scores.get(best_model_name, 0)
            improvement = ensemble_score - best_single

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Best Single Model', 'Ensemble'],
                y=[best_single, ensemble_score],
                marker_color=[self.colors['warning'], self.colors['primary']],
                text=[f'{best_single:.4f}', f'{ensemble_score:.4f}'],
                textposition='auto',
                textfont=dict(size=16, color='white'),
                width=0.5
            ))
            # Add annotation for improvement
            fig.add_annotation(
                x=1, y=ensemble_score + 0.01,
                text=f'{"+" if improvement >= 0 else ""}{improvement:.4f}',
                showarrow=False,
                font=dict(
                    size=18,
                    color=self.colors['success'] if improvement >= 0 else self.colors['danger']
                )
            )
            fig.update_layout(
                title=dict(
                    text=f'Ensemble vs {best_model_name}',
                    font=dict(size=20)
                ),
                yaxis_title=metric_name,
                template=self.template, height=450,
                yaxis=dict(range=[
                    max(0, min(best_single, ensemble_score) - 0.05),
                    min(1.0, max(best_single, ensemble_score) + 0.05)
                ])
            )
            visuals['ensemble_vs_best'] = fig
            self._save_figure(fig, viz_dir, 'ensemble_vs_best')

        # --- 4.4 Confusion Matrix (Classification) or Residuals Plot (Regression) ---
        X = state.get('X')
        y = state.get('y')
        ensemble_model = state.get('ensemble_model')

        if X is not None and y is not None and ensemble_model is not None:
            try:
                if task_type == 'classification':
                    visuals.update(
                        self._create_confusion_matrix(ensemble_model, X, y, viz_dir)
                    )
                else:
                    visuals.update(
                        self._create_residuals_plot(ensemble_model, X, y, viz_dir)
                    )
            except Exception as e:
                self.log(f"  Warning: Confusion/Residual chart failed: {e}")

        # --- 4.5 Model Training Summary Table ---
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Model</b>', f'<b>CV {metric_name}</b>', '<b>Rank</b>'],
                fill_color=self.colors['primary'],
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[
                    model_names,
                    [f'{s:.4f}' for s in scores],
                    list(range(1, len(model_names) + 1))
                ],
                fill_color=[
                    ['#1e293b' if i % 2 == 0 else '#0f172a'
                     for i in range(len(model_names))]
                ],
                font=dict(size=13, color='#e2e8f0'),
                align='left'
            )
        )])
        fig.update_layout(
            title=dict(text='Model Performance Ranking', font=dict(size=20)),
            template=self.template, height=max(300, len(model_names) * 35 + 100)
        )
        visuals['model_ranking_table'] = fig
        self._save_figure(fig, viz_dir, 'model_ranking_table')

        # --- 4.6 Comprehensive Metrics Visualization ---
        comp_metrics = state.get('comprehensive_metrics', {})
        if comp_metrics and 'error' not in comp_metrics:
            if task_type == 'classification':
                # Radar chart for classification metrics
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                metric_vals = []
                for m in metric_names:
                    val = comp_metrics.get(m, 0)
                    metric_vals.append(float(val) if isinstance(val, (int, float)) else 0)

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=metric_vals + [metric_vals[0]],
                    theta=metric_names + [metric_names[0]],
                    fill='toself',
                    fillcolor='rgba(37, 99, 235, 0.25)',
                    line=dict(color=self.colors['primary'], width=3),
                    name='Model Metrics',
                    text=[f'{v:.4f}' for v in metric_vals] + [f'{metric_vals[0]:.4f}'],
                    textposition='top center'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1],
                                        tickfont=dict(size=11)),
                        angularaxis=dict(tickfont=dict(size=13))
                    ),
                    title=dict(
                        text='Classification Metrics Overview',
                        font=dict(size=20)
                    ),
                    template=self.template, height=520,
                    showlegend=False,
                    annotations=[dict(
                        text=(
                            f"Accuracy: {comp_metrics.get('Accuracy', 'N/A')}<br>"
                            f"Precision: {comp_metrics.get('Precision', 'N/A')}<br>"
                            f"Recall: {comp_metrics.get('Recall', 'N/A')}<br>"
                            f"F1-Score: {comp_metrics.get('F1-Score', 'N/A')}<br>"
                            f"ROC-AUC: {comp_metrics.get('ROC-AUC', 'N/A')}"
                        ),
                        x=0.98, y=-0.05, xref='paper', yref='paper',
                        showarrow=False,
                        font=dict(size=11, color='#94a3b8'),
                        align='right'
                    )]
                )
                visuals['comprehensive_metrics'] = fig
                self._save_figure(fig, viz_dir, 'comprehensive_metrics')

            else:
                # Bar chart for regression metrics
                metric_names = ['R²', 'MAE', 'MSE', 'RMSE']
                metric_vals = [
                    float(comp_metrics.get(m, 0)) for m in metric_names
                ]

                bar_colors = [
                    self.colors['primary'],    # R²
                    self.colors['warning'],    # MAE
                    self.colors['danger'],     # MSE
                    self.colors['secondary'],  # RMSE
                ]

                fig = go.Figure(data=[go.Bar(
                    x=metric_names,
                    y=metric_vals,
                    marker_color=bar_colors,
                    text=[f'{v:.4f}' for v in metric_vals],
                    textposition='auto',
                    textfont=dict(size=14, color='white'),
                    width=0.6
                )])
                fig.update_layout(
                    title=dict(
                        text='Regression Metrics Overview',
                        font=dict(size=20)
                    ),
                    yaxis_title='Score',
                    template=self.template, height=500,
                    xaxis=dict(tickfont=dict(size=14)),
                    annotations=[dict(
                        text=(
                            f"R²: {comp_metrics.get('R²', 'N/A')}  |  "
                            f"MAE: {comp_metrics.get('MAE', 'N/A')}  |  "
                            f"MSE: {comp_metrics.get('MSE', 'N/A')}  |  "
                            f"RMSE: {comp_metrics.get('RMSE', 'N/A')}"
                        ),
                        x=0.5, y=1.08, xref='paper', yref='paper',
                        showarrow=False,
                        font=dict(size=12, color='#94a3b8')
                    )]
                )
                visuals['comprehensive_metrics'] = fig
                self._save_figure(fig, viz_dir, 'comprehensive_metrics')

        return visuals

    # =========================================================================
    # CONFUSION MATRIX & RESIDUALS
    # =========================================================================

    def _create_confusion_matrix(self, model, X, y, viz_dir: str) -> Dict:
        """Generate confusion matrix heatmap for classification."""
        visuals = {}
        try:
            # Convert to numpy to avoid feature-name mismatch in VotingClassifier
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            y_arr = y.values if isinstance(y, pd.Series) else y
            try:
                y_pred = cross_val_predict(model, X_arr, y_arr, cv=3)
            except Exception:
                # Fallback: fit on full data and predict (less rigorous but still useful)
                model.fit(X_arr, y_arr)
                y_pred = model.predict(X_arr)
            cm = confusion_matrix(y_arr, y_pred)
            labels = sorted(np.unique(y_arr))

            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

            fig = go.Figure(data=go.Heatmap(
                z=cm_norm,
                x=[str(l) for l in labels],
                y=[str(l) for l in labels],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 14},
                hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>'
            ))
            fig.update_layout(
                title=dict(text='Confusion Matrix (3-Fold CV)', font=dict(size=20)),
                xaxis_title='Predicted',
                yaxis_title='Actual',
                template=self.template,
                height=500, width=550
            )
            visuals['confusion_matrix'] = fig
            self._save_figure(fig, viz_dir, 'confusion_matrix')
        except Exception as e:
            self.log(f"  Confusion matrix failed: {e}")

        return visuals

    def _create_residuals_plot(self, model, X, y, viz_dir: str) -> Dict:
        """Generate residuals plot for regression."""
        visuals = {}
        try:
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            y_arr = y.values if isinstance(y, pd.Series) else y
            try:
                y_pred = cross_val_predict(model, X_arr, y_arr, cv=3)
            except Exception:
                model.fit(X_arr, y_arr)
                y_pred = model.predict(X_arr)
            residuals = y_arr - y_pred

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Predicted vs Actual', 'Residuals Distribution']
            )

            # Predicted vs Actual scatter
            fig.add_trace(
                go.Scatter(
                    x=y_arr, y=y_pred, mode='markers',
                    marker=dict(color=self.colors['primary'], size=5, opacity=0.5),
                    name='Predictions'
                ),
                row=1, col=1
            )
            # Perfect prediction line
            min_val = min(float(y_arr.min()), float(y_pred.min()))
            max_val = max(float(y_arr.max()), float(y_pred.max()))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    line=dict(color=self.colors['danger'], dash='dash', width=2),
                    name='Perfect Prediction'
                ),
                row=1, col=1
            )

            # Residuals histogram
            fig.add_trace(
                go.Histogram(
                    x=residuals, nbinsx=40,
                    marker_color=self.colors['secondary'],
                    opacity=0.8, name='Residuals'
                ),
                row=1, col=2
            )

            fig.update_layout(
                title=dict(text='Regression Analysis (3-Fold CV)', font=dict(size=20)),
                template=self.template, height=450, showlegend=True
            )
            fig.update_xaxes(title_text='Actual', row=1, col=1)
            fig.update_yaxes(title_text='Predicted', row=1, col=1)
            fig.update_xaxes(title_text='Residual', row=1, col=2)
            fig.update_yaxes(title_text='Count', row=1, col=2)

            visuals['residuals_plot'] = fig
            self._save_figure(fig, viz_dir, 'residuals_plot')
        except Exception as e:
            self.log(f"  Residuals plot failed: {e}")

        return visuals

    # =========================================================================
    # DASHBOARD — combines key charts into one view
    # =========================================================================

    def _create_dashboard(self, state: Dict, all_visuals: Dict,
                          viz_dir: str) -> Dict:
        """Create a summary dashboard combining key insights."""
        visuals = {}

        profile = state.get('profile_report', {})
        cv_scores = state.get('cv_scores', {})
        ensemble_score = state.get('ensemble_score', 0)
        best_model = state.get('best_model_name', 'N/A')
        task_type = state.get('task_type', 'classification')
        metric = 'Accuracy' if task_type == 'classification' else 'R²'

        # Dashboard with 4 panels
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Data Quality',
                'Feature Types',
                'Model Scores',
                'Pipeline Summary'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        # Panel 1: Quality Score
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=profile.get('quality_score', 0),
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': self.colors['primary']},
                    'steps': [
                        {'range': [0, 40], 'color': '#FEE2E2'},
                        {'range': [40, 70], 'color': '#FEF3C7'},
                        {'range': [70, 100], 'color': '#D1FAE5'}
                    ]
                }
            ),
            row=1, col=1
        )

        # Panel 2: Column Types Pie
        if profile.get('column_types'):
            type_counts = pd.Series(profile['column_types']).value_counts()
            fig.add_trace(
                go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    marker_colors=self.color_sequence[:len(type_counts)],
                    hole=0.4,
                    textinfo='label+value'
                ),
                row=1, col=2
            )

        # Panel 3: Model Scores Bar — all 4 ranked by CV score
        if cv_scores:
            all_scores_panel = {**cv_scores, '🏆 Ensemble': ensemble_score}
            sorted_models = sorted(all_scores_panel.items(), key=lambda x: x[1], reverse=True)
            names = [m[0] for m in sorted_models]
            vals = [m[1] for m in sorted_models]
            fig.add_trace(
                go.Bar(
                    x=names, y=vals,
                    marker_color=self.color_sequence[:len(names)],
                    text=[f'{v:.3f}' for v in vals],
                    textposition='auto'
                ),
                row=2, col=1
            )

        # Panel 4: Pipeline Summary Table
        n_rows = profile.get('n_rows', '?')
        n_cols = profile.get('n_cols', '?')
        size_str = f"{n_rows:,}" if isinstance(n_rows, int) else str(n_rows)
        all_scores_for_summary = {**cv_scores, 'Ensemble': ensemble_score} if cv_scores else {}
        best_score_overall = max(all_scores_for_summary.values()) if all_scores_for_summary else 0
        summary_rows = [
            ['Dataset Size', f"{size_str} × {n_cols}"],
            ['Task Type', task_type.title()],
            ['Target', state.get('target_column', 'N/A')],
            ['Final Model', best_model],
            [f'{metric} (Final Model)', f"{best_score_overall:.4f}"],
            [f'Ensemble {metric}', f"{ensemble_score:.4f}"],
        ]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color=self.colors['primary'],
                    font=dict(color='white', size=13),
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*summary_rows)),
                    fill_color='#1e293b',
                    font=dict(size=12, color='#e2e8f0'),
                    align='left'
                )
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=dict(
                text='📊 DataPilot AI Pro — Pipeline Dashboard',
                font=dict(size=24)
            ),
            template=self.template,
            height=900,
            showlegend=False
        )

        visuals['dashboard'] = fig
        self._save_figure(fig, viz_dir, 'dashboard')

        return visuals

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _save_figure(self, fig: go.Figure, viz_dir: str, name: str):
        """Save a Plotly figure as interactive HTML and static PNG."""
        try:
            html_path = os.path.join(viz_dir, f'{name}.html')
            fig.write_html(
                html_path,
                include_plotlyjs='cdn',
                full_html=True,
                config={'responsive': True, 'displayModeBar': True},
            )
        except Exception as e:
            self.log(f"  Warning: Could not save HTML for {name}: {e}")

        try:
            png_path = os.path.join(viz_dir, f'{name}.png')
            fig.write_image(png_path, scale=2)
        except Exception:
            # write_image requires kaleido or orca — skip silently if not installed
            pass
