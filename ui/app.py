# ui/app.py

"""
DataPilot AI Pro — Streamlit Web Interface.

TWO-TAB LAYOUT:
  1. 📊 Data Insights  — AI Dashboard Generator (auto charts + chatbot)
  2. 🤖 Data Pipeline  — Full AutoML pipeline (profiling → cleaning → features → models → SHAP)

Usage:
    streamlit run ui/app.py
"""

import os
import re
import sys
import pandas as pd
import numpy as np
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR       = os.path.dirname(os.path.abspath(__file__))

for _p in (PROJECT_ROOT, UI_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import insights_tab  # ui/insights_tab.py — renders the AI Dashboard Generator

# LAZY IMPORTS — heavy ML libraries only loaded when the ML button is clicked
_lazy_loaded = {}

def _get_pipeline_functions():
    if 'orchestrator' not in _lazy_loaded:
        from orchestrator.graph import run_ml_pipeline
        _lazy_loaded['orchestrator'] = {'run_ml_pipeline': run_ml_pipeline}
    return _lazy_loaded['orchestrator']


# =========================================================================
# PAGE CONFIG
# =========================================================================

st.set_page_config(
    page_title="DataPilot AI Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8em;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB, #7C3AED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.15em;
        text-align: center;
        margin-bottom: 30px;
    }
    .success-box {
        background: rgba(5, 150, 105, 0.18);
        border-left: 4px solid #059669;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #a7f3d0;
    }
    .info-box {
        background: rgba(37, 99, 235, 0.18);
        border-left: 4px solid #2563EB;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #bfdbfe;
    }
    .warn-box {
        background: rgba(217, 119, 6, 0.18);
        border-left: 4px solid #D97706;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #fde68a;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    div[data-testid="stHorizontalBlock"] > div { padding: 0 4px; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SESSION STATE
# =========================================================================

for key in ['ml_uploaded_df', 'ml_file_name', 'ml_pipeline_result']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Target column & model selection state ─────────────────────────────────
if 'ml_selected_target' not in st.session_state:
    st.session_state.ml_selected_target = None
if 'ml_selected_algorithm' not in st.session_state:
    st.session_state.ml_selected_algorithm = None


# =========================================================================
# HELPER: Detect task type from target column
# =========================================================================

# Regex to strip common non-numeric formatting characters
_NUMERIC_NOISE_RE = re.compile(r'[$€£¥₹₩₫¢,%\s]')


def _try_coerce_to_numeric(series: pd.Series) -> pd.Series:
    """
    Attempt to coerce a string Series to numeric by stripping common
    formatting characters (currency symbols, commas, %, whitespace, etc.).
    Returns the original series unchanged if <80% of values parse successfully.
    """
    if series.dtype != 'object':
        return series

    cleaned = series.dropna().astype(str)
    # Quick reject: no digits in the data
    if cleaned.str.contains(r'\d', regex=True).sum() < len(cleaned) * 0.5:
        return series

    # Strip formatting noise
    stripped = cleaned.str.replace(_NUMERIC_NOISE_RE, '', regex=True)
    # Handle parenthesised negatives: (500) -> -500
    mask = stripped.str.match(r'^\((.+)\)$', na=False)
    stripped = stripped.where(~mask, '-' + stripped.str.replace(r'[()]', '', regex=True))
    # Strip trailing/leading alphabetic characters (units like kg, lbs, USD)
    stripped = stripped.str.replace(r'[a-zA-Z]+$', '', regex=True)
    stripped = stripped.str.replace(r'^[a-zA-Z]+', '', regex=True)

    numeric = pd.to_numeric(stripped, errors='coerce')
    if numeric.notna().sum() >= len(cleaned) * 0.80:
        return numeric
    return series


def _detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Heuristic to decide classification vs regression based on the target column.

    Rules (mirrors agents/profiler.py logic):
      • First, try to coerce "disguised numeric" strings (e.g. "$1,234.56")
        to actual numbers so they are not misclassified as categorical.
      • object / bool / category dtypes → classification
      • integer with ≤20 unique values   → classification
      • float  with ≤10 unique values    → classification
      • otherwise                         → regression
    """
    col = df[target_col]

    # Try coercing string columns that contain formatted numbers
    if col.dtype == 'object':
        coerced = _try_coerce_to_numeric(col)
        if pd.api.types.is_numeric_dtype(coerced):
            col = coerced

    if col.dtype == 'object' or col.dtype == 'bool' or pd.api.types.is_categorical_dtype(col):
        return 'classification'
    nunique = col.nunique()
    if pd.api.types.is_integer_dtype(col) and nunique <= 20:
        return 'classification'
    if pd.api.types.is_float_dtype(col) and nunique <= 10:
        return 'classification'
    return 'regression'


# =========================================================================
# MODEL CATALOGUES
# =========================================================================

CLASSIFICATION_MODELS = {
    "Logistic Regression":            "LogisticRegression",
    "Gaussian Naïve Bayes":           "GaussianNB",
    "K-Nearest Neighbors (Clf)":      "KNeighborsClassifier",
    "Support Vector Classifier":      "SVC",
    "Decision Tree (Clf)":            "DecisionTreeClassifier",
    "Random Forest (Clf)":            "RandomForestClassifier",
    "Extra Trees (Clf)":              "ExtraTreesClassifier",
    "Gradient Boosting (Clf)":        "GradientBoostingClassifier",
    "XGBoost (Clf)":                  "XGBClassifier",
    "LightGBM (Clf)":                "LGBMClassifier",
    "CatBoost (Clf)":                "CatBoostClassifier",
}

REGRESSION_MODELS = {
    "Ridge Regression":               "Ridge",
    "Lasso Regression":               "Lasso",
    "ElasticNet":                     "ElasticNet",
    "Support Vector Regressor":       "SVR",
    "K-Nearest Neighbors (Reg)":      "KNeighborsRegressor",
    "Decision Tree (Reg)":            "DecisionTreeRegressor",
    "Random Forest (Reg)":            "RandomForestRegressor",
    "Extra Trees (Reg)":              "ExtraTreesRegressor",
    "Gradient Boosting (Reg)":        "GradientBoostingRegressor",
    "XGBoost (Reg)":                  "XGBRegressor",
    "LightGBM (Reg)":                "LGBMRegressor",
    "CatBoost (Reg)":                "CatBoostRegressor",
}


# =========================================================================
# HELPER: Display charts (ML tab)
# =========================================================================

def display_charts(visuals, section_name: str):
    """Display a dict of Plotly figures in a 2-column grid."""
    if not visuals:
        st.info(f"No {section_name} charts available.")
        return
    flat_charts = {}
    for key, value in visuals.items():
        if isinstance(value, dict):
            for chart_name, fig in value.items():
                if hasattr(fig, 'to_json'):
                    flat_charts[f"{key}/{chart_name}"] = fig
        elif hasattr(value, 'to_json'):
            flat_charts[key] = value
    if not flat_charts:
        st.info(f"No displayable charts in {section_name}.")
        return
    items = list(flat_charts.items())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(items):
                name, fig = items[i + j]
                with col:
                    try:
                        st.plotly_chart(fig, use_container_width=True,
                                        key=f"{section_name}_{name}_{i}_{j}")
                    except Exception as e:
                        st.warning(f"Could not display chart '{name}': {e}")


# =========================================================================
# HEADER
# =========================================================================

st.markdown('<div class="main-header">DataPilot AI Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Automated Data Science Platform — Insights, Dashboards & Machine Learning'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================================
# TABS
# =========================================================================

tab_insights, tab_ml = st.tabs(["📊 Data Insights", "🤖 Data Pipeline"])


# =========================================================================
# TAB 1 — DATA INSIGHTS (AI Dashboard Generator)
# =========================================================================

with tab_insights:
    insights_tab.render()


# =========================================================================
# TAB 2 — DATA PIPELINE
# =========================================================================

with tab_ml:
    st.markdown("### Upload Your Dataset")

    ml_uploaded_file = st.file_uploader(
        "Drop a CSV file here",
        type=["csv"],
        label_visibility="collapsed",
        key="ml_uploader",
    )

    if ml_uploaded_file is not None:
        try:
            df = pd.read_csv(ml_uploaded_file)
            st.session_state.ml_uploaded_df  = df
            st.session_state.ml_file_name    = ml_uploaded_file.name
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    ml_df        = st.session_state.ml_uploaded_df
    ml_file_name = st.session_state.ml_file_name or "Dataset"

    if ml_df is not None:
        st.success(f"✅ **{ml_file_name}** loaded — {ml_df.shape[0]:,} rows × {ml_df.shape[1]} columns")

        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(ml_df.head(20), use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(ml_df):,}")
            c2.metric("Columns", len(ml_df.columns))
            c3.metric("Missing Cells", f"{ml_df.isnull().sum().sum():,}")
            c4.metric("Duplicates", f"{ml_df.duplicated().sum():,}")

        # ==================================================================
        # 🎯 TARGET COLUMN SELECTOR  (mandatory — scrollable list)
        # ==================================================================
        st.markdown("---")
        st.markdown("### 🎯 Select Target Column")
        st.markdown(
            '<div class="info-box">'
            'Choose the column you want to predict. '
            'This is <b>required</b> before running the pipeline.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Custom CSS for the scrollable column picker
        st.markdown("""
        <style>
            /* scrollable radio group container */
            div[data-testid="stRadio"] > div[role="radiogroup"] {
                max-height: 260px;
                overflow-y: auto;
                border: 1px solid rgba(250,250,250,0.12);
                border-radius: 10px;
                padding: 8px 12px;
                background: rgba(255,255,255,0.03);
            }
        </style>
        """, unsafe_allow_html=True)

        column_list = list(ml_df.columns)
        target_col = st.radio(
            "Select the target column from your dataset:",
            options=column_list,
            index=None,          # nothing pre-selected → forces explicit choice
            key="ml_target_radio",
            help="Scroll through the list and pick the column you want the model to predict.",
        )

        # Persist selection
        st.session_state.ml_selected_target = target_col

        # ==================================================================
        # After target is chosen → detect task type & show model picker
        # ==================================================================
        detected_task_type = None
        user_selected_model = None   # None ⇒ PPO auto-selects

        if target_col is not None:
            detected_task_type = _detect_task_type(ml_df, target_col)

            # ── Show detected task type ──
            task_emoji = "📂" if detected_task_type == "classification" else "📈"
            task_label = detected_task_type.title()
            st.markdown(
                f'<div class="success-box">'
                f'{task_emoji} Detected task type: <b>{task_label}</b> '
                f'(based on column <code>{target_col}</code> — '
                f'{ml_df[target_col].nunique()} unique values, dtype: {ml_df[target_col].dtype})'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Model Selection ──
            st.markdown("---")
            st.markdown("### 🧠 Select ML Algorithm")
            st.markdown(
                '<div class="info-box">'
                'Pick a specific algorithm or let the <b>PPO Reinforcement Learning agent</b> '
                'automatically select the best model for your data. '
                'If you\'re unsure, the auto-select option is recommended!'
                '</div>',
                unsafe_allow_html=True,
            )

            models_dict = (CLASSIFICATION_MODELS if detected_task_type == "classification"
                           else REGRESSION_MODELS)

            model_display_names = ["🤖 Auto-Select (PPO picks the best model)"] + list(models_dict.keys())

            chosen_display = st.selectbox(
                f"Available {task_label} Models:",
                options=model_display_names,
                index=0,
                key="ml_model_selectbox",
                help=(
                    "Choose 'Auto-Select' to let the PPO Reinforcement Learning agent "
                    "analyse your data and pick the top models automatically. "
                    "Or pick a specific algorithm if you know what you want."
                ),
            )

            if chosen_display != model_display_names[0]:
                user_selected_model = models_dict[chosen_display]
                st.info(f"✅ You selected **{chosen_display}** (`{user_selected_model}`)")
            else:
                st.info("🤖 PPO will analyse your data's meta-features and select the best models automatically.")

        # ==================================================================
        # 🚀 RUN PIPELINE BUTTON (disabled until target is selected)
        # ==================================================================
        st.markdown("---")

        pipeline_disabled = (target_col is None)

        if pipeline_disabled:
            st.warning("⬆️ Please select a **target column** above to enable the pipeline.")

        if st.button(
            "🚀  Run Data Pipeline",
            use_container_width=True,
            type="primary",
            disabled=pipeline_disabled,
            help="Full pipeline: profiling → cleaning → features → models → SHAP/LIME",
        ):
            output_dir = "./output"
            with st.spinner("Running full ML pipeline… (Profiling → Cleaning → Features → Models → Explanations)"):
                try:
                    fns    = _get_pipeline_functions()
                    result = fns["run_ml_pipeline"](
                        df=ml_df,
                        target_column=target_col,
                        dataset_name=ml_file_name,
                        output_dir=output_dir,
                        user_selected_model=user_selected_model,
                    )
                    st.session_state.ml_pipeline_result = result
                except Exception as e:
                    st.error(f"ML Pipeline failed: {e}")

    # ── Display ML results ────────────────────────────────────────────────────
    result = st.session_state.ml_pipeline_result

    if result:
        st.markdown("---")
        st.markdown(
            '<div class="success-box">✅ ML Pipeline completed successfully!</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Model",    result.get("best_model_name", "N/A"))
        col2.metric("📈 Ensemble Score", f"{result.get('ensemble_score', 0):.4f}")
        col3.metric("🎯 Task Type",      result.get("task_type", "N/A").title())
        col4.metric("⭐ Data Quality",
                    f"{result.get('profile_report', {}).get('quality_score', 0)}/100")

        (tab_models, tab_profile, tab_cleaning, tab_features,
         tab_viz, tab_explain, tab_errors, tab_predict) = st.tabs([
            "📊 Models", "📋 Data Profile", "🧹 Cleaning", "🔧 Features",
            "📈 Visualizations", "🔍 Explanations", "⚠️ Error Analysis",
            "🎯 Predict",
        ])

        # ── Models ────────────────────────────────────────────────────────────
        with tab_models:
            cv_scores = result.get("cv_scores", {})
            if cv_scores:
                st.markdown("### Model Performance (Cross-Validated)")
                scores_df = pd.DataFrame([
                    {"Model": name, "CV Score": round(score, 4)}
                    for name, score in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(scores_df, use_container_width=True, hide_index=True)
                st.metric("🥇 Ensemble Score", f"{result.get('ensemble_score', 0):.4f}")

                recs = result.get("model_recommendations", [])
                if recs:
                    st.markdown("#### RL Agent Recommendations")
                    for name, conf in recs:
                        st.write(f"→ **{name}** (confidence: {conf:.1%})")

                overfit = result.get("overfitting_analysis", {})
                if overfit:
                    if overfit.get("is_suspicious"):
                        st.markdown("### ⚠️ Overfitting Warning")
                        st.markdown(f'<div class="warn-box">{overfit["reason"]}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="success-box">✅ {overfit.get("reason", "Scores look healthy.")}</div>',
                            unsafe_allow_html=True)
                    if overfit.get("train_score") is not None:
                        oc1, oc2, oc3 = st.columns(3)
                        oc1.metric("Train Score", f"{overfit['train_score']:.4f}")
                        oc2.metric("CV Score",    f"{overfit['cv_score']:.4f}")
                        oc3.metric("Gap (overfit indicator)", f"{overfit.get('gap', 0):.4f}")

                # ── Comprehensive Metrics ──────────────────────────────────
                comp_metrics = result.get("comprehensive_metrics", {})
                if comp_metrics and 'error' not in comp_metrics:
                    task = comp_metrics.get("task_type", "classification")
                    st.markdown("### 📏 Comprehensive Evaluation Metrics")
                    if task == "classification":
                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                        mc1.metric("✅ Accuracy",  comp_metrics.get("Accuracy", "N/A"))
                        mc2.metric("🎯 Precision", comp_metrics.get("Precision", "N/A"))
                        mc3.metric("📡 Recall",    comp_metrics.get("Recall", "N/A"))
                        mc4.metric("⚖️ F1-Score",  comp_metrics.get("F1-Score", "N/A"))
                        mc5.metric("📈 ROC-AUC",   comp_metrics.get("ROC-AUC", "N/A"))
                    else:
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("📐 R²",   comp_metrics.get("R²", "N/A"))
                        mc2.metric("📏 MAE",  comp_metrics.get("MAE", "N/A"))
                        mc3.metric("📊 MSE",  comp_metrics.get("MSE", "N/A"))
                        mc4.metric("📉 RMSE", comp_metrics.get("RMSE", "N/A"))
            else:
                st.info("No model scores available.")

        # ── Data Profile ──────────────────────────────────────────────────────
        with tab_profile:
            prof = result.get("profile_report", {})
            if prof:
                st.markdown("### Data Profile Summary")
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Rows",          prof.get("n_rows", "N/A"))
                pc2.metric("Columns",       prof.get("n_cols", "N/A"))
                pc3.metric("Quality Score", f"{prof.get('quality_score', 0)}/100")
                pc4.metric("Task Type",     result.get("task_type", "N/A").title())
                st.markdown(f"**Target Column:** `{result.get('target_column', 'N/A')}`")

                col_types = prof.get("column_types", {})
                if col_types:
                    st.markdown("#### Column Types")
                    st.dataframe(
                        pd.DataFrame([{"Column": c, "Type": t} for c, t in col_types.items()]),
                        use_container_width=True, hide_index=True,
                    )

                anomalies = prof.get("describe_anomalies", [])
                if anomalies:
                    st.markdown("### 🔍 Anomalies Detected")
                    st.markdown(
                        '<div class="warn-box">Unusual patterns found by analyzing descriptive statistics.</div>',
                        unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)

                uniformity = prof.get("uniformity_issues", {})
                if uniformity:
                    st.markdown("### 🔄 Non-Uniform Category Values Detected")
                    for col, issues in uniformity.items():
                        with st.expander(f"Column: `{col}` — {len(issues)} inconsistencies"):
                            for canonical, variants in issues.items():
                                st.write(f"  `{canonical}` has variants: {variants}")
                    st.markdown(
                        '<div class="info-box">✅ These have been automatically standardized by the Cleaner.</div>',
                        unsafe_allow_html=True)

                warnings = prof.get("warnings", [])
                if warnings:
                    st.markdown("#### Warnings")
                    for w in warnings:
                        st.write(w)
            else:
                st.info("No profile data available.")

        # ── Cleaning ──────────────────────────────────────────────────────────
        with tab_cleaning:
            cleaning = result.get("cleaning_report", {})
            if cleaning:
                st.markdown("### Cleaning Report")
                cc1, cc2, cc3 = st.columns(3)
                dup = cleaning.get("duplicate_removal", {})
                cc1.metric("Duplicates Removed", dup.get("rows_removed", 0))
                cc2.metric("Rows Remaining",     dup.get("rows_remaining", "N/A"))
                cc3.metric("Columns Imputed",    len(cleaning.get("missing_value_handling", {})))

                mv = cleaning.get("missing_value_handling", {})
                if mv:
                    st.markdown("#### Missing Value Handling")
                    st.dataframe(pd.DataFrame([
                        {"Column": col, "Strategy": info.get("strategy", "N/A"),
                         "Value": str(info.get("value", "N/A"))}
                        for col, info in mv.items()
                    ]), use_container_width=True, hide_index=True)

                outliers = cleaning.get("outlier_handling", {})
                if outliers:
                    st.markdown("#### Outlier Treatment (IQR Method)")
                    st.dataframe(pd.DataFrame([
                        {"Column": col, "N Outliers": info.get("n_outliers", 0),
                         "Outlier %": info.get("outlier_pct", 0),
                         "Action": info.get("action", "N/A"),
                         "Lower Bound": info.get("lower_bound", "N/A"),
                         "Upper Bound": info.get("upper_bound", "N/A")}
                        for col, info in outliers.items()
                    ]), use_container_width=True, hide_index=True)

                cat_std = cleaning.get("category_standardization", {})
                if cat_std:
                    st.markdown("#### Category Standardization")
                    for col, mapping in cat_std.items():
                        with st.expander(f"Column: `{col}`"):
                            for old, new in mapping.items():
                                st.write(f"  `{old}` → `{new}`")
            else:
                st.info("No cleaning report available.")

        # ── Features ──────────────────────────────────────────────────────────
        with tab_features:
            feat_report = result.get("feature_report", {})
            if feat_report:
                st.markdown("### Feature Engineering Report")

                vif_info = feat_report.get("vif_analysis", {})
                if vif_info:
                    st.markdown("#### VIF Multicollinearity Analysis")
                    threshold    = vif_info.get("threshold", 10)
                    removed_vif  = vif_info.get("removed_features", [])
                    final_vif    = vif_info.get("final_vif_scores", {})
                    skip_reason  = vif_info.get("reason", "")
                    high_vif_kept = {f: v for f, v in final_vif.items() if v > threshold}

                    st.markdown(
                        f'<div class="info-box">'
                        f'<b>VIF</b> (Variance Inflation Factor) measures multicollinearity. '
                        f'VIF 1–5 = fine &nbsp;|&nbsp; 5–10 = concerning &nbsp;|&nbsp; >10 = severe. '
                        f'Threshold used: <b>{threshold}</b>. '
                        f'High-VIF features are removed only if enough features remain to maintain predictive power.'
                        f'</div>',
                        unsafe_allow_html=True)

                    if removed_vif:
                        st.markdown(f"**Removed {len(removed_vif)} features due to high VIF:**")
                        st.dataframe(pd.DataFrame(removed_vif), use_container_width=True, hide_index=True)
                    elif skip_reason:
                        # VIF removal was deliberately skipped (e.g. too few features)
                        st.info(f"ℹ️ VIF removal skipped — {skip_reason}")
                        if high_vif_kept:
                            st.warning(
                                f"⚠️ {len(high_vif_kept)} feature(s) have VIF > {threshold} "
                                f"({', '.join(f'{f} = {v}' for f, v in sorted(high_vif_kept.items(), key=lambda x: x[1], reverse=True))}). "
                                f"They are **retained** because the feature set is small — removing them would hurt prediction accuracy more than the collinearity does."
                            )
                    elif high_vif_kept:
                        # Removal ran but high-VIF features survived (kept due to target correlation)
                        st.warning(
                            f"⚠️ {len(high_vif_kept)} feature(s) with VIF > {threshold} were kept "
                            f"because they are highly correlated with the target variable: "
                            + ", ".join(f"**{f}** (VIF={v:.1f})" for f, v in sorted(high_vif_kept.items(), key=lambda x: x[1], reverse=True))
                        )
                    else:
                        st.success("All features have acceptable VIF — no multicollinearity issues.")

                    if final_vif:
                        st.markdown("#### Final VIF Scores")
                        st.dataframe(pd.DataFrame([
                            {"Feature": f, "VIF": v}
                            for f, v in sorted(final_vif.items(), key=lambda x: x[1], reverse=True)
                        ]), use_container_width=True, hide_index=True)

                encoding = feat_report.get("encoding", {})
                if encoding:
                    st.markdown("#### Encoding Applied")
                    st.dataframe(pd.DataFrame([
                        {"Column": col, "Method": info.get("method", "N/A"),
                         "Unique Values": info.get("n_unique", "N/A")}
                        for col, info in encoding.items()
                    ]), use_container_width=True, hide_index=True)

                scaling = feat_report.get("scaling", {})
                if scaling:
                    st.markdown("#### Scaling")
                    st.write(f"**Method:** {scaling.get('method', 'N/A')}")
                    st.write(f"**Reason:** {scaling.get('reason', 'N/A')}")
                    st.write(f"**Columns Scaled:** {scaling.get('n_columns', 0)}")

                dropped = feat_report.get("dropped_columns", [])
                if dropped:
                    st.markdown("#### Dropped Columns")
                    st.write(f"ID columns removed: {dropped}")
            else:
                st.info("No feature engineering report available.")

        # ── Visualizations ────────────────────────────────────────────────────
        with tab_viz:
            display_charts(result.get("visualizations", {}), "ML_Viz")

        # ── Explanations ──────────────────────────────────────────────────────
        with tab_explain:
            explanations = result.get("explanations", {})
            if explanations:
                narrative = explanations.get("global_narrative", "")
                if narrative:
                    # Render markdown directly so **bold** headers display correctly
                    st.markdown(narrative)
                    st.divider()
                importance = explanations.get("shap_importance")
                if importance is not None:
                    st.markdown("### SHAP Feature Importance")
                    st.dataframe(importance.head(15), use_container_width=True, hide_index=True)
                # Exclude summary_dashboard — it duplicates the narrative already shown above
                explain_charts = {
                    k: v for k, v in explanations.get("charts", {}).items()
                    if k != "summary_dashboard"
                }
                display_charts(explain_charts, "Explain")
            else:
                st.info("No explanations available.")

        # ── Error Analysis ────────────────────────────────────────────────────
        with tab_errors:
            error_analysis = result.get("error_analysis", {})
            if error_analysis and error_analysis.get("summary"):
                st.markdown("### Error Analysis")
                task_type = result.get("task_type", "")

                # Summary message
                st.info(error_analysis.get("summary", ""))

                if task_type == "regression":
                    ea_cols = st.columns(3)
                    ea_cols[0].metric("MAE",  f"{error_analysis.get('mae', 0):.4f}")
                    ea_cols[1].metric("RMSE", f"{error_analysis.get('rmse', 0):.4f}")
                    ea_cols[2].metric("Median Error", f"{error_analysis.get('median_error', 0):.4f}")

                    # Error patterns by value range
                    patterns = error_analysis.get("error_patterns", [])
                    if patterns:
                        st.markdown("#### Error by Value Range")
                        st.dataframe(
                            pd.DataFrame(patterns),
                            use_container_width=True, hide_index=True,
                        )

                    # Worst predictions
                    worst = error_analysis.get("worst_samples", [])
                    if worst:
                        st.markdown("#### Top 10 Worst Predictions")
                        st.dataframe(
                            pd.DataFrame(worst),
                            use_container_width=True, hide_index=True,
                        )

                elif task_type == "classification":
                    ea_cols = st.columns(3)
                    ea_cols[0].metric("Total Errors", error_analysis.get("total_errors", 0))
                    ea_cols[1].metric("Error Rate", f"{error_analysis.get('error_rate', 0):.2f}%")
                    ea_cols[2].metric("Classes", len(error_analysis.get("class_errors", [])))

                    # Per-class error breakdown
                    class_errors = error_analysis.get("class_errors", [])
                    if class_errors:
                        st.markdown("#### Per-Class Error Rates")
                        st.dataframe(
                            pd.DataFrame(class_errors),
                            use_container_width=True, hide_index=True,
                        )

                    # Confusion matrix as table
                    cm = error_analysis.get("confusion_matrix")
                    if cm:
                        st.markdown("#### Confusion Matrix")
                        encoders = result.get("encoders", {})
                        if "target" in encoders:
                            labels = list(encoders["target"].classes_)
                        else:
                            labels = [str(i) for i in range(len(cm))]
                        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                        cm_df.index.name = "Actual"
                        cm_df.columns.name = "Predicted"
                        st.dataframe(cm_df, use_container_width=True)
            else:
                st.info("No error analysis available.")


        # ── Predict ───────────────────────────────────────────────────────────
        with tab_predict:
            st.markdown("### 🎯 Predict With Your Own Values")
            st.markdown(
                '<div class="info-box">'
                'Enter your own feature values and the trained model will make a prediction.'
                '</div>',
                unsafe_allow_html=True,
            )

            # Use ORIGINAL column names (before one-hot encoding) so the user
            # sees "m_dep" not "m_dep_0.2 / m_dep_0.3 / ..." etc.
            original_input_columns = result.get("original_input_columns", [])
            feature_names          = result.get("feature_names", [])
            display_cols           = original_input_columns if original_input_columns else feature_names

            if display_cols and result.get("ensemble_model"):
                encoding_info = result.get("feature_report", {}).get("encoding", {})
                column_types  = result.get("profile_report", {}).get("column_types", {})

                st.markdown("#### Enter Feature Values")
                user_inputs = {}
                for i in range(0, len(display_cols), 3):
                    cols_ui = st.columns(3)
                    for j, col_widget in enumerate(cols_ui):
                        fi = i + j
                        if fi < len(display_cols):
                            feat       = display_cols[fi]
                            enc_info   = encoding_info.get(feat, {})
                            categories = enc_info.get("categories", [])
                            with col_widget:
                                if categories:
                                    # Categorical column — dropdown with real options
                                    user_inputs[feat] = st.selectbox(
                                        feat,
                                        options=categories,
                                        key=f"ml_predict_{feat}",
                                    )
                                else:
                                    # Numeric column
                                    user_inputs[feat] = st.number_input(
                                        feat, value=0.0, format="%.4f",
                                        key=f"ml_predict_{feat}",
                                    )

                if st.button("Make Prediction", type="primary", use_container_width=True):
                    try:
                        import numpy as np

                        # ── Step 1: build 1-row DataFrame from user inputs ──
                        df_pred = pd.DataFrame([user_inputs])

                        # ── Step 2: apply same encoding as feature.py ───────
                        encoders = result.get("encoders", {})
                        for col, enc_info in encoding_info.items():
                            if col not in df_pred.columns:
                                continue
                            method = enc_info.get("method", "")

                            if method == "label":
                                encoder = encoders.get(col)
                                if encoder:
                                    val = str(df_pred[col].iloc[0])
                                    df_pred[col] = (
                                        encoder.transform([val])[0]
                                        if val in encoder.classes_ else 0
                                    )

                            elif method == "onehot":
                                val      = str(df_pred[col].iloc[0])
                                new_cols = enc_info.get("new_cols", [])
                                prefix   = col + "_"
                                for new_col in new_cols:
                                    cat_val = new_col[len(prefix):]
                                    df_pred[new_col] = 1 if val == cat_val else 0
                                df_pred = df_pred.drop(columns=[col])

                            elif method == "target":
                                encoder = encoders.get(col)
                                if encoder:
                                    try:
                                        df_pred[col] = encoder.transform(
                                            df_pred[[col]]
                                        )[col].values
                                    except Exception:
                                        df_pred[col] = 0.0

                        # ── Step 3: align to final feature set ─────────────
                        # Handles VIF removal & feature selection automatically.
                        df_pred = df_pred.reindex(columns=feature_names, fill_value=0)

                        # ── Step 4: apply scaling ───────────────────────────
                        scalers     = result.get("scalers", {})
                        num_scaler  = scalers.get("numeric")
                        cols_scaled = (result.get("feature_report", {})
                                       .get("scaling", {})
                                       .get("columns_scaled", []))
                        if num_scaler is not None and cols_scaled:
                            avail = [c for c in cols_scaled if c in df_pred.columns]
                            if avail:
                                df_pred[avail] = num_scaler.transform(df_pred[avail])

                        # ── Step 5: predict ─────────────────────────────────
                        model      = result["ensemble_model"]
                        prediction = model.predict(df_pred)[0]

                        task_type    = result.get("task_type", "")
                        display_pred = prediction
                        if task_type == "classification" and "target" in encoders:
                            try:
                                display_pred = encoders["target"].inverse_transform(
                                    [int(prediction)]
                                )[0]
                            except Exception:
                                pass

                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,#059669,#10B981);'
                            f'border-radius:12px;padding:24px;text-align:center;color:white;">'
                            f'<div style="font-size:1.2em;opacity:0.9;">Predicted Value</div>'
                            f'<div style="font-size:2.5em;font-weight:800;">{display_pred}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        if task_type == "classification" and hasattr(model, "predict_proba"):
                            try:
                                proba   = model.predict_proba(df_pred)[0]
                                classes = (encoders["target"].classes_
                                           if "target" in encoders
                                           else [f"Class {k}" for k in range(len(proba))])
                                st.markdown("#### Prediction Confidence")
                                st.dataframe(
                                    pd.DataFrame({"Class": classes,
                                                  "Probability": [f"{p:.2%}" for p in proba]}),
                                    use_container_width=True, hide_index=True,
                                )
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            else:
                st.info("Train a model first (click **Run Data Pipeline** above).")

        if result.get("errors"):
            with st.expander("⚠️ Pipeline Errors & Warnings"):
                for err in result["errors"]:
                    st.warning(err)

    elif ml_df is None:
        st.markdown(
            '<div class="info-box">'
            '👆 Upload a CSV file above, then select your <b>target column</b> to get started.'
            '</div>',
            unsafe_allow_html=True,
        )


# =========================================================================
# FOOTER
# =========================================================================

st.markdown("---")
st.markdown(
    '<p style="color:#9CA3AF;font-size:0.85em;text-align:center;">'
    'DataPilot AI Pro — Automated Data Science Platform &nbsp;|&nbsp; '
    'Powered by LangGraph · Groq · SHAP · Reinforcement Learning'
    '</p>',
    unsafe_allow_html=True,
)
