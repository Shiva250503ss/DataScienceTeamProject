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
import sys
import pandas as pd
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

        st.markdown("---")
        st.markdown("### Settings")

        with st.expander("⚙️ ML Settings", expanded=False):
            target_col = st.text_input(
                "Target Column (leave blank for auto-detection)",
                value="",
                help="The column you want to predict.",
                key="ml_target_col",
            )
            if not target_col.strip():
                target_col = None

        st.markdown("---")

        if st.button("🤖  Run Data Pipeline", use_container_width=True, type="primary",
                     help="Full pipeline: profiling → cleaning → features → models → SHAP/LIME"):
            output_dir = "./output"
            with st.spinner("Running full ML pipeline… (Profiling → Cleaning → Features → Models → Explanations)"):
                try:
                    fns    = _get_pipeline_functions()
                    result = fns["run_ml_pipeline"](
                        df=ml_df,
                        target_column=target_col,
                        dataset_name=ml_file_name,
                        output_dir=output_dir,
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
         tab_viz, tab_explain, tab_errors, tab_segments, tab_predict) = st.tabs([
            "📊 Models", "📋 Data Profile", "🧹 Cleaning", "🔧 Features",
            "📈 Visualizations", "🔍 Explanations", "⚠️ Error Analysis",
            "📐 Segments", "🎯 Predict",
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
                display_charts(explanations.get("charts", {}), "Explain")
            else:
                st.info("No explanations available.")

        # ── Error Analysis ────────────────────────────────────────────────────
        with tab_errors:
            error_analysis = result.get("error_analysis", {})
            if error_analysis:
                st.markdown("### Error Analysis")
                task_type = result.get("task_type", "")
                if task_type == "regression":
                    ea_cols = st.columns(3)
                    ea_cols[0].metric("MAE",  f"{error_analysis.get('mae', 0):.4f}")
                    ea_cols[1].metric("RMSE", f"{error_analysis.get('rmse', 0):.4f}")
                    ea_cols[2].metric("R²",   f"{error_analysis.get('r2', 0):.4f}")
                    display_charts(error_analysis.get("charts", {}), "ErrorAnalysis")
                elif task_type == "classification":
                    report = error_analysis.get("classification_report", "")
                    if report:
                        st.text(report)
                    cm_fig = error_analysis.get("confusion_matrix_fig")
                    if cm_fig and hasattr(cm_fig, "to_json"):
                        st.plotly_chart(cm_fig, use_container_width=True)
                    display_charts(error_analysis.get("charts", {}), "ErrorAnalysis")
            else:
                st.info("No error analysis available.")

        # ── Segments ──────────────────────────────────────────────────────────
        with tab_segments:
            segment_analysis = result.get("segment_analysis", {})
            segments_list = segment_analysis.get("segments", []) if isinstance(segment_analysis, dict) else []
            if segments_list:
                st.markdown("### Segment Analysis")
                summary = segment_analysis.get("summary", "")
                if summary:
                    st.info(summary)
                # Group segments by column
                cols_seen = []
                for s in segments_list:
                    c = s.get("column", "unknown")
                    if c not in cols_seen:
                        cols_seen.append(c)
                for col in cols_seen:
                    column_segments = [s for s in segments_list if s.get("column") == col]
                    with st.expander(f"Segment by: `{col}`"):
                        st.dataframe(
                            pd.DataFrame(column_segments).drop(columns=["column"], errors="ignore"),
                            use_container_width=True, hide_index=True,
                        )
            else:
                st.info("No segment analysis available.")

        # ── Predict ───────────────────────────────────────────────────────────
        with tab_predict:
            st.markdown("### 🎯 Predict With Your Own Values")
            st.markdown(
                '<div class="info-box">'
                'Enter your own feature values and the trained model will make a prediction.'
                '</div>',
                unsafe_allow_html=True,
            )
            feature_names = result.get("feature_names", [])
            if feature_names and result.get("ensemble_model"):
                st.markdown("#### Enter Feature Values")
                user_inputs = {}
                for i in range(0, len(feature_names), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        fi = i + j
                        if fi < len(feature_names):
                            feat = feature_names[fi]
                            with col:
                                user_inputs[feat] = st.number_input(
                                    feat, value=0.0, format="%.4f",
                                    key=f"ml_predict_{feat}",
                                )

                if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
                    try:
                        import numpy as np
                        input_df = pd.DataFrame([user_inputs])[feature_names]

                        scalers     = result.get("scalers", {})
                        num_scaler  = scalers.get("numeric")
                        cols_scaled = (result.get("feature_report", {})
                                       .get("scaling", {})
                                       .get("columns_scaled", []))
                        if num_scaler is not None and cols_scaled:
                            avail = [c for c in cols_scaled if c in input_df.columns]
                            if avail:
                                input_df[avail] = num_scaler.transform(input_df[avail])

                        model      = result["ensemble_model"]
                        prediction = model.predict(input_df)[0]

                        encoders      = result.get("encoders", {})
                        task_type     = result.get("task_type", "")
                        display_pred  = prediction
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
                                proba   = model.predict_proba(input_df)[0]
                                classes = (encoders["target"].classes_
                                           if "target" in encoders
                                           else [f"Class {i}" for i in range(len(proba))])
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
            '👆 Upload a CSV file above, then click <b>Run Data Pipeline</b> to get started.'
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
