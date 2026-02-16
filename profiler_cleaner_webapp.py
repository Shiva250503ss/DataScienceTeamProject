"""
DataPilot - Profiler & Cleaner Web App (Enhanced)
===================================================

A comprehensive Streamlit web interface for the enhanced Profiler and Cleaner agents.
Now includes: Quality Score, Warnings, Imputation Maps, Outlier Bounds, and Real Landmark Scores!

Run with: streamlit run profiler_cleaner_webapp.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from io import BytesIO
from datapilot.agents.profiler import ProfilerAgent
from datapilot.agents.cleaner import CleanerAgent


# Page configuration
st.set_page_config(
    page_title="DataPilot - Profiler & Cleaner (Enhanced)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_feature_names():
    """Return all 32 meta-feature names (updated for spec compliance)."""
    return [
        # Basic (6)
        "n_samples (normalized)", "n_features (normalized)",
        "n_numeric (ratio)", "n_categorical (ratio)",
        "n_classes (target unique ratio)", "dimensionality",

        # Missing (3)
        "missing_ratio", "cols_with_missing (ratio)", "max_missing_percent",

        # Statistical (10)
        "mean_skewness", "mean_kurtosis", "outlier_ratio",
        "mean_correlation", "cv_mean (coef. of variation)",
        "cv_std", "skew_std", "kurt_std",
        "range_ratio", "zero_ratio",

        # Categorical (3)
        "mean_cardinality", "max_cardinality", "high_cardinality_ratio",

        # Target (3)
        "target_imbalance", "target_skewness", "target_kurtosis",

        # PCA (3)
        "pca_95_components", "pca_50_variance", "intrinsic_dim",

        # Landmarks (4)
        "dt_score", "nb_score", "lr_score", "knn_score"
    ]


def display_quality_score(quality_score):
    """Display quality score with color-coded gauge."""
    if quality_score >= 90:
        color = "green"
        status = "Excellent"
    elif quality_score >= 75:
        color = "lightgreen"
        status = "Good"
    elif quality_score >= 60:
        color = "orange"
        status = "Fair"
    else:
        color = "red"
        status = "Poor"

    st.markdown(f"""
    <div style="padding: 20px; background-color: {color}; border-radius: 10px; text-align: center;">
        <h1 style="color: white; margin: 0;">{quality_score:.1f}/100</h1>
        <h3 style="color: white; margin: 0;">{status}</h3>
    </div>
    """, unsafe_allow_html=True)


def display_warnings(warnings):
    """Display warnings in an alert panel."""
    if not warnings:
        st.success("✅ No data quality warnings detected!")
        return

    st.warning(f"⚠️ {len(warnings)} Warning(s) Detected")
    for i, warning in enumerate(warnings, 1):
        st.markdown(f"**{i}.** {warning}")


def display_meta_features(meta_features):
    """Display meta-features in an organized layout with tabs."""
    feature_names = get_feature_names()

    categories = [
        ("📊 Basic", 0, 6),
        ("❓ Missing", 6, 9),
        ("📈 Statistical", 9, 19),
        ("🏷️ Categorical", 19, 22),
        ("🎯 Target", 22, 25),
        ("🔍 PCA", 25, 28),
        ("⭐ Landmarks", 28, 32),
    ]

    tabs = st.tabs([name for name, _, _ in categories])

    for tab, (category_name, start, end) in zip(tabs, categories):
        with tab:
            # Calculate number of columns based on feature count
            n_features = end - start
            n_cols = min(3, n_features)
            cols = st.columns(n_cols)

            for i, idx in enumerate(range(start, min(end, len(meta_features)))):
                col_idx = i % n_cols
                with cols[col_idx]:
                    st.metric(
                        label=feature_names[idx],
                        value=f"{meta_features[idx]:.4f}",
                        help=f"Feature index: {idx + 1}"
                    )


def display_imputation_strategies(imputation_map):
    """Display imputation strategies in a table."""
    if not imputation_map:
        st.info("ℹ️ No missing values were imputed.")
        return

    st.subheader("🔧 Imputation Strategies Applied")

    # Convert to DataFrame for display
    strategies = []
    for col, strategy in imputation_map.items():
        method = strategy.get('method', 'unknown')
        if method == 'mean' or method == 'median':
            detail = f"{method.capitalize()}: {strategy.get('value', 'N/A'):.4f}"
        elif method == 'mode':
            detail = f"Mode: {strategy.get('value', 'N/A')}"
        elif method == 'knn':
            detail = f"KNN (k={strategy.get('n_neighbors', 5)})"
        elif method == 'indicator+median':
            detail = f"Indicator column + Median: {strategy.get('value', 'N/A'):.4f}"
        elif method == 'unknown':
            detail = "Filled with 'Unknown'"
        elif method == 'missing_category':
            detail = "Filled with 'Missing' category"
        elif method == 'forward_backward_fill':
            detail = "Forward/Backward fill"
        else:
            detail = str(strategy)

        strategies.append({
            'Column': col,
            'Method': method.upper(),
            'Details': detail
        })

    df_strategies = pd.DataFrame(strategies)
    st.dataframe(df_strategies, use_container_width=True, hide_index=True)


def display_outlier_bounds(outlier_bounds):
    """Display outlier handling in a table."""
    if not outlier_bounds:
        st.info("ℹ️ No outliers were detected or handled.")
        return

    st.subheader("🎯 Outlier Handling Applied")

    # Convert to DataFrame for display
    outliers = []
    for col, bounds in outlier_bounds.items():
        method = bounds.get('method', 'unknown')
        if method == 'winsorization':
            detail = f"Capped at [{bounds.get('lower', 'N/A'):.4f}, {bounds.get('upper', 'N/A'):.4f}]"
        elif method == 'remove':
            detail = f"Removed {bounds.get('removed_count', 0)} rows"
        elif method == 'keep':
            reason = bounds.get('reason', 'unknown')
            if reason == 'skewed_distribution':
                detail = "Kept (skewed distribution - valid extremes)"
            elif reason == 'too_many_outliers':
                pct = bounds.get('outlier_pct', 0) * 100
                detail = f"Flagged for review ({pct:.1f}% outliers)"
            else:
                detail = f"Kept ({reason})"
        else:
            detail = str(bounds)

        outliers.append({
            'Column': col,
            'Action': method.upper(),
            'Details': detail
        })

    df_outliers = pd.DataFrame(outliers)
    st.dataframe(df_outliers, use_container_width=True, hide_index=True)


def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')


def convert_meta_to_csv(meta_features):
    """Convert meta-features to CSV for download."""
    feature_names = get_feature_names()
    df = pd.DataFrame({
        'feature_index': range(1, len(meta_features) + 1),
        'feature_name': feature_names[:len(meta_features)],
        'feature_value': meta_features
    })
    return df.to_csv(index=False).encode('utf-8')


def main():
    """Main Streamlit app."""

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<p class="main-header">🔬 DataPilot: Profiler & Cleaner</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enhanced with Quality Scoring, Warnings, and Advanced Cleaning Strategies</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This enhanced tool demonstrates the **Profiler** and **Cleaner** agents from the DataPilot AI Pro project.

        ### 🔬 Profiler Agent
        Extracts **32 meta-features**:
        - Basic statistics
        - Missing value patterns
        - Statistical properties
        - Categorical features
        - Target characteristics
        - PCA analysis
        - Baseline model scores

        ### 🧹 Cleaner Agent
        Intelligently cleans data:
        - Smart missing value handling
        - Advanced outlier detection
        - Distribution-aware strategies
        - Inference-ready outputs

        ### 📊 Task Support
        - ✅ Classification
        - ✅ Regression

        """)

        st.markdown("---")
        st.markdown("**DataPilot AI Pro**  \nBuilt with Streamlit + scikit-learn")

    # Main content
    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

            # Display dataset preview
            with st.expander("📊 Dataset Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                missing = df.isnull().sum().sum()
                st.metric("Missing Values", f"{missing:,}")
            with col4:
                duplicates = df.duplicated().sum()
                st.metric("Duplicates", duplicates)

            # Target column selection
            st.subheader("🎯 Select Target Column")
            target_col = st.selectbox(
                "Target Variable (or leave blank for auto-detection)",
                options=['Auto-detect'] + df.columns.tolist(),
                index=0
            )

            if target_col == 'Auto-detect':
                target_col = None

            # Analyze button
            if st.button("🚀 Analyze & Clean Dataset", type="primary", use_container_width=True):
                with st.spinner("⏳ Analyzing dataset..."):
                    # Run Profiler Agent
                    profiler = ProfilerAgent()
                    profile_report, meta_features, detected_target, task_type, quality_score, warnings = profiler.analyze(df, target_col)

                st.success(f"✅ Profiling complete! Detected task type: **{task_type.upper()}**")

                # Display Quality Score prominently
                st.markdown("---")
                st.subheader("📊 Data Quality Assessment")
                col1, col2 = st.columns([1, 2])

                with col1:
                    display_quality_score(quality_score)

                with col2:
                    st.markdown("### Quality Breakdown")
                    st.metric("Task Type", task_type.upper())
                    st.metric("Target Column", f"'{detected_target}'")
                    st.metric("Overall Quality", f"{quality_score:.1f}/100")

                # Display Warnings
                st.markdown("---")
                st.subheader("⚠️ Data Quality Warnings")
                display_warnings(warnings)

                # Display Profile Report
                st.markdown("---")
                st.subheader("📋 Profile Report")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Dataset Shape", f"{profile_report['dataset_shape'][0]} × {profile_report['dataset_shape'][1]}")
                    st.metric("Numeric Columns", profile_report['numeric_cols'])
                    st.metric("Categorical Columns", profile_report['categorical_cols'])

                with col2:
                    st.metric("DateTime Columns", profile_report['datetime_cols'])
                    st.metric("Text Columns", profile_report['text_cols'])
                    st.metric("ID Columns", profile_report['id_cols'])

                with col3:
                    st.metric("Missing Ratio", f"{profile_report['missing_ratio']:.4f}")
                    st.metric("Duplicates", profile_report['duplicates'])
                    st.metric("Quality Score", f"{quality_score:.1f}/100")

                # Display Meta-Features
                st.markdown("---")
                st.subheader("🔢 32 Meta-Features Extracted")
                display_meta_features(meta_features)

                # Display Landmark Scores separately
                st.markdown("---")
                st.subheader("⭐ Landmark Model Scores (Real Cross-Validated)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Decision Tree", f"{meta_features[28]:.4f}", help="Cross-validated score")
                with col2:
                    st.metric("Naive Bayes", f"{meta_features[29]:.4f}", help="Classification only, 0.5 for regression")
                with col3:
                    st.metric("Logistic/Linear Reg", f"{meta_features[30]:.4f}", help="Cross-validated score")
                with col4:
                    st.metric("K-Nearest Neighbors", f"{meta_features[31]:.4f}", help="Cross-validated score")

                # Run Cleaner Agent
                st.markdown("---")
                with st.spinner("🧹 Cleaning dataset..."):
                    cleaner = CleanerAgent()
                    df_cleaned, cleaning_report, imputation_map, outlier_bounds = cleaner.clean(
                        df, detected_target, task_type, meta_features, profile_report
                    )

                st.success("✅ Cleaning complete!")

                # Display Cleaning Report
                st.subheader("🧹 Cleaning Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Original Rows", cleaning_report['original_shape'][0])
                    st.metric("Final Rows", cleaning_report['final_shape'][0])

                with col2:
                    st.metric("Rows Removed", cleaning_report['rows_removed'])
                    st.metric("Columns Removed", cleaning_report['columns_removed'])

                with col3:
                    st.metric("Duplicates Removed", cleaning_report['duplicates_removed'])
                    st.metric("ID Columns Dropped", len(cleaning_report['id_columns_dropped']))

                with col4:
                    st.metric("Missing Ratio", f"{cleaning_report['missing_ratio']:.4f}")
                    st.metric("Outlier Ratio", f"{cleaning_report['outlier_ratio']:.4f}")

                # Transformations Applied
                if cleaning_report.get('transformations'):
                    with st.expander("🔍 Detailed Transformations Applied", expanded=False):
                        for i, transform in enumerate(cleaning_report['transformations'], 1):
                            st.text(f"{i}. {transform}")

                # Display Imputation Strategies
                st.markdown("---")
                display_imputation_strategies(imputation_map)

                # Display Outlier Bounds
                st.markdown("---")
                display_outlier_bounds(outlier_bounds)

                # Before/After Comparison
                st.markdown("---")
                st.subheader("📊 Before vs After Comparison")
                comparison_df = pd.DataFrame({
                    'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicates'],
                    'Before': [
                        df.shape[0],
                        df.shape[1],
                        df.isnull().sum().sum(),
                        df.duplicated().sum()
                    ],
                    'After': [
                        df_cleaned.shape[0],
                        df_cleaned.shape[1],
                        df_cleaned.isnull().sum().sum(),
                        df_cleaned.duplicated().sum()
                    ],
                    'Change': [
                        df.shape[0] - df_cleaned.shape[0],
                        df.shape[1] - df_cleaned.shape[1],
                        df.isnull().sum().sum() - df_cleaned.isnull().sum().sum(),
                        df.duplicated().sum() - df_cleaned.duplicated().sum()
                    ]
                })
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Display cleaned dataset
                with st.expander("🔍 Cleaned Dataset Preview", expanded=True):
                    st.dataframe(df_cleaned.head(20), use_container_width=True)

                # Download section
                st.markdown("---")
                st.subheader("💾 Download Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    csv_cleaned = convert_df_to_csv(df_cleaned)
                    st.download_button(
                        label="📥 Download Cleaned Dataset",
                        data=csv_cleaned,
                        file_name="cleaned_dataset.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    csv_meta = convert_meta_to_csv(meta_features)
                    st.download_button(
                        label="📥 Download Meta-Features",
                        data=csv_meta,
                        file_name="meta_features.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col3:
                    # Create comprehensive report
                    report_data = {
                        'profile_report': {k: v for k, v in profile_report.items() if k != 'column_types'},
                        'quality_score': quality_score,
                        'warnings': warnings,
                        'cleaning_report': cleaning_report,
                        'meta_features': meta_features.tolist(),
                        'task_type': task_type,
                        'imputation_map': imputation_map,
                        'outlier_bounds': outlier_bounds
                    }
                    json_report = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Full Report",
                        data=json_report,
                        file_name="analysis_report.json",
                        mime="application/json",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.exception(e)

    else:
        # Show message to upload CSV only
        st.info("👆 Please upload a CSV file to get started with data profiling and cleaning.")

        st.markdown("---")
        st.markdown("### 📝 Instructions:")
        st.markdown("""
        1. **Upload your CSV file** using the file uploader above
        2. **Select the target column** (the variable you want to predict)
        3. **Click "Analyze & Clean"** to process your data
        4. **Review the results** including:
           - Quality score and warnings
           - 32 meta-features extracted
           - Cleaning strategies applied
        5. **Download** the cleaned dataset and reports
        """)


if __name__ == "__main__":
    main()
