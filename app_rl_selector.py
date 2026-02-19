"""
================================================================================
RL MODEL SELECTOR - STREAMLIT WEB APPLICATION
================================================================================

A Streamlit web application that uses Reinforcement Learning to automatically
select the best machine learning model for your dataset.

Features:
- Upload CSV dataset
- Automatic task detection (Classification/Regression)
- RL-powered model selection
- Automatic training with the best model
- Performance visualization and results

Usage:
    streamlit run app_rl_selector.py

Author: DataPilot AI Pro
Version: 1.0.0
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import RL Model Selector components
try:
    from rl_model_selector_classification_regression import (
        RLModelSelector,
        RLModelSelectorConfig,
        TaskType,
        MetaFeatureExtractor,
        GPUModelFactory,
        gpu_manager
    )
    RL_AVAILABLE = True
except ImportError as e:
    RL_AVAILABLE = False
    st.error(f"Error importing RL components: {e}")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RL Model Selector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dataset(uploaded_file):
    """Load dataset from uploaded file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

def preprocess_data(X, y, task_type):
    """Preprocess data for training."""
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) > 0:
        X_processed = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col].astype(str))
    else:
        X_processed = X.copy()

    # Fill missing values
    X_processed = X_processed.fillna(X_processed.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    # Encode target if classification
    if task_type == TaskType.CLASSIFICATION:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X_scaled, y_encoded, scaler, le
    else:
        return X_scaled, y.values, scaler, None

def train_and_evaluate(model, X_train, X_test, y_train, y_test, task_type):
    """Train model and evaluate performance."""
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {}

    if task_type == TaskType.CLASSIFICATION:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
    else:
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['predictions'] = y_pred
        metrics['actuals'] = y_test

    metrics['training_time'] = training_time

    return model, metrics

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    return fig

def plot_regression_results(y_true, y_pred, title="Predictions vs Actual"):
    """Plot regression results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Residuals
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🤖 RL Model Selector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Intelligent Model Selection using Reinforcement Learning</div>',
        unsafe_allow_html=True
    )

    # Check if RL components are available
    if not RL_AVAILABLE:
        st.error("⚠️ RL Model Selector components are not available. Please check the installation.")
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # GPU Check
        st.subheader("GPU Status")
        if gpu_manager.cuda_available:
            st.success(f"✅ GPU Available\n\n{gpu_manager.gpu_name}")
        else:
            st.warning("⚠️ No GPU detected. CPU mode will be used (slower).")

        st.markdown("---")

        # Model Options
        st.subheader("Model Selection Options")
        top_k = st.slider("Number of top models to show", 1, 5, 3)

        st.markdown("---")

        # Training Options
        st.subheader("Training Options")
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        use_cv = st.checkbox("Use Cross-Validation", value=False)
        if use_cv:
            cv_folds = st.slider("CV Folds", 3, 10, 5)

        st.markdown("---")

        # About
        st.subheader("About")
        st.info("""
        This application uses Reinforcement Learning (PPO) to automatically
        select the best machine learning model for your dataset.

        **Features:**
        - Automatic task detection
        - Meta-feature extraction
        - GPU-accelerated training
        - Comprehensive evaluation
        """)

    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "🤖 Model Selection", "📊 Results"])

    # TAB 1: Upload Data
    with tab1:
        st.header("Upload Your Dataset")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your dataset. The file should contain features and a target column."
        )

        if uploaded_file is not None:
            # Load dataset
            df, error = load_dataset(uploaded_file)

            if error:
                st.error(f"Error loading dataset: {error}")
                return

            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

            # Store in session state
            st.session_state['df'] = df

            # Display preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))

            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())

            # Column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)

            # Select target column
            st.subheader("Select Target Column")
            target_col = st.selectbox(
                "Choose the column you want to predict",
                df.columns.tolist()
            )

            if target_col:
                st.session_state['target_col'] = target_col

                # Detect task type
                y = df[target_col]
                task_type = TaskType.detect(y)
                st.session_state['task_type'] = task_type

                st.info(f"📋 Detected Task Type: **{task_type.value.upper()}**")

                # Show target distribution
                if task_type == TaskType.CLASSIFICATION:
                    st.subheader("Target Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    y.value_counts().plot(kind='bar', ax=ax)
                    ax.set_title('Target Class Distribution')
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                else:
                    st.subheader("Target Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(y, bins=30, edgecolor='black')
                    ax.set_title('Target Value Distribution')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)

    # TAB 2: Model Selection
    with tab2:
        st.header("🤖 RL Model Selection")

        if 'df' not in st.session_state or 'target_col' not in st.session_state:
            st.warning("⚠️ Please upload a dataset and select a target column first.")
            return

        df = st.session_state['df']
        target_col = st.session_state['target_col']
        task_type = st.session_state['task_type']

        st.info(f"Task Type: **{task_type.value.upper()}**")

        if st.button("🚀 Run Model Selection", type="primary"):
            with st.spinner("Running RL Model Selection..."):
                try:
                    # Initialize RL Selector
                    config = RLModelSelectorConfig()
                    selector = RLModelSelector(config)

                    # Load RL model
                    progress_text = st.empty()
                    progress_text.text("Loading RL model...")

                    if gpu_manager.cuda_available:
                        selector.load(task_type)
                    else:
                        st.warning("GPU not available. Some models may not be available.")
                        # Try to load anyway for CPU mode
                        try:
                            selector.load(task_type)
                        except Exception as e:
                            st.error(f"Error loading RL model: {e}")
                            st.info("💡 Tip: Train the RL model first using the training script.")
                            return

                    progress_text.text("Extracting meta-features...")

                    # Run selection
                    result = selector.select(df, target_col, task_type, top_k=top_k)

                    st.session_state['selection_result'] = result

                    progress_text.text("✅ Model selection complete!")
                    time.sleep(0.5)
                    progress_text.empty()

                    # Display results
                    st.success("✅ Model Selection Complete!")

                    st.subheader(f"🏆 Top {top_k} Recommended Models")

                    for i, (model_name, prob) in enumerate(zip(result['models'], result['probs'])):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{i+1}. {model_name}**")
                        with col2:
                            st.progress(prob)
                            st.caption(f"{prob*100:.1f}%")

                    # All model probabilities
                    with st.expander("📊 All Model Probabilities"):
                        prob_df = pd.DataFrame({
                            'Model': list(result['all'].keys()),
                            'Probability': list(result['all'].values())
                        }).sort_values('Probability', ascending=False)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(prob_df['Model'], prob_df['Probability'])
                        ax.set_xlabel('Probability')
                        ax.set_title('All Model Selection Probabilities')
                        plt.tight_layout()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during model selection: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # TAB 3: Results
    with tab3:
        st.header("📊 Training & Results")

        if 'selection_result' not in st.session_state:
            st.warning("⚠️ Please run model selection first.")
            return

        df = st.session_state['df']
        target_col = st.session_state['target_col']
        task_type = st.session_state['task_type']
        result = st.session_state['selection_result']

        # Select model to train
        st.subheader("Select Model to Train")
        selected_model_name = st.selectbox(
            "Choose a model to train",
            result['models']
        )

        if st.button("🎯 Train Selected Model", type="primary"):
            with st.spinner(f"Training {selected_model_name}..."):
                try:
                    # Prepare data
                    X = df.drop(columns=[target_col])
                    y = df[target_col]

                    # Preprocess
                    X_processed, y_processed, scaler, label_encoder = preprocess_data(X, y, task_type)

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed,
                        test_size=test_size,
                        random_state=42
                    )

                    # Get model
                    config = RLModelSelectorConfig()
                    models = GPUModelFactory.get_models(task_type, config)

                    if selected_model_name not in models:
                        st.error(f"Model {selected_model_name} not available.")
                        return

                    model = models[selected_model_name]

                    # Train and evaluate
                    trained_model, metrics = train_and_evaluate(
                        model, X_train, X_test, y_train, y_test, task_type
                    )

                    st.session_state['trained_model'] = trained_model
                    st.session_state['metrics'] = metrics
                    st.session_state['model_name'] = selected_model_name

                    st.success(f"✅ Model trained successfully in {metrics['training_time']:.2f} seconds!")

                    # Display results
                    st.subheader("📈 Performance Metrics")

                    if task_type == TaskType.CLASSIFICATION:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                        with col4:
                            st.metric("F1 Score", f"{metrics['f1']:.3f}")

                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        fig = plot_confusion_matrix(metrics['confusion_matrix'])
                        st.pyplot(fig)

                        # Classification Report
                        with st.expander("📄 Detailed Classification Report"):
                            st.text(metrics['classification_report'])

                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{metrics['r2']:.3f}")
                        with col2:
                            st.metric("MSE", f"{metrics['mse']:.3f}")
                        with col3:
                            st.metric("RMSE", f"{metrics['rmse']:.3f}")
                        with col4:
                            st.metric("MAE", f"{metrics['mae']:.3f}")

                        # Regression plots
                        st.subheader("Prediction Analysis")
                        fig = plot_regression_results(metrics['actuals'], metrics['predictions'])
                        st.pyplot(fig)

                    # Cross-validation if enabled
                    if use_cv:
                        st.subheader("Cross-Validation Results")
                        with st.spinner("Running cross-validation..."):
                            if task_type == TaskType.CLASSIFICATION:
                                cv_scores = cross_val_score(
                                    model, X_processed, y_processed,
                                    cv=cv_folds, scoring='accuracy'
                                )
                                st.info(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                            else:
                                cv_scores = cross_val_score(
                                    model, X_processed, y_processed,
                                    cv=cv_folds, scoring='r2'
                                )
                                st.info(f"CV R² Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

                except Exception as e:
                    st.error(f"Error during training: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
