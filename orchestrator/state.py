# orchestrator/state.py

"""
Pipeline State Definition for LangGraph Orchestrator.

The state is a TypedDict that flows through all agents in the pipeline.
Each agent reads what it needs and adds its outputs.

State flow:
  raw_data → ProfilerAgent → CleanerAgent → FeatureAgent → ModelerAgent
  → VisualizerAgent → ExplainerAgent → DataAnalyzerAgent (optional)

Every key is documented with its type and which agent produces it.
"""

from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd
import numpy as np


class PipelineState(TypedDict, total=False):
    """
    Complete pipeline state that flows through all agents.

    Naming convention:
      - Keys set by USER: raw_data, target_column, dataset_name, user_prompt
      - Keys set by PROFILER: task_type, profile_report, meta_features
      - Keys set by CLEANER: current_data, cleaning_report, imputation_map, outlier_bounds
      - Keys set by FEATURE: X, y, feature_names, feature_report, encoders, scalers
      - Keys set by MODELER: trained_models, cv_scores, ensemble_model, ensemble_score, best_model_name
      - Keys set by VISUALIZER: visualizations, viz_dir
      - Keys set by EXPLAINER: explanations, explain_dir
      - Keys set by DATA_ANALYZER: data_analysis, analyzer_dir
      - Keys set by ORCHESTRATOR: stage, errors, logs
    """

    # =====================================================================
    # USER INPUTS
    # =====================================================================
    raw_data: Any                    # pd.DataFrame — original uploaded CSV
    target_column: Optional[str]     # Target column name (auto-detected if None)
    dataset_name: str                # Name of the dataset (for display)
    user_prompt: Optional[str]       # Optional user question for data analyzer
    output_dir: str                  # Directory for all outputs

    # =====================================================================
    # PROFILER OUTPUTS
    # =====================================================================
    task_type: str                   # 'classification' or 'regression'
    profile_report: Dict[str, Any]   # {column_types, statistics, quality_score, warnings, n_rows, n_cols}
    meta_features: Any               # np.array of 32 meta-features for RL selector

    # =====================================================================
    # CLEANER OUTPUTS
    # =====================================================================
    current_data: Any                # pd.DataFrame — cleaned data
    cleaning_report: Dict[str, Any]  # {missing_value_handling, outlier_handling, duplicate_removal, transformations}
    imputation_map: Dict[str, Any]   # Column → imputation strategy used
    outlier_bounds: Dict[str, Any]   # Column → (lower, upper) bounds

    # =====================================================================
    # FEATURE AGENT OUTPUTS
    # =====================================================================
    X: Any                           # pd.DataFrame — feature matrix (ready for ML)
    y: Any                           # pd.Series — target vector
    feature_names: List[str]         # List of final feature names
    feature_report: Dict[str, Any]   # {encoding, scaling, feature_selection, new_features, dropped_columns}
    encoders: Dict[str, Any]         # Fitted encoders for each column
    scalers: Dict[str, Any]          # Fitted scalers for each column

    # =====================================================================
    # MODELER OUTPUTS
    # =====================================================================
    trained_models: Dict[str, Any]   # {model_name: fitted_model}
    cv_scores: Dict[str, float]      # {model_name: cv_score}
    ensemble_model: Any              # Fitted VotingClassifier/VotingRegressor
    ensemble_score: float            # Ensemble CV score
    best_model_name: str             # Name of best single model
    model_recommendations: List      # [(model_name, confidence), ...] from RL
    overfitting_analysis: Dict[str, Any]  # Overfitting detection results
    error_analysis: Dict[str, Any]        # Error analysis (where model fails)
    segment_analysis: Dict[str, Any]      # Performance by data segment

    # =====================================================================
    # VISUALIZER OUTPUTS
    # =====================================================================
    visualizations: Dict[str, Any]   # {group_name: {chart_name: plotly Figure}}
    viz_dir: str                     # Path to visualization output directory

    # =====================================================================
    # EXPLAINER OUTPUTS
    # =====================================================================
    explanations: Dict[str, Any]     # {shap_values, shap_importance, narratives, charts}
    explain_dir: str                 # Path to explanation output directory

    # =====================================================================
    # DATA ANALYZER OUTPUTS
    # =====================================================================
    data_context: Dict[str, Any]     # Phase 1 LLM context: {domain, suggested_target, task_type, cleaning_hints, feature_hints, ...}
    data_analysis: Dict[str, Any]    # {domain, summary, insights, charts, narratives, dashboard}
    analyzer_dir: str                # Path to data analysis output directory

    # =====================================================================
    # ORCHESTRATOR METADATA
    # =====================================================================
    stage: str                       # Current pipeline stage
    errors: List[str]                # List of errors encountered
    logs: List[str]                  # Log messages from all agents
    run_mode: str                    # 'ml_pipeline' or 'data_analysis' or 'both'
