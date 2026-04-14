# orchestrator/graph.py

"""
LangGraph Orchestrator — Connects all agents into an intelligent pipeline.

This is the BRAIN of DataPilot AI Pro. It defines the execution graph
that routes data through the correct sequence of agents.

TWO PIPELINES:

  1. ML Pipeline (for data scientists):
     Upload CSV → Profiler → Cleaner → Feature → Modeler → Visualizer → Explainer
     Result: Trained model + performance charts + SHAP/LIME explanations

  2. Data Analysis Pipeline (for managers):
     Upload CSV → DataAnalyzer (LLM-powered insights + dashboards)
     Result: Interactive business intelligence dashboard

  3. Full Pipeline (both):
     Runs ML Pipeline first, then Data Analysis on the same data.

Graph structure (LangGraph):

    START
      ↓
    [route_decision]  ← decides which pipeline to run
      ↓           ↓
    [profiler]   [data_analyzer]  ← data analysis branch
      ↓               ↓
    [cleaner]        END
      ↓
    [feature]
      ↓
    [modeler]
      ↓
    [visualizer]
      ↓
    [explainer]
      ↓
    [data_analyzer]  ← optional, if run_mode='both'
      ↓
    END

Owner: Orchestration Team
"""

import os
import time
import traceback
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END
from orchestrator.state import PipelineState

from agents.profiler import ProfilerAgent
from agents.cleaner import CleanerAgent
from agents.feature import FeatureAgent
from agents.modeler import ModelerAgent
from agents.visualizer import VisualizerAgent
from agents.explainer import ExplainerAgent
from agents.data_analyzer import DataAnalyzerAgent


# =========================================================================
# AGENT INSTANCES (created once, reused across runs)
# =========================================================================

profiler_agent = ProfilerAgent()
cleaner_agent = CleanerAgent()
feature_agent = FeatureAgent()
modeler_agent = ModelerAgent()
visualizer_agent = VisualizerAgent()
explainer_agent = ExplainerAgent()
data_analyzer_agent = DataAnalyzerAgent()


# =========================================================================
# NODE FUNCTIONS
# Each function wraps an agent's execute() with error handling and timing.
# =========================================================================

def run_profiler(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 1: Profile the dataset."""
    print(f"\n{'='*60}")
    print("  STAGE 1/6: DATA PROFILING")
    print(f"{'='*60}")
    start = time.time()
    try:
        # Set initial current_data from raw_data
        if 'current_data' not in state:
            state['current_data'] = state['raw_data'].copy()
        state = profiler_agent.execute(state)
    except Exception as e:
        state.setdefault('errors', []).append(f"Profiler failed: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return state


def run_cleaner(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 2: Clean the data."""
    print(f"\n{'='*60}")
    print("  STAGE 2/6: DATA CLEANING")
    print(f"{'='*60}")
    start = time.time()
    try:
        state = cleaner_agent.execute(state)
    except Exception as e:
        state.setdefault('errors', []).append(f"Cleaner failed: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return state


def run_feature(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 3: Feature engineering."""
    print(f"\n{'='*60}")
    print("  STAGE 3/6: FEATURE ENGINEERING")
    print(f"{'='*60}")
    start = time.time()
    try:
        state = feature_agent.execute(state)
    except Exception as e:
        state.setdefault('errors', []).append(f"Feature agent failed: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return state


def run_modeler(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 4: Model training & selection."""
    print(f"\n{'='*60}")
    print("  STAGE 4/6: MODEL TRAINING")
    print(f"{'='*60}")
    start = time.time()
    try:
        state = modeler_agent.execute(state)
    except Exception as e:
        state.setdefault('errors', []).append(f"Modeler failed: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return state


def run_visualizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 5: Generate ML pipeline visualizations."""
    print(f"\n{'='*60}")
    print("  STAGE 5/6: VISUALIZATION")
    print(f"{'='*60}")
    start = time.time()
    try:
        state = visualizer_agent.execute(state)
    except Exception as e:
        state.setdefault('errors', []).append(f"Visualizer failed: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return state


def run_explainer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 6: Model explainability (SHAP + LIME + LLM narratives)."""
    print(f"\n{'='*60}")
    print("  STAGE 6/6: EXPLAINABILITY")
    print(f"{'='*60}")
    start = time.time()
    try:
        state = explainer_agent.execute(state)
    except Exception as e:
        state.setdefault('errors', []).append(f"Explainer failed: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return state


def run_context_analyzer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 0: LLM context analysis — understands data BEFORE the ML pipeline starts."""
    print(f"\n{'='*60}")
    print("  STAGE 0/6: DATA CONTEXT ANALYSIS (LLM)")
    print(f"{'='*60}")
    start = time.time()
    try:
        state = data_analyzer_agent.execute_context_phase(state)
    except Exception as e:
        state.setdefault('errors', []).append(f"Context analysis failed: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return state


def run_data_analyzer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: LLM-powered data analysis (for managers)."""
    print(f"\n{'='*60}")
    print("  DATA ANALYSIS: LLM-POWERED INSIGHTS")
    print(f"{'='*60}")
    start = time.time()
    try:
        state = data_analyzer_agent.execute(state)
    except Exception as e:
        state.setdefault('errors', []).append(f"Data Analyzer failed: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return state


# =========================================================================
# ROUTING FUNCTION
# =========================================================================

def route_pipeline(state: Dict[str, Any]) -> str:
    """
    Decide which pipeline to run based on run_mode.

    Returns the name of the next node to execute.
    """
    run_mode = state.get('run_mode', 'ml_pipeline')

    if run_mode == 'data_analysis':
        # Manager mode: go straight to data analyzer (no ML needed)
        return 'data_analyzer'
    else:
        # ml_pipeline and both: start with LLM context analysis first
        return 'context_analyzer'


def route_after_explainer(state: Dict[str, Any]) -> str:
    """After ML pipeline, optionally run data analysis."""
    run_mode = state.get('run_mode', 'ml_pipeline')
    if run_mode == 'both':
        return 'data_analyzer'
    return END


# =========================================================================
# GRAPH BUILDER
# =========================================================================

def build_pipeline_graph() -> StateGraph:
    """
    Build the LangGraph execution graph.

    This defines the DAG (Directed Acyclic Graph) of agent execution:

      START → route_decision → profiler → cleaner → feature → modeler
              → visualizer → explainer → [data_analyzer] → END

    Or for data_analysis mode:
      START → route_decision → data_analyzer → END
    """
    # Create the state graph
    graph = StateGraph(dict)

    # Add all nodes
    graph.add_node("context_analyzer", run_context_analyzer)
    graph.add_node("profiler", run_profiler)
    graph.add_node("cleaner", run_cleaner)
    graph.add_node("feature", run_feature)
    graph.add_node("modeler", run_modeler)
    graph.add_node("visualizer", run_visualizer)
    graph.add_node("explainer", run_explainer)
    graph.add_node("data_analyzer", run_data_analyzer)

    # Set the entry point with conditional routing
    graph.set_conditional_entry_point(
        route_pipeline,
        {
            "context_analyzer": "context_analyzer",
            "data_analyzer": "data_analyzer"
        }
    )

    # Context analyzer feeds directly into profiler
    graph.add_edge("context_analyzer", "profiler")

    # ML Pipeline edges (linear sequence)
    graph.add_edge("profiler", "cleaner")
    graph.add_edge("cleaner", "feature")
    graph.add_edge("feature", "modeler")
    graph.add_edge("modeler", "visualizer")
    graph.add_edge("visualizer", "explainer")

    # After explainer: conditionally go to data_analyzer or END
    graph.add_conditional_edges(
        "explainer",
        route_after_explainer,
        {
            "data_analyzer": "data_analyzer",
            END: END
        }
    )

    # Data analyzer always ends
    graph.add_edge("data_analyzer", END)

    return graph


def compile_pipeline():
    """Build and compile the pipeline graph for execution."""
    graph = build_pipeline_graph()
    return graph.compile()


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

def run_ml_pipeline(df, target_column=None, dataset_name='Dataset',
                    output_dir='./output', user_selected_model=None):
    """
    Run the full ML pipeline on a dataset.

    Args:
        df: pandas DataFrame
        target_column: target column name (auto-detected if None)
        dataset_name: name for display
        output_dir: where to save outputs
        user_selected_model: specific sklearn model class name chosen by
            the user (e.g. 'RandomForestClassifier').  When *None* the
            PPO RL agent selects the best models automatically.

    Returns:
        Final pipeline state dict
    """
    pipeline = compile_pipeline()

    initial_state = {
        'raw_data': df,
        'target_column': target_column,
        'dataset_name': dataset_name,
        'output_dir': output_dir,
        'run_mode': 'ml_pipeline',
        'user_selected_model': user_selected_model,
        'errors': [],
        'logs': [],
        'stage': 'start'
    }

    print(f"\n{'#'*60}")
    print(f"  DATAPILOT AI PRO — ML PIPELINE")
    print(f"  Dataset: {dataset_name}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Target: {target_column or 'Auto-detect'}")
    print(f"  Model: {user_selected_model or 'PPO Auto-Select'}")
    print(f"{'#'*60}")

    start = time.time()
    result = pipeline.invoke(initial_state)
    total = time.time() - start

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE in {total:.1f}s")
    if result.get('errors'):
        print(f"  Errors: {len(result['errors'])}")
        for e in result['errors']:
            print(f"    - {e}")
    else:
        print(f"  Status: SUCCESS")
    print(f"  Best model: {result.get('best_model_name', 'N/A')}")
    print(f"  Ensemble score: {result.get('ensemble_score', 0):.4f}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    return result


def run_data_analysis(df, dataset_name='Dataset', user_prompt=None,
                      output_dir='./output'):
    """
    Run only the LLM-powered data analysis (for managers).

    Args:
        df: pandas DataFrame
        dataset_name: name for display
        user_prompt: optional specific question
        output_dir: where to save outputs

    Returns:
        Final pipeline state dict
    """
    pipeline = compile_pipeline()

    initial_state = {
        'raw_data': df,
        'dataset_name': dataset_name,
        'user_prompt': user_prompt,
        'output_dir': output_dir,
        'run_mode': 'data_analysis',
        'errors': [],
        'logs': [],
        'stage': 'start'
    }

    print(f"\n{'#'*60}")
    print(f"  DATAPILOT AI PRO — DATA ANALYSIS")
    print(f"  Dataset: {dataset_name}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    if user_prompt:
        print(f"  User prompt: {user_prompt}")
    print(f"{'#'*60}")

    start = time.time()
    result = pipeline.invoke(initial_state)
    total = time.time() - start

    print(f"\n  Analysis complete in {total:.1f}s")
    print(f"  Dashboard: {result.get('analyzer_dir', output_dir)}/dashboard.html")

    return result


def run_full_pipeline(df, target_column=None, dataset_name='Dataset',
                      output_dir='./output'):
    """
    Run BOTH ML pipeline AND data analysis.

    Args:
        df: pandas DataFrame
        target_column: target column name (auto-detected if None)
        dataset_name: name for display
        output_dir: where to save outputs

    Returns:
        Final pipeline state dict
    """
    pipeline = compile_pipeline()

    initial_state = {
        'raw_data': df,
        'target_column': target_column,
        'dataset_name': dataset_name,
        'output_dir': output_dir,
        'run_mode': 'both',
        'errors': [],
        'logs': [],
        'stage': 'start'
    }

    print(f"\n{'#'*60}")
    print(f"  DATAPILOT AI PRO — FULL PIPELINE")
    print(f"  Dataset: {dataset_name}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Target: {target_column or 'Auto-detect'}")
    print(f"  Modes: ML Pipeline + Data Analysis")
    print(f"{'#'*60}")

    start = time.time()
    result = pipeline.invoke(initial_state)
    total = time.time() - start

    print(f"\n{'#'*60}")
    print(f"  FULL PIPELINE COMPLETE in {total:.1f}s")
    print(f"{'#'*60}")

    return result
