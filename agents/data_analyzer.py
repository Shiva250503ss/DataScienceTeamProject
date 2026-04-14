# agents/data_analyzer.py

"""
Data Analyzer Agent — LLM-Powered Business Intelligence Dashboard Generator.

This agent is SEPARATE from the ML Visualizer. It's designed for MANAGERS
and NON-TECHNICAL users who just want to upload a CSV and get smart insights.

How it works:
  1. User (manager) uploads ANY dataset
  2. Agent sends column names + sample rows + statistics to the LLM
  3. LLM analyzes the data context and returns:
     - What domain is this data from? (sales, HR, healthcare, etc.)
     - What are the KEY INSIGHTS we should look for?
     - What charts should we build for each insight?
  4. Agent auto-generates interactive Plotly dashboards for each insight
  5. LLM writes natural-language summaries explaining each finding
  6. (Optional) User can type a custom prompt like:
     "Show me revenue trends by quarter" → Agent generates that specific chart

This is the "smart analyst" feature — the LLM acts like a data analyst who
knows what questions to ask about any dataset.

Owner: Data Analytics Team
"""

import os
import json
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from agents.base import BaseAgent

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# =========================================================================
# LLM PROMPT TEMPLATES
# =========================================================================

INSIGHT_DISCOVERY_PROMPT = """You are a senior data analyst. A manager just uploaded a dataset and wants automated insights.

DATASET INFO:
- Name: {dataset_name}
- Rows: {n_rows:,} | Columns: {n_cols}
- Columns and types:
{column_info}

SAMPLE DATA (first 5 rows):
{sample_data}

BASIC STATISTICS:
{statistics}

STATISTICAL HIGHLIGHTS (pre-computed):
{stat_highlights}

YOUR TASK:
Analyze this data and suggest 6-8 KEY INSIGHTS that a business manager would want to see.
For each insight, specify what chart to create.

RESPOND IN THIS EXACT JSON FORMAT (no other text):
{{
  "domain": "the business domain (e.g., sales, HR, healthcare, finance)",
  "summary": "one paragraph overview of what this dataset is about",
  "key_metric": "the single most important numeric column for business decisions",
  "key_grouping": "the most meaningful categorical column to segment by",
  "insights": [
    {{
      "title": "Short insight title that states the FINDING (e.g. 'Region X leads with 40% of revenue'), NOT just axis labels",
      "description": "What this insight reveals and why it matters",
      "chart_type": "bar|line|scatter|heatmap|histogram|box|treemap|funnel|area",
      "x_column": "column name for x-axis (or null)",
      "y_column": "column name for y-axis (or null)",
      "color_column": "column for color grouping (or null)",
      "aggregation": "sum|mean|count|median|max|min|none",
      "insight_type": "distribution|trend|comparison|correlation|composition|ranking"
    }}
  ]
}}

RULES:
- Only use column names that ACTUALLY EXIST in the dataset
- Choose chart types that are APPROPRIATE for the data types
- For categorical columns with many values, suggest top 10/15 grouping
- Include at least 1 distribution chart, 1 comparison chart, and 1 composition chart
- Think about what a MANAGER would want to know, not what a data scientist would want
- Focus on actionable business insights
- SKIP insights where data is nearly uniform (all categories within 5% of each other)
- SKIP correlations weaker than r=0.15 — they are meaningless noise
- NEVER use pie charts — use horizontal bar charts instead
- Prefer columns with HIGH variance and BUSINESS meaning as the y-axis metric
- DO NOT pick ID columns, index columns, or near-constant columns as metrics
- key_metric should be the column most useful for business KPIs (revenue, count, score, etc.)
- key_grouping should be a categorical column with 3-15 distinct values
"""

USER_PROMPT_TEMPLATE = """You are a senior data analyst. The user wants a specific visualization.

DATASET INFO:
- Columns: {columns}
- Column types: {column_types}
- Sample values: {sample_values}

USER REQUEST: "{user_prompt}"

YOUR TASK:
Figure out exactly what chart to create from the available columns.

RESPOND IN THIS EXACT JSON FORMAT (no other text):
{{
  "title": "Chart title",
  "description": "What this chart shows",
  "chart_type": "bar|line|pie|scatter|heatmap|histogram|box|treemap|funnel|area",
  "x_column": "column name for x-axis (or null)",
  "y_column": "column name for y-axis (or null)",
  "color_column": "column name for color grouping (or null)",
  "aggregation": "sum|mean|count|median|max|min|none",
  "filter_column": "column to filter on (or null)",
  "filter_value": "value to filter by (or null)",
  "sort": "ascending|descending|none",
  "top_n": null
}}

RULES:
- ONLY use columns that exist: {columns}
- Match the user's intent as closely as possible
- If the user asks about trends, use a line chart
- If the user asks about distribution, use histogram or box plot
- If the user asks about comparison, use bar chart
- If the user asks about composition, use pie or treemap
"""

CONTEXT_ANALYSIS_PROMPT = """You are an expert data scientist examining a new dataset for the first time.
Your job is to understand what this data is about and guide the ML pipeline BEFORE any processing.

DATASET INFO:
- Name: {dataset_name}
- Rows: {n_rows:,} | Columns: {n_cols}
- Columns and sample values:
{column_info}

SAMPLE DATA (first 5 rows):
{sample_data}

BASIC STATISTICS:
{statistics}

YOUR TASK:
Analyze this dataset like a senior data scientist seeing it for the first time.
Identify the domain, the most likely ML target column, and cleaning considerations.

RESPOND IN THIS EXACT JSON FORMAT (no other text):
{{
  "domain": "the business domain (e.g., healthcare, sales, HR, finance, education, e-commerce)",
  "dataset_description": "one sentence describing what this dataset captures",
  "suggested_target": "the most likely target column name for ML prediction",
  "task_type": "classification or regression",
  "target_reasoning": "brief reason why this is the target column",
  "important_features": ["col1", "col2", "col3"],
  "cleaning_hints": {{
    "column_name": "specific cleaning note (e.g., 'zero means missing', 'outliers expected', 'case inconsistency')"
  }},
  "feature_hints": ["useful feature engineering idea 1", "idea 2"],
  "business_context": "what business question this data helps answer"
}}

RULES:
- Only suggest column names that ACTUALLY EXIST in the dataset
- suggested_target must be a real column name from the dataset
- cleaning_hints should only list columns that need special attention
- task_type must be exactly 'classification' or 'regression'
"""

INSIGHT_NARRATIVE_PROMPT = """You are a business analyst writing a brief insight for a manager's dashboard.

CHART: {chart_title}
DATA SUMMARY:
{data_summary}

Write a 2-3 sentence insight in PLAIN BUSINESS ENGLISH. No technical jargon.
Focus on: What does this mean? Is it good or bad? What should the manager do?
"""


class DataAnalyzerAgent(BaseAgent):
    """
    LLM-Powered Data Analyzer for Business Intelligence.

    Two modes of operation:
      1. AUTO MODE: Upload data → LLM discovers insights → generates dashboard
      2. PROMPT MODE: User types a question → LLM generates the specific chart

    Unlike the VisualizerAgent (which shows ML pipeline results), this agent
    is designed for business users who want to understand their RAW data.
    """

    def __init__(self):
        super().__init__("DataAnalyzerAgent")
        self.colors = {
            'primary':   '#2563EB',
            'secondary': '#7C3AED',
            'success':   '#059669',
            'warning':   '#D97706',
            'danger':    '#DC2626',
            'info':      '#0891B2',
        }
        # Soft pastel palette inspired by modern BI dashboards
        self.color_sequence = [
            '#8B9DC3',  # soft purple/search
            '#F5C26B',  # warm yellow/direct
            '#F4A261',  # soft orange/referral
            '#A8D5BA',  # soft green/paid
            '#E76F7A',  # soft red
            '#7EC8E3',  # sky blue
            '#C3B1E1',  # lavender
            '#F9D5A7',  # peach
        ]
        self.template = 'plotly_dark'

    # =====================================================================
    # MAIN ENTRY POINTS
    # =====================================================================

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-analyze a dataset and generate a full insight dashboard.
        
        This is called when a manager uploads a CSV without any specific question.
        The LLM figures out what's interesting in the data.
        """
        self.log("Starting LLM-powered data analysis...")

        df = state.get('raw_data', state.get('current_data'))
        if df is None:
            self.log("No data found in state!")
            return state

        dataset_name = state.get('dataset_name', 'Uploaded Dataset')

        # Create output directory
        output_dir = state.get('output_dir', './output')
        analyzer_dir = os.path.join(output_dir, 'data_analysis')
        os.makedirs(analyzer_dir, exist_ok=True)

        # Step 1: Profile the data (quick stats for the LLM)
        self.log("Step 1: Quick data profiling...")
        data_profile = self._quick_profile(df)

        # Step 2: Ask LLM to discover insights
        self.log("Step 2: Asking LLM to discover insights...")
        insights = self._discover_insights(df, dataset_name, data_profile)

        if not insights:
            self.log("LLM insight discovery failed, using fallback analysis...")
            insights = self._fallback_insights(df, data_profile)

        domain = insights.get('domain', 'General')
        summary = insights.get('summary', '')
        insight_list = insights.get('insights', [])

        self.log(f"  Domain detected: {domain}")
        self.log(f"  Insights discovered: {len(insight_list)}")

        # Step 3: Generate charts for each insight
        self.log("Step 3: Generating insight charts...")
        charts = {}
        narratives = {}

        # Filter insights through significance gate
        significant_insights = []
        for insight in insight_list:
            is_sig, reason = self._is_insight_significant(df, insight)
            if is_sig:
                significant_insights.append(insight)
            else:
                self.log(f"  Filtered out: '{insight.get('title', '?')}' — {reason}")

        if len(significant_insights) < len(insight_list):
            self.log(f"  Kept {len(significant_insights)}/{len(insight_list)} insights after quality filtering")
        insight_list = significant_insights

        for i, insight in enumerate(insight_list):
            chart_name = f"insight_{i+1}"
            self.log(f"  Creating chart {i+1}: {insight.get('title', 'Untitled')}")

            try:
                fig = self._create_insight_chart(df, insight)
                if fig:
                    charts[chart_name] = fig
                    self._save_figure(fig, analyzer_dir, chart_name)

                    # Generate narrative for this chart
                    narrative = self._generate_narrative(df, insight)
                    narratives[chart_name] = narrative
            except Exception as e:
                self.log(f"  Warning: Chart {i+1} failed: {e}")

        # Step 4: Create auto-generated overview charts (always useful)
        self.log("Step 4: Creating overview charts...")
        try:
            overview_charts = self._create_overview_charts(df, data_profile, analyzer_dir)
            charts.update(overview_charts)
        except Exception as e:
            self.log(f"  Overview charts warning (non-fatal): {e}")

        # Step 5: Generate KPIs and key takeaways
        self.log("Step 5: Generating KPIs and key takeaways...")
        kpis = self._generate_kpis(df, data_profile, domain)
        key_takeaways = self._generate_key_takeaways(df, data_profile, insight_list)

        # Assess data quality and add warnings to takeaways
        quality = self._assess_data_quality(df, data_profile)
        if quality.get('warnings'):
            for w in quality['warnings']:
                key_takeaways.append(f'**Data Quality Warning:** {w}')

        # Step 6: Build the combined dashboard
        self.log("Step 6: Building dashboard...")
        dashboard = self._build_dashboard(
            df, domain, summary, insight_list, charts, narratives, analyzer_dir,
            kpis=kpis, key_takeaways=key_takeaways, data_quality=quality
        )

        # Update state
        state['data_analysis'] = {
            'domain': domain,
            'summary': summary,
            'insights': insight_list,
            'charts': charts,
            'narratives': narratives,
            'key_takeaways': key_takeaways,
            'kpis': kpis,
            'data_quality': quality,
            'dashboard': dashboard,
            'profile': data_profile
        }
        state['analyzer_dir'] = analyzer_dir

        self.log(f"Data analysis complete! {len(charts)} charts generated in {analyzer_dir}")
        return state

    def execute_context_phase(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 1 — runs FIRST in the pipeline, before Profiler/Cleaner.

        Sends column names + sample rows to the LLM so it can understand the
        dataset like a human would. Stores the result in state['data_context']
        so every downstream agent can use it.

        Also sets state['target_column'] automatically if the user didn't
        specify one and the LLM finds a confident suggestion.
        """
        self.log("Analyzing data context — domain, target, and cleaning hints...")

        df = state.get('raw_data')
        if df is None:
            self.log("No raw_data in state, skipping context phase.")
            return state

        dataset_name = state.get('dataset_name', 'Dataset')
        profile = self._quick_profile(df)

        # Build column info string
        column_info_lines = []
        for col, info in profile['columns'].items():
            line = f"  - {col} ({info['dtype']}): {info['n_unique']} unique, missing={info['missing_pct']}%"
            if info.get('mean') is not None:
                line += f", range=[{info['min']}, {info['max']}], mean={info['mean']}"
            else:
                samples = ', '.join(info['sample_values'][:3])
                line += f", examples: [{samples}]"
            column_info_lines.append(line)
        column_info = '\n'.join(column_info_lines)

        sample_data = df.head(5).to_string(max_cols=15)
        statistics = df.describe(include='all').to_string()

        prompt = CONTEXT_ANALYSIS_PROMPT.format(
            dataset_name=dataset_name,
            n_rows=len(df),
            n_cols=len(df.columns),
            column_info=column_info,
            sample_data=sample_data,
            statistics=statistics
        )

        context = {}
        try:
            response = self.ask_llm(prompt)
            context = self._parse_json_response(response) or {}
        except Exception as e:
            self.log(f"  Context analysis LLM call failed: {e}")

        if context:
            self.log(f"  Domain: {context.get('domain', 'unknown')}")
            self.log(f"  Suggested target: {context.get('suggested_target', 'unknown')}")
            self.log(f"  Task type: {context.get('task_type', 'unknown')}")

            # Set target_column from LLM if user didn't specify one
            if not state.get('target_column'):
                suggested = context.get('suggested_target')
                if suggested and suggested in df.columns:
                    state['target_column'] = suggested
                    self.log(f"  Set target_column = '{suggested}' (LLM-detected)")
        else:
            self.log("  Context analysis returned no results, pipeline will use defaults.")

        state['data_context'] = context
        state['stage'] = 'context_analyzed'
        return state

    def analyze_with_prompt(self, df: pd.DataFrame, user_prompt: str,
                            output_dir: str = './output/data_analysis') -> Dict:
        """
        Generate a specific visualization based on a user's natural language prompt.
        
        Examples:
          - "Show me sales trends by month"
          - "What's the distribution of customer ages?"
          - "Compare revenue across regions"
          - "Top 10 products by quantity sold"
        
        Args:
            df: The dataset
            user_prompt: Natural language question/request
            output_dir: Where to save the chart
        
        Returns:
            Dict with 'chart' (Plotly Figure), 'title', 'description', 'narrative'
        """
        self.log(f"Processing user prompt: '{user_prompt}'")
        os.makedirs(output_dir, exist_ok=True)

        # Get column info for the LLM
        columns = df.columns.tolist()
        column_types = {col: str(df[col].dtype) for col in columns}
        sample_values = {}
        for col in columns[:20]:  # Limit columns sent to LLM
            uniq = df[col].dropna().unique()
            if len(uniq) <= 10:
                sample_values[col] = [str(v) for v in uniq[:10]]
            else:
                sample_values[col] = [str(v) for v in uniq[:5]] + ['...']

        # Ask LLM
        prompt = USER_PROMPT_TEMPLATE.format(
            columns=columns,
            column_types=json.dumps(column_types, indent=2),
            sample_values=json.dumps(sample_values, indent=2),
            user_prompt=user_prompt
        )

        try:
            response = self.ask_llm(prompt)
            chart_spec = self._parse_json_response(response)
        except Exception as e:
            self.log(f"  LLM prompt parsing failed: {e}")
            chart_spec = None

        if not chart_spec:
            return {'error': 'Could not understand the request. Try rephrasing.'}

        # Generate the chart
        try:
            fig = self._create_insight_chart(df, chart_spec)
            if fig:
                self._save_figure(fig, output_dir, 'user_prompt_chart')

                # Generate narrative
                narrative = self._generate_narrative(df, chart_spec)

                return {
                    'chart': fig,
                    'title': chart_spec.get('title', user_prompt),
                    'description': chart_spec.get('description', ''),
                    'narrative': narrative,
                    'spec': chart_spec
                }
        except Exception as e:
            return {'error': f'Chart generation failed: {e}'}

        return {'error': 'Could not generate chart for this request.'}

    # =====================================================================
    # STEP 1: QUICK DATA PROFILING (for LLM context)
    # =====================================================================

    def _quick_profile(self, df: pd.DataFrame) -> Dict:
        """Generate a quick profile for the LLM to understand the data."""
        profile = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': {},
            'numeric_cols': [],
            'categorical_cols': [],
            'datetime_cols': [],
            'missing_total': int(df.isnull().sum().sum())
        }

        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'n_unique': int(df[col].nunique()),
                'missing_pct': round(df[col].isnull().mean() * 100, 1),
                'sample_values': [str(v) for v in df[col].dropna().unique()[:5]]
            }

            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                col_info['mean'] = round(float(df[col].mean()), 2) if not df[col].isnull().all() else None
                col_info['median'] = round(float(df[col].median()), 2) if not df[col].isnull().all() else None
                col_info['std'] = round(float(df[col].std()), 2) if not df[col].isnull().all() else None
                col_info['min'] = round(float(df[col].min()), 2) if not df[col].isnull().all() else None
                col_info['max'] = round(float(df[col].max()), 2) if not df[col].isnull().all() else None
                profile['numeric_cols'].append(col)
            elif df[col].dtype == 'datetime64[ns]':
                profile['datetime_cols'].append(col)
            else:
                profile['categorical_cols'].append(col)

            # Detect potential datetime in string columns
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head(20))
                    profile['datetime_cols'].append(col)
                except:
                    pass

            profile['columns'][col] = col_info

        return profile

    # =====================================================================
    # STATISTICAL HIGHLIGHTS (grounding data for LLM + insight filtering)
    # =====================================================================

    def _compute_stat_highlights(self, df: pd.DataFrame, profile: Dict) -> str:
        """
        Pre-compute statistical facts so the LLM has GROUNDED data to work with.
        This prevents the LLM from guessing and ensures accuracy.
        """
        lines = []
        numeric_cols = profile.get('numeric_cols', [])
        cat_cols = profile.get('categorical_cols', [])

        # Column importance ranking
        col_scores = self._score_columns(df, profile)
        if col_scores:
            top_cols = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("MOST IMPORTANT COLUMNS (by variance + business relevance):")
            for col, score in top_cols:
                lines.append(f"  - {col}: importance={score:.2f}")

        # Strongest correlations (only meaningful ones)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            if not upper.stack().empty:
                strong_corrs = upper.stack().abs()
                strong_corrs = strong_corrs[strong_corrs > 0.15].sort_values(ascending=False).head(5)
                if len(strong_corrs) > 0:
                    lines.append("\nMEANINGFUL CORRELATIONS (|r| > 0.15):")
                    for (c1, c2), val in strong_corrs.items():
                        actual = corr.loc[c1, c2]
                        lines.append(f"  - {c1} vs {c2}: r={actual:.3f}")
                else:
                    lines.append("\nNO MEANINGFUL CORRELATIONS found (all |r| < 0.15)")

        # Categorical uniformity check
        for col in cat_cols[:5]:
            vc = df[col].value_counts(normalize=True)
            if len(vc) > 1:
                max_pct = vc.iloc[0] * 100
                min_pct = vc.iloc[-1] * 100
                spread = max_pct - min_pct
                if spread < 5:
                    lines.append(f"\nWARNING: '{col}' is nearly UNIFORM — all categories within {spread:.1f}% of each other. Skip insights about this column's distribution.")
                elif max_pct > 40:
                    lines.append(f"\n'{col}': '{vc.index[0]}' dominates at {max_pct:.0f}%")

        # Data quality flags
        quality = self._assess_data_quality(df, profile)
        if quality.get('warnings'):
            lines.append("\nDATA QUALITY WARNINGS:")
            for w in quality['warnings']:
                lines.append(f"  - {w}")

        return '\n'.join(lines) if lines else 'No special highlights.'

    def _score_columns(self, df: pd.DataFrame, profile: Dict) -> Dict[str, float]:
        """
        Score each column by how useful it is for analysis.
        Higher score = more interesting / business-relevant.

        Factors: variance, range, business keyword match, not-ID-like, cardinality.
        """
        scores = {}
        numeric_cols = profile.get('numeric_cols', [])
        cat_cols = profile.get('categorical_cols', [])

        # Business-relevant keywords (boost score)
        high_value_keywords = [
            'revenue', 'sales', 'price', 'amount', 'total', 'cost', 'profit',
            'income', 'salary', 'quantity', 'order', 'payment', 'transaction',
            'rating', 'score', 'count', 'likes', 'views', 'clicks',
            'matches', 'followers', 'subscribers', 'downloads', 'visits',
            'duration', 'time_spent', 'age', 'weight', 'height', 'distance',
        ]
        medium_keywords = [
            'category', 'type', 'region', 'department', 'status', 'gender',
            'segment', 'channel', 'product', 'brand', 'class', 'group',
            'city', 'state', 'country', 'store', 'source', 'platform',
            'level', 'tier', 'plan', 'role', 'industry',
        ]
        id_keywords = ['_id', 'id_', 'uuid', 'guid', 'index', 'key', 'pk']

        for col in numeric_cols:
            name_lower = col.lower()

            # Skip ID-like columns
            if any(kw in name_lower for kw in id_keywords):
                scores[col] = 0.0
                continue
            if name_lower in ('id', 'pk', 'rownum', 'row_num', 'record'):
                scores[col] = 0.0
                continue
            # Skip near-unique columns (likely IDs)
            if df[col].nunique() > 0.9 * len(df) and len(df) > 50:
                scores[col] = 0.0
                continue

            score = 0.0
            std = df[col].std()
            mean = abs(df[col].mean()) + 1e-9
            cv = std / mean  # coefficient of variation

            # Variance score (0-3): higher CV = more interesting
            if cv > 1.0:
                score += 3.0
            elif cv > 0.5:
                score += 2.0
            elif cv > 0.1:
                score += 1.0
            else:
                score += 0.2  # near-constant — boring

            # Range score: wider range = more interesting
            val_range = df[col].max() - df[col].min()
            if val_range > 100:
                score += 1.5
            elif val_range > 10:
                score += 1.0
            elif val_range > 1:
                score += 0.5
            else:
                score += 0.1  # tiny range like 0-6

            # Business keyword boost
            if any(kw in name_lower for kw in high_value_keywords):
                score += 3.0
            elif any(kw in name_lower for kw in medium_keywords):
                score += 1.5

            # Penalize very low cardinality numerics (e.g., 0-6 integers)
            if df[col].nunique() <= 7:
                score *= 0.5

            scores[col] = round(score, 2)

        for col in cat_cols:
            name_lower = col.lower()
            if any(kw in name_lower for kw in id_keywords):
                scores[col] = 0.0
                continue
            if df[col].nunique() > 0.9 * len(df):
                scores[col] = 0.0
                continue

            score = 0.0
            n_unique = df[col].nunique()

            # Best grouping columns have 3-15 categories
            if 3 <= n_unique <= 15:
                score += 2.5
            elif 2 <= n_unique <= 30:
                score += 1.5
            elif n_unique <= 50:
                score += 0.5
            else:
                score += 0.1

            # Business keyword boost
            if any(kw in name_lower for kw in medium_keywords):
                score += 2.0

            # Penalize uniform distributions
            vc = df[col].value_counts(normalize=True)
            if len(vc) > 1:
                spread = (vc.iloc[0] - vc.iloc[-1]) * 100
                if spread < 5:
                    score *= 0.3  # uniform — not interesting

            scores[col] = round(score, 2)

        return scores

    def _assess_data_quality(self, df: pd.DataFrame, profile: Dict) -> Dict:
        """
        Assess data quality and detect synthetic/uniform datasets.
        Returns quality score and warnings.
        """
        warnings = []
        quality_score = 100
        n_rows = len(df)
        numeric_cols = profile.get('numeric_cols', [])
        cat_cols = profile.get('categorical_cols', [])

        # Check for uniform categorical distributions (synthetic data signal)
        uniform_cats = 0
        for col in cat_cols:
            vc = df[col].value_counts(normalize=True)
            if len(vc) > 1:
                spread = (vc.iloc[0] - vc.iloc[-1]) * 100
                if spread < 3:
                    uniform_cats += 1

        if uniform_cats > 0 and len(cat_cols) > 0:
            pct_uniform = uniform_cats / len(cat_cols) * 100
            if pct_uniform > 50:
                warnings.append(
                    f"{uniform_cats}/{len(cat_cols)} categorical columns have nearly "
                    f"uniform distributions. This data may be synthetic or randomly generated. "
                    f"Ranking/composition insights will not be meaningful."
                )
                quality_score -= 20

        # Check for low correlation across all numeric pairs
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            if not upper.stack().empty:
                max_corr = upper.stack().abs().max()
                if max_corr < 0.15:
                    warnings.append(
                        f"No meaningful correlations found between numeric columns "
                        f"(max |r| = {max_corr:.3f}). Scatter plots will show noise, not patterns."
                    )

        # Missing data
        missing_pct = df.isnull().sum().sum() / (n_rows * len(df.columns)) * 100
        if missing_pct > 20:
            warnings.append(f"{missing_pct:.0f}% of data is missing — insights may be unreliable.")
            quality_score -= 15

        # Very small dataset
        if n_rows < 50:
            warnings.append("Very small dataset (<50 rows). Statistical patterns may not be reliable.")
            quality_score -= 10

        return {
            'quality_score': max(0, quality_score),
            'warnings': warnings,
            'is_likely_synthetic': uniform_cats > len(cat_cols) * 0.5 if cat_cols else False,
        }

    def _generate_kpis(self, df: pd.DataFrame, profile: Dict,
                       domain: str = 'General') -> List[Dict]:
        """
        Auto-detect KPIs from the dataset based on domain + column semantics.
        Returns a list of KPI dicts with {name, value, format, trend, description}.
        """
        kpis = []
        col_scores = self._score_columns(df, profile)
        numeric_cols = [c for c in profile.get('numeric_cols', []) if col_scores.get(c, 0) > 0]
        cat_cols = [c for c in profile.get('categorical_cols', []) if col_scores.get(c, 0) > 0]

        # Sort by importance
        numeric_sorted = sorted(numeric_cols, key=lambda c: col_scores.get(c, 0), reverse=True)

        # KPI patterns: detect from column names
        kpi_patterns = {
            'total': {
                'keywords': ['revenue', 'sales', 'amount', 'total', 'income', 'profit',
                             'cost', 'payment', 'transaction', 'order'],
                'agg': 'sum', 'format': ',.0f', 'prefix': 'Total'
            },
            'average': {
                'keywords': ['rating', 'score', 'price', 'age', 'duration', 'time_spent',
                             'salary', 'likes', 'views', 'matches', 'followers'],
                'agg': 'mean', 'format': ',.1f', 'prefix': 'Average'
            },
            'count': {
                'keywords': ['count', 'quantity', 'visits', 'clicks', 'downloads',
                             'subscribers'],
                'agg': 'sum', 'format': ',.0f', 'prefix': 'Total'
            },
        }

        used_cols = set()
        for pattern_name, pattern in kpi_patterns.items():
            for col in numeric_sorted:
                if col in used_cols:
                    continue
                name_lower = col.lower()
                if any(kw in name_lower for kw in pattern['keywords']):
                    if pattern['agg'] == 'sum':
                        value = df[col].sum()
                    elif pattern['agg'] == 'mean':
                        value = df[col].mean()
                    else:
                        value = df[col].sum()

                    kpis.append({
                        'name': f"{pattern['prefix']} {col.replace('_', ' ').title()}",
                        'value': value,
                        'format': pattern['format'],
                        'column': col,
                        'description': f"{pattern['prefix']} across {len(df):,} records"
                    })
                    used_cols.add(col)
                    if len(kpis) >= 6:
                        break
            if len(kpis) >= 6:
                break

        # If we didn't find enough named KPIs, add top-scored numeric columns
        for col in numeric_sorted:
            if len(kpis) >= 4:
                break
            if col in used_cols:
                continue
            # For remaining columns, use the most informative aggregation
            mean_val = df[col].mean()
            total_val = df[col].sum()
            # If mean is small and total is large, show total; otherwise show average
            if total_val > mean_val * 100:
                kpis.append({
                    'name': f"Total {col.replace('_', ' ').title()}",
                    'value': total_val,
                    'format': ',.0f',
                    'column': col,
                    'description': f"Sum across all records"
                })
            else:
                kpis.append({
                    'name': f"Avg {col.replace('_', ' ').title()}",
                    'value': mean_val,
                    'format': ',.1f',
                    'column': col,
                    'description': f"Average value"
                })
            used_cols.add(col)

        # Add a record count KPI
        kpis.insert(0, {
            'name': 'Total Records',
            'value': len(df),
            'format': ',',
            'column': None,
            'description': f'{len(df.columns)} columns analyzed'
        })

        # Add data quality KPI
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        completeness = 100 - missing_pct
        kpis.append({
            'name': 'Data Completeness',
            'value': completeness,
            'format': '.1f',
            'column': None,
            'description': f'{missing_pct:.1f}% missing values'
        })

        return kpis[:6]  # Max 6 KPIs

    def _is_insight_significant(self, df: pd.DataFrame, insight: Dict) -> Tuple[bool, str]:
        """
        Check if an insight is actually meaningful or just noise.
        Returns (is_significant, reason_if_not).

        This is the KEY QUALITY GATE — prevents showing garbage insights.
        """
        itype = insight.get('insight_type', '')
        x_col = insight.get('x_column')
        y_col = insight.get('y_column')
        chart_type = insight.get('chart_type', 'bar')
        agg = insight.get('aggregation', 'none')

        # ── Correlation: skip if |r| < 0.15 ──
        if itype == 'correlation' and chart_type == 'scatter':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        r = df[x_col].corr(df[y_col])
                        if abs(r) < 0.15:
                            return False, f"Correlation too weak (r={r:.3f}), skipping"
                except Exception:
                    pass

        # ── Ranking/Composition: skip if distribution is uniform ──
        if itype in ('ranking', 'composition', 'comparison') and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                try:
                    agg_func = {'sum': 'sum', 'mean': 'mean', 'count': 'count',
                                'median': 'median'}.get(agg, 'sum')
                    grouped = df.groupby(x_col)[y_col].agg(agg_func)
                    if len(grouped) > 1:
                        max_val = grouped.max()
                        min_val = grouped.min()
                        mean_val = grouped.mean()
                        # Check if spread is less than 5% of mean
                        if mean_val > 0 and (max_val - min_val) / mean_val < 0.05:
                            return False, f"Values are nearly uniform (spread < 5% of mean), skipping"
                        # Check if top category is less than 3% ahead
                        total = grouped.sum()
                        if total > 0:
                            top_pct = grouped.max() / total * 100
                            bottom_pct = grouped.min() / total * 100
                            if top_pct - bottom_pct < 3:
                                return False, f"Categories differ by less than 3%, skipping"
                except Exception:
                    pass

        # ── Distribution: skip if near-zero variance ──
        if itype == 'distribution':
            col = x_col or y_col
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                std = df[col].std()
                mean = abs(df[col].mean()) + 1e-9
                if std / mean < 0.01:
                    return False, f"Near-zero variance (CV={std/mean:.4f}), skipping"
                # Skip if only 2-3 unique values — not enough for a histogram
                if df[col].nunique() <= 2:
                    return False, f"Only {df[col].nunique()} unique values, too few for distribution"

        # ── Skip if columns don't exist ──
        for col_key in ['x_column', 'y_column']:
            col = insight.get(col_key)
            if col and col not in df.columns:
                return False, f"Column '{col}' does not exist"

        return True, "OK"

    # =====================================================================
    # STEP 2: LLM INSIGHT DISCOVERY
    # =====================================================================

    def _discover_insights(self, df: pd.DataFrame, dataset_name: str,
                           profile: Dict) -> Optional[Dict]:
        """Ask the LLM what insights to extract from this dataset."""

        # Build column info string
        column_info_lines = []
        for col, info in profile['columns'].items():
            line = f"  - {col} ({info['dtype']}): {info['n_unique']} unique values"
            if info.get('mean') is not None:
                line += f", mean={info['mean']}, range=[{info['min']}, {info['max']}]"
            else:
                samples = ', '.join(info['sample_values'][:3])
                line += f", examples: [{samples}]"
            column_info_lines.append(line)

        column_info = '\n'.join(column_info_lines)

        # Sample data
        sample_data = df.head(5).to_string(max_cols=15)

        # Statistics
        stats_str = df.describe(include='all').to_string()

        # Pre-compute statistical highlights so LLM doesn't guess
        stat_highlights = self._compute_stat_highlights(df, profile)

        prompt = INSIGHT_DISCOVERY_PROMPT.format(
            dataset_name=dataset_name,
            n_rows=profile['n_rows'],
            n_cols=profile['n_cols'],
            column_info=column_info,
            sample_data=sample_data,
            statistics=stats_str,
            stat_highlights=stat_highlights
        )

        try:
            response = self.ask_llm(prompt)
            result = self._parse_json_response(response)

            if result and 'insights' in result:
                # Validate columns exist
                valid_insights = []
                for insight in result['insights']:
                    valid = True
                    for key in ['x_column', 'y_column', 'color_column']:
                        col = insight.get(key)
                        if col and col not in df.columns and col != 'null' and col is not None:
                            # Try case-insensitive match
                            matched = [c for c in df.columns if c.lower() == col.lower()]
                            if matched:
                                insight[key] = matched[0]
                            else:
                                insight[key] = None
                    valid_insights.append(insight)
                result['insights'] = valid_insights
                return result
        except Exception as e:
            self.log(f"  LLM insight discovery failed: {e}")

        return None

    # =====================================================================
    # STEP 3: CHART GENERATION ENGINE
    # =====================================================================

    def _create_insight_chart(self, df: pd.DataFrame, spec: Dict) -> Optional[go.Figure]:
        """
        Create a Plotly chart from an LLM-generated insight specification.
        
        This is the core chart engine that translates LLM instructions into
        actual Plotly visualizations. Handles all chart types and aggregations.
        """
        chart_type = spec.get('chart_type', 'bar')
        title = spec.get('title', 'Insight')
        x_col = spec.get('x_column')
        y_col = spec.get('y_column')
        color_col = spec.get('color_column')
        aggregation = spec.get('aggregation', 'none')
        top_n = spec.get('top_n')
        sort_order = spec.get('sort', 'none')

        # Sanitize None-like strings from LLM
        for var_name in ['x_col', 'y_col', 'color_col']:
            val = locals()[var_name]
            if val in ('null', 'None', 'none', ''):
                locals()[var_name] = None

        # Handle NoneType values properly  
        x_col = None if x_col in ('null', 'None', 'none', '') else x_col
        y_col = None if y_col in ('null', 'None', 'none', '') else y_col
        color_col = None if color_col in ('null', 'None', 'none', '') else color_col

        # Apply filter if specified
        plot_df = df.copy()
        filter_col = spec.get('filter_column')
        filter_val = spec.get('filter_value')
        if filter_col and filter_val and filter_col in plot_df.columns:
            plot_df = plot_df[plot_df[filter_col].astype(str) == str(filter_val)]

        # Apply aggregation if needed
        if aggregation != 'none' and x_col and y_col:
            if x_col in plot_df.columns and y_col in plot_df.columns:
                group_cols = [x_col]
                if color_col and color_col in plot_df.columns:
                    group_cols.append(color_col)

                agg_func = {
                    'sum': 'sum', 'mean': 'mean', 'count': 'count',
                    'median': 'median', 'max': 'max', 'min': 'min'
                }.get(aggregation, 'mean')

                try:
                    plot_df = plot_df.groupby(group_cols, as_index=False).agg(
                        {y_col: agg_func}
                    )
                except:
                    pass  # If aggregation fails, use raw data

        # Apply top_n filter
        if top_n and x_col and y_col:
            try:
                top_n = int(top_n)
                if sort_order == 'ascending':
                    plot_df = plot_df.nsmallest(top_n, y_col)
                else:
                    plot_df = plot_df.nlargest(top_n, y_col)
            except:
                pass

        # Sort
        if sort_order in ('ascending', 'descending') and y_col and y_col in plot_df.columns:
            plot_df = plot_df.sort_values(y_col, ascending=(sort_order == 'ascending'))

        # ---- CHART TYPE DISPATCH ----
        try:
            fig = self._dispatch_chart(chart_type, plot_df, title,
                                       x_col, y_col, color_col, aggregation)
            return fig
        except Exception as e:
            self.log(f"  Chart creation failed for '{title}': {e}")
            return None

    def _dispatch_chart(self, chart_type: str, df: pd.DataFrame, title: str,
                        x_col: Optional[str], y_col: Optional[str],
                        color_col: Optional[str], agg: str) -> go.Figure:
        """
        Route to the correct chart builder based on type.
        Every chart type uses go.Figure directly for full rendering control.
        """
        color_seq = self.color_sequence
        self.log(f"  _dispatch_chart: type={chart_type}, rows={len(df)}, "
                 f"x={x_col}, y={y_col}, color={color_col}, agg={agg}")

        # ── SAFETY: re-aggregate bar data if it looks unaggregated ──
        def _ensure_aggregated(data, x, y, agg_method):
            """If data has many rows per category, aggregate it."""
            if x and y and x in data.columns and y in data.columns:
                if pd.api.types.is_numeric_dtype(data[y]):
                    n_rows = len(data)
                    n_cats = data[x].nunique()
                    if n_rows > n_cats * 1.5:  # looks unaggregated
                        func = agg_method if agg_method in ('sum','mean','median','max','min','count') else 'sum'
                        data = data.groupby(x, as_index=False)[y].agg(func)
                        self.log(f"    Re-aggregated: {n_rows} rows → {len(data)} rows ({func})")
            return data

        # ── PIE → HORIZONTAL BAR ──
        if chart_type == 'pie':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                bar_df = _ensure_aggregated(df, x_col, y_col, 'sum')
                bar_df = bar_df.sort_values(y_col, ascending=True)
                vals = bar_df[y_col].values
                cats = bar_df[x_col].astype(str).values
                total = vals.sum()
                fig = go.Figure(go.Bar(
                    x=vals, y=cats, orientation='h',
                    marker_color=[color_seq[i % len(color_seq)] for i in range(len(vals))],
                    text=[f'{v:,.0f}' for v in vals],
                    textposition='outside',
                    textfont=dict(size=12, color='#374151'),
                    width=0.5 if len(vals) <= 5 else 0.7,
                ))
                if total > 0 and len(vals) > 0:
                    top_pct = vals[-1] / total * 100
                    fig.add_annotation(
                        x=vals[-1], y=cats[-1],
                        text=f"  ← {top_pct:.0f}% of total",
                        showarrow=False, font=dict(size=11, color=color_seq[4]),
                        xanchor='left',
                    )
                xmax = vals.max()
                fig.update_layout(xaxis=dict(range=[0, xmax * 1.2]))
            else:
                fig = go.Figure()

        # ── BAR CHART ──
        elif chart_type == 'bar':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                if color_col and color_col in df.columns:
                    # Grouped bars: aggregate per (x, color) group
                    func = agg if agg in ('sum','mean','median','max','min','count') else 'mean'
                    grp = df.groupby([x_col, color_col], as_index=False)[y_col].agg(func)
                    fig = px.bar(grp, x=x_col, y=y_col, color=color_col,
                                 color_discrete_sequence=color_seq, barmode='group')
                else:
                    # Simple bars — always aggregate first
                    bar_df = _ensure_aggregated(df, x_col, y_col, agg)
                    bar_df = bar_df.sort_values(y_col, ascending=False)
                    vals = bar_df[y_col].values.astype(float)
                    cats = bar_df[x_col].astype(str).values
                    self.log(f"    Bar values: {list(zip(cats, vals))}")
                    # Format numbers
                    if vals.max() >= 1000:
                        texts = [f'{v:,.0f}' for v in vals]
                    elif vals.max() >= 10:
                        texts = [f'{v:.1f}' for v in vals]
                    else:
                        texts = [f'{v:.2f}' for v in vals]
                    fig = go.Figure(go.Bar(
                        x=cats, y=vals,
                        marker_color=[color_seq[i % len(color_seq)] for i in range(len(vals))],
                        text=texts, textposition='outside',
                        textfont=dict(size=12, color='#374151'),
                    ))
                n_cats = df[x_col].nunique()
                bargap = 0.45 if n_cats <= 4 else 0.35 if n_cats <= 8 else 0.25
                fig.update_layout(bargap=bargap)
            else:
                fig = go.Figure()

        # ── LINE CHART ──
        elif chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, color=color_col,
                          title=title, color_discrete_sequence=color_seq, markers=True)
            if y_col and y_col in df.columns and len(df) >= 2:
                try:
                    first_val = df[y_col].iloc[0]
                    last_val = df[y_col].iloc[-1]
                    change = ((last_val - first_val) / (abs(first_val) + 1e-9)) * 100
                    arrow = "▲" if change > 0 else "▼"
                    clr = '#059669' if change > 0 else '#DC2626'
                    fig.add_annotation(
                        x=df[x_col].iloc[-1] if x_col and x_col in df.columns else len(df)-1,
                        y=last_val, text=f" {arrow} {abs(change):.0f}%",
                        showarrow=False, font=dict(size=12, color=clr, weight='bold'),
                        xanchor='left',
                    )
                except Exception:
                    pass

        # ── SCATTER PLOT ──
        elif chart_type == 'scatter':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                cols_needed = [c for c in [x_col, y_col, color_col] if c and c in df.columns]
                sdf = df[cols_needed].dropna().copy()
                if len(sdf) > 5000:
                    sdf = sdf.sample(5000, random_state=42)
                n = len(sdf)
                msz = 9 if n < 200 else 6 if n < 1000 else 4
                self.log(f"    Scatter: {n} points, x={x_col}, y={y_col}, color={color_col}")

                # Use px.scatter — guaranteed SVG rendering in all Plotly versions
                fig = px.scatter(
                    sdf, x=x_col, y=y_col,
                    color=color_col if color_col and color_col in sdf.columns else None,
                    color_discrete_sequence=color_seq,
                    opacity=0.8,
                )
                # Marker size
                fig.update_traces(marker=dict(size=msz), selector=dict(mode='markers'))

                # Add trendline manually (avoids statsmodels dependency)
                try:
                    xv = sdf[x_col].values.astype(float)
                    yv = sdf[y_col].values.astype(float)
                    valid = np.isfinite(xv) & np.isfinite(yv)
                    if valid.sum() > 2:
                        z = np.polyfit(xv[valid], yv[valid], 1)
                        x_line = np.linspace(xv[valid].min(), xv[valid].max(), 80)
                        y_line = np.polyval(z, x_line)
                        fig.add_trace(go.Scatter(
                            x=x_line, y=y_line, mode='lines',
                            line=dict(color='#DC2626', width=2, dash='dash'),
                            name='Trend', showlegend=False,
                        ))
                        # Correlation badge
                        r = np.corrcoef(xv[valid], yv[valid])[0, 1]
                        strength = 'Strong' if abs(r) > 0.6 else 'Moderate' if abs(r) > 0.3 else 'Weak'
                        fig.add_annotation(
                            xref='paper', yref='paper', x=0.98, y=0.02,
                            text=f"r = {r:.2f} ({strength})", showarrow=False,
                            font=dict(size=12, color='#6B7280'),
                            bgcolor='white', bordercolor='#E5E7EB', borderwidth=1,
                            borderpad=4, xanchor='right', yanchor='bottom',
                        )
                except Exception as e:
                    self.log(f"    Trendline error: {e}")
            else:
                fig = go.Figure()

        # ── HISTOGRAM ──
        elif chart_type == 'histogram':
            col = x_col or y_col
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                values = df[col].dropna()
                n_unique = values.nunique()
                val_range = values.max() - values.min()
                self.log(f"    Histogram: col={col}, n={len(values)}, range={val_range:.2f}, unique={n_unique}")

                # Smart bin count
                if val_range <= 5:
                    nbins = min(15, max(8, n_unique // 2))
                elif val_range <= 20:
                    nbins = 20
                else:
                    nbins = min(35, max(15, int(np.sqrt(len(values)))))

                hist_color = color_col if color_col and color_col in df.columns else None

                # Use px.histogram — reliable across Plotly versions
                fig = px.histogram(
                    df, x=col,
                    color=hist_color,
                    nbins=nbins,
                    opacity=0.85,
                    color_discrete_sequence=color_seq,
                    barmode='overlay' if hist_color else None,
                )
                fig.update_layout(bargap=0.05, yaxis_title='Count')

                # Mean line
                mean_val = values.mean()
                fig.add_vline(
                    x=mean_val, line_dash='dot', line_color='#DC2626', line_width=2,
                    annotation_text=f"Mean: {mean_val:.2f}",
                    annotation_font=dict(size=12, color='#DC2626'),
                    annotation_position='top right',
                )
            else:
                fig = go.Figure()

        # ── BOX PLOT ──
        elif chart_type == 'box':
            fig = px.box(df, x=x_col, y=y_col, color=color_col,
                         title=title, color_discrete_sequence=color_seq)

        # ── HEATMAP ──
        elif chart_type == 'heatmap':
            if x_col and y_col:
                pivot = df.pivot_table(index=y_col, columns=x_col, aggfunc='size', fill_value=0)
                fig = px.imshow(pivot, title=title, color_continuous_scale='Blues', text_auto=True)
            else:
                numeric_df = df.select_dtypes(include=[np.number])
                corr = numeric_df.corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale='RdBu_r', zmin=-1, zmax=1,
                    text=np.round(corr.values, 2), texttemplate='%{text:.2f}',
                    textfont=dict(size=11),
                ))

        # ── TREEMAP ──
        elif chart_type == 'treemap':
            if x_col and y_col:
                fig = px.treemap(df, path=[x_col], values=y_col,
                                 title=title, color_discrete_sequence=color_seq)
            else:
                fig = go.Figure()

        # ── FUNNEL ──
        elif chart_type == 'funnel':
            fig = px.funnel(df, x=y_col, y=x_col,
                            title=title, color_discrete_sequence=color_seq)

        # ── AREA ──
        elif chart_type == 'area':
            fig = px.area(df, x=x_col, y=y_col, color=color_col,
                          title=title, color_discrete_sequence=color_seq)

        # ── FALLBACK ──
        else:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                         title=title, color_discrete_sequence=color_seq)

        # ══════════════════════════════════════════════════════════════
        # CLEAN STYLING — applied to all charts
        # ══════════════════════════════════════════════════════════════
        fig.update_layout(
            template='plotly_dark',
            height=460,
            title=dict(
                text=title,
                font=dict(size=16, family='Segoe UI, sans-serif'),
                x=0.01, xanchor='left', y=0.97,
            ),
            font=dict(family='Segoe UI, sans-serif', size=12),
            margin=dict(t=55, b=45, l=55, r=30),
            legend=dict(
                orientation='h', yanchor='bottom', y=1.02,
                xanchor='left', x=0, font=dict(size=11),
                bgcolor='rgba(0,0,0,0)', borderwidth=0,
            ),
            hoverlabel=dict(
                bgcolor='white', font_size=12, bordercolor='#E5E7EB',
                font_family='Segoe UI, sans-serif'
            ),
        )
        fig.update_xaxes(
            showgrid=False, showline=False,
            title_font=dict(size=12, color='#9CA3AF'),
            tickfont=dict(size=11, color='#6B7280'),
        )
        fig.update_yaxes(
            showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=0.5,
            showline=False,
            title_font=dict(size=12),
            tickfont=dict(size=11),
        )

        return fig

    def _apply_clean_layout(self, fig: go.Figure, title: str) -> go.Figure:
        """Apply the standard decluttered layout to any figure."""
        fig.update_layout(
            template='plotly_dark',
            height=460,
            title=dict(
                text=title,
                font=dict(size=16, family='Segoe UI, sans-serif'),
                x=0.01, xanchor='left', y=0.97,
            ),
            font=dict(family='Segoe UI, sans-serif', size=12),
            margin=dict(t=55, b=40, l=50, r=25),
            legend=dict(
                orientation='h', yanchor='bottom', y=1.02,
                xanchor='left', x=0, font=dict(size=11),
                bgcolor='rgba(0,0,0,0)',
                borderwidth=0,
            ),
            hoverlabel=dict(
                font_size=12,
                font_family='Segoe UI, sans-serif'
            ),
        )
        fig.update_xaxes(
            showgrid=False, showline=False,
            title_font=dict(size=12),
            tickfont=dict(size=11),
        )
        fig.update_yaxes(
            showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=0.5,
            showline=False,
            title_font=dict(size=12),
            tickfont=dict(size=11),
        )
        return fig

    # =====================================================================
    # STEP 4: OVERVIEW CHARTS (always generated)
    # =====================================================================

    def _create_overview_charts(self, df: pd.DataFrame, profile: Dict,
                                output_dir: str) -> Dict:
        """
        Create standard overview charts — decluttered, with action titles.

        Principles applied:
        - NO PIE CHARTS (replaced with horizontal bars)
        - Action titles tell the finding
        - Minimal non-data ink
        - Annotations on key data points
        """
        charts = {}
        _base = dict(
            font=dict(family='Segoe UI, sans-serif', size=12, color='#6B7280'),
            paper_bgcolor='white', plot_bgcolor='white',
            hoverlabel=dict(bgcolor='white', font_size=12, bordercolor='#E5E7EB'),
        )

        # --- Overview 1: Data Completeness ---
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
        complete_pct = 100 - missing_pct
        avg_complete = complete_pct.mean()
        worst_col = complete_pct.idxmin()
        worst_pct = complete_pct[worst_col]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=complete_pct.index, x=complete_pct.values,
            orientation='h', name='Complete',
            marker_color='#A8D5BA',
        ))
        fig.add_trace(go.Bar(
            y=missing_pct.index, x=missing_pct.values,
            orientation='h', name='Missing',
            marker_color='#E76F7A',
        ))
        # Action title
        if worst_pct < 80:
            title_text = f'"{worst_col}" has {100-worst_pct:.0f}% missing data — needs attention'
        elif avg_complete > 95:
            title_text = f'Data quality is strong — {avg_complete:.0f}% average completeness'
        else:
            title_text = f'Average completeness: {avg_complete:.0f}% across all columns'

        fig.update_layout(
            title=dict(text=title_text, font=dict(size=15, color='#1F2937')),
            barmode='stack',
            height=max(350, len(df.columns) * 28),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                        bgcolor='rgba(0,0,0,0)', borderwidth=0),
            margin=dict(t=55, b=35, l=120, r=25),
            xaxis=dict(showgrid=False, showline=False, title=''),
            yaxis=dict(showgrid=False, showline=False),
            **_base
        )
        charts['overview_completeness'] = fig
        self._save_figure(fig, output_dir, 'overview_completeness')

        # --- Overview 2: Numeric Summary (Box plots) ---
        numeric_cols = profile['numeric_cols']
        if numeric_cols:
            n_show = min(10, len(numeric_cols))
            fig = go.Figure()
            for i, col in enumerate(numeric_cols[:n_show]):
                fig.add_trace(go.Box(
                    y=df[col].dropna(), name=col,
                    marker_color=self.color_sequence[i % len(self.color_sequence)],
                    line_width=1.5,
                ))
            # Action title based on most variable column
            try:
                most_var = max(numeric_cols[:n_show],
                               key=lambda c: df[c].std() / (abs(df[c].mean()) + 1e-9))
                fig_title = f'{most_var} has the widest spread among numeric features'
            except Exception:
                fig_title = 'Numeric column distributions'

            fig.update_layout(
                title=dict(text=fig_title, font=dict(size=15)),
                template='plotly_dark', height=480, showlegend=False,
                margin=dict(t=55, b=40, l=50, r=25),
            )
            charts['overview_numeric'] = fig
            self._save_figure(fig, output_dir, 'overview_numeric')

        # --- Overview 3: Categorical Value Counts (HORIZONTAL BARS — not pies) ---
        cat_cols = profile['categorical_cols']
        if cat_cols:
            n_show = min(4, len(cat_cols))
            fig = make_subplots(
                rows=n_show, cols=1,
                subplot_titles=[f'{col} — top categories' for col in cat_cols[:n_show]],
                vertical_spacing=0.12,
            )
            for i, col in enumerate(cat_cols[:n_show]):
                vc = df[col].value_counts().head(8).sort_values()
                fig.add_trace(
                    go.Bar(
                        x=vc.values,
                        y=vc.index.astype(str),
                        orientation='h',
                        marker_color=self.color_sequence[i % len(self.color_sequence)],
                        text=[f'{v:,}' for v in vc.values],
                        textposition='outside',
                        textfont=dict(size=10),
                        showlegend=False,
                    ),
                    row=i+1, col=1
                )
            fig.update_layout(
                title=dict(text='Categorical columns — most frequent values',
                           font=dict(size=15)),
                height=200 * n_show + 80,
                template='plotly_dark',
                margin=dict(t=55, b=30, l=120, r=40),
            )
            fig.update_xaxes(showgrid=False, showline=False)
            fig.update_yaxes(showgrid=False, showline=False)
            charts['overview_categorical'] = fig
            self._save_figure(fig, output_dir, 'overview_categorical')

        # --- Overview 4: Correlation Matrix ---
        if len(numeric_cols) > 1:
            n_corr = min(15, len(numeric_cols))
            corr = df[numeric_cols[:n_corr]].corr()

            # Find strongest pair for title
            try:
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                if not upper.stack().empty:
                    max_idx = upper.stack().abs().idxmax()
                    max_val = corr.loc[max_idx[0], max_idx[1]]
                    corr_title = f'Strongest link: {max_idx[0]} ↔ {max_idx[1]} (r={max_val:.2f})'
                else:
                    corr_title = 'Feature correlations'
            except Exception:
                corr_title = 'Feature correlations'

            z = corr.values
            labels = corr.columns.tolist()
            fig = go.Figure(go.Heatmap(
                z=z, x=labels, y=labels,
                colorscale='RdBu_r', zmin=-1, zmax=1,
                text=[[f'{v:.2f}' for v in row] for row in z],
                texttemplate='%{text}',
                textfont=dict(size=10),
                colorbar=dict(thickness=15, len=0.9),
            ))
            fig.update_layout(
                title=dict(text=corr_title, font=dict(size=15, color='#1F2937')),
                template='plotly_dark', height=500,
                margin=dict(t=55, b=60, l=80, r=25),
                xaxis=dict(tickangle=-45),
                **_base
            )
            charts['overview_correlation'] = fig
            self._save_figure(fig, output_dir, 'overview_correlation')

        return charts

    # =====================================================================
    # NARRATIVE GENERATION
    # =====================================================================

    def _generate_narrative(self, df: pd.DataFrame, insight: Dict) -> str:
        """Generate a plain-English story for an insight (LLM or statistical fallback)."""
        title = insight.get('title', 'Chart')
        x_col = insight.get('x_column')
        y_col = insight.get('y_column')
        color_col = insight.get('color_column')
        chart_type = insight.get('chart_type', 'bar')
        agg = insight.get('aggregation', 'none')

        # Build a quick data summary for the LLM
        data_summary_parts = [f"Chart type: {chart_type}"]

        if x_col and x_col in df.columns:
            if df[x_col].dtype in ['int64', 'float64']:
                data_summary_parts.append(
                    f"X-axis ({x_col}): range {df[x_col].min():.2f} to {df[x_col].max():.2f}"
                )
            else:
                top_vals = df[x_col].value_counts().head(5)
                data_summary_parts.append(
                    f"X-axis ({x_col}): top values = {dict(top_vals)}"
                )

        if y_col and y_col in df.columns:
            if df[y_col].dtype in ['int64', 'float64']:
                data_summary_parts.append(
                    f"Y-axis ({y_col}): mean={df[y_col].mean():.2f}, "
                    f"min={df[y_col].min():.2f}, max={df[y_col].max():.2f}"
                )

        data_summary = '\n'.join(data_summary_parts)

        prompt = INSIGHT_NARRATIVE_PROMPT.format(
            chart_title=title,
            data_summary=data_summary
        )

        try:
            narrative = self.ask_llm(prompt)
            if narrative and len(narrative.strip()) > 20:
                return narrative.strip()
        except:
            pass

        # ── Statistical fallback narrative (no LLM needed) ──
        return self._statistical_narrative(df, insight)

    def _statistical_narrative(self, df: pd.DataFrame, insight: Dict) -> str:
        """
        Generate a data-driven narrative using STORY STRUCTURE:
            BEGINNING → setup / context (what are we looking at?)
            MIDDLE    → conflict / evidence (what does the data show?)
            END       → resolution / call to action (so what? what should you do?)

        Principles: Never assume the reader will interpret data themselves.
        State the conclusion EXPLICITLY. Include the "so what?" implication.
        """
        x_col = insight.get('x_column')
        y_col = insight.get('y_column')
        color_col = insight.get('color_column')
        chart_type = insight.get('chart_type', 'bar')
        agg = insight.get('aggregation', 'none')
        itype = insight.get('insight_type', '')

        parts = []

        # ── Ranking / Comparison charts ──
        if itype in ('ranking', 'comparison') and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                try:
                    agg_func = {'sum': 'sum', 'mean': 'mean', 'count': 'count',
                                'median': 'median'}.get(agg, 'sum')
                    grouped = df.groupby(x_col)[y_col].agg(agg_func).sort_values(ascending=False)
                    top = grouped.index[0] if len(grouped) > 0 else 'N/A'
                    top_val = grouped.iloc[0] if len(grouped) > 0 else 0
                    bottom = grouped.index[-1] if len(grouped) > 1 else 'N/A'
                    bottom_val = grouped.iloc[-1] if len(grouped) > 1 else 0
                    total = grouped.sum()
                    top_pct = (top_val / total * 100) if total > 0 else 0

                    # BEGINNING: context
                    parts.append(
                        f'Looking at {agg} {y_col} across {len(grouped)} {x_col} categories:'
                    )
                    # MIDDLE: evidence
                    parts.append(
                        f'"{top}" dominates with {top_val:,.1f} ({top_pct:.0f}% of total).'
                    )
                    if len(grouped) > 1 and bottom_val > 0:
                        ratio = top_val / bottom_val if bottom_val != 0 else 0
                        parts.append(
                            f'The gap is significant — "{top}" is {ratio:.1f}x higher than '
                            f'"{bottom}" ({bottom_val:,.1f}).'
                        )
                    # END: so what + recommendation
                    if top_pct > 40:
                        parts.append(
                            f'Action: High concentration risk — "{top}" accounts for nearly '
                            f'half of all {y_col}. Consider diversifying or investigating why.'
                        )
                    elif len(grouped) > 3 and top_pct < 20:
                        parts.append(
                            f'The distribution is relatively even across categories — '
                            f'no single {x_col} dominates.'
                        )
                except Exception:
                    pass

        # ── Distribution charts ──
        elif itype == 'distribution' and (x_col or y_col):
            col = x_col or y_col
            if col in df.columns and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                skew = (mean_val - median_val) / (std_val + 1e-9)

                # BEGINNING
                parts.append(
                    f'{col} ranges from {df[col].min():,.1f} to {df[col].max():,.1f} '
                    f'across {len(df):,} records.'
                )
                # MIDDLE
                if abs(skew) > 0.3:
                    direction = 'higher' if skew > 0 else 'lower'
                    parts.append(
                        f'The distribution is skewed — the average ({mean_val:,.1f}) is pulled '
                        f'toward {direction} values compared to the median ({median_val:,.1f}). '
                        f'This typically means a few extreme values are inflating the average.'
                    )
                else:
                    parts.append(
                        f'Values cluster symmetrically around {median_val:,.1f} '
                        f'(std dev: {std_val:,.1f}).'
                    )
                # END: implication
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                n_outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
                if n_outliers > 0:
                    parts.append(
                        f'Watch out: {n_outliers:,} outliers detected '
                        f'({n_outliers / len(df) * 100:.1f}% of data). '
                        f'These may need investigation or filtering before analysis.'
                    )

        # ── Composition charts ──
        elif itype == 'composition' and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                try:
                    grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                    top = grouped.index[0]
                    total = grouped.sum()
                    top_pct = (grouped.iloc[0] / total * 100) if total > 0 else 0
                    top3 = grouped.head(3)
                    top3_pct = (top3.sum() / total * 100) if total > 0 else 0

                    # BEGINNING
                    parts.append(
                        f'Breaking down total {y_col} ({total:,.0f}) by {x_col}:'
                    )
                    # MIDDLE
                    parts.append(
                        f'"{top}" holds the largest share at {top_pct:.0f}%.'
                    )
                    if len(grouped) > 2:
                        top3_names = ', '.join(f'"{n}"' for n in top3.index)
                        parts.append(
                            f'The top 3 ({top3_names}) together control {top3_pct:.0f}% — '
                            f'leaving only {100 - top3_pct:.0f}% for the remaining {len(grouped) - 3} categories.'
                        )
                    # END
                    if top3_pct > 80:
                        parts.append(
                            f'This is highly concentrated. The bottom categories contribute '
                            f'very little — consider whether they warrant continued investment.'
                        )
                except Exception:
                    pass

        # ── Correlation charts ──
        elif itype == 'correlation':
            if chart_type == 'heatmap':
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) >= 2:
                    corr = numeric_df.corr()
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    if not upper.stack().empty:
                        max_corr = upper.stack().abs().idxmax()
                        val = corr.loc[max_corr[0], max_corr[1]]
                        direction = 'increase together' if val > 0 else 'move in opposite directions'
                        # BEGINNING + MIDDLE
                        parts.append(
                            f'Scanning all feature pairs for relationships: '
                            f'the strongest link is between "{max_corr[0]}" and '
                            f'"{max_corr[1]}" (r={val:.2f}) — they tend to {direction}.'
                        )
                        # END
                        if abs(val) > 0.7:
                            parts.append(
                                f'This is a strong correlation. If building a predictive model, '
                                f'consider whether both features are needed (multicollinearity).'
                            )
            elif x_col and y_col and x_col in df.columns and y_col in df.columns:
                try:
                    corr_val = df[x_col].corr(df[y_col])
                    strength = 'strong' if abs(corr_val) > 0.6 else 'moderate' if abs(corr_val) > 0.3 else 'weak'
                    direction = 'positive' if corr_val > 0 else 'negative'
                    # BEGINNING + MIDDLE
                    parts.append(
                        f'The relationship between {x_col} and {y_col} is {strength} '
                        f'and {direction} (r={corr_val:.2f}).'
                    )
                    # END
                    if abs(corr_val) > 0.5:
                        parts.append(
                            f'This suggests a meaningful link — changes in {x_col} may help '
                            f'predict {y_col}. Worth investigating for strategic decisions.'
                        )
                    else:
                        parts.append(
                            f'These variables are mostly independent. '
                            f'Don\'t assume one drives the other.'
                        )
                except Exception:
                    pass

        # ── Trend charts ──
        elif itype == 'trend' and x_col and y_col:
            if y_col in df.columns:
                try:
                    first_val = df[y_col].iloc[0]
                    last_val = df[y_col].iloc[-1]
                    change = ((last_val - first_val) / (abs(first_val) + 1e-9)) * 100
                    direction = 'increased' if change > 0 else 'decreased'
                    # BEGINNING + MIDDLE
                    parts.append(
                        f'{y_col} {direction} by {abs(change):.0f}% over this period '
                        f'(from {first_val:,.1f} to {last_val:,.1f}).'
                    )
                    # END
                    if abs(change) > 20:
                        parts.append(
                            f'This is a significant shift. Investigate what changed '
                            f'during this period and whether the trend is accelerating or stabilizing.'
                        )
                    else:
                        parts.append(f'The trend is relatively flat — no major changes detected.')
                except Exception:
                    parts.append(insight.get('description', f'This chart shows how {y_col} changes over time.'))
            else:
                parts.append(insight.get('description', f'This chart shows how {y_col} changes over time.'))

        # Fallback
        if not parts:
            desc = insight.get('description', '')
            return desc if desc else 'This chart provides a visual overview of the data.'

        return ' '.join(parts)

    def _generate_key_takeaways(self, df: pd.DataFrame, profile: Dict,
                                 insight_list: List[Dict]) -> List[str]:
        """
        Generate 4-6 key takeaways with "SO WHAT?" framing.

        Structure for each takeaway:
          FACT → IMPLICATION → RECOMMENDED ACTION

        Principles: Don't just state numbers. State what they MEAN for the
        business and what the reader should DO about it. Use bold for the
        action/recommendation. Repetition reinforces the core message.
        """
        takeaways = []
        numeric_cols = [c for c in profile.get('numeric_cols', []) if c in df.columns]
        cat_cols = [c for c in profile.get('categorical_cols', []) if c in df.columns]
        n_rows = profile.get('n_rows', len(df))
        n_cols = profile.get('n_cols', len(df.columns))

        # 1. Dataset scope — sets context for everything below
        takeaways.append(
            f'📊 **Dataset scope:** {n_rows:,} records across {n_cols} columns '
            f'({len(numeric_cols)} numeric, {len(cat_cols)} categorical). '
            f'This is {"a robust sample for analysis" if n_rows > 500 else "a small dataset — interpret patterns with caution"}.'
        )

        # 2. Data quality — directly affects trust in the insights
        missing_total = int(df.isnull().sum().sum())
        missing_pct = missing_total / (n_rows * n_cols) * 100 if (n_rows * n_cols) > 0 else 0
        if missing_pct < 1:
            takeaways.append(
                '✅ **Data quality is excellent** — less than 1% missing values. '
                'You can trust these insights without worrying about gaps in the data.'
            )
        elif missing_pct < 5:
            takeaways.append(
                f'⚠️ **{missing_pct:.1f}% of data is missing.** This is acceptable, but '
                f'check if the gaps are random or concentrated in specific columns before '
                f'making decisions based on affected metrics.'
            )
        else:
            worst_col = df.isnull().sum().idxmax()
            worst_pct = df[worst_col].isnull().mean() * 100
            takeaways.append(
                f'🚨 **{missing_pct:.1f}% of data is missing** — the worst is '
                f'"{worst_col}" ({worst_pct:.0f}% missing). **Action needed:** fill or '
                f'exclude this column before relying on analysis that includes it.'
            )

        # 3. Key numeric insight — most IMPORTANT metric (not just most volatile)
        col_scores = self._score_columns(df, profile)
        scored_numerics = [c for c in numeric_cols if col_scores.get(c, 0) > 1.0]
        if not scored_numerics:
            scored_numerics = numeric_cols
        if scored_numerics:
            best_col = max(scored_numerics, key=lambda c: col_scores.get(c, 0))
            mean_val = df[best_col].mean()
            std_val = df[best_col].std()
            cv = std_val / (abs(mean_val) + 1e-9) * 100
            takeaways.append(
                f'📈 **{best_col} shows the highest variability** (CV: {cv:.0f}%) — '
                f'average {mean_val:,.1f} but ranges from {df[best_col].min():,.1f} to '
                f'{df[best_col].max():,.1f}. **Investigate** what causes this spread; '
                f'it may reveal distinct customer segments or operational inconsistencies.'
            )

        # 4. Dominant category — concentration risk or opportunity (skip if uniform)
        if cat_cols:
            for cat in cat_cols:
                vc = df[cat].value_counts()
                if len(vc) < 2:
                    continue
                top_pct = vc.iloc[0] / len(df) * 100
                bottom_pct = vc.iloc[-1] / len(df) * 100
                # Skip if distribution is nearly uniform (< 5% spread)
                if top_pct - bottom_pct < 5:
                    continue
                if top_pct > 30:
                    takeaways.append(
                        f'**"{vc.index[0]}" dominates {cat}** with {top_pct:.0f}% of all records. '
                        f'{"This concentration creates risk — if this segment underperforms, the overall numbers suffer. Consider diversification." if top_pct > 50 else "Monitor this segment closely; it drives the majority of your metrics."}'
                    )
                    break

        # 5. Strongest correlation — only if meaningful (|r| > 0.2)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            if not upper.stack().empty:
                max_idx = upper.stack().abs().idxmax()
                max_val = corr_matrix.loc[max_idx[0], max_idx[1]]
                if abs(max_val) > 0.2:
                    direction = 'increase together' if max_val > 0 else 'move in opposite directions'
                    strength = 'strongly' if abs(max_val) > 0.5 else 'moderately'
                    takeaways.append(
                        f'**{max_idx[0]} and {max_idx[1]} are {strength} linked** '
                        f'(r={max_val:.2f} — they {direction}). '
                        f'**Use this:** if you can influence one, the other will likely follow. '
                        f'For predictive models, you may only need one of these features.'
                    )
                else:
                    takeaways.append(
                        f'**No strong correlations found** between numeric features '
                        f'(max |r| = {abs(max_val):.2f}). Each feature captures independent information.'
                    )

        # 6. Outlier warning — actionable data quality flag
        for col in numeric_cols[:5]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                n_outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
                outlier_pct = n_outliers / len(df) * 100
                if outlier_pct > 5:
                    takeaways.append(
                        f'⚡ **{col} has {n_outliers:,} outliers** ({outlier_pct:.0f}% of data). '
                        f'These extreme values will skew averages and model training. '
                        f'**Recommendation:** review if these are data errors or genuine extremes, '
                        f'then decide whether to cap, remove, or analyze them separately.'
                    )
                    break

        return takeaways[:6]

    # =====================================================================
    # FALLBACK INSIGHTS (when LLM fails)
    # =====================================================================

    def _fallback_insights(self, df: pd.DataFrame, profile: Dict) -> Dict:
        """
        Generate smart, meaningful insights WITHOUT the LLM.

        Design Principles Applied:
        - ACTION TITLES: "Region X leads with 40% of revenue" not "Revenue by Region"
        - BARS > PIES: all composition charts use horizontal bars
        - CONTEXT FIRST: titles tell the finding; descriptions explain why it matters
        - CHART SELECTION: match chart type to data relationship (ranking→bar, trend→line)
        - COGNITIVE LOAD: max 8 insights, clear hierarchy, no junk charts
        - QUALITY GATE: skip uniform/boring insights automatically
        """
        insights = []
        all_numeric = profile['numeric_cols']
        all_cat = profile['categorical_cols']
        datetime_cols = profile.get('datetime_cols', [])

        # ── Use column scoring system for smart selection ──
        col_scores = self._score_columns(df, profile)
        data_quality = self._assess_data_quality(df, profile)

        # Filter columns using scores (skip anything scored 0)
        numeric_cols = [c for c in all_numeric if col_scores.get(c, 0) > 0 and df[c].std() > 0]
        cat_cols = [c for c in all_cat
                    if col_scores.get(c, 0) > 0
                    and 1 < df[c].nunique() <= 50
                    and c not in datetime_cols]

        best_cat = [c for c in cat_cols if df[c].nunique() <= 15] or cat_cols

        # Sort numeric columns by importance score (NOT just CV)
        numeric_sorted = sorted(numeric_cols,
                                key=lambda c: col_scores.get(c, 0),
                                reverse=True)

        # ── Pick the best value column using scores ──
        value_col = numeric_sorted[0] if numeric_sorted else None

        # ── Pick the best grouping column using scores ──
        cat_sorted = sorted(best_cat, key=lambda c: col_scores.get(c, 0), reverse=True)
        group_col = cat_sorted[0] if cat_sorted else None

        # ── Secondary grouping column ──
        color_col = None
        if len(cat_sorted) >= 2:
            color_col = cat_sorted[1]

        # Log what was selected
        self.log(f"  Smart column selection:")
        self.log(f"    value_col = {value_col} (score: {col_scores.get(value_col, 0):.1f})")
        self.log(f"    group_col = {group_col} (score: {col_scores.get(group_col, 0):.1f})")
        if color_col:
            self.log(f"    color_col = {color_col} (score: {col_scores.get(color_col, 0):.1f})")
        if data_quality.get('is_likely_synthetic'):
            self.log(f"    WARNING: Data appears synthetic — limiting composition/ranking insights")

        # ══════════════════════════════════════════════════════════════
        # HELPER: generate ACTION TITLES from data
        # (Principle: "title should tell the finding, not the axis")
        # ══════════════════════════════════════════════════════════════
        def _action_title_ranking(x, y, agg_func='sum'):
            """Generate a title like: 'Category X leads with 42% of total Y'."""
            try:
                grouped = df.groupby(x)[y].agg(agg_func).sort_values(ascending=False)
                top = grouped.index[0]
                top_val = grouped.iloc[0]
                total = grouped.sum()
                pct = (top_val / total * 100) if total > 0 else 0
                return f'"{top}" leads {x} with {pct:.0f}% of total {y}'
            except Exception:
                return f'Total {y} by {x}'

        def _action_title_comparison(x, y, agg_func='mean'):
            """Generate: 'Average Y is 2.3x higher in Category A than B'."""
            try:
                grouped = df.groupby(x)[y].agg(agg_func).sort_values(ascending=False)
                if len(grouped) >= 2:
                    top, bottom = grouped.index[0], grouped.index[-1]
                    ratio = grouped.iloc[0] / (grouped.iloc[-1] + 1e-9)
                    if ratio > 1.3:
                        return f'Average {y} is {ratio:.1f}x higher in "{top}" than "{bottom}"'
                    else:
                        return f'Average {y} is similar across {x} categories'
                return f'Average {y} by {x}'
            except Exception:
                return f'Average {y} by {x}'

        def _action_title_distribution(col):
            """Generate: 'Most values cluster around $45K with outliers up to $200K'."""
            try:
                mean_v = df[col].mean()
                median_v = df[col].median()
                max_v = df[col].max()
                skew = (mean_v - median_v) / (df[col].std() + 1e-9)
                if abs(skew) > 0.3:
                    direction = 'right-skewed toward higher values' if skew > 0 else 'left-skewed toward lower values'
                    return f'{col} distribution is {direction} (median: {median_v:,.0f})'
                else:
                    return f'{col} centers around {median_v:,.0f} (range: {df[col].min():,.0f}–{max_v:,.0f})'
            except Exception:
                return f'Distribution of {col}'

        def _action_title_correlation(x, y):
            """Generate: 'Strong positive link between X and Y (r=0.82)'."""
            try:
                r = df[x].corr(df[y])
                strength = 'Strong' if abs(r) > 0.6 else 'Moderate' if abs(r) > 0.3 else 'Weak'
                direction = 'positive' if r > 0 else 'negative'
                return f'{strength} {direction} link between {x} and {y} (r={r:.2f})'
            except Exception:
                return f'Relationship: {x} vs {y}'

        # ══════════════════════════════════════════════════════════════
        # BUILD INSIGHTS (each with action title + appropriate chart)
        # ══════════════════════════════════════════════════════════════

        # 1. Ranking: Total value by category (HORIZONTAL BAR — not pie)
        if value_col and group_col:
            insights.append({
                'title': _action_title_ranking(group_col, value_col, 'sum'),
                'description': f'Sorted from highest to lowest. Shows which {group_col} '
                               f'contributes the most {value_col} overall.',
                'chart_type': 'bar',
                'x_column': group_col,
                'y_column': value_col,
                'color_column': None,
                'aggregation': 'sum',
                'sort': 'descending',
                'insight_type': 'ranking'
            })

        # 2. Comparison: Average with color breakdown
        if value_col and group_col and color_col:
            insights.append({
                'title': _action_title_comparison(group_col, value_col, 'mean'),
                'description': f'Compares average {value_col} across {group_col}, '
                               f'broken down by {color_col}. Look for uneven gaps.',
                'chart_type': 'bar',
                'x_column': group_col,
                'y_column': value_col,
                'color_column': color_col,
                'aggregation': 'mean',
                'insight_type': 'comparison'
            })
        elif value_col and group_col:
            insights.append({
                'title': _action_title_comparison(group_col, value_col, 'mean'),
                'description': f'Which {group_col} has the highest average {value_col}?',
                'chart_type': 'bar',
                'x_column': group_col,
                'y_column': value_col,
                'color_column': None,
                'aggregation': 'mean',
                'sort': 'descending',
                'insight_type': 'comparison'
            })

        # 3. Composition: share breakdown (HORIZONTAL BAR — replaces pie)
        if group_col and value_col:
            insights.append({
                'title': _action_title_ranking(group_col, value_col, 'sum'),
                'description': f'Horizontal bar showing each {group_col}\'s share of '
                               f'total {value_col}. Leader is annotated.',
                'chart_type': 'pie',  # _dispatch_chart converts this to horiz bar
                'x_column': group_col,
                'y_column': value_col,
                'color_column': None,
                'aggregation': 'sum',
                'insight_type': 'composition'
            })

        # 4. Distribution of key numeric column
        if value_col:
            insights.append({
                'title': _action_title_distribution(value_col),
                'description': f'Histogram with mean line. Look for clusters, gaps, '
                               f'or long tails that indicate outliers.',
                'chart_type': 'histogram',
                'x_column': value_col,
                'y_column': None,
                'color_column': group_col if group_col and df[group_col].nunique() <= 8 else None,
                'aggregation': 'none',
                'insight_type': 'distribution'
            })

        # 5. Box plot: spread comparison across groups
        if value_col and group_col and df[group_col].nunique() <= 12:
            try:
                grouped = df.groupby(group_col)[value_col]
                max_spread = grouped.std().max()
                min_spread = grouped.std().min()
                ratio = max_spread / (min_spread + 1e-9)
                if ratio > 2:
                    box_title = f'{value_col} variability differs widely across {group_col} (up to {ratio:.0f}x)'
                else:
                    box_title = f'{value_col} spread is consistent across {group_col} groups'
            except Exception:
                box_title = f'{value_col} spread across {group_col}'

            insights.append({
                'title': box_title,
                'description': f'Box plots show median (line), 25th–75th percentile (box), '
                               f'and outliers (dots). Wider boxes = more variability.',
                'chart_type': 'box',
                'x_column': group_col,
                'y_column': value_col,
                'color_column': None,
                'aggregation': 'none',
                'insight_type': 'distribution'
            })

        # 6. Scatter plot: relationship between two numerics
        if len(numeric_cols) >= 2:
            x_num = numeric_sorted[0]
            y_num = numeric_sorted[1] if numeric_sorted[1] != x_num else (
                numeric_sorted[2] if len(numeric_sorted) > 2 else numeric_sorted[0]
            )
            if x_num != y_num:
                insights.append({
                    'title': _action_title_correlation(x_num, y_num),
                    'description': f'Each dot is one record. Trendline shows direction. '
                                   f'Correlation coefficient displayed on chart.',
                    'chart_type': 'scatter',
                    'x_column': x_num,
                    'y_column': y_num,
                    'color_column': group_col if group_col and df[group_col].nunique() <= 8 else None,
                    'aggregation': 'none',
                    'insight_type': 'correlation'
                })

        # 7. Time trend (if datetime exists)
        if datetime_cols and value_col:
            date_col = datetime_cols[0]
            try:
                first_val = df[value_col].iloc[0]
                last_val = df[value_col].iloc[-1]
                change = ((last_val - first_val) / (abs(first_val) + 1e-9)) * 100
                arrow = "up" if change > 0 else "down"
                trend_title = f'{value_col} trended {arrow} {abs(change):.0f}% over this period'
            except Exception:
                trend_title = f'{value_col} trend over time'

            insights.append({
                'title': trend_title,
                'description': f'Line chart with end-point annotation. '
                               f'Look for sustained trends, seasonality, or sudden shifts.',
                'chart_type': 'line',
                'x_column': date_col,
                'y_column': value_col,
                'color_column': group_col if group_col and df[group_col].nunique() <= 6 else None,
                'aggregation': 'mean',
                'insight_type': 'trend'
            })

        # 8. Heatmap: correlation between all numeric features
        if len(numeric_cols) >= 3:
            # Find strongest correlation pair for title
            try:
                corr = df[numeric_cols].corr()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                if not upper.stack().empty:
                    max_idx = upper.stack().abs().idxmax()
                    max_val = corr.loc[max_idx[0], max_idx[1]]
                    heat_title = f'Strongest link: {max_idx[0]} & {max_idx[1]} (r={max_val:.2f})'
                else:
                    heat_title = 'Feature correlations are weak overall'
            except Exception:
                heat_title = 'Feature Correlation Heatmap'

            insights.append({
                'title': heat_title,
                'description': 'Blue = move together. Red = move opposite. '
                               'Near zero = no relationship. Focus on the darkest cells.',
                'chart_type': 'heatmap',
                'x_column': None,
                'y_column': None,
                'color_column': None,
                'aggregation': 'none',
                'insight_type': 'correlation'
            })

        # Build summary
        n_rows = profile['n_rows']
        n_cols = profile['n_cols']
        summary_parts = [f'Analysis of {n_rows:,} records across {n_cols} columns.']
        if value_col:
            summary_parts.append(
                f'Key metric: {value_col} '
                f'(avg: {df[value_col].mean():.1f}, '
                f'total: {df[value_col].sum():,.0f}).'
            )
        if group_col:
            summary_parts.append(
                f'Main grouping: {group_col} '
                f'({df[group_col].nunique()} categories).'
            )

        return {
            'domain': 'General',
            'summary': ' '.join(summary_parts),
            'insights': insights
        }

    # =====================================================================
    # DASHBOARD BUILDER
    # =====================================================================

    def _build_dashboard(self, df: pd.DataFrame, domain: str, summary: str,
                         insight_list: List[Dict], charts: Dict,
                         narratives: Dict, output_dir: str,
                         kpis: List[Dict] = None, key_takeaways: List[str] = None,
                         data_quality: Dict = None) -> go.Figure:
        """Build a professional, production-ready HTML dashboard."""

        kpis = kpis or []
        key_takeaways = key_takeaways or []
        data_quality = data_quality or {}
        badge_colors = {
            'ranking': '#2563EB', 'comparison': '#7C3AED',
            'composition': '#059669', 'distribution': '#D97706',
            'correlation': '#DC2626', 'trend': '#0891B2',
            'overview': '#6B7280', 'general': '#6B7280',
        }

        html_parts = ['''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DataPilot AI Pro — Data Analysis Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
         background: #0F172A; color: #E2E8F0; line-height: 1.6; }

  /* Header */
  .header { background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            padding: 48px 40px 32px; border-bottom: 1px solid #1E293B; }
  .header h1 { font-size: 2.2em; font-weight: 800; color: #F8FAFC;
               margin-bottom: 8px; letter-spacing: -0.5px; }
  .header .summary { font-size: 1.05em; color: #94A3B8; max-width: 800px;
                     line-height: 1.7; }
  .header .meta { display: flex; gap: 24px; margin-top: 16px; flex-wrap: wrap; }
  .header .meta-item { font-size: 0.85em; color: #64748B; }
  .header .meta-item b { color: #CBD5E1; }

  /* KPI Cards */
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 16px; padding: 24px 40px; background: #0F172A; }
  .kpi-card { background: #1E293B; border-radius: 12px; padding: 20px 24px;
              border: 1px solid #334155; transition: border-color 0.2s; }
  .kpi-card:hover { border-color: #3B82F6; }
  .kpi-value { font-size: 1.8em; font-weight: 700; color: #F8FAFC;
               letter-spacing: -0.5px; }
  .kpi-name { font-size: 0.85em; color: #94A3B8; margin-top: 4px; }
  .kpi-desc { font-size: 0.75em; color: #64748B; margin-top: 2px; }

  /* Container */
  .container { max-width: 1400px; margin: 0 auto; padding: 32px 40px; }

  /* Section headers */
  .section-title { font-size: 1.4em; font-weight: 700; color: #F1F5F9;
                   margin: 32px 0 16px 0; padding-bottom: 8px;
                   border-bottom: 2px solid #1E293B; }

  /* Quality warnings */
  .warnings { background: #1C1917; border: 1px solid #92400E; border-radius: 10px;
              padding: 16px 20px; margin-bottom: 24px; }
  .warnings h3 { color: #FBBF24; font-size: 1em; margin-bottom: 8px; }
  .warnings p { color: #D6D3D1; font-size: 0.9em; margin: 4px 0; }

  /* Takeaways */
  .takeaways { background: #1E293B; border-radius: 12px; padding: 24px;
               margin-bottom: 28px; border: 1px solid #334155; }
  .takeaway-item { padding: 8px 0; color: #CBD5E1; font-size: 0.95em;
                   border-bottom: 1px solid #1E293B; line-height: 1.7; }
  .takeaway-item:last-child { border-bottom: none; }
  .takeaway-item b, .takeaway-item strong { color: #F8FAFC; }

  /* Insight cards */
  .insight-card { background: #1E293B; border-radius: 12px; margin-bottom: 24px;
                  border: 1px solid #334155; overflow: hidden;
                  transition: border-color 0.2s; }
  .insight-card:hover { border-color: #475569; }
  .insight-header { padding: 18px 24px 8px; display: flex; align-items: center; gap: 12px; }
  .insight-title { font-size: 1.15em; font-weight: 600; color: #F1F5F9; flex: 1; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
           font-size: 0.7em; font-weight: 600; text-transform: uppercase;
           letter-spacing: 0.5px; color: white; }
  .insight-desc { padding: 0 24px 12px; font-size: 0.88em; color: #94A3B8; }
  .chart-container { padding: 0 8px; }
  .narrative { background: #0F172A; border-top: 1px solid #334155;
               padding: 16px 24px; font-size: 0.9em; color: #94A3B8;
               line-height: 1.7; }
  .narrative .label { font-weight: 600; margin-right: 4px; }

  /* Grid for overview charts */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

  /* Footer */
  footer { text-align: center; padding: 40px; color: #475569; font-size: 0.85em;
           border-top: 1px solid #1E293B; margin-top: 40px; }
  footer a { color: #3B82F6; text-decoration: none; }
</style>
</head><body>''']

        # ── HEADER ──
        n_insights = len([k for k in charts if k.startswith('insight_')])
        html_parts.append(f'''
<div class="header">
  <h1>Analysis Complete — {n_insights} Insights Discovered</h1>
  <div class="summary">{summary}</div>
  <div class="meta">
    <span class="meta-item">Domain: <b>{domain}</b></span>
    <span class="meta-item">Charts: <b>{len(charts)}</b></span>
    <span class="meta-item">Records: <b>{len(df):,}</b></span>
    <span class="meta-item">Columns: <b>{len(df.columns)}</b></span>
  </div>
</div>''')

        # ── KPI CARDS ──
        if kpis:
            html_parts.append('<div class="kpi-row">')
            for kpi in kpis:
                fmt = kpi.get('format', ',.0f')
                try:
                    formatted = f'{kpi["value"]:{fmt}}'
                except (ValueError, KeyError):
                    formatted = str(kpi.get('value', ''))
                html_parts.append(f'''
  <div class="kpi-card">
    <div class="kpi-value">{formatted}</div>
    <div class="kpi-name">{kpi.get("name", "")}</div>
    <div class="kpi-desc">{kpi.get("description", "")}</div>
  </div>''')
            html_parts.append('</div>')

        html_parts.append('<div class="container">')

        # ── DATA QUALITY WARNINGS ──
        if data_quality.get('warnings'):
            html_parts.append('<div class="warnings">')
            html_parts.append('<h3>Data Quality Warnings</h3>')
            for w in data_quality['warnings']:
                html_parts.append(f'<p>{w}</p>')
            html_parts.append('</div>')

        # ── KEY TAKEAWAYS ──
        if key_takeaways:
            html_parts.append('<div class="section-title">Key Takeaways</div>')
            html_parts.append('<div class="takeaways">')
            for t in key_takeaways:
                # Convert markdown bold to HTML bold
                t_html = t.replace('**', '<b>', 1).replace('**', '</b>', 1)
                while '**' in t_html:
                    t_html = t_html.replace('**', '<b>', 1).replace('**', '</b>', 1)
                html_parts.append(f'<div class="takeaway-item">{t_html}</div>')
            html_parts.append('</div>')

        # ── INSIGHT CARDS ──
        html_parts.append('<div class="section-title">Insights Dashboard</div>')

        chart_idx = 0
        insight_keys = [k for k in charts if k.startswith('insight_')]
        for name in insight_keys:
            fig = charts[name]
            idx = int(name.split('_')[1]) - 1
            insight_info = insight_list[idx] if idx < len(insight_list) else {}

            title = insight_info.get('title', name.replace('_', ' ').title())
            desc = insight_info.get('description', '')
            itype = insight_info.get('insight_type', 'general')
            narrative = narratives.get(name, '')
            badge_bg = badge_colors.get(itype, '#6B7280')

            chart_div_id = f'chart_{chart_idx}'
            chart_json = fig.to_json()

            # Override chart background for dark theme
            html_parts.append(f'''
<div class="insight-card">
  <div class="insight-header">
    <span class="badge" style="background:{badge_bg};">{itype}</span>
    <span class="insight-title">{title}</span>
  </div>
  <div class="insight-desc">{desc}</div>
  <div class="chart-container" id="{chart_div_id}"></div>
  <script>
    (function() {{
      var spec = {chart_json};
      spec.layout.paper_bgcolor = '#1E293B';
      spec.layout.plot_bgcolor = '#1E293B';
      spec.layout.font = spec.layout.font || {{}};
      spec.layout.font.color = '#94A3B8';
      if (spec.layout.title && spec.layout.title.font) spec.layout.title.font.color = '#F1F5F9';
      if (spec.layout.xaxis) {{ spec.layout.xaxis.gridcolor = '#334155'; spec.layout.xaxis.tickfont = {{color: '#94A3B8'}}; }}
      if (spec.layout.yaxis) {{ spec.layout.yaxis.gridcolor = '#334155'; spec.layout.yaxis.tickfont = {{color: '#94A3B8'}}; }}
      if (spec.layout.legend) spec.layout.legend.font = {{color: '#CBD5E1'}};
      Plotly.newPlot("{chart_div_id}", spec.data, spec.layout, {{responsive: true, displayModeBar: false}});
    }})();
  </script>
  {f'<div class="narrative"><span class="label" style="color:{badge_bg};">Insight:</span> {narrative}</div>' if narrative else ''}
</div>''')
            chart_idx += 1

        # ── OVERVIEW CHARTS (2-col grid) ──
        overview_keys = [k for k in charts if k.startswith('overview_')]
        if overview_keys:
            html_parts.append('<div class="section-title">Data Overview</div>')
            html_parts.append('<div class="grid-2">')
            for name in overview_keys:
                fig = charts[name]
                chart_div_id = f'chart_{chart_idx}'
                chart_json = fig.to_json()
                readable_name = name.replace('overview_', '').replace('_', ' ').title()

                html_parts.append(f'''
<div class="insight-card">
  <div class="insight-header">
    <span class="badge" style="background:#6B7280;">overview</span>
    <span class="insight-title">{readable_name}</span>
  </div>
  <div class="chart-container" id="{chart_div_id}"></div>
  <script>
    (function() {{
      var spec = {chart_json};
      spec.layout.paper_bgcolor = '#1E293B';
      spec.layout.plot_bgcolor = '#1E293B';
      spec.layout.font = spec.layout.font || {{}};
      spec.layout.font.color = '#94A3B8';
      if (spec.layout.title && spec.layout.title.font) spec.layout.title.font.color = '#F1F5F9';
      if (spec.layout.xaxis) {{ spec.layout.xaxis.gridcolor = '#334155'; spec.layout.xaxis.tickfont = {{color: '#94A3B8'}}; }}
      if (spec.layout.yaxis) {{ spec.layout.yaxis.gridcolor = '#334155'; spec.layout.yaxis.tickfont = {{color: '#94A3B8'}}; }}
      if (spec.layout.legend) spec.layout.legend.font = {{color: '#CBD5E1'}};
      Plotly.newPlot("{chart_div_id}", spec.data, spec.layout, {{responsive: true, displayModeBar: false}});
    }})();
  </script>
</div>''')
                chart_idx += 1
            html_parts.append('</div>')  # grid-2

        html_parts.append('</div>')  # container

        html_parts.append('''
<footer>
  Generated by <a href="#">DataPilot AI Pro</a> — LLM-Powered Data Intelligence
</footer>
</body></html>''')

        # Save dashboard HTML
        dashboard_path = os.path.join(output_dir, 'dashboard.html')
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))

        self.log(f"  Dashboard saved to {dashboard_path}")
        return {'path': dashboard_path}

    # =====================================================================
    # UTILITIES
    # =====================================================================

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from an LLM response that may contain extra text."""
        # Try direct parsing first
        try:
            return json.loads(response)
        except:
            pass

        # Try to find JSON block in the response
        # Look for { ... } pattern
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
            r'(\{[\s\S]*\})',                  # { ... }
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    continue

        return None

    def _save_figure(self, fig: go.Figure, output_dir: str, name: str):
        """Save a Plotly figure as HTML and optionally PNG."""
        try:
            html_path = os.path.join(output_dir, f'{name}.html')
            fig.write_html(html_path, include_plotlyjs='cdn', full_html=True,
                           config={'responsive': True, 'displayModeBar': True})
        except Exception as e:
            self.log(f"  Warning: Could not save HTML for {name}: {e}")

        try:
            png_path = os.path.join(output_dir, f'{name}.png')
            fig.write_image(png_path, scale=2)
        except:
            pass  # Needs kaleido — skip if not installed
