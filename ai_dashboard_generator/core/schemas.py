from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DerivedMetric:
    """A metric that was automatically computed from existing columns."""
    name: str           # new column name (e.g. "duration_hours")
    description: str    # human-readable explanation
    from_cols: List[str] = field(default_factory=list)  # source columns
    unit: str = ""      # "hours", "days", "hour (0-23)", "day name", …


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    id_like_columns: List[str] = field(default_factory=list)
    # Columns with very long average string length — free text, bad for charts
    text_columns: List[str] = field(default_factory=list)
    # Average string length per categorical column (used to filter free-text cols)
    avg_string_lengths: Dict[str, float] = field(default_factory=dict)
    missing_summary: Dict[str, float] = field(default_factory=dict)
    unique_summary: Dict[str, int] = field(default_factory=dict)
    # Top unique values per categorical column (up to 100), used for LLM context
    top_values: Dict[str, List[str]] = field(default_factory=dict)
    # Value frequency counts per categorical column (top 100), used for LLM prompts
    value_frequencies: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Multi-sheet Excel metadata: {sheet_name: {rows, columns, dtypes}}
    sheet_metadata: Dict[str, dict] = field(default_factory=dict)
    # Auto-derived metrics computed during ingestion
    derived_metrics: List[DerivedMetric] = field(default_factory=list)


@dataclass
class SemanticGroup:
    """Represents an LLM-generated grouping of raw categorical values."""
    source_column: str          # original column, e.g. "device"
    derived_column: str         # new column name, e.g. "device_category"
    mapping: Dict[str, str]     # {"Canon EOS": "Camera", "USB-C Charger": "Charger"}
    coverage: float             # fraction of non-null rows that got mapped (0-1)
    confidence: float           # LLM-reported confidence (0-1)


@dataclass
class ChartSpec:
    title: str
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    agg: Optional[str] = None
    top_n: Optional[int] = None
    description: Optional[str] = None
    time_granularity: Optional[str] = None
    sort_by: Optional[str] = None
    # Row filter: only plot rows where filter_col == filter_val (exact or fuzzy)
    filter_col: Optional[str] = None
    filter_val: Optional[str] = None
    # Multi-value OR filter: rows where filter_col value is in this list
    filter_values: List[str] = field(default_factory=list)


@dataclass
class DashboardSpec:
    kpis: List[Dict[str, Any]] = field(default_factory=list)
    charts: List[ChartSpec] = field(default_factory=list)


@dataclass
class ChatResponse:
    """Response from the chat engine."""
    text: str                                                   # Markdown narrative
    charts: List[ChartSpec] = field(default_factory=list)      # Zero or more charts
