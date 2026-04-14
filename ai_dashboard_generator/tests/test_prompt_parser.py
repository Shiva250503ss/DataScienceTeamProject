import pandas as pd
import pytest

from core.schemas import DatasetProfile, ChatResponse
from services.prompt_parser import PromptParser


def _make_profile(**kwargs):
    """Helper to build a DatasetProfile with sensible defaults."""
    defaults = dict(
        row_count=100,
        column_count=5,
        numeric_columns=["sales", "profit"],
        categorical_columns=["category", "region"],
        datetime_columns=["order_date"],
        id_like_columns=[],
        text_columns=[],
    )
    defaults.update(kwargs)
    return DatasetProfile(**defaults)


def _make_df():
    return pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "category": ["Electronics", "Clothing", "Books"] * 10,
        "region": ["North", "South", "East"] * 10,
        "sales": [100, 200, 50] * 10,
        "profit": [20, 40, 10] * 10,
    })


# ── Chart type detection ──────────────────────────────────────────────

class TestChartTypeDetection:
    def test_trend_prompt(self):
        profile = _make_profile()
        resp = PromptParser.parse("show monthly sales trend", profile)
        assert isinstance(resp, ChatResponse)
        assert len(resp.charts) == 1
        assert resp.charts[0].chart_type == "time_series"
        assert resp.charts[0].x == "order_date"
        assert resp.charts[0].y == "sales"

    def test_bar_compare(self):
        profile = _make_profile()
        resp = PromptParser.parse("compare category by sales", profile)
        assert resp.charts[0].chart_type == "bar"
        assert resp.charts[0].x == "category"
        assert resp.charts[0].y == "sales"

    def test_histogram(self):
        profile = _make_profile()
        resp = PromptParser.parse("distribution of profit", profile)
        assert resp.charts[0].chart_type == "histogram"
        assert resp.charts[0].x == "profit"

    def test_scatter(self):
        profile = _make_profile()
        resp = PromptParser.parse("relationship between sales and profit", profile)
        assert resp.charts[0].chart_type == "scatter"

    def test_pie_chart(self):
        profile = _make_profile()
        resp = PromptParser.parse("pie chart of category", profile)
        assert resp.charts[0].chart_type == "pie"
        assert resp.charts[0].x == "category"

    def test_heatmap(self):
        profile = _make_profile()
        resp = PromptParser.parse("show correlation heatmap", profile)
        assert resp.charts[0].chart_type == "heatmap"

    def test_count_by_category(self):
        profile = _make_profile()
        resp = PromptParser.parse("how many records per region", profile)
        assert resp.charts[0].chart_type == "bar"
        assert resp.charts[0].agg == "count"


# ── Aggregation detection ─────────────────────────────────────────────

class TestAggregationDetection:
    def test_average(self):
        profile = _make_profile()
        resp = PromptParser.parse("average sales by category", profile)
        assert resp.charts[0].agg == "mean"

    def test_total(self):
        profile = _make_profile()
        resp = PromptParser.parse("total profit by region", profile)
        assert resp.charts[0].agg == "sum"

    def test_top_n(self):
        profile = _make_profile()
        resp = PromptParser.parse("top 5 category by sales", profile)
        assert resp.charts[0].top_n == 5


# ── Stat answers ──────────────────────────────────────────────────────

class TestStatAnswers:
    def test_total_stat(self):
        profile = _make_profile(categorical_columns=[], datetime_columns=[])
        df = _make_df()
        resp = PromptParser.parse("what is the total sales", profile, df)
        assert "Total" in resp.text
        assert resp.charts  # should include a histogram

    def test_average_stat(self):
        profile = _make_profile(categorical_columns=[], datetime_columns=[])
        df = _make_df()
        resp = PromptParser.parse("what is the average profit", profile, df)
        assert "Average" in resp.text


# ── Meta queries ──────────────────────────────────────────────────────

class TestMetaQueries:
    def test_summary(self):
        profile = _make_profile()
        resp = PromptParser.parse("summary", profile)
        assert "Overview" in resp.text
        assert not resp.charts

    def test_columns_query(self):
        profile = _make_profile()
        resp = PromptParser.parse("what columns are there", profile)
        assert "Numeric" in resp.text

    def test_help_query(self):
        profile = _make_profile()
        resp = PromptParser.parse("what can I do with this data", profile)
        assert "ask me" in resp.text.lower() or "trend" in resp.text.lower()

    def test_empty_prompt(self):
        profile = _make_profile()
        resp = PromptParser.parse("", profile)
        assert resp.text  # any non-empty guidance message is acceptable


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_numeric_columns(self):
        profile = _make_profile(numeric_columns=[])
        resp = PromptParser.parse("show me something", profile)
        assert isinstance(resp, ChatResponse)

    def test_no_categorical_columns(self):
        profile = _make_profile(categorical_columns=[])
        resp = PromptParser.parse("compare something", profile)
        assert isinstance(resp, ChatResponse)

    def test_no_datetime_columns(self):
        profile = _make_profile(datetime_columns=[])
        resp = PromptParser.parse("show trend", profile)
        assert isinstance(resp, ChatResponse)
        assert "No datetime" in resp.text

    def test_only_one_numeric(self):
        profile = _make_profile(numeric_columns=["sales"])
        resp = PromptParser.parse("scatter plot", profile)
        assert "Need at least 2" in resp.text

    def test_vague_query(self):
        profile = _make_profile()
        resp = PromptParser.parse("what we can do with data", profile)
        assert isinstance(resp, ChatResponse)
        assert resp.text  # Should return help, not crash


# ── Chart rendering integration ───────────────────────────────────────

class TestChartRendering:
    def test_all_chart_types_render(self):
        """Every chart spec produced by the parser should render without error."""
        from services.chart_builder import SmartChartBuilder

        profile = _make_profile()
        df = _make_df()

        prompts = [
            "show monthly sales trend",
            "compare category by sales",
            "distribution of profit",
            "relationship between sales and profit",
            "pie chart of region",
            "how many records per category",
            "correlation heatmap",
            "top 3 category by sales",
            "average profit by region",
        ]

        for p in prompts:
            resp = PromptParser.parse(p, profile, df)
            for chart_spec in resp.charts:
                fig = SmartChartBuilder.build(df, chart_spec)
                assert fig is not None, f"Chart failed for prompt: {p}"
