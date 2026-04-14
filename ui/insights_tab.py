# ui/insights_tab.py
"""
Renders the AI Dashboard Generator (Data Insights) inside the main DataPilot UI.
Call render() inside a Streamlit tab — do NOT call set_page_config here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Ensure ai_dashboard_generator is importable ──────────────────────────────
_DG_ROOT = Path(__file__).resolve().parents[1] / "ai_dashboard_generator"
if str(_DG_ROOT) not in sys.path:
    sys.path.insert(0, str(_DG_ROOT))

# The project root also has a `utils/` package (utils/config.py) which gets
# cached in sys.modules before ai_dashboard_generator's utils is reachable.
# Clear any cached 'utils', 'services', and 'core' entries that came from
# outside _DG_ROOT so they are re-imported from the correct location.
for _mod in list(sys.modules.keys()):
    if _mod in ("utils", "services", "core") or any(
        _mod.startswith(p) for p in ("utils.", "services.", "core.")
    ):
        _cached_file = getattr(sys.modules[_mod], "__file__", None) or ""
        if str(_DG_ROOT) not in _cached_file:
            del sys.modules[_mod]

from core.config import MAX_ROWS_FOR_PREVIEW
from core.state import init_session_state
from services.chart_builder import SmartChartBuilder as ChartBuilder
from services.dashboard_generator import SmartDashboardGenerator as DashboardGenerator
from services.data_cleaner import DataCleaner
from services.data_loader import DataLoader
from services.data_profiler import DataProfiler
from services.llm_chat_engine import LLMChatEngine
from services.metric_deriver import MetricDeriver
from utils.formatting import format_number

# ── Pre-load project-root utils.config for the ML pipeline ───────────────────
# ai_dashboard_generator/utils/ is now the active 'utils' package, so
# agents/base.py's `from utils.config import config` would fail.
# Injecting it explicitly into sys.modules fixes the import without disturbing
# the dashboard's utils.formatting / utils.helpers resolution.
import importlib.util as _ilu
_PR_UTILS_CONFIG = Path(__file__).resolve().parents[1] / "utils" / "config.py"
if "utils.config" not in sys.modules:
    _spec = _ilu.spec_from_file_location("utils.config", str(_PR_UTILS_CONFIG))
    _cfg_mod = _ilu.module_from_spec(_spec)
    sys.modules["utils.config"] = _cfg_mod
    _spec.loader.exec_module(_cfg_mod)
del _ilu, _PR_UTILS_CONFIG, _spec, _cfg_mod

# ── Chart key counter (namespaced to avoid collision with main app) ───────────
_dg_chart_counter = 0


def _next_key() -> str:
    global _dg_chart_counter
    _dg_chart_counter += 1
    return f"dg_chart_{_dg_chart_counter}"


# ── Custom Chart Builder ──────────────────────────────────────────────────────

_CHART_COLORS: dict = {
    "Default":         None,
    "Pastel":          px.colors.qualitative.Pastel,
    "Bold":            px.colors.qualitative.Bold,
    "Vivid":           px.colors.qualitative.Vivid,
    "Colorblind Safe": px.colors.qualitative.Safe,
    "Ocean Blues":     px.colors.sequential.Blues,
    "Warm Sunset":     px.colors.sequential.Sunset,
}


def _col_type(col: str, df: pd.DataFrame, profile) -> str:
    """Classify a column as 'datetime', 'numeric', or 'categorical'."""
    if col in profile.datetime_columns:
        return "datetime"
    try:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return "datetime"
        if pd.api.types.is_numeric_dtype(df[col]):
            return "numeric"
    except Exception:
        pass
    return "categorical"


def _chart_types_for(x: Optional[str], y: Optional[str],
                     df: pd.DataFrame, profile) -> List[str]:
    """Return the ordered list of sensible chart types for the chosen (x, y) pair."""
    if not x:
        return ["— pick an X-axis first —"]

    xk = _col_type(x, df, profile)
    yk = _col_type(y, df, profile) if y else None

    if xk == "datetime":
        if yk == "numeric":
            return ["Line", "Area", "Bar"]
        return ["Line (count)", "Bar (count)"]

    if xk == "categorical":
        if yk == "numeric":
            return ["Bar", "Horizontal Bar", "Pie", "Box", "Violin", "Funnel"]
        return ["Bar (count)", "Pie (count)"]

    # xk == "numeric"
    if yk == "numeric":
        return ["Scatter", "Line", "Bar", "Box", "Violin"]
    if yk == "categorical":
        return ["Box", "Violin", "Strip"]
    return ["Histogram", "Box", "Violin"]


def _build_custom_chart(
    df: pd.DataFrame,
    x: str,
    y: Optional[str],
    chart_type: str,
    color_seq: Optional[list],
    group_by: Optional[str],
    profile,
):
    """Build and return a Plotly figure from the Custom Chart Builder selections."""
    ct = chart_type.lower()

    # Build title
    title_parts = [chart_type, x]
    if y:
        title_parts.append(f"vs {y}")
    if group_by:
        title_parts.append(f"by {group_by}")
    title = " · ".join(title_parts)

    kw: dict = {"title": title}
    if color_seq:
        kw["color_discrete_sequence"] = color_seq

    # ── Histogram ─────────────────────────────────────────────────────────────
    if ct == "histogram":
        cols = [x] + ([group_by] if group_by else [])
        w = df[cols].copy()
        w[x] = pd.to_numeric(w[x], errors="coerce")
        w = w.dropna(subset=[x])
        if group_by:
            kw["color"] = group_by
        return px.histogram(w, x=x, nbins=30, **kw)

    # ── Scatter ────────────────────────────────────────────────────────────────
    if ct == "scatter":
        if not y:
            raise ValueError("Scatter requires a Y-axis column.")
        cols = [x, y] + ([group_by] if group_by else [])
        w = df[cols].copy()
        w[x] = pd.to_numeric(w[x], errors="coerce")
        w[y] = pd.to_numeric(w[y], errors="coerce")
        w = w.dropna(subset=[x, y])
        if group_by:
            kw["color"] = group_by
        return px.scatter(w, x=x, y=y, **kw)

    # ── Box / Violin / Strip ───────────────────────────────────────────────────
    if ct in ("box", "violin", "strip"):
        target = y if y else x
        cols = [target] + ([group_by] if group_by else [])
        w = df[cols].dropna(subset=[target]).copy()
        if group_by:
            kw["color"] = group_by
        if ct == "box":
            return px.box(w, x=group_by, y=target, **kw)
        if ct == "violin":
            return px.violin(w, x=group_by, y=target, **kw)
        return px.strip(w, x=group_by, y=target, **kw)

    # ── Pie / Pie (count) ──────────────────────────────────────────────────────
    if ct in ("pie", "pie (count)"):
        pkw = {"title": title}   # pie ignores color_discrete_sequence
        w = df[[x]].dropna(subset=[x]).copy()
        if y and ct == "pie":
            w[y] = pd.to_numeric(df.loc[w.index, y], errors="coerce")
            w = w.dropna(subset=[y])
            agg = w.groupby(x, as_index=False)[y].sum().nlargest(15, y)
            return px.pie(agg, names=x, values=y, **pkw)
        result = w[x].value_counts().head(15).reset_index()
        result.columns = [x, "_count"]
        return px.pie(result, names=x, values="_count", **pkw)

    # ── Bar / Horizontal Bar / Bar (count) / Funnel ────────────────────────────
    if ct in ("bar", "horizontal bar", "bar (count)", "funnel"):
        if y and ct != "bar (count)":
            cols = [x, y] + ([group_by] if group_by else [])
            w = df[cols].copy()
            w[y] = pd.to_numeric(w[y], errors="coerce")
            w = w.dropna(subset=[x, y])
            if group_by:
                agg = w.groupby([x, group_by], as_index=False)[y].sum()
                kw["color"] = group_by
            else:
                agg = w.groupby(x, as_index=False)[y].sum()
                agg = agg.sort_values(y, ascending=False).head(30)
            if ct == "funnel":
                return px.funnel(agg, x=y, y=x, **kw)
            if ct == "horizontal bar":
                return px.bar(agg, x=y, y=x, orientation="h", text_auto=True, **kw)
            return px.bar(agg, x=x, y=y, text_auto=True, **kw)
        else:  # count
            cols = [x] + ([group_by] if group_by else [])
            w = df[cols].dropna(subset=[x]).copy()
            if group_by:
                result = w.groupby([x, group_by]).size().reset_index(name="_count")
                kw["color"] = group_by
            else:
                result = w[x].value_counts().head(30).reset_index()
                result.columns = [x, "_count"]
            return px.bar(result, x=x, y="_count", text_auto=True, **kw)

    # ── Line / Area (datetime or numeric x) ───────────────────────────────────
    if ct in ("line", "area", "line (count)"):
        xk = _col_type(x, df, profile)
        if xk == "datetime":
            w = df.copy()
            w[x] = pd.to_datetime(w[x], errors="coerce")
            w = w.dropna(subset=[x])
            w["_period"] = w[x].dt.to_period("M").astype(str)
            if y and ct != "line (count)":
                w[y] = pd.to_numeric(w[y], errors="coerce")
                w = w.dropna(subset=[y])
                grp_cols = ["_period"] + ([group_by] if group_by else [])
                agg = (w.groupby(grp_cols, as_index=False)[y]
                         .sum().sort_values("_period"))
                agg.rename(columns={"_period": x}, inplace=True)
                if group_by:
                    kw["color"] = group_by
                if ct == "area":
                    return px.area(agg, x=x, y=y, **kw)
                return px.line(agg, x=x, y=y, markers=True, **kw)
            else:
                result = (w.groupby("_period").size()
                           .reset_index(name="_count")
                           .sort_values("_period"))
                result.rename(columns={"_period": x}, inplace=True)
                return px.line(result, x=x, y="_count", markers=True, **kw)
        else:  # numeric x
            if not y:
                raise ValueError("Line/Area requires a Y-axis column for numeric X.")
            cols = [x, y] + ([group_by] if group_by else [])
            w = df[cols].copy()
            w[x] = pd.to_numeric(w[x], errors="coerce")
            w[y] = pd.to_numeric(w[y], errors="coerce")
            w = w.dropna(subset=[x, y]).sort_values(x)
            if group_by:
                kw["color"] = group_by
            if ct == "area":
                return px.area(w, x=x, y=y, **kw)
            return px.line(w, x=x, y=y, markers=True, **kw)

    raise ValueError(f"Unsupported chart type: {chart_type!r}")


def _render_custom_chart_builder(enriched_df: pd.DataFrame, profile) -> None:
    """Render the 5-control Custom Chart Builder section."""
    st.divider()
    st.subheader("🛠️ Custom Chart Builder")
    st.caption(
        "Choose your axes, then pick a chart type, color theme, and optional group-by column."
    )

    all_cols = list(enriched_df.columns)

    # ── 5 selectboxes in a single row ─────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown("**X-Axis**")
        x_raw = st.selectbox(
            "X-Axis", ["— select —"] + all_cols,
            key="cb_x", label_visibility="collapsed"
        )

    with c2:
        st.markdown("**Y-Axis**")
        y_raw = st.selectbox(
            "Y-Axis", ["None (count)"] + all_cols,
            key="cb_y", label_visibility="collapsed"
        )

    x_col = x_raw if x_raw != "— select —" else None
    y_col = y_raw if y_raw != "None (count)" else None

    # Dynamic chart types based on X / Y column types
    available_types = _chart_types_for(x_col, y_col, enriched_df, profile)

    with c3:
        st.markdown("**Chart Type**")
        chart_type = st.selectbox(
            "Chart Type", available_types,
            key="cb_type", label_visibility="collapsed"
        )

    with c4:
        st.markdown("**Color Theme**")
        color_name = st.selectbox(
            "Color Theme", list(_CHART_COLORS.keys()),
            key="cb_color", label_visibility="collapsed"
        )
        color_seq = _CHART_COLORS[color_name]

    with c5:
        st.markdown("**Group By**")
        groupby_candidates = [
            c for c in all_cols
            if c not in (x_col, y_col)
            and enriched_df[c].dtype == object
            and enriched_df[c].nunique() <= 30
        ]
        group_raw = st.selectbox(
            "Group By", ["None"] + groupby_candidates,
            key="cb_groupby", label_visibility="collapsed"
        )
        group_by = group_raw if group_raw != "None" else None

    # ── Render chart ──────────────────────────────────────────────────────────
    if x_col and not chart_type.startswith("—"):
        try:
            fig = _build_custom_chart(
                df=enriched_df,
                x=x_col,
                y=y_col,
                chart_type=chart_type,
                color_seq=color_seq,
                group_by=group_by,
                profile=profile,
            )
            st.plotly_chart(fig, use_container_width=True, key=_next_key())
        except Exception as exc:
            st.warning(f"Could not render chart: {exc}")
    else:
        st.info("Select an X-axis column above to build your chart.")


def render() -> None:
    """Render the full AI Dashboard Generator UI inside the current Streamlit tab."""

    # ── Groq credentials from .env / environment ─────────────────────────────
    import os
    from dotenv import load_dotenv
    load_dotenv()
    _gemini_key   = os.getenv("GEMINI_API_KEY", "")
    _gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    _groq_key     = os.getenv("GROQ_API_KEY", "")
    _groq_model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    init_session_state(st)

    # ── Auto-connect LLM on first load — Gemini primary, Groq fallback ───────
    if st.session_state.get("chat_llm_client") is None:
        from services.llm_clients import GeminiClient, GroqClient
        if _gemini_key:
            _fallback = GroqClient(api_key=_groq_key, model=_groq_model) if _groq_key else None
            st.session_state.chat_llm_client = GeminiClient(
                api_key=_gemini_key, model=_gemini_model, fallback_client=_fallback
            )
            st.session_state.chat_llm_model = f"{_gemini_model} (+ Groq fallback)" if _fallback else _gemini_model
        elif _groq_key:
            st.session_state.chat_llm_client = GroqClient(api_key=_groq_key, model=_groq_model)
            st.session_state.chat_llm_model  = _groq_model

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="dg_uploader"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash  = hashlib.md5(file_bytes).hexdigest()

        if file_hash != st.session_state.get("uploaded_hash"):
            with st.spinner("Loading and profiling dataset…"):
                try:
                    df, sheet_metadata = DataLoader.load_file(uploaded_file)
                    df = DataCleaner.clean(df)
                    df, derived_metrics = MetricDeriver.derive(df)
                    profile = DataProfiler.profile(df)
                    profile.sheet_metadata  = sheet_metadata
                    profile.derived_metrics = derived_metrics
                    enriched_df, dashboard_spec = DashboardGenerator.generate(df, profile)

                    st.session_state.df             = df
                    st.session_state.enriched_df    = enriched_df
                    st.session_state.profile        = profile
                    st.session_state.dashboard_spec = dashboard_spec
                    st.session_state.uploaded_name  = uploaded_file.name
                    st.session_state.uploaded_hash  = file_hash
                    st.session_state.llm_enhanced   = False
                    st.session_state.chat_messages  = []

                    if derived_metrics:
                        st.success(
                            f"Auto-derived {len(derived_metrics)} metric(s): "
                            + ", ".join(f"**{m.name}**" for m in derived_metrics)
                        )
                    if sheet_metadata and len(sheet_metadata) > 1:
                        st.info(
                            f"Excel file has {len(sheet_metadata)} sheets: "
                            + ", ".join(f"'{s}'" for s in sheet_metadata)
                            + ". All sheets were combined."
                        )
                except Exception as exc:
                    st.error(f"Error loading file: {exc}")
                    return

    if st.session_state.df is None:
        st.info("Upload a CSV or Excel file above to get started.")
        return

    df             = st.session_state.df
    _edf           = st.session_state.get("enriched_df")
    enriched_df    = _edf if _edf is not None else df
    profile        = st.session_state.profile
    dashboard_spec = st.session_state.dashboard_spec

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        if st.button("Refresh dashboard", use_container_width=True,
                     help="Force re-process the current file", key="dg_refresh"):
            for k in ["df", "enriched_df", "profile", "dashboard_spec",
                      "uploaded_name", "uploaded_hash", "llm_enhanced",
                      "chat_messages", "chat_llm_client", "chat_llm_model"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.subheader("Dataset summary")
        st.write(f"**File:** {st.session_state.uploaded_name}")
        st.write(f"**Rows:** {profile.row_count:,}")
        st.write(f"**Columns:** {profile.column_count}")
        st.write(f"**Numeric:** {len(profile.numeric_columns)}")
        st.write(f"**Categorical:** {len(profile.categorical_columns)}")
        st.write(f"**Datetime:** {len(profile.datetime_columns)}")

        if profile.derived_metrics:
            st.write(f"**Derived metrics:** {len(profile.derived_metrics)}")
            for dm in profile.derived_metrics:
                st.caption(f"• {dm.name} ({dm.unit})")

        if profile.sheet_metadata and len(profile.sheet_metadata) > 1:
            st.divider()
            st.write("**Excel sheets:**")
            for sname, smeta in profile.sheet_metadata.items():
                st.caption(f"• '{sname}': {smeta['rows']:,} rows")

        st.divider()
        st.subheader("🤖 Chat LLM")
        st.caption("Gemini 2.5 Flash (primary) · Groq fallback · auto-loaded from .env")

        if st.session_state.get("chat_llm_model"):
            st.success(f"Connected: **{st.session_state.chat_llm_model}**")
        else:
            st.warning("No LLM connected — add GEMINI_API_KEY or GROQ_API_KEY to .env")

        with st.expander("Manual override", expanded=False):
            chat_gemini_key = st.text_input(
                "Gemini API key", value=_gemini_key, type="password",
                placeholder="AIza...", key="dg_chat_gemini_key",
            )
            chat_groq_key = st.text_input(
                "Groq API key (fallback)", value=_groq_key, type="password",
                placeholder="gsk_...", key="dg_chat_groq_key",
            )
            if st.button("Reconnect LLM", use_container_width=True, key="dg_connect_chat"):
                from services.llm_clients import GeminiClient, GroqClient
                if chat_gemini_key:
                    _fb = GroqClient(api_key=chat_groq_key, model=_groq_model) if chat_groq_key else None
                    st.session_state.chat_llm_client = GeminiClient(
                        api_key=chat_gemini_key, model=_gemini_model, fallback_client=_fb
                    )
                    st.session_state.chat_llm_model = (
                        f"{_gemini_model} (+ Groq fallback)" if _fb else _gemini_model
                    )
                elif chat_groq_key:
                    st.session_state.chat_llm_client = GroqClient(api_key=chat_groq_key, model=_groq_model)
                    st.session_state.chat_llm_model  = _groq_model
                st.rerun()

        st.divider()
        st.subheader("📊 Dashboard Enhancement")
        st.caption("Improve auto-generated charts with LLM analysis.")

        provider = st.selectbox(
            "Provider",
            ["None (rule-based)", "Gemini (primary)", "Groq (fallback)"],
            key="dg_provider",
        )

        if provider == "Gemini (primary)":
            dash_gemini_key = st.text_input(
                "Gemini API key", value=_gemini_key, type="password",
                placeholder="AIza...", key="dg_gemini_key",
            )
            if dash_gemini_key and st.button("Apply to Dashboard", use_container_width=True, key="dg_apply_gemini"):
                from services.llm_clients import GeminiClient, GroqClient
                _fb = GroqClient(api_key=_groq_key, model=_groq_model) if _groq_key else None
                _tmp = GeminiClient(api_key=dash_gemini_key, model=_gemini_model, fallback_client=_fb)
                n = len(profile.numeric_columns) + len(profile.categorical_columns)
                with st.spinner(f"Analyzing {n} columns with Gemini…"):
                    try:
                        new_edf, new_spec = DashboardGenerator.generate(
                            df, profile, llm_client=_tmp, llm_model=_gemini_model
                        )
                        st.session_state.enriched_df    = new_edf
                        st.session_state.dashboard_spec = new_spec
                        st.session_state.llm_enhanced   = True
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Enhancement error: {exc}")

        elif provider == "Groq (fallback)":
            dash_groq_key = st.text_input(
                "Groq API key", value=_groq_key, type="password",
                placeholder="gsk_...", key="dg_groq_key",
            )
            dash_groq_model = st.selectbox(
                "Model",
                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                key="dg_groq_model",
            )
            if dash_groq_key and st.button("Apply to Dashboard", use_container_width=True, key="dg_apply_groq"):
                from services.llm_clients import GroqClient
                _tmp = GroqClient(api_key=dash_groq_key, model=dash_groq_model)
                n = len(profile.numeric_columns) + len(profile.categorical_columns)
                with st.spinner(f"Analyzing {n} columns with Groq…"):
                    try:
                        new_edf, new_spec = DashboardGenerator.generate(
                            df, profile, llm_client=_tmp, llm_model=dash_groq_model
                        )
                        st.session_state.enriched_df    = new_edf
                        st.session_state.dashboard_spec = new_spec
                        st.session_state.llm_enhanced   = True
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Groq error: {exc}")

        if st.session_state.get("llm_enhanced"):
            st.success("Dashboard: LLM enhanced ✓")
            new_cols = [c for c in enriched_df.columns if c not in df.columns]
            if new_cols:
                st.caption(f"Semantic columns added: {', '.join(new_cols)}")

    # ── Reload after potential LLM update ────────────────────────────────────
    dashboard_spec = st.session_state.dashboard_spec
    _edf2          = st.session_state.get("enriched_df")
    enriched_df    = _edf2 if _edf2 is not None else df

    # ── Data preview ─────────────────────────────────────────────────────────
    st.subheader("Data preview")
    st.dataframe(df.head(MAX_ROWS_FOR_PREVIEW), use_container_width=True)

    # ── KPI cards ────────────────────────────────────────────────────────────
    if dashboard_spec.kpis:
        st.subheader("KPIs")
        kpi_cols = st.columns(min(len(dashboard_spec.kpis), 6))
        for col, kpi in zip(kpi_cols, dashboard_spec.kpis):
            col.metric(kpi["label"], format_number(kpi["value"]))

    # ── Auto dashboard ────────────────────────────────────────────────────────
    st.subheader("Auto dashboard")
    for spec in dashboard_spec.charts:
        try:
            fig = ChartBuilder.build(enriched_df, spec)
            if spec.description:
                st.caption(spec.description)
            st.plotly_chart(fig, use_container_width=True, key=_next_key())
        except Exception as exc:
            st.warning(f"Could not build chart '{spec.title}': {exc}")

    # ── Custom Chart Builder ──────────────────────────────────────────────────
    _render_custom_chart_builder(enriched_df, profile)

    # ── Chat interface ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Chat with your data")

    chat_client = st.session_state.get("chat_llm_client")
    chat_model  = st.session_state.get("chat_llm_model")

    if chat_model:
        st.caption(
            f"Powered by **{chat_model}** — ask anything: "
            "'show sales by month', 'average value by category', 'compare trends'"
        )
    else:
        st.caption("No LLM connected. Add a Groq API key in the sidebar to enable AI chat.")

    extra_cols   = [c for c in enriched_df.columns if c not in df.columns]
    chat_profile = profile
    if extra_cols:
        new_num = [c for c in extra_cols if c in enriched_df.select_dtypes(include="number").columns]
        new_cat = [c for c in extra_cols if c not in new_num]
        chat_profile = dataclasses.replace(
            profile,
            numeric_columns=profile.numeric_columns + new_num,
            categorical_columns=profile.categorical_columns + new_cat,
        )

    # Render full chat history (all previous messages)
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg.get("text"):
                st.markdown(msg["text"])
            for cs in msg.get("charts", []):
                try:
                    fig = ChartBuilder.build(enriched_df, cs)
                    st.plotly_chart(fig, use_container_width=True, key=_next_key())
                except Exception as exc:
                    st.warning(f"Could not render chart: {exc}")

    # Input always at the bottom — process, store, rerun so next render is clean
    if prompt := st.chat_input("Ask anything about your data…", key="dg_chat"):
        st.session_state.chat_messages.append({"role": "user", "text": prompt, "charts": []})
        with st.spinner("Thinking…"):
            try:
                response = LLMChatEngine.chat(
                    user_prompt=prompt,
                    df=enriched_df,
                    profile=chat_profile,
                    llm_client=chat_client,
                    llm_model=chat_model,
                    history=st.session_state.chat_messages[:-1],
                    file_name=st.session_state.get("uploaded_name", "dataset"),
                )
                rendered = []
                for cs in response.charts:
                    try:
                        ChartBuilder.build(enriched_df, cs)  # validate
                        rendered.append(cs)
                    except Exception:
                        pass
                st.session_state.chat_messages.append({
                    "role": "assistant", "text": response.text, "charts": rendered,
                })
            except Exception as exc:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "text": f"Sorry, I couldn't process that: {exc}",
                    "charts": [],
                })
        st.rerun()

    # ── Dataset profile expander ──────────────────────────────────────────────
    with st.expander("Dataset profile"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Numeric columns:**", profile.numeric_columns)
            st.write("**Datetime columns:**", profile.datetime_columns)
            if profile.derived_metrics:
                st.write("**Derived metrics:**")
                for dm in profile.derived_metrics:
                    st.write(f"  • `{dm.name}` ({dm.unit}) — {dm.description}")
        with c2:
            st.write("**Categorical columns:**", profile.categorical_columns)
            st.write("**ID-like columns:**", profile.id_like_columns)
            if profile.sheet_metadata:
                st.write("**Sheets:**", list(profile.sheet_metadata.keys()))
