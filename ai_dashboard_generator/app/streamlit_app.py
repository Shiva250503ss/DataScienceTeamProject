from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
import hashlib

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st

from core.config import APP_TITLE, MAX_ROWS_FOR_PREVIEW
from core.state import init_session_state
from services.chart_builder import SmartChartBuilder as ChartBuilder
from services.dashboard_generator import SmartDashboardGenerator as DashboardGenerator
from services.data_cleaner import DataCleaner
from services.data_loader import DataLoader
from services.data_profiler import DataProfiler
from services.llm_chat_engine import LLMChatEngine
from services.metric_deriver import MetricDeriver
from utils.formatting import format_number

# ── Global chart key counter — prevents duplicate Streamlit element IDs ──────
_chart_key_counter = 0


def _next_chart_key() -> str:
    global _chart_key_counter
    _chart_key_counter += 1
    return f"plotly_chart_{_chart_key_counter}"


# ── Read Groq credentials from secrets if available ──────────────────────────
_DEFAULT_GROQ_KEY   = st.secrets.get("GROQ_API_KEY", "")
_DEFAULT_GROQ_MODEL = st.secrets.get("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")


st.set_page_config(page_title=APP_TITLE, layout="wide")
init_session_state(st)

# ── Auto-connect Groq chat LLM from secrets on first load ────────────────────
if _DEFAULT_GROQ_KEY and st.session_state.get("chat_llm_client") is None:
    from services.llm_clients import GroqClient
    st.session_state.chat_llm_client = GroqClient(api_key=_DEFAULT_GROQ_KEY, model=_DEFAULT_GROQ_MODEL)
    st.session_state.chat_llm_model  = _DEFAULT_GROQ_MODEL

st.title(APP_TITLE)
st.caption("Upload any dataset and generate smart charts automatically.")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash  = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.get("uploaded_hash"):
        with st.spinner("Loading and profiling dataset…"):
            try:
                # ── 1. Load (all sheets for Excel) ───────────────────────────
                df, sheet_metadata = DataLoader.load_file(uploaded_file)

                # ── 2. Clean ─────────────────────────────────────────────────
                df = DataCleaner.clean(df)

                # ── 3. Derive metrics (duration, hour-of-day, etc.) ──────────
                df, derived_metrics = MetricDeriver.derive(df)

                # ── 4. Profile ───────────────────────────────────────────────
                profile = DataProfiler.profile(df)
                # Inject loader-level metadata into the profile
                profile.sheet_metadata  = sheet_metadata
                profile.derived_metrics = derived_metrics

                # ── 5. Dashboard generation ───────────────────────────────────
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
                st.error(f"Error while loading file: {exc}")

if st.session_state.df is not None:
    df            = st.session_state.df
    _edf          = st.session_state.get("enriched_df")
    enriched_df   = _edf if _edf is not None else df
    profile       = st.session_state.profile
    dashboard_spec = st.session_state.dashboard_spec

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        if st.button("Refresh dashboard", use_container_width=True,
                     help="Force re-process the current file"):
            for key in ["df", "enriched_df", "profile", "dashboard_spec",
                        "uploaded_name", "uploaded_hash", "llm_enhanced",
                        "chat_messages", "chat_llm_client", "chat_llm_model"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.subheader("Dataset summary")
        st.write(f"**File:** {st.session_state.uploaded_name}")
        st.write(f"**Rows:** {profile.row_count:,}")
        st.write(f"**Columns:** {profile.column_count}")
        st.write(f"**Numeric:** {len(profile.numeric_columns)}")
        st.write(f"**Categorical:** {len(profile.categorical_columns)}")
        st.write(f"**Datetime:** {len(profile.datetime_columns)}")

        # Show derived metrics in sidebar
        if profile.derived_metrics:
            st.write(f"**Derived metrics:** {len(profile.derived_metrics)}")
            for dm in profile.derived_metrics:
                st.caption(f"• {dm.name} ({dm.unit})")

        # Show sheet list for Excel files
        if profile.sheet_metadata and len(profile.sheet_metadata) > 1:
            st.divider()
            st.write("**Excel sheets:**")
            for sname, smeta in profile.sheet_metadata.items():
                st.caption(f"• '{sname}': {smeta['rows']:,} rows")

        st.divider()
        st.subheader("Dashboard Enhancement")
        st.caption("Improve auto-generated charts with LLM analysis. Chat always uses Groq by default.")

        provider = st.selectbox(
            "Provider",
            ["None (rule-based)", "Groq (free API)"],
            help="Applies LLM to improve dashboard chart quality.",
        )

        # ── Groq ──────────────────────────────────────────────────────────────
        if provider == "Groq (free API)":
            groq_key = st.text_input(
                "Groq API key",
                value=_DEFAULT_GROQ_KEY,
                type="password",
                placeholder="gsk_...",
                help="Free key at console.groq.com",
            )
            groq_model = st.selectbox(
                "Model",
                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                index=0,
            )

            if groq_key and st.button("Apply to Dashboard", use_container_width=True):
                from services.llm_clients import GroqClient
                _tmp_client = GroqClient(api_key=groq_key, model=groq_model)
                n_cols = len(profile.numeric_columns) + len(profile.categorical_columns)
                with st.spinner(f"Analyzing {n_cols} columns with Groq…"):
                    try:
                        new_enriched_df, new_spec = DashboardGenerator.generate(
                            df, profile,
                            llm_client=_tmp_client,
                            llm_model=groq_model,
                        )
                        st.session_state.enriched_df    = new_enriched_df
                        st.session_state.dashboard_spec = new_spec
                        st.session_state.llm_enhanced   = True
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Groq error: {exc}")

        # ── Chat LLM status ───────────────────────────────────────────────────
        st.divider()
        st.success(f"Chat LLM: **{st.session_state.chat_llm_model}** (auto-connected)")
        if st.session_state.get("llm_enhanced"):
            st.success("Dashboard: LLM enhanced")
            new_cols = [c for c in enriched_df.columns if c not in df.columns]
            if new_cols:
                st.caption(f"Semantic columns added: {', '.join(new_cols)}")

    # ── Reload after potential LLM update ────────────────────────────────────
    dashboard_spec = st.session_state.dashboard_spec
    _edf = st.session_state.get("enriched_df")
    enriched_df = _edf if _edf is not None else df

    # ── Preview ───────────────────────────────────────────────────────────────
    st.subheader("Data preview")
    st.dataframe(df.head(MAX_ROWS_FOR_PREVIEW), use_container_width=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
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
            st.plotly_chart(fig, use_container_width=True, key=_next_chart_key())
        except Exception as exc:
            st.warning(f"Could not build chart '{spec.title}': {exc}")

    # ── Chat interface ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Chat with your data")

    chat_llm_client = st.session_state.get("chat_llm_client")
    chat_llm_model  = st.session_state.get("chat_llm_model")

    st.caption(
        f"Powered by **{chat_llm_model}** — ask anything: "
        "'show charger usage', 'average duration by type', 'compare monitor checkouts by month'"
    )

    # Build chat profile — includes semantic/derived columns added to enriched_df
    extra_cols = [c for c in enriched_df.columns if c not in df.columns]
    chat_profile = profile
    if extra_cols:
        # Add any new numeric or categorical derived columns to the profile
        new_numeric = [
            c for c in extra_cols
            if c in enriched_df.select_dtypes(include="number").columns
        ]
        new_cat = [
            c for c in extra_cols
            if c not in new_numeric
        ]
        chat_profile = dataclasses.replace(
            profile,
            numeric_columns=profile.numeric_columns + new_numeric,
            categorical_columns=profile.categorical_columns + new_cat,
        )

    # ── Render chat history ───────────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg.get("text"):
                st.markdown(msg["text"])
            for chart_spec in msg.get("charts", []):
                try:
                    fig = ChartBuilder.build(enriched_df, chart_spec)
                    st.plotly_chart(fig, use_container_width=True, key=_next_chart_key())
                except Exception as exc:
                    st.warning(f"Could not render chart: {exc}")

    # ── New user message ──────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask anything about your data…"):
        st.session_state.chat_messages.append({"role": "user", "text": prompt, "charts": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    response = LLMChatEngine.chat(
                        user_prompt=prompt,
                        df=enriched_df,
                        profile=chat_profile,
                        llm_client=chat_llm_client,
                        llm_model=chat_llm_model,
                        history=st.session_state.chat_messages[:-1],
                        file_name=st.session_state.get("uploaded_name", "dataset"),
                    )

                if response.text:
                    st.markdown(response.text)

                rendered_specs = []
                for chart_spec in response.charts:
                    try:
                        fig = ChartBuilder.build(enriched_df, chart_spec)
                        st.plotly_chart(fig, use_container_width=True, key=_next_chart_key())
                        rendered_specs.append(chart_spec)
                    except Exception as exc:
                        st.warning(f"Could not build chart: {exc}")

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "text": response.text,
                    "charts": rendered_specs,
                })

            except Exception as exc:
                error_msg = f"Sorry, I couldn't process that: {exc}"
                st.error(error_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "text": error_msg,
                    "charts": [],
                })

    # ── Dataset profile expander ──────────────────────────────────────────────
    with st.expander("Dataset profile"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numeric columns:**", profile.numeric_columns)
            st.write("**Datetime columns:**", profile.datetime_columns)
            if profile.derived_metrics:
                st.write("**Derived metrics:**")
                for dm in profile.derived_metrics:
                    st.write(f"  • `{dm.name}` ({dm.unit}) — {dm.description}")
        with col2:
            st.write("**Categorical columns:**", profile.categorical_columns)
            st.write("**ID-like columns:**", profile.id_like_columns)
            if profile.sheet_metadata:
                st.write("**Sheets:**", list(profile.sheet_metadata.keys()))

else:
    st.info("Upload a CSV or Excel file to begin.")
