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

# ── Chart key counter (namespaced to avoid collision with main app) ───────────
_dg_chart_counter = 0


def _next_key() -> str:
    global _dg_chart_counter
    _dg_chart_counter += 1
    return f"dg_chart_{_dg_chart_counter}"


def render() -> None:
    """Render the full AI Dashboard Generator UI inside the current Streamlit tab."""

    # ── Groq credentials from Streamlit secrets (optional) ───────────────────
    try:
        _groq_key   = st.secrets.get("GROQ_API_KEY", "")
        _groq_model = st.secrets.get("GROQ_DEFAULT_MODEL", "llama-3.1-8b-instant")
    except Exception:
        _groq_key   = ""
        _groq_model = "llama-3.3-70b-versatile"

    init_session_state(st)

    # ── Auto-connect Groq chat LLM on first load ─────────────────────────────
    if _groq_key and st.session_state.get("chat_llm_client") is None:
        from services.llm_clients import GroqClient
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
        st.caption("Connect Groq to power the chatbot.")

        chat_key_input = st.text_input(
            "Groq API key",
            value=_groq_key or (st.session_state.get("chat_llm_client") and "") or "",
            type="password",
            placeholder="gsk_...",
            help="Free key at console.groq.com",
            key="dg_chat_groq_key",
        )
        chat_model_input = st.selectbox(
            "Chat model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            key="dg_chat_model_select",
        )
        if chat_key_input and st.button("Connect Chat LLM", use_container_width=True, key="dg_connect_chat"):
            from services.llm_clients import GroqClient
            st.session_state.chat_llm_client = GroqClient(api_key=chat_key_input, model=chat_model_input)
            st.session_state.chat_llm_model  = chat_model_input
            st.rerun()

        if st.session_state.get("chat_llm_model"):
            st.success(f"Connected: **{st.session_state.chat_llm_model}**")
        else:
            st.warning("Chat not connected — enter a Groq key above.")

        st.divider()
        st.subheader("📊 Dashboard Enhancement")
        st.caption("Improve auto-generated charts with LLM analysis.")

        provider = st.selectbox(
            "Provider",
            ["None (rule-based)", "Groq (free API)", "Ollama (free, local)"],
            key="dg_provider",
        )

        if provider == "Groq (free API)":
            dash_groq_key = st.text_input(
                "Groq API key", value=_groq_key, type="password",
                placeholder="gsk_...", help="Free key at console.groq.com",
                key="dg_groq_key",
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

        elif provider == "Ollama (free, local)":
            ollama_model = st.text_input(
                "Model", value="llama3.2", help="Run: ollama pull llama3.2",
                key="dg_ollama_model",
            )
            if st.button("Apply to Dashboard", use_container_width=True, key="dg_apply_ollama"):
                from services.llm_clients import OllamaClient
                _tmp = OllamaClient(model=ollama_model)
                n = len(profile.numeric_columns) + len(profile.categorical_columns)
                with st.spinner(f"Analyzing {n} columns with Ollama…"):
                    try:
                        new_edf, new_spec = DashboardGenerator.generate(
                            df, profile, llm_client=_tmp, llm_model=ollama_model
                        )
                        st.session_state.enriched_df    = new_edf
                        st.session_state.dashboard_spec = new_spec
                        st.session_state.llm_enhanced   = True
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ollama error: {exc}")
                        st.caption("Is Ollama running? Start it with: ollama serve")

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
