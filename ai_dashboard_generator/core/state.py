from typing import Optional
import pandas as pd
from core.schemas import DatasetProfile, DashboardSpec


def init_session_state(st):
    defaults = {
        "df": None,
        "enriched_df": None,
        "profile": None,
        "dashboard_spec": None,
        "uploaded_name": None,
        "uploaded_hash": None,
        "llm_enhanced": False,
        "chat_messages": [],       # list of {"role": "user"|"assistant", "text": str, "charts": list}
        "chat_llm_client": None,   # LLM client for the chatbot (Groq)
        "chat_llm_model": None,    # model name for the chatbot
        "sheet_metadata": {},      # {sheet_name: {rows, columns, dtypes}} for Excel
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_df(st) -> Optional[pd.DataFrame]:
    return st.session_state.get("df")


def get_profile(st) -> Optional[DatasetProfile]:
    return st.session_state.get("profile")


def get_dashboard_spec(st) -> Optional[DashboardSpec]:
    return st.session_state.get("dashboard_spec")
