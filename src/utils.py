"""Shared helpers: paths, Streamlit session state, styling and export utilities."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_CSV = DATA_DIR / "sample_documents.csv"

PALETTE = {"blue": "#1F5FA8", "green": "#2E8B57", "light": "#EAF3FB", "white": "#FFFFFF"}
COLOR_SEQUENCE = ["#1F5FA8", "#2E8B57", "#4FA3D1", "#7BC47F", "#14456F", "#1E6F45",
                  "#8EC5E8", "#A7D8B0", "#3A7CA5", "#58A574"]

# Keys kept in st.session_state so work survives page switches.
STATE_DEFAULTS: dict[str, Any] = {
    "raw_df": None, "text_column": None, "dataset_name": None, "clean_info": None,
    "pre_config": None, "pre_result": None, "vec_result": None,
    "lda_result": None, "coherence_sweep": None, "interpretations": None,
}


def init_state() -> None:
    """Create all session-state keys with defaults (idempotent)."""
    import streamlit as st
    for key, value in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def setup_page(title: str, icon: str = "📚") -> None:
    """Common page configuration, theme CSS and state initialisation."""
    import streamlit as st
    st.set_page_config(page_title=f"{title} | LDA AML23702", page_icon=icon, layout="wide")
    init_state()
    st.markdown(
        f"""<style>
        h1, h2, h3 {{ color: {PALETTE['blue']}; }}
        [data-testid="stMetric"] {{ background:{PALETTE['light']}; border-left:4px solid {PALETTE['green']};
            padding:0.6rem 0.9rem; border-radius:6px; }}
        [data-testid="stSidebar"] {{ background:{PALETTE['light']}; }}
        </style>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### AML23702 · LDA Topic Modeling")
    st.sidebar.caption("Advance NLP · CO4 · L3")
    ready = [("Dataset", st.session_state.raw_df is not None),
             ("Preprocessed", st.session_state.pre_result is not None),
             ("Vectorized", st.session_state.vec_result is not None),
             ("LDA trained", st.session_state.lda_result is not None)]
    for name, ok in ready:
        st.sidebar.write(("✅ " if ok else "⬜ ") + name)
    st.title(title)


def require(key: str, message: str) -> Any:
    """Stop the page with a helpful message when a prerequisite step is missing."""
    import streamlit as st
    value = st.session_state.get(key)
    if value is None:
        st.warning(message)
        st.stop()
    return value


def df_to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=index).encode("utf-8")


def json_bytes(obj: Any) -> bytes:
    """Serialise a JSON-compatible object to bytes."""
    return json.dumps(obj, indent=2, default=str).encode("utf-8")


def fig_png_bytes(fig) -> bytes:
    """PNG bytes for a Matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    return buf.getvalue()


def plotly_html_bytes(fig) -> bytes:
    """Standalone HTML bytes for a Plotly figure (CDN-loaded plotly.js)."""
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


_STAGES = ["raw_df", "pre_result", "vec_result", "lda_result"]
_DOWNSTREAM = {
    "dataset": ["pre_config", "pre_result", "vec_result", "lda_result", "coherence_sweep", "interpretations"],
    "preprocess": ["vec_result", "lda_result", "coherence_sweep", "interpretations"],
    "vectorize": ["lda_result", "coherence_sweep", "interpretations"],
}


def reset_downstream(stage: str) -> None:
    """Clear session results that depend on a step that was just re-run."""
    import streamlit as st
    for key in _DOWNSTREAM.get(stage, []):
        st.session_state[key] = None
