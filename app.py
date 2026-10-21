"""Project details page. Run with: streamlit run app.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.utils import setup_page

setup_page("Details", "🧭")
ss = st.session_state

st.markdown("**Advanced Natural Language Processing (AML23702)** · Department of AI & ML · CO4 · Bloom's L3")
st.write("This application applies **Latent Dirichlet Allocation (LDA)** to a document collection, "
         "discovers latent topics, and presents the results in one end-to-end workflow.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", len(ss.raw_df) if ss.raw_df is not None else "-")
c2.metric("Vocabulary", len(ss.vec_result.dictionary) if ss.vec_result is not None else "-")
c3.metric("Topics", ss.lda_result.model.num_topics if ss.lda_result is not None else "-")
c4.metric("Coherence (c_v)", f"{ss.lda_result.coherence:.3f}" if ss.lda_result is not None else "-")

st.subheader("Workflow")
st.markdown("""
1. Select the sample collection or upload a CSV/TXT file.
2. Configure preprocessing and LDA parameters.
3. Press **Run complete LDA modelling** on the *LDA Modelling* page.
4. Review preprocessing, the document-term matrix, topics, document assignments, and visualizations.
""")

st.subheader("About the model")
st.info("The primary model is Gensim's `LdaModel`, which uses **online variational Bayes**. "
        "It does **not** use Gibbs sampling (that is used by tools such as MALLET). "
        "See *Theory and Methodology* for the comparison.")
st.caption("Results stay in this browser session while you switch pages. Refreshing the browser tab clears them.")
