"""Single-screen, end-to-end LDA modelling workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import DatasetError, clean_dataset, dataset_stats, guess_text_column, load_csv, load_sample, load_txt
from src.lda_model import LDAConfig, document_topic_matrix, dominant_topics, save_model, topic_words_frame, train_lda
from src.preprocessing import NLTK_HELP, PreprocessConfig, run_preprocessing
from src.topic_interpretation import interpret_all
from src.utils import OUTPUT_DIR, df_to_csv_bytes, reset_downstream, setup_page
from src.vectorization import build_gensim_representation, dtm_frame, sparsity, top_terms
from src.visualization import doc_topic_heatmap_fig, topic_prevalence_fig, topic_word_heatmap_fig, top_words_fig, word_frequency_fig

setup_page("LDA Modelling", "🤖")
ss = st.session_state

st.write("Configure the options below and press the button. Every stage runs in order on this screen.")

source = st.radio(
    "Dataset",
    ["Built-in sample collection", "Upload CSV", "Upload TXT"],
    horizontal=True,
)
raw, dataset_name = None, None
try:
    if source == "Built-in sample collection":
        raw, dataset_name = load_sample(), "sample_documents.csv (built-in)"
    elif source == "Upload CSV":
        upload = st.file_uploader("CSV file with a text column", type=["csv"])
        if upload is not None:
            raw, dataset_name = load_csv(upload), upload.name
    else:
        st.caption("Separate TXT documents with blank lines.")
        upload = st.file_uploader("TXT file", type=["txt"])
        if upload is not None:
            raw, dataset_name = load_txt(upload), upload.name
except DatasetError as exc:
    st.error(f"Could not load the dataset: {exc}")

if raw is None:
    st.info("Choose a dataset to configure the modelling run.")
    st.stop()

try:
    default_column = guess_text_column(raw)
except DatasetError as exc:
    st.error(str(exc))
    st.stop()

text_column = st.selectbox(
    "Text column",
    list(raw.columns),
    index=list(raw.columns).index(default_column),
)

with st.expander("Model settings", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        lemmatize = st.checkbox("Lemmatize with WordNet", value=True)
        min_token_len = st.number_input("Minimum token length", 1, 10, 3)
        min_df = st.number_input("Minimum document frequency", 1, 1000, 1)
        max_df = st.slider("Maximum document frequency", 0.1, 1.0, 1.0, 0.05)
        custom_stopwords = st.text_input("Custom stopwords (comma separated)")
    with c2:
        default_topics = min(6, max(2, len(raw) // 3))
        topics = st.number_input("Number of topics (K)", 2, 50, default_topics)
        passes = st.number_input("Passes", 1, 200, 20)
        iterations = st.number_input("Iterations", 1, 2000, 200)
        seed = st.number_input("Random seed", 0, 2**31 - 1, 42)
        min_word_prob = st.slider("Minimum topic-word probability", 0.0, 0.1, 0.0, 0.005)

run = st.button("Run complete LDA modelling", type="primary", width="stretch")

if run:
    try:
        df, clean_info = clean_dataset(raw, text_column)
        ss.raw_df, ss.text_column, ss.dataset_name, ss.clean_info = df, text_column, dataset_name, clean_info
        reset_downstream("dataset")

        progress = st.progress(0.0, text="1/4 Loading and cleaning the dataset...")
        pre_config = PreprocessConfig(
            lemmatize=lemmatize,
            min_token_len=int(min_token_len),
            custom_stopwords=[word.strip() for word in custom_stopwords.split(",") if word.strip()],
            min_df=int(min_df),
            max_df=float(max_df),
        )
        pre = run_preprocessing(
            df[text_column].tolist(),
            pre_config,
            df["doc_id"].tolist(),
            progress=lambda value: progress.progress(value * 0.25, text="2/4 Preprocessing documents..."),
        )
        ss.pre_config, ss.pre_result = pre_config, pre

        progress.progress(0.5, text="3/4 Building the document-term matrix...")
        vec = build_gensim_representation(pre.tokens, pre.doc_ids)
        ss.vec_result = vec

        config = LDAConfig(
            num_topics=int(topics),
            passes=int(passes),
            iterations=int(iterations),
            random_seed=int(seed),
            min_word_prob=float(min_word_prob),
        )
        progress.progress(0.75, text="4/4 Training and evaluating the LDA model...")
        lda = train_lda(vec, config)
        ss.lda_result = lda
        ss.interpretations = None
        save_model(lda, OUTPUT_DIR / "lda_model")
        progress.progress(1.0, text="Complete LDA modelling finished.")
        st.success("All stages completed successfully. The results are shown below.")
    except (DatasetError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Modelling could not be completed: {exc}")

if ss.pre_result is None or ss.vec_result is None or ss.lda_result is None:
    st.stop()

pre, vec, res = ss.pre_result, ss.vec_result, ss.lda_result
if pre.warnings:
    for warning in pre.warnings:
        st.warning(warning)
    with st.expander("NLTK setup help"):
        st.write(NLTK_HELP)

st.header("Results")
metrics = st.columns(6)
metrics[0].metric("Documents", len(ss.raw_df))
metrics[1].metric("After preprocessing", len(pre.doc_ids))
metrics[2].metric("Vocabulary", len(vec.dictionary))
metrics[3].metric("Topics", res.model.num_topics)
metrics[4].metric("Coherence", "n/a" if res.coherence != res.coherence else f"{res.coherence:.4f}")
metrics[5].metric("Perplexity", f"{res.perplexity:.1f}")

with st.expander("1. Dataset details", expanded=True):
    stats = dataset_stats(ss.raw_df, ss.text_column)
    c = st.columns(4)
    c[0].metric("Rows loaded", ss.clean_info.original_rows)
    c[1].metric("Missing/blank removed", ss.clean_info.missing_removed)
    c[2].metric("Duplicates removed", ss.clean_info.duplicates_removed)
    c[3].metric("Mean words", f"{stats['mean_words']:.1f}")
    st.dataframe(ss.raw_df.head(50), width="stretch")

with st.expander("2. Preprocessing results", expanded=True):
    c = st.columns(4)
    c[0].metric("Documents processed", pre.n_input)
    c[1].metric("Empty documents removed", pre.n_empty_removed)
    c[2].metric("Vocabulary before filter", pre.vocab_before)
    c[3].metric("Vocabulary after filter", pre.vocab_after)
    st.dataframe(pre.to_frame(), width="stretch")
    st.download_button("Download preprocessed corpus", df_to_csv_bytes(pre.to_frame()), "preprocessed_corpus.csv", "text/csv")

with st.expander("3. Document-term matrix", expanded=True):
    dtm = dtm_frame(vec)
    c = st.columns(3)
    c[0].metric("Matrix shape", f"{dtm.shape[0]} x {dtm.shape[1]}")
    c[1].metric("Sparsity", f"{sparsity(dtm):.1%}")
    c[2].metric("Most frequent term", top_terms(vec, 1).iloc[0]["term"])
    st.dataframe(dtm.loc[:, dtm.sum().sort_values(ascending=False).head(25).index].head(15), width="stretch")
    st.dataframe(top_terms(vec, 20), width="stretch")
    st.download_button("Download document-term matrix", df_to_csv_bytes(dtm, index=True), "document_term_matrix.csv", "text/csv")

with st.expander("4. Topics and interpretations", expanded=True):
    topic_words = topic_words_frame(res, 10)
    interpretations = interpret_all({
        topic: list(topic_words[topic_words.topic == topic].sort_values("rank").word)
        for topic in range(res.model.num_topics)
    })
    for topic in range(res.model.num_topics):
        rows = topic_words[topic_words.topic == topic].sort_values("rank")
        st.subheader(f"Topic {topic}: {', '.join(rows.word.head(7))}")
        left, right = st.columns([1, 1.3])
        left.dataframe(rows[["rank", "word", "probability"]], hide_index=True, width="stretch")
        right.plotly_chart(top_words_fig(res, topic, min(10, len(rows))), width="stretch", key=f"topic_words_{topic}")
        st.caption(interpretations[topic])
    st.download_button("Download topic-word probabilities", df_to_csv_bytes(topic_words), "topic_word_probabilities.csv", "text/csv")

with st.expander("5. Document analysis", expanded=True):
    theta = document_topic_matrix(res)
    dominant = dominant_topics(theta)
    original = dict(zip(pre.doc_ids, pre.original))
    table = theta.merge(dominant, on="doc_id")
    table.insert(1, "original_text", table["doc_id"].map(original))
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download document-topic results", df_to_csv_bytes(table), "document_topic_distribution.csv", "text/csv")

with st.expander("6. Visualizations", expanded=True):
    st.plotly_chart(topic_prevalence_fig(res), width="stretch")
    st.plotly_chart(topic_word_heatmap_fig(res), width="stretch")
    st.plotly_chart(doc_topic_heatmap_fig(res), width="stretch")
    st.plotly_chart(word_frequency_fig(res), width="stretch")
