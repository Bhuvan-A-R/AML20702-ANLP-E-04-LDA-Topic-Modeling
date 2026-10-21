"""Builders for downloadable deliverables (dictionary, visualization bundle, full results ZIP)."""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from . import visualization as viz
from .lda_model import (LDAResult, document_topic_matrix, dominant_topics, model_zip_bytes,
                        topic_word_matrix, topic_words_frame)
from .preprocessing import PreprocessResult
from .utils import df_to_csv_bytes, fig_png_bytes, json_bytes, plotly_html_bytes
from .vectorization import VectorizationResult, dtm_frame, vocabulary_frame


def dictionary_bytes(res: LDAResult) -> bytes:
    """Serialized Gensim dictionary."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dictionary.gensim"
        res.dictionary.save(str(path))
        return path.read_bytes()


def config_dict(res: LDAResult) -> dict:
    """Model configuration plus key results as a JSON-compatible dict."""
    return {**res.config.to_dict(), "coherence": res.coherence, "perplexity": res.perplexity,
            "train_seconds": res.train_seconds, "vocab_size": len(res.dictionary), "n_documents": len(res.corpus)}


def visualization_zip_bytes(res: LDAResult, sweep: pd.DataFrame | None = None) -> bytes:
    """ZIP of Plotly HTML and Matplotlib PNG files for all charts."""
    files: dict[str, bytes] = {
        "topic_prevalence.html": plotly_html_bytes(viz.topic_prevalence_fig(res)),
        "topic_word_heatmap.html": plotly_html_bytes(viz.topic_word_heatmap_fig(res)),
        "document_topic_heatmap.html": plotly_html_bytes(viz.doc_topic_heatmap_fig(res)),
        "word_frequency.html": plotly_html_bytes(viz.word_frequency_fig(res)),
        "topic_prevalence.png": fig_png_bytes(viz.mpl_topic_prevalence(res)),
        "top_words.png": fig_png_bytes(viz.mpl_top_words_grid(res)),
        "topic_word_heatmap.png": fig_png_bytes(viz.mpl_topic_word_heatmap(res)),
        "document_topic_heatmap.png": fig_png_bytes(viz.mpl_doc_topic_heatmap(res)),
        "word_frequency.png": fig_png_bytes(viz.mpl_word_frequency(res)),
    }
    for k in range(res.model.num_topics):
        files[f"topic_{k}_top_words.html"] = plotly_html_bytes(viz.top_words_fig(res, k))
    if sweep is not None and not sweep.empty:
        files["coherence_comparison.html"] = plotly_html_bytes(viz.coherence_fig(sweep))
        files["coherence_comparison.png"] = fig_png_bytes(viz.mpl_coherence(sweep))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    import matplotlib.pyplot as plt
    plt.close("all")
    return buf.getvalue()


def results_zip_bytes(*, pre: PreprocessResult, vec: VectorizationResult, res: LDAResult,
                      sweep: pd.DataFrame | None = None, report_md: str | None = None) -> bytes:
    """Everything in one archive: CSVs, model, dictionary, config, visualizations, report."""
    theta = document_topic_matrix(res)
    dom = dominant_topics(theta)
    doc_topic = theta.merge(dom, on="doc_id")
    files = {
        "preprocessed_corpus.csv": df_to_csv_bytes(pre.to_frame()),
        "vocabulary.csv": df_to_csv_bytes(vocabulary_frame(vec)),
        "document_term_matrix.csv": df_to_csv_bytes(dtm_frame(vec), index=True),
        "document_topic_distribution.csv": df_to_csv_bytes(doc_topic),
        "topic_word_probabilities.csv": df_to_csv_bytes(topic_words_frame(res, 15)),
        "topic_word_matrix_phi.csv": df_to_csv_bytes(topic_word_matrix(res), index=True),
        "model_config.json": json_bytes(config_dict(res)),
        "dictionary.gensim": dictionary_bytes(res),
        "lda_model.zip": model_zip_bytes(res),
        "visualizations.zip": visualization_zip_bytes(res, sweep),
    }
    if report_md:
        files["topic_interpretation_report.md"] = report_md.encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()
