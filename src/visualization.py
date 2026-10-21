"""Plotly (interactive, HTML-exportable) and Matplotlib (PNG-exportable) visualizations, plus pyLDAvis."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .lda_model import LDAResult, document_topic_matrix, dominant_topics, topic_word_matrix, topic_words_frame
from .utils import COLOR_SEQUENCE

BLUE_GREEN = [[0, "#FFFFFF"], [0.5, "#4FA3D1"], [1, "#1E6F45"]]


def _layout(fig: go.Figure, title: str, height: int = 450) -> go.Figure:
    fig.update_layout(title=title, height=height, template="plotly_white", autosize=True,
                      margin=dict(l=40, r=20, t=60, b=40), font=dict(size=13))
    return fig


# ----------------------------- Plotly figures -----------------------------
def topic_prevalence_fig(res: LDAResult) -> go.Figure:
    dom = dominant_topics(document_topic_matrix(res))
    counts = dom["dominant_topic"].value_counts().reindex(range(res.model.num_topics), fill_value=0)
    fig = go.Figure(go.Bar(x=[f"Topic {k}" for k in counts.index], y=counts.values,
                           marker_color=[COLOR_SEQUENCE[k % 10] for k in counts.index], text=counts.values))
    fig.update_yaxes(title="Documents (dominant topic)")
    return _layout(fig, "Topic prevalence across the collection")


def top_words_fig(res: LDAResult, topic: int, topn: int = 10) -> go.Figure:
    words = res.model.show_topic(topic, topn=topn)[::-1]
    fig = go.Figure(go.Bar(x=[p for _, p in words], y=[w for w, _ in words], orientation="h",
                           marker_color=COLOR_SEQUENCE[topic % 10]))
    fig.update_xaxes(title="P(word | topic)")
    return _layout(fig, f"Topic {topic}: top {topn} words", 400)


def topic_word_heatmap_fig(res: LDAResult, topn: int = 8) -> go.Figure:
    phi = topic_word_matrix(res)
    words = list(dict.fromkeys(w for k in range(len(phi)) for w, _ in res.model.show_topic(k, topn)))
    fig = px.imshow(phi[words], color_continuous_scale=BLUE_GREEN, aspect="auto",
                    labels=dict(x="Word", y="Topic", color="P(w|t)"))
    return _layout(fig, "Topic-word probability heatmap (union of top words)", 450)


def doc_topic_heatmap_fig(res: LDAResult, max_docs: int = 60) -> go.Figure:
    theta = document_topic_matrix(res).head(max_docs)
    z = theta.drop(columns="doc_id")
    fig = px.imshow(z.to_numpy(), x=[f"Topic {k}" for k in range(z.shape[1])],
                    y=[f"Doc {i}" for i in theta["doc_id"]], color_continuous_scale=BLUE_GREEN,
                    aspect="auto", labels=dict(x="Topic", y="Document", color="P(t|d)"))
    return _layout(fig, f"Document-topic probability heatmap (first {len(theta)} documents)", 700)


def word_frequency_fig(res: LDAResult, n: int = 20) -> go.Figure:
    d = res.dictionary
    top = sorted(d.cfs.items(), key=lambda kv: kv[1], reverse=True)[:n][::-1]
    fig = go.Figure(go.Bar(x=[c for _, c in top], y=[d[i] for i, _ in top], orientation="h",
                           marker_color="#1F5FA8"))
    fig.update_xaxes(title="Corpus frequency")
    return _layout(fig, f"Top {n} most frequent words", 520)


def coherence_fig(sweep: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=sweep["num_topics"], y=sweep["coherence"], mode="lines+markers",
                               line=dict(color="#2E8B57"), marker=dict(size=9)))
    fig.update_xaxes(title="Number of topics", dtick=1)
    fig.update_yaxes(title="Coherence (c_v)")
    return _layout(fig, "Topic coherence vs. number of topics")


# ---------------------------- Matplotlib figures ----------------------------
def mpl_topic_prevalence(res: LDAResult):
    dom = dominant_topics(document_topic_matrix(res))
    counts = dom["dominant_topic"].value_counts().reindex(range(res.model.num_topics), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"T{k}" for k in counts.index], counts.values, color=[COLOR_SEQUENCE[k % 10] for k in counts.index])
    ax.set(title="Topic prevalence (dominant topic)", xlabel="Topic", ylabel="Documents")
    return fig


def mpl_top_words_grid(res: LDAResult, topn: int = 10):
    k = res.model.num_topics
    cols = min(3, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    for t in range(rows * cols):
        ax = axes[t // cols][t % cols]
        if t >= k:
            ax.axis("off")
            continue
        w = res.model.show_topic(t, topn)[::-1]
        ax.barh([a for a, _ in w], [b for _, b in w], color=COLOR_SEQUENCE[t % 10])
        ax.set_title(f"Topic {t}")
    fig.suptitle("Top words per topic", y=1.0)
    fig.tight_layout()
    return fig


def mpl_topic_word_heatmap(res: LDAResult, topn: int = 8):
    phi = topic_word_matrix(res)
    words = list(dict.fromkeys(w for t in range(len(phi)) for w, _ in res.model.show_topic(t, topn)))
    fig, ax = plt.subplots(figsize=(max(8, len(words) * 0.28), 0.6 * len(phi) + 2))
    im = ax.imshow(phi[words].to_numpy(), aspect="auto", cmap="GnBu")
    ax.set_xticks(range(len(words)), words, rotation=90)
    ax.set_yticks(range(len(phi)), phi.index)
    fig.colorbar(im, label="P(word | topic)")
    ax.set_title("Topic-word probability heatmap")
    return fig


def mpl_doc_topic_heatmap(res: LDAResult, max_docs: int = 60):
    theta = document_topic_matrix(res).head(max_docs)
    z = theta.drop(columns="doc_id").to_numpy()
    fig, ax = plt.subplots(figsize=(6, max(4, len(z) * 0.18)))
    im = ax.imshow(z, aspect="auto", cmap="GnBu")
    ax.set_xticks(range(z.shape[1]), [f"T{k}" for k in range(z.shape[1])])
    ax.set_yticks(range(len(z)), theta["doc_id"].astype(str), fontsize=6)
    fig.colorbar(im, label="P(topic | doc)")
    ax.set_title("Document-topic probability heatmap")
    return fig


def mpl_word_frequency(res: LDAResult, n: int = 20):
    d = res.dictionary
    top = sorted(d.cfs.items(), key=lambda kv: kv[1], reverse=True)[:n][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([d[i] for i, _ in top], [c for _, c in top], color="#1F5FA8")
    ax.set(title=f"Top {n} words", xlabel="Corpus frequency")
    return fig


def mpl_coherence(sweep: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["num_topics"], sweep["coherence"], "o-", color="#2E8B57")
    ax.set(title="Coherence vs. number of topics", xlabel="Topics", ylabel="c_v coherence")
    return fig


# ------------------------------- pyLDAvis -------------------------------
def pyldavis_html(res: LDAResult) -> str:
    """Interactive pyLDAvis page as an HTML string. Raises RuntimeError on failure."""
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = gm.prepare(res.model, res.corpus, res.dictionary, sort_topics=False)
        return pyLDAvis.prepared_data_to_html(data)
    except Exception as exc:
        raise RuntimeError(f"pyLDAvis could not be generated: {exc}") from exc
