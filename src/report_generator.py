"""Markdown/TXT academic report built from the *actual* results of the trained model."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .lda_model import LDAResult, dominant_topics
from .preprocessing import PreprocessResult
from .vectorization import VectorizationResult

REFERENCES = [
    "Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.",
    "Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. PNAS, 101(suppl. 1), 5228-5235.",
    "Hoffman, M., Bach, F., & Blei, D. (2010). Online learning for Latent Dirichlet Allocation. NeurIPS 23.",
    "Roder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. WSDM '15.",
    "Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C., & Blei, D. (2009). Reading tea leaves: How humans interpret topic models. NeurIPS 22.",
    "Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. LREC Workshop on New Challenges for NLP Frameworks.",
    "McCallum, A. K. (2002). MALLET: A machine learning for language toolkit. http://mallet.cs.umass.edu",
]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    if df.empty:
        return "_(no rows)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return "\n".join(lines)


def build_report(*, title: str, author: str, dataset_name: str, n_raw: int, pre: PreprocessResult,
                 vec: VectorizationResult, res: LDAResult, topic_words: pd.DataFrame,
                 interpretations: dict[int, str], theta: pd.DataFrame,
                 sweep: pd.DataFrame | None = None, doc_texts: dict[int, str] | None = None) -> str:
    """Compose the full 19-section report using real values from the session."""
    cfg, k = res.config, res.model.num_topics
    dom = dominant_topics(theta)
    prevalence = dom["dominant_topic"].value_counts().reindex(range(k), fill_value=0)
    mean_conf = float(dom["dominant_probability"].mean())
    low_conf = int((dom["dominant_probability"] < 0.5).sum())
    tw_rows = []
    for t in range(k):
        words = topic_words[topic_words.topic == t].sort_values("rank")
        tw_rows.append({"Topic": t, "Top words (probability)": ", ".join(f"{w} ({p:.3f})" for w, p in zip(words.word, words.probability)),
                        "Docs (dominant)": int(prevalence[t])})
    interp = "\n".join(f"- **Topic {t}:** {txt}" for t, txt in interpretations.items())
    dtm_cells = vec.shape[0] * vec.shape[1]
    nnz = sum(len(b) for b in vec.corpus)
    sweep_txt = ("Coherence was computed for several topic counts:\n\n" + md_table(sweep) if sweep is not None and not sweep.empty
                 else "No coherence sweep was run in this session; only the coherence of the chosen model is reported.")
    best = ""
    if sweep is not None and not sweep.empty and sweep["coherence"].notna().any():
        row = sweep.loc[sweep["coherence"].idxmax()]
        best = f"\n\nThe highest coherence in the sweep was {row['coherence']:.4f} at K = {int(row['num_topics'])}. Coherence is a guide, not proof of quality."
    examples = ""
    if doc_texts:
        ex = dom.sort_values("dominant_probability", ascending=False).head(3)
        examples = "\n".join(f"- Doc {int(r.doc_id)} -> Topic {int(r.dominant_topic)} (p = {r.dominant_probability:.2f}): "
                             f"\"{doc_texts.get(int(r.doc_id), '')[:140]}...\"" for r in ex.itertuples())
    return f"""# {title}

**Course:** Advance Natural Language Processing (AML23702) | **Department:** Artificial Intelligence and Machine Learning
**Exercise:** Latent Dirichlet Allocation for topic modeling | **CO:** CO4 | **Bloom's level:** L3
**Author:** {author or 'N/A'} | **Date:** {date.today().isoformat()}

## 1. Abstract
This report applies Latent Dirichlet Allocation (LDA) to the dataset "{dataset_name}" ({pre.n_input} input documents, {len(vec.corpus)} used for modeling). After preprocessing, a vocabulary of {len(vec.dictionary)} terms was modeled with {k} topics using Gensim's LdaModel. The model reached a c_v coherence of {res.coherence:.4f} and trained in {res.train_seconds:.2f} s. All numbers below come from the actual run.

## 2. Introduction
Topic models discover latent thematic structure in unlabeled text. LDA treats every document as a mixture of topics and every topic as a probability distribution over words.

## 3. Objectives
- Convert raw documents into a document-term representation.
- Train an LDA model and inspect topic-word and document-topic distributions.
- Evaluate topics with coherence and perplexity and discuss their limitations.
- Relate the results to the probabilistic and sampling-based theory behind LDA.

## 4. Dataset Description
- Source: {dataset_name}
- Rows loaded: {n_raw}; documents entering preprocessing: {pre.n_input}
- Documents removed as empty after preprocessing: {pre.n_empty_removed}
- Documents in the final corpus: {len(vec.corpus)}

## 5. Text Preprocessing Methodology
Lowercasing; removal of URLs, HTML tags, punctuation and digits; regex tokenization; NLTK English stopword removal; removal of numeric tokens and tokens shorter than {pre.config.min_token_len} characters; lemmatization: {'on (WordNet)' if pre.config.lemmatize else 'off'}; custom stopwords: {', '.join(pre.config.custom_stopwords) or 'none'}; document-frequency filter: min_df = {pre.config.min_df}, max_df = {pre.config.max_df}.
Vocabulary size: {pre.vocab_before} (before DF filtering) -> {pre.vocab_after} (after).{(' Warnings: ' + ' '.join(pre.warnings)) if pre.warnings else ''}

## 6. Document-Term Representation
A Gensim `Dictionary` and Bag-of-Words corpus were built (min_df = {vec.min_df}, max_df = {vec.max_df}). The matrix has {vec.shape[0]} documents x {vec.shape[1]} terms with {nnz} non-zero entries (sparsity {1 - nnz / dtm_cells:.2%}). Bag-of-Words counts, rather than TF-IDF, are used because LDA models integer word counts.

## 7. LDA Model Architecture and Parameters
| Parameter | Value |
|---|---|
| Number of topics (K) | {k} |
| Passes | {cfg.passes} |
| Iterations | {cfg.iterations} |
| Random seed | {cfg.random_seed} |
| Alpha | {cfg.alpha} |
| Eta | {cfg.eta} |
| Chunk size | {cfg.chunksize} |
| Min word probability (display) | {cfg.min_word_prob} |

## 8. Probabilistic Topic Modeling Methodology
For each document d: draw theta_d ~ Dirichlet(alpha); for each word position draw a topic z ~ Categorical(theta_d), then a word w ~ Categorical(phi_z), where phi_k ~ Dirichlet(eta). Training estimates the posterior over theta, phi and z given the observed words.

## 9. Sampling-Based Inference Explanation
The primary model, Gensim `LdaModel`, uses **online variational Bayes**, a deterministic optimisation of an approximate posterior, **not Gibbs sampling**. Collapsed Gibbs sampling (e.g. in MALLET) instead resamples each word's topic assignment from its conditional distribution and averages over samples. Both approximate the same intractable posterior and can give different topics. No Gibbs sampler was used in this run.

## 10. Experimental Results
- Coherence ({cfg.coherence_measure}): **{res.coherence:.4f}**
- Perplexity (2^-bound, on training data): **{res.perplexity:.2f}** (per-word bound {res.log_perplexity_bound:.4f})
- Training time: **{res.train_seconds:.2f} s**
- Mean dominant-topic probability: **{mean_conf:.3f}**; documents with dominant probability < 0.5: **{low_conf}** of {len(dom)}

## 11. Topic-Word Tables
{md_table(pd.DataFrame(tw_rows))}

## 12. Document-Topic Analysis
Topic prevalence (number of documents whose dominant topic is each topic):

{md_table(pd.DataFrame({'Topic': prevalence.index, 'Documents': prevalence.values, 'Share': (prevalence.values / max(1, len(dom)))}), '{:.2%}')}

Most confidently assigned documents:
{examples or '_(document texts not supplied)_'}

A document can contain several topics; the dominant topic is only the one with the highest estimated probability.

## 13. Topic Coherence Results
{sweep_txt}{best}

## 14. Visualization Findings
The Visualizations page provides topic prevalence, top-word bars, topic-word and document-topic heatmaps, word frequencies and (optionally) pyLDAvis. In this run the largest topic is Topic {int(prevalence.idxmax())} ({int(prevalence.max())} documents) and the smallest is Topic {int(prevalence.idxmin())} ({int(prevalence.min())} documents).

## 15. Discussion and Interpretation
Automatically generated, keyword-based interpretations (heuristic suggestions, **not** validated labels):

{interp}

Topics are statistical word co-occurrence patterns. A human must read the top words and representative documents before naming a topic.

## 16. Limitations
- Short documents give little co-occurrence evidence; small corpora give unstable topics.
- K must be chosen by the user and coherence/perplexity only partly reflect human judgement.
- Bag-of-Words ignores word order; results depend on vocabulary and preprocessing choices.
- Overlapping themes may be merged or split across topics; results vary with the random seed.

## 17. Conclusion
LDA with K = {k} produced {k} topics with c_v coherence {res.coherence:.4f} on {len(vec.corpus)} documents. The topic-word and document-topic distributions show how probabilistic mixtures summarise the collection, subject to human validation.

## 18. References
""" + "\n".join(f"{i}. {r}" for i, r in enumerate(REFERENCES, 1)) + "\n"


def sample_report_structure() -> str:
    """Skeleton of the report that is filled with real values after training."""
    heads = ["Abstract", "Introduction", "Objectives", "Dataset Description", "Text Preprocessing Methodology",
             "Document-Term Representation", "LDA Model Architecture and Parameters",
             "Probabilistic Topic Modeling Methodology", "Sampling-Based Inference Explanation",
             "Experimental Results", "Topic-Word Tables", "Document-Topic Analysis", "Topic Coherence Results",
             "Visualization Findings", "Discussion and Interpretation", "Limitations", "Conclusion", "References"]
    body = "\n\n".join(f"## {i}. {h}\n_<generated from the trained model>_" for i, h in enumerate(heads, 1))
    return ("# <Report title>\n\n**Course / Exercise / CO / Author / Date** (title page block)\n\n" + body +
            "\n\n> Sections are filled automatically with actual results (K, coherence, perplexity, topic words, "
            "prevalence, preprocessing statistics) once a model has been trained.\n")
