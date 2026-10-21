"""Document-term representations: Gensim Dictionary/BoW (primary) and scikit-learn CountVectorizer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from gensim import matutils
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer

BOW_VS_TFIDF = """
**Bag-of-Words (BoW)** stores the raw count of each term in each document. **TF-IDF** rescales
counts by `tf(t,d) x log(N / df(t))`, down-weighting terms that occur in many documents.
LDA's generative model *draws word counts from multinomial distributions*, so it expects integer
counts. TF-IDF values are real-valued weights with no such probabilistic interpretation, which is
why this project trains LDA on **BoW**. TF-IDF is better suited to retrieval and similarity tasks.
"""


@dataclass
class VectorizationResult:
    """Gensim dictionary + BoW corpus aligned with the preprocessed documents."""
    dictionary: Dictionary
    corpus: list[list[tuple[int, int]]]
    tokens: list[list[str]]       # tokens restricted to the final dictionary
    doc_ids: list[int]
    min_df: int
    max_df: float
    vocab_before_filter: int

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.corpus), len(self.dictionary)


def build_gensim_representation(tokens: list[list[str]], doc_ids: list[int],
                                min_df: int = 1, max_df: float = 1.0) -> VectorizationResult:
    """Build Dictionary + BoW corpus; drop documents that become empty after filtering."""
    if min_df < 1 or not 0 < max_df <= 1:
        raise ValueError("min_df must be >= 1 and max_df in (0, 1].")
    if not tokens:
        raise ValueError("No tokenized documents supplied.")
    dictionary = Dictionary(tokens)
    before = len(dictionary)
    if before == 0:
        raise ValueError("Empty vocabulary: no tokens available.")
    dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=None)
    dictionary.compactify()
    if len(dictionary) == 0:
        raise ValueError("Vocabulary is empty after DF filtering; relax min_df/max_df.")
    kept_tokens, kept_ids, corpus = [], [], []
    for toks, did in zip(tokens, doc_ids):
        bow = dictionary.doc2bow(toks)
        if bow:
            corpus.append(bow)
            kept_ids.append(did)
            kept_tokens.append([t for t in toks if t in dictionary.token2id])
    if not corpus:
        raise ValueError("All documents are empty after DF filtering.")
    return VectorizationResult(dictionary, corpus, kept_tokens, kept_ids, min_df, max_df, before)


def vocabulary_frame(vec: VectorizationResult) -> pd.DataFrame:
    """Vocabulary with term id, corpus frequency and document frequency."""
    d = vec.dictionary
    rows = [{"term_id": i, "term": t, "corpus_frequency": d.cfs.get(i, 0), "document_frequency": d.dfs.get(i, 0)}
            for t, i in d.token2id.items()]
    return pd.DataFrame(rows).sort_values("corpus_frequency", ascending=False).reset_index(drop=True)


def dtm_frame(vec: VectorizationResult) -> pd.DataFrame:
    """Dense document-term matrix (rows = documents, columns = terms) from the Gensim corpus."""
    n_terms = len(vec.dictionary)
    mat = matutils.corpus2dense(vec.corpus, num_terms=n_terms, num_docs=len(vec.corpus)).T.astype(int)
    cols = [vec.dictionary[i] for i in range(n_terms)]
    return pd.DataFrame(mat, index=[f"doc_{i}" for i in vec.doc_ids], columns=cols)


def top_terms(vec: VectorizationResult, n: int = 20) -> pd.DataFrame:
    """Most frequent terms in the corpus."""
    return vocabulary_frame(vec).head(n)[["term", "corpus_frequency", "document_frequency"]]


def build_count_vectorizer(texts: list[str], min_df: int = 1, max_df: float = 1.0) -> tuple[pd.DataFrame, CountVectorizer]:
    """Alternative representation: scikit-learn CountVectorizer on cleaned text."""
    if not texts:
        raise ValueError("No documents supplied.")
    cv = CountVectorizer(min_df=min_df, max_df=max_df, token_pattern=r"(?u)\b[a-z]{2,}\b")
    try:
        mat = cv.fit_transform(texts)
    except ValueError as exc:
        raise ValueError(f"CountVectorizer failed (empty vocabulary?): {exc}") from exc
    df = pd.DataFrame(mat.toarray(), columns=cv.get_feature_names_out())
    return df, cv


def sparsity(matrix: pd.DataFrame) -> float:
    """Fraction of zero cells in a document-term matrix."""
    arr = matrix.to_numpy()
    return float(1.0 - np.count_nonzero(arr) / arr.size) if arr.size else 0.0
