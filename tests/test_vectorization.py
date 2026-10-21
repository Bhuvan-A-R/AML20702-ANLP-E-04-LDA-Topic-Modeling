import numpy as np
import pytest

from src.vectorization import (build_count_vectorizer, build_gensim_representation, dtm_frame, top_terms,
                               vocabulary_frame)


def test_gensim_representation_shapes(vec, pre):
    assert vec.shape[0] == len(vec.doc_ids) <= len(pre.tokens)
    assert vec.shape[1] == len(vec.dictionary) > 0
    assert all(len(bow) > 0 for bow in vec.corpus)


def test_dtm_matches_corpus_counts(vec):
    dtm = dtm_frame(vec)
    assert dtm.shape == vec.shape
    assert int(dtm.to_numpy().sum()) == sum(c for bow in vec.corpus for _, c in bow)


def test_vocabulary_and_top_terms_columns(vec):
    vocab = vocabulary_frame(vec)
    assert list(vocab.columns) == ["term_id", "term", "corpus_frequency", "document_frequency"]
    top = top_terms(vec, 5)
    assert len(top) == 5 and top.corpus_frequency.is_monotonic_decreasing


def test_df_filtering_changes_vocabulary(pre):
    loose = build_gensim_representation(pre.tokens, pre.doc_ids, 1, 1.0)
    strict = build_gensim_representation(pre.tokens, pre.doc_ids, 3, 0.4)
    assert strict.shape[1] < loose.shape[1]


def test_invalid_and_empty_inputs():
    with pytest.raises(ValueError):
        build_gensim_representation([], [])
    with pytest.raises(ValueError):
        build_gensim_representation([["a"]], [0], min_df=0)
    with pytest.raises(ValueError):
        build_gensim_representation([["a"], ["b"]], [0, 1], min_df=5)


def test_count_vectorizer(pre):
    df, cv = build_count_vectorizer(pre.cleaned)
    assert df.shape[0] == len(pre.cleaned) and df.shape[1] == len(cv.get_feature_names_out())
    with pytest.raises(ValueError):
        build_count_vectorizer(["", ""])
