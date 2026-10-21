import pandas as pd
import pytest

from src.data_loader import DatasetError, clean_dataset, load_csv, load_sample, load_txt
from src.preprocessing import (PreprocessConfig, apply_df_filter, clean_text, run_preprocessing, tokenize)


def test_sample_dataset_size_and_themes():
    df = load_sample()
    assert len(df) >= 40 and df["theme"].nunique() == 6
    assert {"doc_id", "theme", "text"} <= set(df.columns)


def test_clean_text_removes_urls_html_digits_punctuation():
    out = clean_text("Visit https://x.org <b>NOW</b>!!! In 2024, AI-based tools: 100%.")
    assert "http" not in out and "<" not in out and "2024" not in out and "!" not in out
    assert out == out.lower()


def test_tokenize_and_filters():
    res = run_preprocessing(["The cats are running to 123 big houses", "An AI at it"],
                            PreprocessConfig(lemmatize=False))
    assert "the" not in res.tokens[0] and "123" not in res.tokens[0]
    assert all(len(t) >= 3 for toks in res.tokens for t in toks)
    assert res.n_empty_removed == 1  # second document is empty after filtering


def test_custom_stopwords_and_lemmatization():
    res = run_preprocessing(["models training models", "model data"],
                            PreprocessConfig(lemmatize=True, custom_stopwords=["data"]))
    flat = [t for toks in res.tokens for t in toks]
    assert "data" not in flat
    assert "models" not in flat or res.warnings  # lemmatized unless WordNet is unavailable


def test_df_filter():
    toks = [["a1", "common", "x"], ["common", "y"], ["common", "x"]]
    out = apply_df_filter(toks, min_df=2, max_df=0.9)
    assert out == [["x"], [], ["x"]]


def test_pipeline_on_sample(pre, sample_df):
    assert len(pre.tokens) == len(pre.doc_ids) == len(pre.original) <= len(sample_df)
    assert pre.vocab_after <= pre.vocab_before and pre.vocab_after > 0


def test_all_empty_raises():
    with pytest.raises(ValueError):
        run_preprocessing(["the and of", "12 34"], PreprocessConfig())


def test_invalid_config():
    with pytest.raises(ValueError):
        PreprocessConfig(max_df=0).validate()


def test_clean_dataset_missing_and_duplicates():
    df = pd.DataFrame({"text": ["hello world", "hello world", None, "  ", "other doc"], "x": range(5)})
    out, info = clean_dataset(df, "text")
    assert info.duplicates_removed == 1 and info.missing_removed == 2 and len(out) == 2
    assert list(out["doc_id"]) == [0, 1]


def test_clean_dataset_keeps_existing_doc_id():
    out, _ = clean_dataset(pd.DataFrame({"doc_id": [7, 8], "text": ["a b", "c d"]}), "text")
    assert "original_doc_id" in out.columns and list(out.doc_id) == [0, 1]


def test_load_csv_txt_and_errors():
    assert len(load_csv(b"id,text\n1,hello there\n2,second doc\n")) == 2
    docs = load_txt(b"First doc line one\nline two\n\nSecond doc\n\n\nThird doc")
    assert len(docs) == 3 and "line two" in docs.text[0]
    with pytest.raises(DatasetError):
        load_csv(b"")
    with pytest.raises(DatasetError):
        load_txt(b"\n\n  \n")
    with pytest.raises(DatasetError):
        clean_dataset(pd.DataFrame({"text": [None, None]}), "text")
