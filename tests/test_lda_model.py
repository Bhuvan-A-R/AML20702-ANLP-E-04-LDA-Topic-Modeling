import json
import zipfile
import io

import numpy as np
import pandas as pd
import pytest

from src.exports import results_zip_bytes
from src.lda_model import (LDAConfig, coherence_sweep, document_topic_matrix, dominant_topics, infer_document,
                           load_model, model_zip_bytes, save_model, topic_word_matrix, topic_words_frame, train_lda)
from src.report_generator import build_report, sample_report_structure
from src.topic_interpretation import interpret_all, interpret_topic


@pytest.fixture(scope="module")
def res(vec):
    return train_lda(vec, LDAConfig(num_topics=4, passes=10, iterations=100))


def test_training_outputs(res, vec):
    assert res.model.num_topics == 4 and res.train_seconds > 0
    assert np.isfinite(res.perplexity) and -1 <= res.coherence <= 1


def test_topic_word_distributions_sum_to_one(res):
    phi = topic_word_matrix(res)
    assert phi.shape[0] == 4 and np.allclose(phi.sum(axis=1), 1.0, atol=1e-5)
    tw = topic_words_frame(res, 10)
    assert list(tw.columns) == ["topic", "rank", "word", "probability"] and len(tw) == 40


def test_every_document_has_valid_distribution(res, vec):
    theta = document_topic_matrix(res)
    assert len(theta) == len(vec.corpus)
    assert np.allclose(theta.drop(columns="doc_id").sum(axis=1), 1.0, atol=1e-4)
    dom = dominant_topics(theta)
    assert dom.dominant_topic.between(0, 3).all() and dom.dominant_probability.between(0, 1).all()


def test_reproducible_with_seed(vec):
    a = train_lda(vec, LDAConfig(num_topics=3, passes=5)).model.get_topics()
    b = train_lda(vec, LDAConfig(num_topics=3, passes=5)).model.get_topics()
    assert np.allclose(a, b)


@pytest.mark.parametrize("kwargs", [dict(num_topics=1), dict(num_topics=500), dict(passes=0), dict(alpha="bogus"),
                                    dict(alpha=-1.0), dict(eta="asymmetric"), dict(min_word_prob=1.5)])
def test_invalid_config_rejected(vec, kwargs):
    with pytest.raises(ValueError):
        train_lda(vec, LDAConfig(**kwargs))


def test_custom_priors_train(vec):
    r = train_lda(vec, LDAConfig(num_topics=3, passes=3, alpha=0.1, eta=0.01))
    assert r.model.num_topics == 3


def test_save_load_and_inference(res, tmp_path):
    save_model(res, tmp_path)
    model, dictionary, cfg = load_model(tmp_path)
    assert cfg["num_topics"] == 4 and len(dictionary) == len(res.dictionary)
    dist = infer_document(model, dictionary, ["patient", "doctor", "hospital", "treatment"])
    assert len(dist) == 4 and abs(sum(dist) - 1) < 1e-4
    with pytest.raises(ValueError):
        infer_document(model, dictionary, ["zzzunknownzzz"])
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing")


def test_model_zip_reloadable(res, tmp_path):
    with zipfile.ZipFile(io.BytesIO(model_zip_bytes(res))) as zf:
        zf.extractall(tmp_path)
    model, _, _ = load_model(tmp_path)
    assert model.num_topics == 4


def test_coherence_sweep(vec):
    sw = coherence_sweep(vec, LDAConfig(passes=3), range(2, 4))
    assert list(sw.num_topics) == [2, 3] and {"coherence", "perplexity"} <= set(sw.columns)


def test_interpretation_is_hedged():
    text = interpret_topic(["patient", "doctor", "hospital", "treatment"])
    assert "healthcare" in text and "human validation" in text
    assert "No predefined theme" in interpret_topic(["zzz", "qqq"])


def test_report_uses_real_results_and_exports(res, vec, pre):
    tw = topic_words_frame(res, 15)
    theta = document_topic_matrix(res)
    interp = interpret_all({t: list(tw[tw.topic == t].word) for t in range(4)})
    md = build_report(title="T", author="A", dataset_name="sample", n_raw=54, pre=pre, vec=vec, res=res,
                      topic_words=tw, interpretations=interp, theta=theta)
    assert f"{res.coherence:.4f}" in md and "online variational Bayes" in md and "not Gibbs sampling" in md
    for i in range(1, 19):
        assert f"## {i}. " in md
    assert "## 18. References" in md and "Blei" in md
    assert "Abstract" in sample_report_structure()
    with zipfile.ZipFile(io.BytesIO(results_zip_bytes(pre=pre, vec=vec, res=res, report_md=md))) as zf:
        names = set(zf.namelist())
        assert {"preprocessed_corpus.csv", "vocabulary.csv", "document_term_matrix.csv",
                "document_topic_distribution.csv", "topic_word_probabilities.csv", "model_config.json",
                "dictionary.gensim", "lda_model.zip", "visualizations.zip",
                "topic_interpretation_report.md"} <= names
        dt = pd.read_csv(zf.open("document_topic_distribution.csv"))
        assert {"doc_id", "topic_0", "dominant_topic", "dominant_probability"} <= set(dt.columns)
        assert json.loads(zf.read("model_config.json"))["num_topics"] == 4
