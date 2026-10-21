"""Gensim LDA training, evaluation, inference, persistence and coherence sweeps.

Note: gensim.models.LdaModel uses *online variational Bayes* (Hoffman et al., 2010),
not Gibbs sampling. See the Theory page for the distinction.
"""
from __future__ import annotations

import io
import json
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

from .vectorization import VectorizationResult

AlphaEta = Union[str, float]


@dataclass
class LDAConfig:
    """Training hyper-parameters (validated by `validate`)."""
    num_topics: int = 6
    passes: int = 20
    iterations: int = 200
    random_seed: int = 42
    alpha: AlphaEta = "symmetric"     # 'symmetric' | 'asymmetric' | 'auto' | positive float
    eta: AlphaEta = "symmetric"       # 'symmetric' | 'auto' | positive float
    min_word_prob: float = 0.0        # display threshold for topic words
    chunksize: int = 2000
    coherence_measure: str = "c_v"

    def validate(self, n_docs: int, vocab_size: int) -> list[str]:
        """Raise ValueError on invalid input; return non-fatal warnings."""
        warnings: list[str] = []
        if not 2 <= self.num_topics <= 50:
            raise ValueError("Number of topics must be between 2 and 50.")
        if self.num_topics > n_docs:
            raise ValueError(f"Number of topics ({self.num_topics}) exceeds the number of documents ({n_docs}).")
        if self.num_topics > vocab_size:
            raise ValueError(f"Number of topics ({self.num_topics}) exceeds vocabulary size ({vocab_size}).")
        if self.num_topics > max(2, n_docs // 3):
            warnings.append("Many topics for so few documents: topics may be unstable or redundant.")
        if self.passes < 1 or self.iterations < 1:
            raise ValueError("Passes and iterations must be at least 1.")
        if self.chunksize < 1:
            raise ValueError("Chunk size must be positive.")
        for name, val, allowed in (("alpha", self.alpha, {"symmetric", "asymmetric", "auto"}),
                                   ("eta", self.eta, {"symmetric", "auto"})):
            if isinstance(val, str):
                if val not in allowed:
                    raise ValueError(f"{name} must be a positive number or one of {sorted(allowed)}.")
            elif not float(val) > 0:
                raise ValueError(f"{name} must be strictly positive.")
        if not 0.0 <= self.min_word_prob < 1.0:
            raise ValueError("Minimum word probability must be in [0, 1).")
        return warnings

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LDAResult:
    """Everything produced by one training run."""
    model: LdaModel
    dictionary: Dictionary
    corpus: list
    tokens: list[list[str]]
    doc_ids: list[int]
    config: LDAConfig
    coherence: float
    perplexity: float
    log_perplexity_bound: float
    train_seconds: float
    warnings: list[str]


def _prior(value: AlphaEta) -> AlphaEta:
    """Convert numeric strings from the UI to floats, keep keywords."""
    if isinstance(value, str) and value not in {"symmetric", "asymmetric", "auto"}:
        return float(value)
    return value


def compute_coherence(model: LdaModel, tokens: list[list[str]], dictionary: Dictionary,
                      measure: str = "c_v") -> float:
    """Topic coherence (single process for cross-platform safety)."""
    try:
        cm = CoherenceModel(model=model, texts=tokens, dictionary=dictionary, coherence=measure, processes=1)
        score = float(cm.get_coherence())
        return score if np.isfinite(score) else float("nan")
    except Exception:
        return float("nan")


def train_lda(vec: VectorizationResult, config: LDAConfig) -> LDAResult:
    """Validate inputs, train LdaModel on BoW counts, and evaluate it."""
    warnings = config.validate(len(vec.corpus), len(vec.dictionary))
    start = time.perf_counter()
    try:
        model = LdaModel(corpus=vec.corpus, id2word=vec.dictionary, num_topics=config.num_topics,
                         passes=config.passes, iterations=config.iterations,
                         random_state=config.random_seed, alpha=_prior(config.alpha),
                         eta=_prior(config.eta), chunksize=config.chunksize,
                         eval_every=None)
    except Exception as exc:
        raise RuntimeError(f"LDA training failed: {exc}") from exc
    elapsed = time.perf_counter() - start
    coh = compute_coherence(model, vec.tokens, vec.dictionary, config.coherence_measure)
    bound = float(model.log_perplexity(vec.corpus))
    return LDAResult(model, vec.dictionary, vec.corpus, vec.tokens, vec.doc_ids, config,
                     coh, float(np.exp2(-bound)), bound, elapsed, warnings)


def topic_word_matrix(res: LDAResult) -> pd.DataFrame:
    """phi: topics x vocabulary probability matrix (rows sum to 1)."""
    phi = res.model.get_topics()
    cols = [res.dictionary[i] for i in range(phi.shape[1])]
    return pd.DataFrame(phi, index=[f"Topic {k}" for k in range(phi.shape[0])], columns=cols)


def topic_words_frame(res: LDAResult, topn: int = 15) -> pd.DataFrame:
    """Long-format ranked topic-word table filtered by the minimum probability threshold."""
    rows = []
    for k in range(res.model.num_topics):
        for rank, (word, p) in enumerate(res.model.show_topic(k, topn=topn), start=1):
            if p >= res.config.min_word_prob:
                rows.append({"topic": k, "rank": rank, "word": word, "probability": float(p)})
    return pd.DataFrame(rows, columns=["topic", "rank", "word", "probability"])


def document_topic_matrix(res: LDAResult) -> pd.DataFrame:
    """theta: documents x topics probability matrix (rows sum to ~1)."""
    k = res.model.num_topics
    theta = np.zeros((len(res.corpus), k))
    for i, bow in enumerate(res.corpus):
        for t, p in res.model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, t] = p
    df = pd.DataFrame(theta, columns=[f"topic_{j}" for j in range(k)])
    df.insert(0, "doc_id", res.doc_ids)
    return df


def dominant_topics(theta: pd.DataFrame) -> pd.DataFrame:
    """Dominant topic and its probability for each document."""
    vals = theta.drop(columns="doc_id").to_numpy()
    return pd.DataFrame({"doc_id": theta["doc_id"], "dominant_topic": vals.argmax(axis=1),
                         "dominant_probability": vals.max(axis=1)})


def infer_document(model: LdaModel, dictionary: Dictionary, tokens: list[str]) -> list[float]:
    """Topic distribution for a new tokenized document using a trained model."""
    bow = dictionary.doc2bow(tokens)
    if not bow:
        raise ValueError("None of the tokens appear in the model vocabulary.")
    dist = dict(model.get_document_topics(bow, minimum_probability=0.0))
    return [float(dist.get(k, 0.0)) for k in range(model.num_topics)]


def coherence_sweep(vec: VectorizationResult, base: LDAConfig, topic_range: range,
                    progress=None) -> pd.DataFrame:
    """Train a model for each topic count and record coherence and perplexity."""
    rows = []
    valid = [k for k in topic_range if 2 <= k <= min(len(vec.corpus), len(vec.dictionary))]
    if not valid:
        raise ValueError("Topic range is invalid for this dataset.")
    for i, k in enumerate(valid):
        cfg = LDAConfig(**{**base.to_dict(), "num_topics": k})
        res = train_lda(vec, cfg)
        rows.append({"num_topics": k, "coherence": res.coherence, "perplexity": res.perplexity})
        if progress:
            progress((i + 1) / len(valid))
    return pd.DataFrame(rows)


def save_model(res: LDAResult, directory: str | Path) -> Path:
    """Persist model, dictionary and configuration into `directory`."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    res.model.save(str(d / "lda.model"))
    res.dictionary.save(str(d / "dictionary.gensim"))
    meta = {**res.config.to_dict(), "coherence": res.coherence, "perplexity": res.perplexity,
            "train_seconds": res.train_seconds, "vocab_size": len(res.dictionary), "n_documents": len(res.corpus)}
    (d / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def load_model(directory: str | Path) -> tuple[LdaModel, Dictionary, dict]:
    """Reload a saved model, dictionary and config for future inference."""
    d = Path(directory)
    for name in ("lda.model", "dictionary.gensim", "config.json"):
        if not (d / name).exists():
            raise FileNotFoundError(f"Missing {name} in {d}")
    model = LdaModel.load(str(d / "lda.model"))
    dictionary = Dictionary.load(str(d / "dictionary.gensim"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return model, dictionary, cfg


def model_zip_bytes(res: LDAResult) -> bytes:
    """ZIP archive (model files + dictionary + config) suitable for download and reload."""
    with tempfile.TemporaryDirectory() as tmp:
        save_model(res, tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(Path(tmp).iterdir()):
                zf.write(f, f.name)
        return buf.getvalue()
