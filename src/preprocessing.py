"""Text preprocessing pipeline: cleaning, tokenization, stopwords, lemmatization, DF filtering."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

import pandas as pd

NLTK_HELP = (
    "NLTK resources could not be downloaded (no internet or blocked network). Fix: connect to the "
    "internet and run  python -m nltk.downloader stopwords wordnet omw-1.4  . Offline: download the "
    "nltk_data folder on another machine and copy it to ~/nltk_data. The app continues with a built-in "
    "scikit-learn stopword list (and without lemmatization) until the resources are available."
)

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_HTML = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")  # drops punctuation, digits and special characters
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z]+")


@dataclass
class PreprocessConfig:
    """User-configurable preprocessing settings."""
    lemmatize: bool = True
    min_token_len: int = 3
    custom_stopwords: list[str] = field(default_factory=list)
    min_df: int = 1          # absolute doc count
    max_df: float = 1.0      # fraction of documents

    def validate(self) -> None:
        if self.min_token_len < 1:
            raise ValueError("Minimum token length must be at least 1.")
        if self.min_df < 1:
            raise ValueError("min_df must be >= 1.")
        if not 0 < self.max_df <= 1:
            raise ValueError("max_df must be in (0, 1].")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreprocessResult:
    """Output of the preprocessing pipeline (aligned lists, one entry per kept document)."""
    doc_ids: list[int]
    original: list[str]
    cleaned: list[str]
    tokens: list[list[str]]
    n_input: int
    n_empty_removed: int
    vocab_before: int
    vocab_after: int
    warnings: list[str]
    config: PreprocessConfig

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"doc_id": self.doc_ids, "original_text": self.original,
                             "cleaned_text": self.cleaned,
                             "tokens": [" ".join(t) for t in self.tokens],
                             "n_tokens": [len(t) for t in self.tokens]})


def load_nltk_resources(need_lemmatizer: bool) -> tuple[set[str], object | None, list[str]]:
    """Return (stopwords, lemmatizer-or-None, warnings), downloading NLTK data if needed."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    warnings: list[str] = []
    stop: set[str]
    try:
        try:
            stop = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stop = set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop = set(ENGLISH_STOP_WORDS)
        warnings.append("NLTK stopwords unavailable. " + NLTK_HELP)
    lemmatizer = None
    if need_lemmatizer:
        try:
            lem = WordNetLemmatizer()
            try:
                lem.lemmatize("tests")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                lem.lemmatize("tests")
            lemmatizer = lem
        except Exception:
            warnings.append("NLTK WordNet unavailable; lemmatization skipped. " + NLTK_HELP)
    return stop, lemmatizer, warnings


def clean_text(text: str) -> str:
    """Lowercase; strip URLs, HTML tags, punctuation, digits and special characters."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(cleaned: str) -> list[str]:
    """Regex tokenization of already-cleaned text (needs no NLTK data)."""
    return _TOKEN.findall(cleaned)


def preprocess_tokens(cleaned: str, stop: set[str], lemmatizer, min_len: int) -> list[str]:
    """Tokenize, drop stopwords/short/numeric tokens, optionally lemmatize."""
    out: list[str] = []
    for tok in tokenize(cleaned):
        if tok.isdigit() or tok in stop:
            continue
        if lemmatizer is not None:
            tok = lemmatizer.lemmatize(tok)
        if len(tok) < min_len or tok in stop:
            continue
        out.append(tok)
    return out


def apply_df_filter(token_lists: list[list[str]], min_df: int, max_df: float) -> list[list[str]]:
    """Remove tokens whose document frequency is outside [min_df, max_df * N]."""
    n = len(token_lists)
    if n == 0 or (min_df <= 1 and max_df >= 1.0):
        return token_lists
    df_counts = Counter(t for toks in token_lists for t in set(toks))
    max_abs = max_df * n
    keep = {t for t, c in df_counts.items() if c >= min_df and c <= max_abs}
    return [[t for t in toks if t in keep] for toks in token_lists]


def run_preprocessing(texts: list[str], config: PreprocessConfig | None = None,
                      doc_ids: list[int] | None = None, progress=None) -> PreprocessResult:
    """Run the full pipeline. `progress` is an optional callable(fraction)."""
    config = config or PreprocessConfig()
    config.validate()
    if not texts:
        raise ValueError("No documents to preprocess.")
    doc_ids = list(doc_ids) if doc_ids is not None else list(range(len(texts)))
    stop, lemmatizer, warnings = load_nltk_resources(config.lemmatize)
    stop |= {w.strip().lower() for w in config.custom_stopwords if w.strip()}

    cleaned_all, token_all = [], []
    n = len(texts)
    for i, text in enumerate(texts):
        cleaned = clean_text(text)
        cleaned_all.append(cleaned)
        token_all.append(preprocess_tokens(cleaned, stop, lemmatizer, config.min_token_len))
        if progress and (i % 10 == 0 or i == n - 1):
            progress((i + 1) / n)

    vocab_before = len({t for toks in token_all for t in toks})
    token_all = apply_df_filter(token_all, config.min_df, config.max_df)
    keep_idx = [i for i, toks in enumerate(token_all) if toks]
    n_empty = n - len(keep_idx)
    if not keep_idx:
        raise ValueError("All documents are empty after preprocessing. Relax the filters "
                         "(lower min_df, raise max_df, reduce custom stopwords).")
    tokens = [token_all[i] for i in keep_idx]
    vocab_after = len({t for toks in tokens for t in toks})
    cleaned_kept = [" ".join(t) for t in tokens]
    return PreprocessResult(
        doc_ids=[doc_ids[i] for i in keep_idx], original=[texts[i] for i in keep_idx],
        cleaned=cleaned_kept, tokens=tokens, n_input=n, n_empty_removed=n_empty,
        vocab_before=vocab_before, vocab_after=vocab_after, warnings=warnings, config=config)
