import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.data_loader import clean_dataset, load_sample
from src.preprocessing import PreprocessConfig, run_preprocessing
from src.vectorization import build_gensim_representation


@pytest.fixture(scope="session")
def sample_df():
    df, _ = clean_dataset(load_sample(), "text")
    return df


@pytest.fixture(scope="session")
def pre(sample_df):
    return run_preprocessing(sample_df["text"].tolist(), PreprocessConfig(min_df=2, max_df=0.5),
                             sample_df["doc_id"].tolist())


@pytest.fixture(scope="session")
def vec(pre):
    return build_gensim_representation(pre.tokens, pre.doc_ids)
