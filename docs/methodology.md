# Project Methodology

1. **Dataset management**: load the bundled CSV, or upload CSV/TXT; choose the text column; drop missing and duplicate documents (`src/data_loader.py`).
2. **Preprocessing** (`src/preprocessing.py`): lowercase, remove URLs/HTML/punctuation/digits, regex tokenization, NLTK stopwords (+ custom), drop numeric and short tokens, optional WordNet lemmatization, optional min/max document-frequency filter. NLTK download failures fall back to scikit-learn's stopword list with instructions.
3. **Document-term representation** (`src/vectorization.py`): Gensim `Dictionary` + Bag-of-Words (primary) with `filter_extremes`; scikit-learn `CountVectorizer` shown as an alternative. Documents that become empty are dropped and `doc_id`s stay aligned.
4. **Training** (`src/lda_model.py`): `gensim.models.LdaModel` (online variational Bayes) with configurable K, passes, iterations, seed, alpha, eta. Inputs are validated (e.g. K must not exceed document count or vocabulary size).
5. **Evaluation**: c_v coherence (`CoherenceModel`, single process), perplexity from `log_perplexity`, training time, optional coherence sweep across K.
6. **Interpretation** (`src/topic_interpretation.py`): ranked topic words with probabilities plus a heuristic, hedged description. Human validation is required.
7. **Analysis & visualization** (`src/visualization.py`): theta table, dominant topics, prevalence, heatmaps, word frequency, coherence curve, optional pyLDAvis.
8. **Reporting** (`src/report_generator.py`): Markdown report filled with the actual session results.
9. **Persistence**: model, dictionary and `config.json` saved/zipped so they can be reloaded with `load_model` for inference.

Reproducibility: fixed random seed (default 42) for LDA; preprocessing is deterministic.
