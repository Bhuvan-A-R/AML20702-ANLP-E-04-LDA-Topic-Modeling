# LDA Topic Modeling: AML23702

An end-to-end Streamlit application that applies **Latent Dirichlet Allocation (LDA)** to a document collection, discovers topics, and explains them with probabilistic theory.

## Academic exercise details
| | |
|---|---|
| Department | Artificial Intelligence and Machine Learning |
| Course | Advance Natural Language Processing (AML23702) |
| Exercise | Apply LDA to a document collection and interpret the discovered topics using probabilistic and sampling-based methods |
| CO / Bloom's level | CO4 / L3 |

> **Inference note:** the primary model is Gensim's `LdaModel`, which uses **online variational Bayes**, *not* Gibbs sampling. Collapsed Gibbs sampling (e.g. MALLET) is explained in the Theory page; no Gibbs sampler is included in this project.

## Features
- **Dataset management**: built-in 54-document sample (6 themes), CSV upload with column selection, TXT upload (blank-line separated), missing/duplicate handling, statistics.
- **Preprocessing**: lowercase, URL/HTML/punctuation/digit removal, tokenization, NLTK stopwords (+ custom), short/numeric token removal, optional lemmatization, min/max document-frequency filtering, graceful NLTK-failure fallback.
- **Document-term representation**: Gensim `Dictionary` + Bag-of-Words (primary), scikit-learn `CountVectorizer` (alternative), BoW vs TF-IDF explanation, CSV export.
- **LDA training**: K, passes, iterations, seed, alpha, eta, minimum word probability; validation; coherence (c_v), perplexity, training time; saved model, dictionary and config.
- **Topic explorer**: ranked words with probabilities, bar charts, heuristic (hedged) interpretations, CSV export.
- **Document analysis**: dominant topic, probability, full distribution, filtering, prevalence chart, CSV export.
- **Visualizations**: topic distribution, top words, topic-word and document-topic heatmaps, word frequencies, coherence vs K, optional pyLDAvis; HTML/PNG downloads.
- **Theory page**, **Markdown/TXT report** built from real results, and **one-click export** of every deliverable.

## Technologies
Python 3.10+ · Streamlit · Gensim · scikit-learn · NLTK · Pandas · NumPy · Matplotlib · Plotly · pyLDAvis · pytest

## Project structure
```
LDA_Topic_Modeling_AML23702/
├── app.py                      # Home / overview (entry point)
├── requirements.txt  README.md  LICENSE  .gitignore
├── .streamlit/config.toml      # blue/green theme, upload limit
├── data/sample_documents.csv   # 54 documents, columns: doc_id, theme, text
├── src/
│   ├── data_loader.py  preprocessing.py  vectorization.py  lda_model.py
│   ├── topic_interpretation.py  visualization.py  report_generator.py
│   ├── exports.py  utils.py  __init__.py
├── pages/                      # 1-9 as specified + 10_Download_Results.py
├── tests/                      # test_preprocessing / test_vectorization / test_lda_model
├── outputs/.gitkeep            # generated results go here (git-ignored)
└── docs/                       # methodology.md, theory.md, sample_report.md
```

## Installation
An internet connection is needed for the first dependency install and for NLTK resource downloads (stopwords, wordnet).
Python 3.10 – 3.12 is recommended (the pinned SciPy range required by Gensim has no wheels for newer Python versions).

**Windows (PowerShell / CMD)**
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
(PowerShell may need: `Set-ExecutionPolicy -Scope Process RemoteSigned` before activating.)

**Linux / macOS**
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501. NLTK data downloads automatically on first preprocessing run; to pre-download:
`python -m nltk.downloader stopwords wordnet omw-1.4`

## Usage
1. **Dataset Explorer**: keep the built-in sample or upload your own file.
2. **Text Preprocessing**: set options and press *Run preprocessing*.
3. **Document-Term Matrix**: press *Build* (set DF filters if desired).
4. **LDA Model Training**: choose K and parameters, press *Train*. Optionally run the coherence comparison.
5. Explore **Topic Explorer**, **Document Analysis**, **Visualizations**; read **Theory**.
6. **Generate Report**, then **Download Results**.

Progress is kept in the browser session while you switch pages. Refreshing the tab clears it, so download your results first.

### Uploading a custom dataset
- **CSV**: any file with at least one text column (UTF-8 recommended). If there are several columns, choose the text column in the dropdown.
- **TXT**: documents separated by a blank line.
- Missing/blank and exact-duplicate documents are removed automatically. For useful topics, use at least ~50 documents of 30+ words.

Sample CSV format (`data/sample_documents.csv`):
```
doc_id,theme,text
1,Cybersecurity,"Phishing attacks trick users into revealing passwords ..."
```

### Reloading a saved model for inference
Training saves to `outputs/lda_model/` (and the app offers a ZIP). Then:
```python
from src.lda_model import load_model, infer_document
model, dictionary, cfg = load_model("outputs/lda_model")
print(infer_document(model, dictionary, ["patient", "doctor", "treatment"]))
```
(New text must be preprocessed the same way, using the functions in `src/preprocessing.py`.)

## Interpreting topic-word distributions
Each topic is a probability distribution over words (rows of φ sum to 1). Read the top words together, then read the highest-probability documents for that topic. The app's automatic interpretation is a keyword heuristic only. **Always validate topics by hand.** A document is a mixture of topics; the dominant topic is only the largest component.

## LDA parameters
| Parameter | Effect |
|---|---|
| Number of topics K | Granularity; too low merges themes, too high fragments them. Compare coherence across K. |
| Alpha (α) | Document-topic prior. Small → few topics per document; large → many. `auto` learns it. |
| Eta (η/β) | Topic-word prior. Small → focused topics; large → broad topics. |
| Passes / iterations | Convergence effort of online variational Bayes. |
| Seed | Reproducibility. |
| Min word probability | Hides low-probability words in display. |

## Running the tests
```
pytest -q
```

## Troubleshooting
- **NLTK download fails**: run `python -m nltk.downloader stopwords wordnet omw-1.4` with internet access, or copy an `nltk_data` folder to your home directory. Until then the app uses scikit-learn's stopword list and skips lemmatization (a warning is shown).
- **`pip install` fails for SciPy/Gensim**: use Python 3.10–3.12 in a fresh virtual environment.
- **"All documents are empty after preprocessing"**: lower min_df, raise max_df, or remove custom stopwords.
- **"Number of topics exceeds documents/vocabulary"**: reduce K or load more data.
- **Odd or repeated topics**: use more/longer documents, adjust DF filters, change K, or increase passes.
- **pyLDAvis fails**: it is optional; the rest of the app is unaffected.
- **Port in use**: `streamlit run app.py --server.port 8502`.

## References and learning resources
- Blei, Ng & Jordan (2003). *Latent Dirichlet Allocation.* JMLR 3, 993–1022.
- Griffiths & Steyvers (2004). *Finding scientific topics.* PNAS 101.
- Hoffman, Bach & Blei (2010). *Online learning for LDA.* NeurIPS.
- Röder, Both & Hinneburg (2015). *Exploring the space of topic coherence measures.* WSDM.
- Chang et al. (2009). *Reading tea leaves.* NeurIPS.
- Gensim docs: https://radimrehurek.com/gensim/models/ldamodel.html · Streamlit docs: https://docs.streamlit.io
