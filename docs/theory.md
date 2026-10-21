# Theory and Methodology: Latent Dirichlet Allocation

## 1. What is LDA?
Latent Dirichlet Allocation (Blei, Ng & Jordan, 2003) is a **generative probabilistic model** of a text collection. It assumes every document is a *mixture of topics* and every topic is a *probability distribution over the vocabulary*. The topics are **latent**: never observed, only inferred from word co-occurrence.

## 2. Documents, topics and words
- **Word** `w`: an item from a vocabulary of size $V$.
- **Document** `d`: a bag of $N_d$ words (order ignored).
- **Topic** `k`: a distribution over the $V$ words. There are $K$ topics.

In the app: the *Document-Term Matrix* page shows the observed words; the *Topic Explorer* shows the inferred topics.

## 3. θ and φ
- **θ (theta), document-topic distribution:** $\theta_d = (\theta_{d1},\dots,\theta_{dK})$, with $\sum_k \theta_{dk}=1$. Shown on the *Document Analysis* page and in the document-topic heatmap.
- **φ (phi), topic-word distribution:** $\phi_k = (\phi_{k1},\dots,\phi_{kV})$, with $\sum_v \phi_{kv}=1$. Shown as the top-word bar charts and the topic-word heatmap.

## 4. Dirichlet priors α and η
$$\theta_d \sim \mathrm{Dirichlet}(\alpha), \qquad \phi_k \sim \mathrm{Dirichlet}(\eta)$$
- **α (alpha)** controls how many topics a document mixes. Small α (< 1) gives documents dominated by few topics; large α gives more uniform mixtures.
- **η (eta, often called β)** controls how many words a topic uses. Small η gives sparse, focused topics; large η gives broad topics.
- Gensim accepts `'symmetric'`, `'asymmetric'`, `'auto'` (learned from data) or a number.

## 5. The generative process
For each topic $k=1..K$: draw $\phi_k \sim \mathrm{Dir}(\eta)$.
For each document $d$:
1. Draw $\theta_d \sim \mathrm{Dir}(\alpha)$.
2. For each word position $n$:
   - draw a topic $z_{dn} \sim \mathrm{Categorical}(\theta_d)$;
   - draw a word $w_{dn} \sim \mathrm{Categorical}(\phi_{z_{dn}})$.

The probability of a word in a document is therefore
$$p(w \mid d) = \sum_{k=1}^{K} \theta_{dk}\,\phi_{kw}.$$
Topic assignment is probabilistic: the **dominant topic** reported by the app is $\arg\max_k \theta_{dk}$, but a document still contains the other topics with smaller probability.

## 6. Why inference is approximate
We observe only words and want the posterior
$$p(\theta, \phi, z \mid w, \alpha, \eta) = \frac{p(\theta,\phi,z,w \mid \alpha,\eta)}{p(w \mid \alpha,\eta)}.$$
The denominator (evidence) couples all latent variables and is intractable, so approximate inference is needed. The two standard families are **sampling** (MCMC) and **variational inference**.

## 7. Gibbs sampling
In **collapsed Gibbs sampling** (Griffiths & Steyvers, 2004) $\theta$ and $\phi$ are integrated out and only topic assignments $z$ are sampled. Each word's topic is resampled from
$$p(z_i = k \mid z_{-i}, w) \;\propto\; \frac{n_{k,w_i}^{-i} + \eta}{n_{k,\cdot}^{-i} + V\eta}\,\bigl(n_{d,k}^{-i} + \alpha\bigr).$$
Here $n_{k,w}$ counts word $w$ assigned to topic $k$, $n_{d,k}$ counts words in document $d$ assigned to topic $k$, and $-i$ excludes the current word. After burn-in, the counts give estimates of $\theta$ and $\phi$. The sampler is asymptotically exact but can be slow and is stochastic.

## 8. Variational inference (what Gensim uses)
Variational inference replaces the posterior with a simpler family $q$ and maximises the evidence lower bound (ELBO):
$$\log p(w) \;\ge\; \mathbb{E}_q[\log p(w,z,\theta,\phi)] - \mathbb{E}_q[\log q].$$
> **Important:** Gensim's standard `LdaModel` implements **online variational Bayes** (Hoffman, Bach & Blei, 2010), processing the corpus in mini-batches. It does **not** use Gibbs sampling. Collapsed Gibbs sampling is available in other tools such as **MALLET**. This project trains only the variational model; no Gibbs sampler is included or simulated.

| | Collapsed Gibbs sampling | Variational Bayes (Gensim) |
|---|---|---|
| Idea | Draw samples from the exact posterior | Optimise a simple approximation |
| Output | Samples, averaged | Deterministic parameters |
| Speed / scaling | Slower, harder to parallelise | Faster, streams large corpora |
| Bias | Asymptotically unbiased | Biased by the chosen family |

## 9. Evaluation: coherence and perplexity
- **Perplexity** $=\exp(-\text{per-word log-likelihood})$ (Gensim reports a bound). Lower is better on held-out data, but it often disagrees with human judgement (Chang et al., 2009). In this app it is computed on the *training* corpus, so it indicates fit, not generalisation.
- **Coherence (c_v)** scores how often a topic's top words co-occur in the texts (sliding windows, NPMI and cosine similarity). Higher is generally better, but it depends on preprocessing, the number of top words and the corpus, and it can be unstable on small data.

Neither metric proves that a topic is meaningful; use them to compare models, then read the topics.

## 10. Topics are not labels
A topic is just a probability distribution over words. The app's text interpretations are keyword heuristics. A human must inspect top words and representative documents before naming a topic, and some topics will be mixed or uninterpretable.

## 11. Limitations
- **Short documents**: few words per document give weak co-occurrence evidence.
- **Overlapping topics**: similar themes may merge; one theme may split.
- **Vocabulary dependence**: preprocessing, stopwords and DF filters change the topics.
- **Bag-of-Words**: word order and semantics are ignored.
- **K must be chosen**, and results depend on the random seed and priors.

## References
1. Blei, Ng, Jordan (2003). Latent Dirichlet Allocation. JMLR 3.
2. Griffiths, Steyvers (2004). Finding scientific topics. PNAS 101.
3. Hoffman, Bach, Blei (2010). Online learning for LDA. NeurIPS.
4. Roder, Both, Hinneburg (2015). Exploring the space of topic coherence measures. WSDM.
5. Chang et al. (2009). Reading tea leaves. NeurIPS.
