# Evaluation Methodologies for Code Retrieval Systems and Context Engines: A Structured Survey

> **Purpose**: Research-grade reference for designing evaluation frameworks for code retrieval and context-augmented AI systems.

---

## 1. BEIR: Benchmarking Information Retrieval (Zero-Shot)

**Citation**: Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS 2021 Datasets and Benchmarks Track*. arXiv:2104.08663.

### Primary Metric

**nDCG@10** (Normalized Discounted Cumulative Gain at rank 10) — chosen because it "provides a good balance suitable for both tasks involving binary and graded relevance judgements" (Thakur et al., 2021).

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

$$
\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

where IDCG@k is the ideal (maximum possible) DCG@k.

### Additional Metrics (available in BEIR framework)
| Metric | Formula | Use Case |
|--------|---------|----------|
| **MAP** (Mean Average Precision) | $\text{AP} = \frac{1}{\|R\|} \sum_{k=1}^{n} P(k) \cdot rel(k)$ | Binary relevance, recall-oriented |
| **MRR** (Mean Reciprocal Rank) | $\text{MRR} = \frac{1}{\|Q\|} \sum_{i=1}^{\|Q\|} \frac{1}{\text{rank}_i}$ | Single correct answer tasks |
| **Recall@k** | $\frac{\|\\{\text{relevant docs in top-}k\\}\|}{\|\\{\text{all relevant docs}\\}\|}$ | Coverage assessment |
| **Precision@k** | $\frac{\|\\{\text{relevant docs in top-}k\\}\|}{k}$ | Precision at cutoff |

### 18 Datasets Across 9 Task Types

| Task Type | Datasets |
|-----------|----------|
| Bio-Medical IR | TREC-COVID, NFCorpus, BioASQ |
| Question Answering | Natural Questions (NQ), HotpotQA, FiQA-2018 |
| Tweet Retrieval | Signal-1M |
| News Retrieval | TREC-NEWS, Robust04 |
| Argument Retrieval | ArguAna, Touché-2020 |
| Duplicate Question | CQADupStack, Quora |
| Entity Retrieval | DBPedia |
| Citation Prediction | SCIDOCS |
| Fact Verification | FEVER, Climate-FEVER, SciFact |

### Evaluation Protocol
- **Zero-shot**: Models pre-trained/fine-tuned on MS MARCO are evaluated directly on all 18 datasets *without task-specific fine-tuning*.
- **Model-agnostic**: Supports lexical (BM25), sparse, dense, late-interaction, and re-ranking architectures.
- **Standardized format**: Queries, documents, and relevance judgments (qrels) in universal TREC format.

---

## 2. MTEB: Massive Text Embedding Benchmark

**Citation**: Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. *EACL 2023*. arXiv:2210.07316.

### Scope
**8 embedding task types**, **58 datasets**, **112 languages**.

### Task-Specific Metrics

| Task | Metric | Method |
|------|--------|--------|
| **Retrieval** | nDCG@10 (primary), MRR@k, MAP@k, Recall@k, P@k | Cosine similarity ranking on embedded queries/docs |
| **Reranking** | MAP (primary), MRR@k | Re-rank candidates by cosine similarity |
| **Classification** | Accuracy | Train logistic regression on embeddings |
| **Clustering** | V-measure | Mini-batch k-means on embeddings |
| **Pair Classification** | Average Precision | Binary label assignment on text pairs |
| **STS** | Spearman correlation | Cosine similarity vs. gold scores |
| **Bitext Mining** | F1 | Cross-lingual sentence matching |
| **Summarization** | Spearman correlation | Summary quality via embedding similarity |

### Code Search in MTEB
MTEB **does not include code-specific search tasks** (no CodeSearchNet or similar). Its 15 retrieval datasets are sourced from BEIR and cover general-domain IR. For code-specific embedding evaluation, **CoIR** (Code Information Retrieval Benchmark) is the dedicated benchmark.

---

## 3. LLM-as-Judge Evaluation

### 3.1 Foundational Paper

**Citation**: Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks Track*. arXiv:2306.05685.

**Key Findings**:
- GPT-4 achieves **>80% agreement** with human preferences, matching the human-human agreement rate of **81%**.
- Three identified biases:
  - **Position bias**: GPT-4 shows 65% consistency when responses are swapped; improved to 66.2% with response renaming.
  - **Verbosity bias**: All judges favor longer responses. GPT-3.5/Claude-v1 fail 91.3% on repetitive list attacks; GPT-4 fails only 8.7%.
  - **Self-enhancement bias**: GPT-4 boosts its own win rate by ~10%.

### 3.2 Position Bias Analysis

**Citation**: Shi, W., et al. (2025). Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. arXiv:2406.07791.

- Analyzed **15 LLM judges** on MT-Bench/DevBench across **>150K instances**.
- Bias is **judge-dependent and task-dependent** (e.g., Claude-3.5-Sonnet achieves PC=0.82±0.14 on MT-Bench pairwise).
- Introduced metrics:
  - **Position Consistency (PC)**: Fraction of consistent preferences when swapping positions: $PC = \frac{|V_{\text{consistent}}|}{n}$
  - **Preference Fairness (PF)**: Directional bias magnitude (e.g., GPT-4o PF = −0.12).
  - **Repetition Stability (RS)**: RS > 0.95 for top judges (Claude-3.5-Sonnet).

### 3.3 Best Practices

| Practice | Details | Source |
|----------|---------|--------|
| **Pairwise over pointwise** | More reliable preference elicitation | Zheng et al. 2023 |
| **Multiple heterogeneous judges** | Majority vote or weighted Elo aggregation | Zheng et al. 2023 |
| **Reference-based grading** | Reduces failure from 70% → 15% with CoT + reference | Shi et al. 2025 |
| **Position swapping** | Present A/B in both orders, require consistency | Standard practice |
| **Calibration samples** | Include known-quality samples to detect drift | Community consensus |

---

## 4. Statistical Significance in IR Evaluation

### 4.1 Recommended Tests

**Primary references**:
- Sakai, T. (2006). Evaluating Evaluation Metrics based on the Bootstrap. *SIGIR 2006*.
- Sakai, T. (2014). Statistical Reform in Information Retrieval? *SIGIR Forum*.
- Urbano, J., Lima, H., & Hanjalic, A. (2019). Statistical Significance Testing in Information Retrieval: An Empirical Analysis of Type I, Type II, and Type III Errors. *SIGIR 2019*.

| Test | Formula | When to Use | Notes |
|------|---------|-------------|-------|
| **Paired t-test** | $t = \frac{\bar{D}}{s_D / \sqrt{n}}$, df = n−1 | Default recommendation (Urbano et al.) | Maintains Type I error at nominal α |
| **Permutation/Randomization test** | Compute test statistic under all (or 10⁶) random label swaps | Gold standard reference test | Non-parametric; exact under H₀ |
| **Bootstrap test** | Resample with replacement, compute CI | Confidence intervals for metrics | *Caution*: tends toward small p-values (Urbano et al.) |
| **Wilcoxon signed-rank** | Rank differences, sum positive/negative ranks | Non-parametric alternative | Inflates Type I errors in some cases |
| **Sign test** | Counts of positive/negative differences | Distribution-free, low power | Last resort |

### 4.2 Practical Guidelines (from top IR venues)

- **Significance threshold**: p < 0.05 standard; p < 0.01 for strong claims.
- **Multiple comparisons**: Bonferroni correction: $\alpha_{\text{adj}} = \alpha / m$ for $m$ comparisons. Holm-Bonferroni as less conservative alternative.
- **Effect size**: Report Cohen's d or mean difference $\Delta = |\mu_1 - \mu_2| / \sigma$ alongside p-values.
- **Minimum topics/queries**: **n ≥ 50** for reliable power at α = 0.05 with small effects (δ = 0.01–0.1). n = 25 risks high Type II/III errors. n = 100 ensures robust results.
- **Preferred approach** (Urbano et al. 2019): Paired t-test for simplicity and power; permutation test as reference. Both maintain Type I error across AP, nDCG@20, P@10, RR at topic sizes 25–100.
- **TREC/SIGIR convention**: ~65% of papers use t-tests, ~25% Wilcoxon, with increasing adoption of permutation tests.

---

## 5. Code Search Benchmarks

### 5.1 CodeSearchNet

**Citation**: Husain, H., Wu, H.-H., Gazit, T., Allamanis, M., & Brockschmidt, M. (2019). CodeSearchNet Challenge: Evaluating the State of Semantic Code Search. arXiv:1909.09436.

| Property | Value |
|----------|-------|
| Corpus size | ~6M functions from GitHub |
| With documentation | ~2M paired functions |
| Languages | Go, Java, JavaScript, PHP, Python, Ruby |
| Evaluation (standard) | MRR on test sets with 999 distractors per query |
| Evaluation (challenge) | NDCG on 99 NL queries, ~4K expert-annotated relevance labels (0–3 scale) |
| Splits | 80/10/10 train/valid/test |

### 5.2 CoSQA

**Citation**: Huang, J., Tang, D., Shou, L., Gong, M., Xu, K., Jiang, D., Zhou, M., & Duan, N. (2021). CoSQA: 20,000+ Web Queries for Code Search and Question Answering. *ACL 2021*.

- **20,604 human-annotated** NL query–code pairs (Python).
- Uses **real-world web search queries** (vs. CodeSearchNet's doc-derived queries).
- Two-stage annotation: code-search intent check → relevance (binary), ≥3 annotators per pair.
- Improves CodeBERT code QA accuracy by 5.1% when used for training.

### 5.3 AdvTest

**Citation**: Lu, S., Guo, D., Ren, S., Huang, J., Svyatkovskiy, A., Blanco, A., Clement, C., Drain, D., Jiang, D., Tang, D., Li, G., Zhou, L., Shou, L., Zhou, L., Tufano, M., Gong, M., Zhou, M., Duan, N., Sundaresan, N., Deng, S. K., Fu, S., & Liu, S. (2021). CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation. *NeurIPS 2021 Datasets and Benchmarks*. arXiv:2102.04664.

- Adversarial Python test set: **19,210 examples** (251K train / 9.6K dev).
- **Construction**: Normalizes function/variable names (e.g., `func`, `arg_i`) to remove lexical cues.
- Uses full test set as candidates per query (vs. prior 1K subsample).
- Drops CodeBERT MRR from **0.869 → 0.507** — tests genuine semantic understanding.

### 5.4 CRUXEval

**Citation**: Zeng, A., et al. (2024). CRUXEval: Code Reasoning, Understanding, and Execution Evaluation.

- **800 short Python functions** (3–13 lines, 75–300 chars) with input-output pairs.
- Generated by Code Llama 34B, filtered for no syntax errors, small args, runtime < 2s.
- Evaluates **code reasoning and execution prediction** via pass@1.
- Uses bootstrap sampling for statistical significance.

### 5.5 Metrics Beyond NDCG/MRR

| Metric | Formula | Use Case |
|--------|---------|----------|
| **MAP@k** | $\frac{1}{\|Q\|} \sum_{q} \text{AP}_q@k$ | Mean precision across ranked results |
| **Recall@k** | Relevant retrieved in top-k / total relevant | Coverage of relevant results |
| **Precision@k** | Relevant in top-k / k | Precision at cutoff |
| **Success Rate@k** (SR@k) | $\frac{1}{\|Q\|} \sum_{q} \mathbb{1}[\text{relevant in top-}k]$ | Binary: any relevant result found? |
| **FirstRank** | Rank of first relevant result | Latency-sensitive applications |
| **pass@k** | $1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$ | Code execution correctness (Chen et al. 2021, Codex paper) |

### 5.6 Recent Benchmarks (2023–2024)

**CodeRAG-Bench** evaluates code retrieval in the RAG pipeline for code generation across general programming, open-domain, and repository-level tasks (DS-1000, SWE-Bench), retrieving from GitHub, StackOverflow, and documentation. Shows retrieval gains for GPT-4 but diminishing returns on hard tasks.

---

## 6. Context Quality Metrics for RAG

### 6.1 RAGAS Framework

**Citation**: Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.

| Metric | Definition | Formula |
|--------|------------|---------|
| **Context Precision** | Fraction of retrieved docs that are relevant (position-weighted) | $\frac{\sum_{i=1}^{k} rel_i \cdot P(i)}{\sum_{i=1}^{k} P(i)}$ where $P(i) = 1/i$ (position discount), $rel_i$ is binary relevance (LLM-judged) |
| **Context Recall** | Fraction of ground-truth relevant items retrieved | $\frac{\sum_{i \in \text{retrieved}} rel_i^{gt}}{|\text{gt}|}$ |
| **Faithfulness** | Fraction of answer claims supported by context | LLM extracts claims from answer → verifies each against context → score ∈ [0,1] per claim, averaged |
| **Answer Relevance** | Semantic similarity of answer to query | LLM generates 3–5 questions from answer → cosine similarity of question embeddings to query |

**Implementation**: `pip install ragas`, uses LLM-as-judge (GPT-4 or open-source) with chain-of-thought decomposition.

### 6.2 ARES

**Citation**: Saad-Falcon, J., et al. (2023). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. arXiv:2311.09476.

| Aspect | RAGAS | ARES |
|--------|-------|------|
| **Judge** | Prompts LLM directly | Fine-tunes DeBERTa-v3-Large judges |
| **Training data** | None (zero-shot) | Synthetic in-domain data (FLAN-T5-XXL) |
| **Calibration** | None | Prediction-Powered Inference (PPI) with 150–300 human annotations for 95% CI |
| **Performance** | Baseline | +59.3pp context relevance, +14.4pp answer relevance over RAGAS on KILT/SuperGLUE |
| **System ranking** | N/A | Kendall's τ > 0.9 for ranking RAG configurations |

### 6.3 eRAG

**Citation**: Salemi, A., & Zamani, H. (2024). eRAG: Evaluating Retrieval-Augmented Generation at the Document Level. *SIGIR 2025*.

- Evaluates **per-document** relevance: runs RAG LLM on each retrieved doc alone, scores with downstream metric (EM / F1 / ROUGE).
- Aggregates via MAP/NDCG.
- High correlation with end-to-end performance (τ up to 0.7); 2–3× faster, 50× less GPU memory.

### 6.4 RGB Benchmark

**Citation**: Chen, J., et al. (2024). Benchmarking Large Language Models in Retrieval-Augmented Generation. *AAAI 2024*.

Evaluates four RAG abilities:
| Ability | Metric | Details |
|---------|--------|---------|
| Noise robustness | Accuracy at noise ratios 0–0.8 | Acc drops 20%+ at 80% noise |
| Negative rejection | Rejection rate on irrelevant-only context | Can the model refuse to answer? |
| Information integration | Multi-document QA accuracy | Synthesize across passages |
| Counterfactual robustness | Error detection/correction rate | Handle contradictory context |

### 6.5 Other Frameworks

| Framework | Key Differentiator |
|-----------|--------------------|
| **TruLens** | End-to-end RAG eval; CI/CD integration; benchmarks vs. RAGAS/DeepEval on NDCG@3, top-1 accuracy |
| **DeepEval** | Modular LLM evals; G-Eval for hallucinations; claim decomposition for faithfulness |

---

## 7. CodeBERTScore and Semantic Similarity Metrics for Code

### 7.1 CodeBERTScore

**Citation**: Zhou, S., Alon, U., Agarwal, S., & Neubig, G. (2023). CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code. arXiv:2302.05527.

**Method**: Extends BERTScore by (a) encoding *both* the NL input and code with code-specific pretrained models (CodeBERT further pretrained on CodeParrot corpora for Java, Python, C, C++, JavaScript) and (b) computing cosine similarity only between non-punctuation code tokens.

$$
P = \frac{1}{|\hat{y}[\hat{m}]|} \sum_{\hat{y}_j \in \hat{y}[\hat{m}]} \max_{y^*_i \in y^*[m^*]} \cos(\mathbf{h}_{y^*_i}, \mathbf{h}_{\hat{y}_j})
$$

$$
R = \frac{1}{|y^*[m^*]|} \sum_{y^*_i \in y^*[m^*]} \max_{\hat{y}_j \in \hat{y}[\hat{m}]} \cos(\mathbf{h}_{y^*_i}, \mathbf{h}_{\hat{y}_j})
$$

$$
F_1 = \frac{2 \cdot P \cdot R}{P + R}, \quad F_3 = \frac{10 \cdot P \cdot R}{9P + R}
$$

**Reported correlations** (with human preference):
| Benchmark | Kendall τ | Spearman ρ |
|-----------|-----------|------------|
| CoNaLa | **0.517** | **0.662** |
| HumanEval (Java) | **0.553** | — |
| Outperforms baselines | by up to 10% | — |

Deeper layers (7–11) yield best correlations.

### 7.2 BERTScore (Original)

**Citation**: Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*. arXiv:1904.09675.

Same greedy matching formulation as CodeBERTScore but using general-purpose BERT/RoBERTa embeddings. Optional IDF weighting:

$$
w_t = -\log\left(\frac{df(t)}{N}\right)
$$

Applied to weight token importance in P/R/F1 computation. Typical correlation with human judgment: ~0.59 (vs. BLEU ~0.47–0.50).

### 7.3 CodeBLEU

**Citation**: Ren, S., Guo, D., Lu, S., Zhou, L., Liu, S., Tang, D., Sundaresan, N., Zhou, M., Blanco, A., & Ma, S. (2020). CodeBLEU: A Method for Automatic Evaluation of Code Synthesis. arXiv:2009.10297.

$$
\text{CodeBLEU} = \alpha \cdot \text{BLEU} + \beta \cdot \text{BLEU}_{\text{weight}} + \gamma \cdot \text{Match}_{\text{ast}} + \delta \cdot \text{Match}_{\text{df}}
$$

| Component | Description |
|-----------|-------------|
| BLEU | Standard n-gram precision |
| BLEU_weight | Keywords weighted ×5 in unigram precision |
| Match_ast | Sub-tree matching on parsed ASTs (leaves masked): $\frac{\text{Count}_{\text{clip}}(T_{\text{cand}})}{\text{Count}(T_{\text{ref}})}$ |
| Match_df | Data-flow graph edge matching normalized by variable appearance order |

Default weights: α = β = γ = δ = 0.25. Optimal: ~0.1/0.1/0.4/0.4.

Pearson correlation with human scores: **0.977** (text-to-code), **0.970** (translation), **0.979** (refinement).

### 7.4 CrystalBLEU

**Citation**: Eghbali, A. & Pradel, M. (2022). CrystalBLEU: Precisely and Efficiently Measuring the Similarity of Code. *ASE 2022*.

Modifies BLEU by removing **trivially shared n-grams** (top 500 most common per dataset/language) from matched counts before computing BLEU precision. Better distinguishability between similar/dissimilar code pairs.

### 7.5 Embedding-Based Cosine Similarity

Models: CodeBERT, UniXcoder, StarCoder, CodeLlama embeddings.

$$
\text{sim}(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1^\top \mathbf{e}_2}{\|\mathbf{e}_1\| \cdot \|\mathbf{e}_2\|}
$$

Used as the primary ranking function in dense retrieval. No standardized human-correlation benchmarks for raw cosine similarity in code search.

### 7.6 Edit Distance Variants

| Method | Description |
|--------|-------------|
| **Levenshtein distance** | Token-level insertions, deletions, substitutions |
| **Tree Edit Distance (TED)** | Min-cost node operations on ASTs (APTED algorithm); captures structural similarity |

---

## 8. Inter-Annotator Agreement for LLM Judges

### 8.1 Cohen's Kappa (2 raters)

**Reference**: Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46.

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $p_o$ = observed agreement, $p_e = \sum_k \frac{n_{k1}}{N} \cdot \frac{n_{k2}}{N}$ (expected agreement by chance).

**Interpretation** (Landis & Koch, 1977):

| κ Value | Interpretation |
|---------|----------------|
| < 0.00 | Poor (less than chance) |
| 0.00–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost perfect |

**In LLM judge literature**: Typical GPT-4 vs. human κ = 0.3–0.5 (conservative due to chance correction), while raw agreement > 80%.

### 8.2 Weighted Kappa (Ordinal Scales)

$$
\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} \cdot x_{ij}}{\sum_{i,j} w_{ij} \cdot m_{ij}}
$$

Weight schemes:
- **Linear**: $w_{ij} = |i - j|$
- **Quadratic**: $w_{ij} = (i - j)^2$

Interpretation (Fleiss, 2003): > 0.75 excellent; 0.40–0.75 fair to good; < 0.40 poor.

Suited for LLM judges producing ordinal ratings (e.g., 1–5 quality scores).

### 8.3 Fleiss' Kappa (>2 raters)

**Reference**: Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin*, 76(5), 378–382.

$$
\kappa = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e}
$$

where $\bar{P} = \frac{1}{N} \sum_i P_i$, $P_i = \frac{1}{n(n-1)} \sum_j n_{ij}(n_{ij} - 1)$, $\bar{P}_e = \sum_j p_j^2$, $p_j = \frac{1}{Nn} \sum_i n_{ij}$.

Used when evaluating agreement among **multiple LLM judges** simultaneously.

### 8.4 Krippendorff's Alpha

**Reference**: Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. *Annenberg School for Communication*.

$$
\alpha = 1 - \frac{D_o}{D_e}
$$

where $D_o$ = observed disagreement, $D_e$ = expected disagreement.

**Advantages over Cohen's κ**: Handles missing data, >2 raters, nominal/ordinal/interval/ratio scales, no rater-pairing assumption.

**Interpretation thresholds** (Krippendorff):
| α Value | Interpretation |
|---------|----------------|
| ≥ 0.800 | Reliable |
| 0.667–0.800 | Tentatively reliable |
| < 0.667 | Unreliable |

Reported: α = 0.79 in "Judge's Verdict" (arXiv:2510.09738, 2025), indicating substantial agreement between LLM and human judges.

### 8.5 Alternatives

| Metric | When to Use |
|--------|-------------|
| **Scott's π** | 2 raters, uses joint marginals (vs. Cohen's individual marginals) |
| **Gwet's AC1** | Robust to prevalence/bias paradox; less sensitive to skewed marginals |

### 8.6 Position Bias Quantification

| Metric | Definition | Source |
|--------|------------|--------|
| **Position Consistency (PC)** | $PC = \frac{|V_{\text{consistent}}|}{n}$ — fraction of consistent preferences when swapping A/B positions | Shi et al. 2025 |
| **Preference Fairness (PF)** | Net directional bias (e.g., PF = −0.12 means slight first-position preference) | Shi et al. 2025 |
| **Repetition Stability (RS)** | Consistency across repeated identical queries (RS > 0.95 = reliable) | Shi et al. 2025 |

---

## Summary: Recommended Evaluation Stack for a Code Retrieval / Context Engine

| Layer | Metrics | Benchmarks | Statistical Tests |
|-------|---------|------------|-------------------|
| **Retrieval quality** | nDCG@10, MRR, MAP, Recall@k, Success@k | BEIR (general), CodeSearchNet, CoSQA, AdvTest | Paired t-test (primary), permutation test (reference), n ≥ 50 queries, Bonferroni for multiple comparisons |
| **Embedding quality** | MTEB tasks, code-specific cosine similarity | MTEB, CoIR | Same as above |
| **Code similarity** | CodeBERTScore (F1/F3), CodeBLEU, CrystalBLEU | CoNaLa, HumanEval | Report Kendall τ and Spearman ρ with human judgments |
| **Context quality (RAG)** | RAGAS (context precision/recall, faithfulness, answer relevance), ARES | RGB benchmark, KILT | Bootstrap CI for metric estimates |
| **LLM-as-Judge** | Pairwise preference, Likert scoring, reference-based grading | MT-Bench, Chatbot Arena | Cohen's κ / Krippendorff's α for agreement; PC/PF/RS for bias |
| **End-to-end** | pass@k, task completion rate | CRUXEval, SWE-Bench, CodeRAG-Bench | Bootstrap for pass@k CI |

---

## Key Citations (Consolidated)

1. Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS 2021 D&B*. arXiv:2104.08663.
2. Muennighoff et al. (2023). MTEB: Massive Text Embedding Benchmark. *EACL 2023*. arXiv:2210.07316.
3. Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 D&B*. arXiv:2306.05685.
4. Shi et al. (2025). Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. arXiv:2406.07791.
5. Sakai, T. (2006). Evaluating Evaluation Metrics based on the Bootstrap. *SIGIR 2006*.
6. Sakai, T. (2014). Statistical Reform in Information Retrieval? *SIGIR Forum*.
7. Urbano, Lima, & Hanjalic (2019). Statistical Significance Testing in Information Retrieval: An Empirical Analysis of Type I, Type II, and Type III Errors. *SIGIR 2019*.
8. Husain et al. (2019). CodeSearchNet Challenge. arXiv:1909.09436.
9. Huang et al. (2021). CoSQA: 20,000+ Web Queries for Code Search and Question Answering. *ACL 2021*.
10. Lu et al. (2021). CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation. *NeurIPS 2021 D&B*. arXiv:2102.04664.
11. Zeng et al. (2024). CRUXEval: Code Reasoning, Understanding, and Execution Evaluation.
12. Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
13. Saad-Falcon et al. (2023). ARES: An Automated Evaluation Framework for RAG Systems. arXiv:2311.09476.
14. Salemi & Zamani (2024). eRAG: Evaluating Retrieval-Augmented Generation at the Document Level. *SIGIR 2025*.
15. Chen et al. (2024). Benchmarking Large Language Models in Retrieval-Augmented Generation. *AAAI 2024*.
16. Zhou et al. (2023). CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code. arXiv:2302.05527.
17. Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*. arXiv:1904.09675.
18. Ren et al. (2020). CodeBLEU: A Method for Automatic Evaluation of Code Synthesis. arXiv:2009.10297.
19. Eghbali & Pradel (2022). CrystalBLEU: Precisely and Efficiently Measuring the Similarity of Code. *ASE 2022*.
20. Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1).
21. Landis & Koch (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1).
22. Fleiss, J. L. (1971). Measuring Nominal Scale Agreement Among Many Raters. *Psychological Bulletin*, 76(5).
23. Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability.
24. Chen et al. (2021). Evaluating Large Language Models Trained on Code [Codex]. arXiv:2107.03374.
