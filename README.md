# NLP Group 45 – Assignment Report

## Part 1 – Word Embedding

### Question 1(a): Vocabulary Size

The vocabulary constructed from the training data contains 7,479 tokens, comprising of the following:

- 7,477 unique tokens from the training corpus
- 2 special tokens: `<pad>` and `<unk>`

The training set has 4,361 documents with a total of 45,591 tokens. We applied a minimum frequency threshold of 1, meaning all tokens appearing at least once were included in the vocabulary.

The tokenisation method used was spaCy (`en_core_web_sm`) with lowercasing enabled.

---

### Question 1(b): Out-of-Vocabulary (OOV) Analysis

When comparing our vocabulary against the `GloVe 6B-100d` pretrained embeddings, we found the following:

- 197 unique OOV words (tokens present in training data but absent from GloVe)
- 215 total OOV token occurrences across the training set

The OOV distribution by topic category (fine-grained labels) is as follows:

| Topic Category | Unique OOV | Total Occurrences |
| -------------- | ---------- | ----------------- |
| DESC:def       | 40         | 40                |
| HUM:ind        | 37         | 39                |
| ENTY:other     | 21         | 21                |
| DESC:desc      | 11         | 11                |
| DESC:manner    | 10         | 10                |
| DESC:reason    | 10         | 11                |
| ENTY:cremat    | 10         | 13                |
| NUM:count      | 9          | 10                |
| LOC:other      | 8          | 8                 |
| ENTY:food      | 5          | 5                 |
| HUM:gr         | 5          | 5                 |
| LOC:city       | 5          | 5                 |
| Others         | 26         | 37                |

The highest OOV counts appear in `DESC:def` (definition questions) and `HUM:ind` (individual person questions), probably because of domain-specific terminology and proper nouns that are underrepresented in the general-domain `GloVe` corpus.

---

### Question 1(c): OOV Mitigation Strategy

The strategy used was global mean vector initialization, which is as follows:

1. Computing the global mean of all in-vocabulary embedding vectors (excluding special tokens `<pad>`, `<unk>` and OOV tokens themselves)
2. Assigning this mean vector to all OOV tokens as their initial representation
3. Keeping embeddings trainable so the model can adapt OOV vectors during training

The reasoning for this is that it provides a more meaningful initialization than random noise, positions OOV tokens near the semantic "centroid" of the vocabulary, and allows the model to fine-tune these vectors based on task-specific context during training. It's also computationally efficient and simple to implement.

Code snippet for the mitigation strategy is in `src/embeddings.py`:

```python
def mitigate_oov_embeddings(
    embedding_matrix: np.ndarray,
    vocabulary: Vocabulary,
    oov_tokens: set[str],
) -> Mapping[str, np.ndarray]:

    mitigated: dict[str, np.ndarray] = {}

    if not oov_tokens:
        return mitigated

    # computing the mean of all in-vocabulary embeddings (excluding specials and OOV)
    in_vocab_indices = [
        idx
        for token, idx in vocabulary.token_to_index.items()
        if token not in {"<pad>", "<unk>"} and token not in oov_tokens
    ]

    if not in_vocab_indices:
        return mitigated

    mean_vector = embedding_matrix[in_vocab_indices].mean(axis=0)

    # assigning the mean vector to all OOV tokens
    for token in sorted(oov_tokens):
        token_index = vocabulary.token_to_index.get(token)
        if token_index is None:
            continue
        embedding_matrix[token_index] = mean_vector.copy()
        mitigated[token] = mean_vector.copy()

    return mitigated
```

This approach successfully mitigated 197 OOV tokens plus 2 special tokens (`<pad>`, `<unk>`), which resulted in 199 initialized vectors.

---

### Question 1(d): Embedding Visualization

We selected the top 20 most frequent tokens from each topic category (after removing stopwords) and projected their `GloVe` embeddings into 2D space using both `PCA` and `t-SNE` (perplexity=15).

<img src="plots/part1_top_tokens_pca.png" width="400" alt="PCA projection of top tokens" />
<img src="plots/part1_top_tokens_tsne.png" width="400" alt="t-SNE projection of top tokens" />

#### Some key observations:

- Tokens from similar categories (e.g., NUM-related topics) show spatial proximity, indicating that `GloVe` captures meaningful semantic relationships
- Question-specific tokens (e.g., `who`, `what`, `when`) cluster distinctly, reflecting their differing semantic roles
- Some categories overlap (particularly DESC subcategories), which is expected since definition and description questions share similar vocabulary
- `t-SNE` reveals more granular local structure while `PCA` preserves global variance; both confirm reasonable semantic organization

The visualizations suggest that pretrained `GloVe` embeddings provide a solid foundation for topic classification, with question-type-specific tokens occupying distinct semantic regions.

## Part 2 – RNN Baseline

- [x] 2(a) Best configuration (see `02_rnn.ipynb`).
- [x] 2(b) Regularisation sweep complete with "no regularisation" as control. Don't know about this one but think we have an ok grid?
- [x] 2(c) Training curves. Validation accuracy sort of plateaus near the best epoch, maybe double check this.
- [x] 2(d) Sentence pooling strategies compared (`pooling_df`). Did mean, max and attention variants. They seem to be improving the accuracy.
- [x] 2(e) Topic-wise accuracy.

<img src="plots/part2_rnn_baseline_curves.png" width="380" alt="RNN baseline curves" />
<img src="plots/part2_topic_accuracy.png" width="380" alt="Baseline topic accuracy" />

## Part 3 – Enhancements

- [x] 3.1 biRNN experiments (biLSTM, biGRU) with training curves. Double check, especially the RNN score which doesnt behave like it used to.
- [x] 3.2 CNN experiment
- [ ] 3.3
- [ ] 3.4

    <img src="plots/part3_bilstm_curves.png" width="300" alt="biLSTM curves" />
    <img src="plots/part3_bigru_curves.png" width="300" alt="biGRU curves" />
    <img src="plots/part3_cnn_curves.png" width="300" alt="CNN curves" />

    <img src="plots/part3_bilstm_topic_accuracy.png" width="300" alt="biLSTM topic accuracy (not plotted!)" />
    <img src="plots/part3_bigru_topic_accuracy.png" width="300" alt="biGRU topic accuracy" />
    <img src="plots/part3_cnn_topic_accuracy.png" width="300" alt="CNN topic accuracy" />
