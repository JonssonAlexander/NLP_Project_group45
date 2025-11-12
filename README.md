# NLP Project – TREC Question Classification

This repository houses the coursework implementation for SC4002, Group 45.  


---

## Repository Layout

| Path | Purpose |
| ---- | ------- |
| `configs/data.yaml` | Central data/tokenisation settings (paths, split ratio, coarse vs fine labels). |
| `configs/embeddings.yaml` | Location and dimensionality of the pretrained GloVe vectors. |
| `configs/experiments/part2_rnn.yaml` | Reference hyperparameters for the Part 2 RNN sweep. |
| `data/raw` | Expected home for `train_5500.label` and `TREC_10.label` (auto-downloaded). |
| `notebooks/01_embeddings.ipynb` | Part 1 analysis notebook (vocabulary + embedding probes). |
| `notebooks/02_rnn.ipynb` | Part 2 training/evaluation for the baseline RNN. |
| `notebooks/03_enhancements.ipynb` & `03_using_diff_models.ipynb` | Part 3 model explorations. |
| `src/` | Modules shared across notebooks (config, preprocessing, dataloaders, models, plotting, reports, training, evaluation). |

---

## Environment Setup

1. **Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. **Embeddings**  
   Place `glove.6B.100d.txt` under `data/glove/` and update
   `configs/embeddings.yaml:file_path` if you store it elsewhere.
3. **Sanity check the pipeline**
   ```bash
   python scripts/verify_data_pipeline.py
   ```
   The script downloads any missing TREC files, tokenises the corpus with spaCy, and prints the
   split sizes plus vocabulary count so you know the notebooks will load correctly.

---

## Data & Preprocessing

- Dataset parameters (train/test filenames, shuffle seed, coarse label flag) live in
  `configs/data.yaml` and are consumed via `src.config.load_data_config`.
- `src/data_io.py` handles downloads directly from the official UIUC endpoints.
- `src/preprocessing.py` exposes `build_simple_tokeniser`, which defaults to spaCy
  (`en_core_web_sm`, lowercasing enabled). Falling back to whitespace tokenisation is possible
  by flipping `tokenisation.use_spacy` in the config.
- `src/dataset_pipeline.prepare_tokenised_splits` ties everything together:
  download (if needed) → shuffle → train/validation split → tokenise → return `TokenisedDatasets`.

---

## Running the Pipeline

### 1. Prepare data and vocab once
```python
from pathlib import Path
from src.config import load_data_config, load_embedding_config
from src.dataset_pipeline import prepare_tokenised_splits
from src.reports import build_vocabulary_report
from src.embeddings import load_glove_embeddings

data_cfg = load_data_config(Path("configs/data.yaml"))
splits = prepare_tokenised_splits(data_cfg)
vocab_report = build_vocabulary_report(
    tokenised_dataset=splits.train,
    min_freq=data_cfg.vocabulary_min_freq,
    specials=data_cfg.vocabulary_specials,
)
embed_cfg = load_embedding_config(Path("configs/embeddings.yaml"))
embedding_result = load_glove_embeddings(embed_cfg, vocab_report.vocabulary)
```

### 2. Train the max-pooling RNN baseline programmatically
```python
from src.training import RNNExperimentConfig, train_rnn_model
from src.evaluation import evaluate_model

config = RNNExperimentConfig(
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.0,
    grad_clip=1.0,
    hidden_dim=128,
    dropout=0.0,
    pooling="max",
    optimizer="adam",
    early_stopping_patience=3,
)

history, model, label_to_idx, loaders = train_rnn_model(
    config=config,
    splits=splits,
    vocabulary=vocab_report.vocabulary,
    embedding_result=embedding_result,
)
result = evaluate_model(model, loaders.test)
print(f"Test accuracy: {result.accuracy:.3f}")
```

### 3. Use the notebooks for analysis
- `01_embeddings.ipynb` → reproduce Part 1 tables and generate the PCA/t-SNE plots.
- `02_rnn.ipynb` → run the hyperparameter sweeps, pooling comparisons, and topic-level charts.
- `03_enhancements.ipynb` & `03_using_diff_models.ipynb` → iterate on CNN/biRNN/RCNN ideas,
  persisting plots under `plots/`.

> **Tip:** All notebooks import from `src/` using relative paths, so launching `jupyter lab`
> or `jupyter notebook` from the repository root avoids import issues.
