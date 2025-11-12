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
