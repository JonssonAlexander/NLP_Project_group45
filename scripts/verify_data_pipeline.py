"""
Quick sanity check for the dataset pipeline.

Run this script to confirm the TREC files download correctly and the tokenised
splits look reasonable. Keeping it lightweight ensures we can trust the
notebook setup before doing heavier experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.config import load_data_config
from src.dataset_pipeline import prepare_tokenised_splits
from src.embeddings import build_vocabulary


def main() -> None:
    config = load_data_config(Path("configs/data.yaml"))
    splits = prepare_tokenised_splits(config)

    print("Train examples:", len(splits.train))
    print("Validation examples:", len(splits.validation))
    print("Test examples:", len(splits.test))

    vocabulary = build_vocabulary(
        (tokens for _, tokens in splits.train),
        min_freq=config.vocabulary_min_freq,
        specials=config.vocabulary_specials,
    )
    print("Vocabulary size:", len(vocabulary))


if __name__ == "__main__":
    main()
