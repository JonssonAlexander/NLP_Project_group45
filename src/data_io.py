"""
Data ingestion utilities for the TREC question classification assignment.

This module keeps file-system concerns isolated from the notebooks so that
experiments can focus on modelling logic. Functions here should stay
lightweight, deterministic, and easy to test.
"""

from __future__ import annotations

import logging
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

TREC_TRAIN_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
TREC_TEST_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"


@dataclass(frozen=True)
class TrecDatasetPaths:
    """
    Container for locating the TREC dataset files.

    The defaults match the filenames provided by the official release.
    """

    root: Path
    train_filename: str = "train_5500.label"
    test_filename: str = "TREC_10.label"

    @property
    def train_path(self) -> Path:
        return self.root / self.train_filename

    @property
    def test_path(self) -> Path:
        return self.root / self.test_filename


def ensure_dataset_local(paths: TrecDatasetPaths, overwrite: bool = False) -> None:
    """
    Download the official TREC dataset files if they are not already present.

    This function performs simple HTTP downloads so the notebooks can assume
    the files exist locally. Pass `overwrite=True` to refresh existing copies.
    """

    paths.root.mkdir(parents=True, exist_ok=True)

    def _download(url: str, destination: Path) -> None:
        if destination.exists() and not overwrite:
            LOGGER.info("Dataset file already present: %s", destination)
            return
        LOGGER.info("Downloading %s -> %s", url, destination)
        urllib.request.urlretrieve(url, destination)  # noqa: S310

    _download(TREC_TRAIN_URL, paths.train_path)
    _download(TREC_TEST_URL, paths.test_path)


def read_labeled_questions(
    path: Path,
    encoding: str = "latin-1",
    fine_grained: bool = False,
) -> Iterable[Tuple[str, str]]:
    """
    Parse a raw TREC file into (label, question) tuples.

    The assignment data stores labels and questions on each line separated
    by a single whitespace after the label (e.g. "DESC:misc What is ...").
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    with path.open("r", encoding=encoding) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            label, question = line.split(" ", 1)
            
            if not fine_grained:
                label = label.split(":")[0]
            
            yield label, question


def split_train_validation(
    examples: Iterable[Tuple[str, str]],
    train_ratio: float,
) -> Tuple[list[Tuple[str, str]], list[Tuple[str, str]]]:
    """
    Split data into train and validation sets.

    This uses a simple list conversion to support reproducibility. The
    caller should shuffle upstream using a seeded RNG when randomness
    is required.
    """

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    sequence = list(examples)
    cutoff = int(len(sequence) * train_ratio)
    return sequence[:cutoff], sequence[cutoff:]


def shuffle_examples(
    examples: Sequence[Tuple[str, str]],
    seed: int,
) -> List[Tuple[str, str]]:
    """
    Return a shuffled copy of the provided examples using the supplied seed.
    """

    materialised = list(examples)
    random.Random(seed).shuffle(materialised)
    return materialised
