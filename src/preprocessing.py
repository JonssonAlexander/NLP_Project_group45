"""
Text preprocessing helpers shared across notebooks.

Keep tokenisation, normalisation, and vocabulary utilities here so the
notebooks can reuse consistent logic without duplicating code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .data_io import shuffle_examples

Tokeniser = Callable[[str], Sequence[str]]


@dataclass(frozen=True)
class TokenisationConfig:
    """
    Configuration describing how questions should be tokenised.

    The `language_model` field mirrors the spaCy model name used in the
    assignment instructions.
    """

    use_spacy: bool = True
    language_model: str = "en_core_web_sm"
    lower: bool = True


def build_simple_tokeniser(config: TokenisationConfig) -> Tokeniser:
    """
    Create a lightweight tokeniser based on the provided configuration.

    For early experimentation we default to a whitespace split to avoid
    heavy dependencies. The spaCy-backed implementation can be plugged in
    later once the environment is ready.
    """

    if config.use_spacy:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError(
                "spaCy is requested but not installed. "
                "Install `spacy` and the `en_core_web_sm` model."
            ) from exc

        try:
            nlp = spacy.load(config.language_model, disable=("parser", "ner", "textcat"))
        except OSError as exc:  # pragma: no cover - model availability
            raise RuntimeError(
                f"spaCy model '{config.language_model}' is not available. "
                "Download it with `python -m spacy download en_core_web_sm`."
            ) from exc

        def spacy_tokeniser(text: str) -> Sequence[str]:
            doc = nlp(text.lower() if config.lower else text)
            return [token.text for token in doc if not token.is_space]

        return spacy_tokeniser

    def whitespace_tokeniser(text: str) -> Sequence[str]:
        value = text.lower() if config.lower else text
        return value.split()

    return whitespace_tokeniser


def strip_stopwords(tokens: Iterable[str], stopwords: set[str]) -> list[str]:
    """
    Remove stopwords while preserving token order.
    """

    return [token for token in tokens if token not in stopwords]


def prepare_tokenised_dataset(
    labeled_examples: Iterable[tuple[str, str]],
    tokeniser: Tokeniser,
    shuffle_seed: int | None = None,
) -> list[tuple[str, list[str]]]:
    """
    Tokenise labelled examples, optionally shuffling for randomness.
    """

    materialised = list(labeled_examples)
    if shuffle_seed is not None:
        materialised = shuffle_examples(materialised, shuffle_seed)

    tokenised_dataset: list[tuple[str, list[str]]] = []
    for label, text in materialised:
        tokens = list(tokeniser(text))
        tokenised_dataset.append((label, tokens))
    return tokenised_dataset
