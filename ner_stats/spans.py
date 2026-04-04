"""Utilities for converting token-level BIO annotations into character spans."""

from typing import List, Sequence, Tuple


CharSpan = Tuple[str, int, int]


def bio_to_character_spans(
    text: str,
    tokens: Sequence[str],
    labels: Sequence[str],
) -> List[CharSpan]:
    """
    Convert token-level BIO annotations to character-level entity spans.

    Spans are returned as (entity_label, start, end) tuples, where `end` is
    exclusive and `entity_label` is stripped of any BIO prefix.
    """
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have the same length.")

    spans: List[CharSpan] = []
    cursor = 0
    current_label = None
    current_start = None
    current_end = None

    def close_current_span() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is not None and current_start is not None and current_end is not None:
            spans.append((current_label, current_start, current_end))
        current_label = None
        current_start = None
        current_end = None

    for token, label in zip(tokens, labels):
        if token == "":
            raise ValueError("tokens must not contain empty strings.")

        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        token_start = text.find(token, cursor)
        if token_start == -1:
            raise ValueError(f"Token '{token}' could not be aligned with the provided text.")

        token_end = token_start + len(token)

        if label == "O":
            close_current_span()
        else:
            entity_label = label
            if entity_label.startswith("B-"):
                entity_label = entity_label[2:]
                close_current_span()
                current_label = entity_label
                current_start = token_start
                current_end = token_end
            elif entity_label.startswith("I-"):
                entity_label = entity_label[2:]
                if current_label != entity_label or current_start is None:
                    close_current_span()
                    current_label = entity_label
                    current_start = token_start
                current_end = token_end
            else:
                close_current_span()
                current_label = entity_label
                current_start = token_start
                current_end = token_end

        cursor = token_end

    close_current_span()
    return spans