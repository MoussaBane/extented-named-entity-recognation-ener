"""
Utility functions for reading INCEpTION-style CoNLL files.

Expected format per line:
TOKEN  ...  NER_TAG

Where:
- TOKEN   = surface form
- NER_TAG = BIO-style tag (e.g., B-PERSON, I-PERSON, O)
Sentences are separated by blank lines.
"""

from typing import List, Tuple


TokenLabel = Tuple[str, str]           # (token, label)
Sentence = List[TokenLabel]            # list of (token, label)


def read_conll_file(path: str) -> List[Sentence]:
    """
    Read a CoNLL file and return a list of sentences.
    Each sentence is a list of (token, label) pairs.
    """
    sentences: List[Sentence] = []
    current: Sentence = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue

            parts = line.split()
            if len(parts) < 2:
                # Skip malformed lines silently
                continue

            token = parts[0]
            label = parts[-1]  # NER tag is assumed to be the last column
            current.append((token, label))

    if current:
        sentences.append(current)

    return sentences
