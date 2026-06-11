"""Character-level CNN encoder for hybrid BERT + character NER models."""

from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


def build_char_vocab(sentences: Sequence[Sequence[str]], min_freq: int = 1) -> Dict[str, int]:
    """Build a character vocabulary from token sequences.

    Returns a dict with reserved entries ``<PAD>=0`` and ``<UNK>=1`` followed
    by all characters that appear at least *min_freq* times, sorted lexicographically.
    """
    counts: Counter = Counter(ch for sent in sentences for word in sent for ch in word)
    char2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for ch, cnt in sorted(counts.items()):
        if cnt >= min_freq:
            char2id[ch] = len(char2id)
    return char2id


def words_to_char_ids(
    words: Sequence[str],
    char2id: Dict[str, int],
    max_chars: int = 32,
) -> List[List[int]]:
    """Convert a list of words to padded character-ID sequences.

    Each word is truncated or right-padded with zeros to exactly *max_chars*.
    Unknown characters map to the ``<UNK>`` id (default 1).
    """
    unk = char2id.get("<UNK>", 1)
    pad = char2id.get("<PAD>", 0)
    result = []
    for word in words:
        ids = [char2id.get(ch, unk) for ch in word[:max_chars]]
        ids += [pad] * (max_chars - len(ids))
        result.append(ids)
    return result


class CharCNNEncoder(nn.Module):
    """Character-level CNN that produces a fixed-size vector for each input word.

    Applies multiple parallel 1-D convolutions (different kernel widths) over
    the character-embedding sequence for each word, then max-pools each filter
    map to a single value.  The concatenated pool values form the character
    representation.

    Parameters
    ----------
    vocab_size : int
        Size of the character vocabulary (including PAD and UNK).
    char_emb_dim : int
        Dimension of the character embeddings.  Default: 30.
    num_filters : int
        Number of convolutional filters per kernel size.  Default: 50.
    kernel_sizes : list of int
        Widths of the parallel convolutions.  Default: [2, 3, 4].
    dropout : float
        Dropout applied to the final output vector.  Default: 0.1.
    """

    def __init__(
        self,
        vocab_size: int,
        char_emb_dim: int = 30,
        num_filters: int = 50,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]
        self.output_dim: int = num_filters * len(kernel_sizes)
        self.char_embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_emb_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Map character-ID sequences to word-level character representations.

        Parameters
        ----------
        char_ids : torch.Tensor of shape ``(N, max_chars)``
            Long tensor of character IDs for *N* words.

        Returns
        -------
        torch.Tensor of shape ``(N, output_dim)``
        """
        embs = self.char_embedding(char_ids)       # (N, max_chars, char_emb_dim)
        embs = embs.transpose(1, 2)                # (N, char_emb_dim, max_chars)
        pooled = [
            torch.relu(conv(embs)).max(dim=2).values  # (N, num_filters)
            for conv in self.convolutions
        ]
        return self.dropout(torch.cat(pooled, dim=-1))  # (N, output_dim)
