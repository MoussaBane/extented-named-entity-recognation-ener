"""Utilities for extracting token embeddings with Hugging Face Transformers."""

from typing import List, Sequence, Tuple, Optional

import torch
from transformers import AutoModel, AutoTokenizer


class TransformerEmbedder:
    """Reusable embedder that keeps tokenizer/model in memory and supports device placement.

    Parameters
    ----------
    model_name: str
        Hugging Face model id or local directory.
    device: str or torch.device
        Device to place the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, model_name: str = "dbmdz/bert-base-turkish-cased", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = torch.device(device if device is not None else ("cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        try:
            self.model.to(self.device)
        except Exception:
            # If device placement fails, keep model on default device
            pass

    def encode_sentence_subwords(self, sentence: str) -> Tuple[List[str], torch.Tensor]:
        """Return non-special subword tokens and embeddings from last hidden state.

        Returns tokens (list of strings) and embeddings tensor with shape
        (num_subwords, hidden_size). The tensor is on the same device as the model.
        """
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            return_special_tokens_mask=True,
        )

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        last_hidden = outputs.last_hidden_state.squeeze(0)
        special_mask = encoded["special_tokens_mask"].squeeze(0).bool()
        token_embeddings = last_hidden[~special_mask]
        token_ids = encoded["input_ids"].squeeze(0)[~special_mask]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
        return tokens, token_embeddings

    def encode_words(self, words: Sequence[str], max_length: int = 256) -> Tuple[List[str], torch.Tensor]:
        """Return one embedding per original word using first-subtoken alignment.

        This mirrors common token-classification label alignment where labels are
        applied to the first subword and the remaining subwords are ignored.
        The returned embeddings tensor is placed on the embedder's device.
        """
        encoded = self.tokenizer(
            list(words),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        hidden = outputs.last_hidden_state.squeeze(0)
        word_ids = encoded.word_ids(batch_index=0)

        aligned_tokens: List[str] = []
        aligned_embeddings: List[torch.Tensor] = []
        previous_word_id = None
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word_id:
                previous_word_id = word_id
                continue

            aligned_tokens.append(words[word_id])
            aligned_embeddings.append(hidden[token_index])
            previous_word_id = word_id

        if not aligned_embeddings:
            hidden_size = self.model.config.hidden_size
            return [], torch.empty((0, hidden_size), dtype=hidden.dtype, device=self.device)

        return aligned_tokens, torch.stack(aligned_embeddings, dim=0)


def get_last_hidden_token_embeddings(
    sentence: str,
    model_name: str = "dbmdz/bert-base-turkish-cased",
    embedder: Optional[TransformerEmbedder] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Return token embeddings from the model's last hidden layer.

    If an `embedder` instance is provided it will be reused; otherwise a
    temporary `TransformerEmbedder` is created with `model_name` and `device`.
    The returned tensor excludes special tokens and has shape (tokens, hidden_size).
    """
    if embedder is None:
        embedder = TransformerEmbedder(model_name=model_name, device=device)
    _, embeddings = embedder.encode_sentence_subwords(sentence)
    return embeddings


def get_last_hidden_tokens_and_embeddings(
    sentence: str,
    model_name: str = "dbmdz/bert-base-turkish-cased",
    embedder: Optional[TransformerEmbedder] = None,
    device: Optional[str] = None,
) -> Tuple[List[str], torch.Tensor]:
    """
    Return token strings and their last-hidden-layer embeddings.

    If `embedder` is provided it will be used; otherwise a temporary
    `TransformerEmbedder` is created.
    """
    if embedder is None:
        embedder = TransformerEmbedder(model_name=model_name, device=device)
    return embedder.encode_sentence_subwords(sentence)


def get_word_aligned_embeddings(
    words: Sequence[str],
    model_name: str = "dbmdz/bert-base-turkish-cased",
    embedder: Optional[TransformerEmbedder] = None,
    max_length: int = 256,
    device: Optional[str] = None,
) -> Tuple[List[str], torch.Tensor]:
    """Return first-subtoken-aligned embeddings for an input word sequence.

    If an `embedder` is provided it will be used; otherwise a temporary
    `TransformerEmbedder` is created with `model_name` and `device`.
    """
    if embedder is None:
        embedder = TransformerEmbedder(model_name=model_name, device=device)
    return embedder.encode_words(words=words, max_length=max_length)
