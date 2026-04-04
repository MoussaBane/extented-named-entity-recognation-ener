"""Utilities for extracting token embeddings with Hugging Face Transformers."""

from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


def get_last_hidden_token_embeddings(
    sentence: str,
    model_name: str = "dbmdz/bert-base-turkish-cased",
) -> torch.Tensor:
    """
    Return token embeddings from the model's last hidden layer.

    The returned tensor excludes special tokens (e.g., [CLS], [SEP]) and has shape:
    (tokens, hidden_size)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        return_special_tokens_mask=True,
    )

    model_inputs = {
        key: value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    last_hidden = outputs.last_hidden_state.squeeze(0)
    special_mask = encoded["special_tokens_mask"].squeeze(0).bool()
    token_embeddings = last_hidden[~special_mask]

    return token_embeddings


def get_last_hidden_tokens_and_embeddings(
    sentence: str,
    model_name: str = "dbmdz/bert-base-turkish-cased",
) -> Tuple[List[str], torch.Tensor]:
    """
    Return token strings and their last-hidden-layer embeddings.

    The returned tokens exclude special tokens such as [CLS] and [SEP].
    The embeddings tensor has shape (tokens, hidden_size).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        return_special_tokens_mask=True,
    )

    model_inputs = {
        key: value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    last_hidden = outputs.last_hidden_state.squeeze(0)
    special_mask = encoded["special_tokens_mask"].squeeze(0).bool()
    token_embeddings = last_hidden[~special_mask]
    token_ids = encoded["input_ids"].squeeze(0)[~special_mask]
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())

    return tokens, token_embeddings
