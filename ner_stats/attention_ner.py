"""Attention-based NER model with explicit learned Q, K, V projection matrices.

Architecture
------------
BERT encoder (frozen or fine-tuned)
    ↓  last hidden states  (B, T, d_bert)
Learned self-attention head
    Q = h · W_Q   K = h · W_K   V = h · W_V      (B, T, d_head)
    scores = Q Kᵀ / √d_head     (B, T, T)
    attn   = softmax(scores + padding_mask)
    ctx    = attn · V            (B, T, d_head)
    out    = LayerNorm(h + ctx·W_O)   residual projection back to d_bert
    ↓
Dropout + Linear → num_labels
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput


# ---------------------------------------------------------------------------
# Self-attention head with explicit W_Q, W_K, W_V matrices
# ---------------------------------------------------------------------------

class NERSelfAttentionHead(nn.Module):
    """Single-head scaled dot-product attention with learnable Q/K/V projections."""

    def __init__(self, d_model: int, d_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_head = d_head

        # The three projection matrices that the model learns
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)

        # Output projection back to d_model for residual connection
        self.W_O = nn.Linear(d_head, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, d_model)
            attention_mask: (B, T)  1 for real tokens, 0 for padding

        Returns:
            output       : (B, T, d_model)  attended & residual-projected states
            attn_weights : (B, T, T)        attention probability matrix
        """
        Q = self.W_Q(hidden_states)  # (B, T, d_head)
        K = self.W_K(hidden_states)  # (B, T, d_head)
        V = self.W_V(hidden_states)  # (B, T, d_head)

        # Scaled dot-product attention scores
        scale = math.sqrt(self.d_head)
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # (B, T, T)

        # Mask padding positions with a large negative value
        if attention_mask is not None:
            # (B, 1, T) broadcast over query dimension
            pad_mask = (1.0 - attention_mask.unsqueeze(1).float()) * -1e9
            scores = scores + pad_mask

        attn_weights = F.softmax(scores, dim=-1)           # (B, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.bmm(attn_weights, V)               # (B, T, d_head)
        context = self.W_O(context)                        # (B, T, d_model)

        # Residual + layer normalisation
        output = self.layer_norm(hidden_states + context)  # (B, T, d_model)
        return output, attn_weights


# ---------------------------------------------------------------------------
# Full NER model: BERT + attention head + classifier
# ---------------------------------------------------------------------------

class AttentionNERModel(nn.Module):
    """BERT encoder with a task-specific self-attention head for token classification.

    The attention head's W_Q, W_K, W_V matrices are learned entirely from the
    NER training signal; BERT weights can be frozen or co-trained.
    """

    def __init__(
        self,
        bert_model_name: str,
        num_labels: int,
        d_head: int = 256,
        attn_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels

        self.bert = AutoModel.from_pretrained(bert_model_name)
        d_model: int = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.attention_head = NERSelfAttentionHead(d_model, d_head, dropout=attn_dropout)
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # (B, T, d_bert) — last hidden states from BERT
        h = bert_outputs.last_hidden_state

        # Apply learned Q/K/V self-attention
        attended, attn_weights = self.attention_head(h, attention_mask)

        logits = self.classifier(self.dropout(attended))  # (B, T, num_labels)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=(attn_weights,),
        )

    def get_entity_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (attended_hidden_states, attn_weights) without classification."""
        with torch.no_grad():
            bert_out = self.bert(input_ids, attention_mask, token_type_ids)
            attended, attn_weights = self.attention_head(
                bert_out.last_hidden_state, attention_mask
            )
        return attended, attn_weights
