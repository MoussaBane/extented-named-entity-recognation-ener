"""Supervised Contrastive Loss and projection head for entity-aware NER training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Given a batch of (embedding, label) pairs, the loss pulls embeddings that
    share the same entity label together and pushes differently-labelled
    embeddings apart.  Tokens marked with label ``-100`` are silently ignored.

    The formulation follows the original paper:

    .. math::

        L = \\sum_{i} \\frac{-1}{|P(i)|}
            \\sum_{p \\in P(i)}
            \\log \\frac{\\exp(z_i \\cdot z_p / \\tau)}
                       {\\sum_{a \\neq i} \\exp(z_i \\cdot z_a / \\tau)}

    where :math:`P(i)` is the set of indices with the same label as anchor *i*
    and :math:`\\tau` is the temperature.

    Parameters
    ----------
    temperature : float
        Softmax temperature :math:`\\tau`.  Default: 0.07.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the supervised contrastive loss.

        Parameters
        ----------
        embeddings : torch.Tensor of shape ``(N, D)``
            L2-normalised embeddings (caller is responsible for normalisation).
        labels : torch.Tensor of shape ``(N,)``
            Integer class ids.  Entries equal to ``-100`` are excluded.

        Returns
        -------
        torch.Tensor
            Scalar loss value (zero-gradient-safe when no positives exist).
        """
        valid = labels != -100
        if valid.sum() < 2:
            # No valid pairs — return zero while keeping the gradient graph alive.
            return embeddings.sum() * 0.0

        emb = F.normalize(embeddings[valid], dim=1)
        lab = labels[valid]
        n = emb.shape[0]

        sim = torch.matmul(emb, emb.T) / self.temperature   # (n, n)
        # Numerical stability: subtract per-row maximum (does not change softmax).
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(n, dtype=torch.bool, device=lab.device)
        pos_mask = (lab.unsqueeze(0) == lab.unsqueeze(1)) & ~eye  # same label, ≠ self

        exp_sim = torch.exp(sim)
        # Denominator: all other tokens (exclude self).
        denom = (exp_sim * (~eye).float()).sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_prob = sim - torch.log(denom)  # (n, n)

        n_pos = pos_mask.float().sum(dim=1)                              # (n,)
        loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos.clamp(min=1)

        has_pos = n_pos > 0
        if has_pos.sum() == 0:
            return embeddings.sum() * 0.0

        return loss_per[has_pos].mean()


class ProjectionHead(nn.Module):
    """Two-layer MLP that maps BERT hidden states to a contrastive embedding space.

    The output is L2-normalised so cosine similarity equals the dot product.

    Parameters
    ----------
    input_dim : int
        Dimension of the input (BERT hidden size, typically 768).
    hidden_dim : int
        Dimension of the hidden layer.  Default: 256.
    output_dim : int
        Dimension of the contrastive output space.  Default: 128.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise the input.

        Parameters
        ----------
        x : torch.Tensor of shape ``(..., input_dim)``

        Returns
        -------
        torch.Tensor of shape ``(..., output_dim)``
        """
        return F.normalize(self.net(x), dim=-1)
