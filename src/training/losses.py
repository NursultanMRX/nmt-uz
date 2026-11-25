"""Softmax-collapse-free log-softmax and cross-entropy.

PyTorch port of Barron, Gnanapragasam & Tschannen, "A Fix for Softmax and
Cross-Entropy" (arXiv:2610.31642). The target (or max) logit is subtracted
exactly, so its gradient survives even when the model is very confident.
"""

import torch
import torch.nn.functional as F


def stable_log_softmax(z: torch.Tensor) -> torch.Tensor:
    is_max = F.one_hot(z.argmax(-1), z.size(-1)).bool()
    shifted = torch.where(is_max, 0.0, z - z.max(-1, keepdim=True).values)
    s = torch.where(is_max, 0.0, shifted.exp()).sum(-1, keepdim=True)
    return shifted - torch.log1p(s)


def stable_cross_entropy(
    z: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Per-row loss, no reduction. z: (N, V) logits, target: (N,) class ids."""
    is_t = F.one_hot(target, z.size(-1)).bool()
    z_t = z.gather(-1, target[:, None])
    shifted = torch.where(is_t, 0.0, z - z_t)
    b = shifted.max(-1, keepdim=True).values  # >= 0 because the target entry is 0
    s = torch.where(is_t, 0.0, (shifted - b).exp()).sum(-1)
    b = b[:, 0]
    nll = b + torch.log1p(s + torch.expm1(-b))
    if label_smoothing == 0.0:
        return nll
    smooth = -stable_log_softmax(z).mean(-1)
    return (1.0 - label_smoothing) * nll + label_smoothing * smooth
