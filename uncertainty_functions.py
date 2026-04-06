"""This file defines various uncertainty functions that can be used to adjust the conformal thresholds"""
import torch


def logit_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of prediction distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def smx_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of prediction distribution from softmax scores."""
    return -(x * x.log()).sum(1)


# ===== NEW: Alternative uncertainty functions =====
def gini_impurity(x: torch.Tensor) -> torch.Tensor:
    """Gini impurity from logits: 1 - Σp². Bounded in [0, 1)."""
    p = x.softmax(1)
    return 1 - (p ** 2).sum(1)


def top2_margin(x: torch.Tensor) -> torch.Tensor:
    """Top-2 margin uncertainty from logits: 1 - (p1 - p2).
    Small margin = uncertain = high value. Bounded in [0, 1]."""
    p = x.softmax(1)
    top2 = p.topk(2, dim=1).values
    return 1 - (top2[:, 0] - top2[:, 1])
# ===== NEW: Normalized versions — mapped to [0, ∞) to match entropy's range =====
def gini_normalized(x: torch.Tensor) -> torch.Tensor:
    """Normalized Gini: -log(1 - gini) from logits. Maps [0,1) -> [0, ∞)."""
    p = x.softmax(1)
    gini = 1 - (p ** 2).sum(1)
    return -torch.log(1 - gini + 1e-10)


def top2_margin_normalized(x: torch.Tensor) -> torch.Tensor:
    """Normalized top-2 margin: -log(p1 - p2) from logits. Maps (0,1] -> [0, ∞).
    Small margin = uncertain = high value."""
    p = x.softmax(1)
    top2 = p.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return -torch.log(margin + 1e-10)
# ===== END NEW =====
