from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def mixup(
    data: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixup data and labels.

    Args:
        data: torch.Tensor, torch.float
            shape => (N, C, T), (N, C, H, W) or (N, C, T, H, W)
        labels: torch.Tensor, torch.long, (N, )
        alpha: hypter-parameter which controls the strength of interpolation
            between feature-target pairs as in the original paper.
    Return:
        mixed_data: the shape is the same as data of Args.
        labels: original labels
        labels[index]: randomly permuted labels
        lamb: lambda sampled from beta distribution
    Reference:
        H. Zhang+ "mixup: BEYOND EMPIRICAL RISK MINIMIZATION" in ICLR2018
    """

    device = data.device
    batch_size = data.shape[0]

    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    index = torch.randperm(batch_size).to(device)

    mixed_data = lamb * data + (1 - lamb) * data[index]

    return mixed_data, labels, labels[index], lamb


class MixupCrossentropy(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )

    def forward(
        self,
        preds: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
        lamb: float,
    ) -> torch.Tensor:

        if lamb < 0 or 1.0 < lamb:
            raise (ValueError("lamb must be 0 <= lamb <= 1"))

        loss = lamb * self.criterion(preds, labels1)
        loss += (1 - lamb) * self.criterion(preds, labels2)

        return loss
