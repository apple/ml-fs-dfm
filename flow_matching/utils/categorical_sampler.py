import torch
from torch import Tensor


def categorical(probs: Tensor) -> Tensor:
    r"""Categorical sampler according to weights in the last dimension of ``probs`` using :func:`torch.multinomial`.

    Args:
        probs (Tensor): probabilities.

    Returns:
        Tensor: Samples.
    """

    return torch.multinomial(probs.flatten(0, -2), 1, replacement=True).view(
        *probs.shape[:-1]
    )


def categorical_argmax(probs: Tensor) -> Tensor:
    return torch.argmax(probs, dim=-1)
