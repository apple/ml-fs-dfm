from abc import ABC, abstractmethod

from torch import nn, Tensor


class Solver(ABC, nn.Module):
    """Abstract base class for solvers."""

    @abstractmethod
    def sample(self, x_0: Tensor = None) -> Tensor:
        ...
