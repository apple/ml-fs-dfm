import torch
from torch import Tensor


def get_nearest_times(time_grid: Tensor, t_discretization: Tensor) -> Tensor:
    distances = torch.cdist(
        time_grid.unsqueeze(1),
        t_discretization.unsqueeze(1),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    nearest_indices = distances.argmin(dim=1)

    return t_discretization[nearest_indices]
