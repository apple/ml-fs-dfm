#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from torch.nn.modules.loss import _Loss
from flow_matching.loss import MixturePathGeneralizedKL


class AdaptiveCEKLLoss(_Loss):
    def __init__(self, path, temperature=1.0, gamma=1.0, reduction="none"):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.reduction = reduction
        self.kl_fn = MixturePathGeneralizedKL(path=path, reduction=reduction)

    def forward(self, student_logits, x_next, path_sample, dt):
        """
        Args:
            student_logits: [B, L, V]
            x_next: [B, L]
            path_sample: namespace with fields x_t [B, L], t [B]
            dt: [B, 1]
        Returns:
            loss: scalar tensor
            dict_loss: dict of components
        """

        # Cross-Entropy Loss
        logits_flat = student_logits.view(-1, student_logits.size(-1))  # [B*L, V]
        targets_flat = x_next.view(-1)  # [B*L]
        ce_loss_all = F.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        )  # [B*L]
        ce_loss = ce_loss_all.view_as(x_next).mean(dim=1)  # [B]

        # KL Loss
        kl_loss = self.kl_fn(
            logits=student_logits / self.temperature,
            x_1=x_next,
            x_t=path_sample.x_t,
            t=path_sample.t,
        ).mean(
            dim=1
        )  # [B]

        # Weighting based on dt
        alpha = (dt / 1.0).clamp(0, 1).pow(self.gamma).squeeze(-1)  # [B]

        # Combined Loss
        loss_per_example = (1.0 - alpha) * ce_loss + alpha * kl_loss
        total_loss = loss_per_example.mean()

        dict_loss = {
            "ce_loss": ce_loss.mean().item(),
            "kl_loss": kl_loss.mean().item(),
            "alpha": alpha.mean().item(),
        }

        return total_loss, dict_loss
