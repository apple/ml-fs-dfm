import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class ForwardKLDistillationLoss(_Loss):
    """
    τ^2 * KL( p_T^(τ) || p_S^(τ) ) computed from teacher & student logits.

    Args
    ----
    tau : float
        Temperature used on both teacher & student logits. Final loss is multiplied by tau**2.
    topk : int or None
        If set, restrict KL to teacher's top-k support (renormalized).
    topp : float or None
        If set (0< topp <=1), restrict KL to teacher's top-p (nucleus) support (renormalized).
        If both topk and topp are provided, topk takes precedence.
    confidence_weighting : bool
        If True, weight each time-step by (1 - H(p_T)/log(|support|)), up-weighting confident targets.
    reduction : {'mean','sum','none'}
        Reduction across batch/time. 'mean' is a masked mean if mask is provided.
    eps : float
        Numerical epsilon.

    Forward Inputs
    --------------
    teacher_logits : FloatTensor [..., V]
    student_logits : FloatTensor [..., V]
    mask : FloatTensor/BoolTensor [...], broadcastable to per-token loss (no vocab dim).
           Use 1/True for valid tokens (e.g., to ignore padding).
    """
    def __init__(self, tau=1.0, topk=None, topp=None,
                 confidence_weighting=False, reduction='mean', eps=1e-12):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.tau = float(tau)
        self.topk = topk
        self.topp = topp
        self.confidence_weighting = confidence_weighting
        self.reduction = reduction
        self.eps = eps

    def _softmax_with_temp(self, logits):
        return F.log_softmax(logits / self.tau, dim=-1)

    def _build_support_mask_topk(self, teacher_logits):
        V = teacher_logits.size(-1)
        k = min(self.topk, V)
        _, idx = torch.topk(teacher_logits, k=k, dim=-1)
        mask = torch.zeros_like(teacher_logits, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return mask

    def _build_support_mask_topp(self, logp_T):
        # logp_T: [..., V] (already temperature-scaled)
        p = logp_T.exp()
        p_sorted, idx = p.sort(dim=-1, descending=True)
        cdf = p_sorted.cumsum(dim=-1)

        # Keep tokens while cumulative prob <= topp, always keep the first token
        keep_sorted = (cdf <= self.topp)
        keep_sorted[..., 0] = True

        # Map back to original indices
        mask = torch.zeros_like(p, dtype=torch.bool)
        mask.scatter_(-1, idx, keep_sorted)
        return mask

    def forward(self, teacher_logits, student_logits, mask=None, do_not_apply_softmax=False):
        assert teacher_logits.shape == student_logits.shape, "Shapes must match"
        assert teacher_logits.size(-1) == student_logits.size(-1), "Last dim = vocab"
        V = teacher_logits.size(-1)

        # Temperature-scaled log-probs
        if do_not_apply_softmax == False:
            t_logp = self._softmax_with_temp(teacher_logits)      # [..., V]
            s_logp = self._softmax_with_temp(student_logits)      # [..., V]
        else:
            t_logp = teacher_logits
            s_logp = student_logits

        # Build support restriction (optional)
        support_mask = None
        if self.topk is not None:
            support_mask = self._build_support_mask_topk(teacher_logits)
        elif (self.topp is not None) and (0.0 < self.topp <= 1.0):
            support_mask = self._build_support_mask_topp(t_logp)

        # Convert to probs (for masking & entropy); keep logs for stability
        pT_full = t_logp.exp()
        pS_full = s_logp.exp()

        if support_mask is not None:
            # Zero-out outside support and renormalize on support
            pT_masked = torch.where(support_mask, pT_full, torch.zeros_like(pT_full))
            pS_masked = torch.where(support_mask, pS_full, torch.zeros_like(pS_full))
            ZT = pT_masked.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            ZS = pS_masked.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            pT = pT_masked / ZT
            pS = pS_masked / ZS

            log_pT = (pT.clamp_min(self.eps)).log()
            log_pS = (pS.clamp_min(self.eps)).log()

            # Effective support size per position (for confidence weighting)
            support_size = support_mask.sum(dim=-1).clamp_min(1).to(pT.dtype)
        else:
            pT, pS = pT_full, pS_full
            log_pT, log_pS = t_logp, s_logp
            support_size = torch.full_like(pT[..., 0], V, dtype=pT.dtype)

        # Per-token forward KL
        kl_per_token = (pT * (log_pT - log_pS)).sum(dim=-1)   # [...], no vocab dim

        # Optional confidence weighting: w = 1 - H(p_T)/log(|support|)
        if self.confidence_weighting:
            H = -(pT * log_pT).sum(dim=-1)                    # [...]
            Hmax = support_size.clamp_min(2).log()            # avoid log(1)=0
            w = 1.0 - (H / Hmax).clamp(0.0, 1.0)
            kl_per_token = kl_per_token * w

        # Apply token mask (e.g., to drop padding)
        if mask is not None:
            mask = mask.to(kl_per_token.dtype)
            kl_per_token = kl_per_token * mask
            denom = mask.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(kl_per_token.numel(), device=kl_per_token.device, dtype=kl_per_token.dtype)

        # Scale by tau**2 (Hinton et al.)
        kl_per_token = (self.tau ** 2) * kl_per_token

        # Reduction
        if self.reduction == 'none':
            return kl_per_token
        elif self.reduction == 'sum':
            return kl_per_token.sum()
        else:  # 'mean' -> masked mean over positions
            if mask is not None:
                return kl_per_token.sum() / denom
            else:
                return kl_per_token.mean()
