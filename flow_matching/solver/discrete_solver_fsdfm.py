from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union, Type

import torch
from torch import Tensor
from torch.nn import functional as F

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper
from flow_matching.solver.utils import get_nearest_times

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class BaseJumperSolver(Solver):
    def __init__(
        self, model, path, vocabulary_size, source_distribution_p=None, mask_token=-1
    ):
        super().__init__()
        self.model = model
        self.path = path
        if mask_token == -1:
            self.vocabulary_size = vocabulary_size
        else:
            self.vocabulary_size = vocabulary_size + 1
        self.mask_token = mask_token
        self.source_distribution_p = source_distribution_p

    def finite_probs_to_generator_differentiable(
        self,
        probs: torch.Tensor,  # [B, L, V]  – raw teacher/student logits
        x_ref: torch.Tensor,  # [B, L]     – current tokens  x_t
        dt_seg: torch.Tensor,  # [B, 1]     – step size Δt
        t: torch.Tensor,  # [B]        – current time  t
        eps: float = 1e-8,
        barrier: float = 5.0,
        can_apply_dt: bool = True,
    ) -> torch.Tensor:
        return self.finite_probs_to_generator(
            probs, x_ref, dt_seg, t, eps, barrier, can_apply_dt
        )

    def finite_logits_to_generator(
        self,
        logits: torch.Tensor,  # [B, L, V]  FVF logits
        x_ref: torch.Tensor,  # [B, L]     current tokens  x_t
        dt_seg: torch.Tensor,  # [B, 1]     step size  dt
        t: torch.Tensor,  # [B]        (kept for API symmetry)
        eps: float = 1e-8,
        can_apply_dt: bool = True,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)  # [B,L,V]

        flow_u = self.finite_probs_to_generator(
            probs=probs,
            x_ref=x_ref,
            dt_seg=dt_seg,
            t=t,
            eps=eps,
            can_apply_dt=can_apply_dt,
        )
        return flow_u

    def finite_probs_to_generator(
        self,
        probs: torch.Tensor,  # [B, L, V]  FVF logits
        x_ref: torch.Tensor,  # [B, L]     current tokens  x_t
        dt_seg: torch.Tensor,  # [B, 1]     step size  dt
        t: torch.Tensor,  # [B]        (kept for API symmetry)
        eps: float = 1e-8,
        barrier: float = 5.0,  # ← “energy” cost for crumbs
        can_apply_dt: bool = True,
    ) -> torch.Tensor:
        eye = F.one_hot(x_ref, num_classes=self.vocabulary_size).to(probs)

        if can_apply_dt:
            dt_seg = dt_seg.clamp_min(eps)  # [B]
            u_raw = (probs - eye) / dt_seg[:, None, None]  # [B,L,V]
        else:
            u_raw = probs - eye

        flow_u = u_raw * (1.0 - eye)  # zero diag
        flow_u.clamp_min_(0.0)  # safety clip
        return flow_u

    def _compute_flow_u(self, x_t, x_1, t, div_free):
        scheduler_output = self.path.scheduler(t=t)
        k_t = scheduler_output.alpha_t
        d_k_t = scheduler_output.d_alpha_t

        delta_1 = F.one_hot(x_1, num_classes=self.vocabulary_size).to(k_t.dtype)
        scale = (d_k_t / (1 - k_t)).view(-1, 1, 1)
        u = scale * delta_1

        if div_free > 0:
            p_0 = self.source_distribution_p[(None,) * x_t.dim()]
            u = u + div_free * d_k_t / (k_t * (1 - k_t)) * (
                (1 - k_t) * p_0 + k_t * delta_1
            )

        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
        u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

        return u

    def logit_to_flow(
        self, logits: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, vocab_size: int
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)  # [B, L, V]
        delta_t = F.one_hot(x_t, num_classes=vocab_size).to(
            dtype=probs.dtype
        )  # [B, L, V]
        flow_u = probs * (1.0 - delta_t)  # zero out self-loop

        return flow_u

    def _step(
        self,
        x_t,
        u,
        h,
        strategy,
        dtype,
        unmask_change=False,
        controlled_unmasking=False,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        x_init,
        step_size,
        div_free=0.0,
        dtype_categorical=torch.float32,
        time_grid=torch.tensor([0.0, 1.0]),
        return_intermediates=False,
        unmask_change=False,
        can_apply_dt: bool = True,
        controlled_unmasking=False,
        verbose=False,
        **model_extras,
    ):

        time_grid = time_grid.to(device=x_init.device)

        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
            t_final = time_grid[-1].item()
        else:
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )
            if return_intermediates:
                order = torch.argsort(time_grid)
                time_grid = get_nearest_times(time_grid, t_discretization)

        x_t = x_init.clone()
        res = [x_init.clone()] if return_intermediates else []
        steps_counter = 0

        ctx = (
            tqdm(total=t_final, desc=f"NFE: {steps_counter}")
            if verbose and TQDM_AVAILABLE
            else nullcontext()
        )

        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]
                if "dt_list" in model_extras:
                    model_extras["dt"] = model_extras["dt_list"][:, i]
                else:
                    model_extras["dt"] = h.repeat(x_t.shape[0])

                p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)

                if i == n_steps - 1:
                    # eye   = F.one_hot(x_init, num_classes=self.vocabulary_size).to(p_1t)
                    # p_1t = p_1t * (1.0 - eye)
                    if self.mask_token != -1:
                        p_1t[..., self.mask_token] = 0.0
                        Z = p_1t.sum(dim=-1, keepdim=True)
                        if self.mask_token != -1:
                            fill = 1.0 / (self.vocabulary_size - 1)
                        else:
                            fill = 1.0 / self.vocabulary_size
                        p_1t = torch.where(
                            Z >= 0 - 1e-10, p_1t / Z, torch.full_like(p_1t, fill)
                        )

                    x_1 = categorical(p_1t.to(dtype=dtype_categorical))
                    if self.mask_token != -1 and not unmask_change:
                        still_masked = x_t == self.mask_token  # boolean [B, L]
                        x_t[still_masked] = x_1[still_masked]  # overwrite only MASKs
                    else:
                        x_t = x_1
                else:
                    u = self.finite_probs_to_generator(
                        p_1t, x_t, h, t=t, can_apply_dt=can_apply_dt
                    )
                    x_t = self._step(
                        x_t,
                        u,
                        h,
                        strategy=self.__class__.__name__,
                        dtype=dtype_categorical,
                        unmask_change=unmask_change,
                        controlled_unmasking=controlled_unmasking,
                        p_1t=p_1t,
                        t=t,
                    )

                if return_intermediates and (t + h in time_grid):
                    res.append(x_t.clone())
                if verbose and TQDM_AVAILABLE:
                    ctx.n = t.item() + h.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")
                steps_counter += 1

        return (
            torch.stack(res, dim=0)[order]
            if return_intermediates and step_size is not None
            else (torch.stack(res, dim=0) if return_intermediates else x_t)
        )

    @torch.no_grad()
    def sample_masked(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        *,
        mask_token_id: Optional[int] = None,
        edit_mask: Optional[Tensor] = None,  # bool mask, same shape as x_init
        div_free: Union[float, Callable[[float], float]] = 0.0,  # kept for API symmetry
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        unmask_change: bool = False,
        controlled_unmasking: bool = False,
        can_apply_dt: bool = True,
        **model_extras,
    ) -> Tensor:
        """
        Masked version of `sample`:
        - Only positions marked editable are allowed to jump/change.
        - Non-editable positions are frozen to their initial values.
        - On the final step, we collapse to x_1 **only** for editable coords.
        """
        time_grid = time_grid.to(device=x_init.device)

        # Determine which positions are editable
        if edit_mask is None:
            assert (
                mask_token_id is not None
            ), "Provide either edit_mask or mask_token_id."
            editable = x_init == mask_token_id
        else:
            editable = edit_mask.to(dtype=torch.bool, device=x_init.device)
            assert editable.shape == x_init.shape, "edit_mask must match x_init.shape"

        # Fast path: nothing to edit
        if not editable.any():
            if return_intermediates:
                return torch.stack([x_init.clone()], dim=0)
            return x_init.clone()

        # Time discretization
        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
            t_final = time_grid[-1].item()
        else:
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )
            if return_intermediates:
                order = torch.argsort(time_grid)
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        x_t = x_init.clone()
        res = [x_init.clone()] if return_intermediates else []
        steps_counter = 0

        if verbose and not TQDM_AVAILABLE:
            raise ImportError("tqdm is required for verbose mode. Please install it.")
        ctx = (
            tqdm(total=t_final, desc=f"NFE: {steps_counter}")
            if (verbose and TQDM_AVAILABLE)
            else nullcontext()
        )

        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # Expose dt to the model (your pipelines use either 'dt_list' or 'step_size_list')
                if "dt_list" in model_extras:
                    model_extras["dt"] = model_extras["dt_list"][:, i]
                elif "step_size_list" in model_extras:
                    model_extras["step_size_model"] = model_extras["step_size_list"][
                        :, i
                    ]
                else:
                    # default: broadcast this h as dt
                    model_extras["dt"] = h.repeat(x_t.shape[0])

                # Model posterior p_{1|t}(· | x_t)
                p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)

                if i == n_steps - 1:
                    # Final collapse: sample x_1 and write ONLY editable positions
                    if (
                        mask_token_id is not None
                        and 0 <= mask_token_id < self.vocabulary_size
                    ):
                        p_1t = p_1t.clone()
                        p_1t[..., mask_token_id] = 0.0
                        Z = p_1t.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        p_1t = p_1t / Z

                    x_1 = categorical(p_1t.to(dtype=dtype_categorical))
                    x_t = torch.where(editable, x_1, x_t)
                else:
                    # Build finite-generator u and zero out non-editable coords
                    u = self.finite_probs_to_generator(
                        probs=p_1t, x_ref=x_t, dt_seg=h, t=t, can_apply_dt=can_apply_dt
                    )
                    u = u * editable[..., None]

                    # One jump step with your subclass strategy
                    x_t = self._step(
                        x_t,
                        u,
                        h,
                        strategy=self.__class__.__name__,
                        dtype=dtype_categorical,
                        unmask_change=unmask_change,
                        controlled_unmasking=controlled_unmasking,
                        p_1t=None,
                        t=None,
                    )

                if return_intermediates and (t + h in time_grid):
                    res.append(x_t.clone())

                if verbose and TQDM_AVAILABLE:
                    ctx.n = t.item() + h.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")
                steps_counter += 1

        if return_intermediates:
            if step_size is None:
                return torch.stack(res, dim=0)
            else:
                return torch.stack(res, dim=0)[order]
        else:
            return x_t


class MixtureDiscreteEleurSolverWithCumulativeScalar(BaseJumperSolver):
    # --------------------------------------------------------------------- #
    # 1.  Turn teacher/student logits into the **average velocity generator**
    # --------------------------------------------------------------------- #
    @staticmethod
    def _alpha_from_kappa(k_t, k_th, dt, eps=1e-8):
        """
        Closed-form α(t,Δt)  =  (ln[(1-k_t)/(1-k_th)]) / Δt
        Shapes: k_t, k_th, dt     →  broadcastable
        """
        num = torch.log(torch.clamp(1.0 - k_t, eps, 1.0))
        den = torch.log(torch.clamp(1.0 - k_th, eps, 1.0))
        return (num - den) / (dt + eps)  # [B] or [B,1]

    def finite_probs_to_generator(
        self,
        probs: torch.Tensor,  # [B, L, V]  – raw teacher/student probs
        x_ref: torch.Tensor,  # [B, L]     – current tokens  x_t
        dt_seg: torch.Tensor,  # [B, 1]     – step size Δt
        t: torch.Tensor,  # [B]        – current time  t
        eps: float = 1e-8,
        barrier: float = 5.0,
        can_apply_dt: bool = True,
    ) -> torch.Tensor:
        """
        Returns ū_{t,Δt}(x,z)  with shape [B, L, V].
        """
        # 2. one-hot for the 'stay' channel δ_z
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(-1, x_ref.unsqueeze(-1), 1.0)

        # 3. α(t,Δt) from scheduler
        k_t = self.path.scheduler(t=t).alpha_t  # [B]   or [B,1]
        k_th = self.path.scheduler(t=t + dt_seg).alpha_t
        alpha = self._alpha_from_kappa(k_t, k_th, dt_seg, eps)  # [B,1] or [B]
        alpha = alpha.view(-1, 1, 1)  # broadcast to [B,1,1]

        # 4. analytic average velocity
        u = alpha * (probs - one_hot)  # [B, L, V]
        u = u * (1.0 - one_hot)  # zero diag
        return u

    # --------------------------------------------------------------------- #
    # 2.  One Euler jump using that generator
    # --------------------------------------------------------------------- #
    def _step(
        self,
        x_t: torch.Tensor,  # [B, L]      current tokens
        u: torch.Tensor,  # [B, L, V]   average velocity
        h: torch.Tensor,  # [B, 1] or scalar  – the same Δt you used
        strategy: str = "",
        dtype: torch.dtype = torch.float32,
        unmask_change: bool = False,
        controlled_unmasking: bool = False,
        p_1t: torch.Tensor = None,  # optional teacher probs if you need them
        t: torch.Tensor = None,  # current time  (kept for API symmetry)
    ) -> torch.Tensor:
        """
        Poisson-jump simulation for one slice of width h.

        We treat *u* as un-normalised **rates**: positive entries are jump
        rates to new tokens, the (single) negative entry is the stay-rate.
        """

        # --- 1. total outgoing intensity λ = Σ_{x≠z} u⁺(x) ----------------
        u_pos = torch.clamp(u, min=0.0)  # keep only positive part
        intensity = u_pos.sum(dim=-1).clamp_min(1e-12)  # [B, L]

        # --- 2. Bernoulli draw: does a jump happen in this slice? ----------
        jump_prob = 1.0 - torch.exp(
            -h.unsqueeze(-1) * intensity
        )  # P(jump in Δt)  [B, L]
        mask_jump = torch.rand_like(jump_prob) < jump_prob

        # --- 3. Sample new tokens where a jump occurs ---------------------
        x_t_new = x_t.clone()
        if mask_jump.any():
            # draw from categorical ∝ u⁺
            u_pos = u_pos / intensity.unsqueeze(-1)
            new_tokens = categorical(u_pos[mask_jump].to(dtype=dtype))
            x_t_new[mask_jump] = new_tokens

        return x_t_new


class MixtureDiscreteEulerSolver(BaseJumperSolver):
    def _step(
        self,
        x_t,
        u,
        h,
        strategy="",
        dtype=torch.float32,
        unmask_change=False,
        controlled_unmasking=False,
        p_1t=None,
        t=None,
    ):
        """
        Fully vectorized _step to support controlled unmasking based on confidence scores.
        """
        eye = F.one_hot(x_t, num_classes=self.vocabulary_size).bool()
        off_diag = u.masked_fill(eye, 0.0).clamp_min(1e-9)
        λ = off_diag.sum(-1).clamp_min(1e-8)  # [B, L]

        k = torch.poisson((λ * h.unsqueeze(-1)).clamp_max(50))

        x_next = x_t.clone()

        if self.mask_token != -1 and not unmask_change:
            still_masked = x_t == self.mask_token
            mask = (k > 0) & still_masked
        else:
            mask = k > 0

        if mask.any():
            probs = off_diag / λ.unsqueeze(-1)

            if controlled_unmasking:
                token_confidences, _ = probs.max(dim=-1)
                # allowed_unmasks = (mask.sum(dim=1).float() * h).clamp_min(1).long()
                allowed_unmasks = (
                    torch.minimum(mask.sum(dim=1).float(), h * x_t.size(1))
                    .clamp_min(1)
                    .long()
                )

                sorted_confidences, sorted_indices = torch.sort(
                    token_confidences.masked_fill(~mask, -float("inf")),
                    dim=-1,
                    descending=True,
                )

                batch_indices = torch.arange(mask.size(0), device=x_t.device).unsqueeze(
                    1
                )
                thresholds = sorted_confidences[
                    batch_indices,
                    allowed_unmasks.clamp_max(mask.size(1) - 1).unsqueeze(1),
                ]

                selected_mask = mask & (token_confidences > thresholds)

                selected_probs = probs[selected_mask]
                dest = torch.multinomial(selected_probs.to(dtype), 1).squeeze(-1)
                x_next[selected_mask] = dest

            else:
                masked_probs = probs[mask].float()
                dest = torch.multinomial(masked_probs.to(dtype), 1).squeeze(-1)
                x_next[mask] = dest

        return x_next

    def _step_org(
        self,
        x_t,
        u,
        h,
        strategy="",
        dtype=torch.float32,
        unmask_change=False,
        controlled_unmasking=False,
        p_1t=None,
    ):
        """
        Poisson thinning version:
        - Each token draws  k ~ Poisson(λ h)  jumps for the whole step.
        - If k == 0  → stay.
        - If k >= 1  → sample ONE destination from the row probs (last jump view).
        This keeps first–order accuracy but allows >1 potential jump per macro step.
        """
        eye = F.one_hot(x_t, num_classes=self.vocabulary_size).bool()  # for safety
        off_diag = u.masked_fill(eye, 0.0)  # ensure u_ii = 0
        λ = off_diag.sum(-1)  # [B, L] row intensity
        k = torch.poisson((λ * h.unsqueeze(-1)).clamp_max(50))  # Poisson jump counts
        if self.mask_token != -1 and not unmask_change:
            still_masked = x_t == self.mask_token  # jump only while masked
            mask = (k > 0) & still_masked  # final boolean mask
        else:
            mask = (k > 0).bool()
        x_next = x_t.clone()
        if mask.any():
            # ---- destination probabilities   p_ij = u_ij / λ_i ----
            probs = off_diag[mask] / λ[mask].unsqueeze(-1)  # normalise
            dest = torch.multinomial(probs.to(dtype), 1).squeeze(-1)
            x_next[mask] = dest

        return x_next

    def _step_1(
        self,
        x_t,
        u,
        h,
        strategy="",
        dtype=torch.float32,
        unmask_change=False,
        controlled_unmasking=False,
        p_1t=None,
    ):
        """
        Optimized _step to support controlled unmasking based on confidence scores (vectorized).
        """
        eye = F.one_hot(x_t, num_classes=self.vocabulary_size).bool()
        off_diag = u.masked_fill(eye, 0.0)
        λ = off_diag.sum(-1)  # [B, L]

        k = torch.poisson((λ * h.unsqueeze(-1)).clamp_max(50))

        x_next = x_t.clone()

        if self.mask_token != -1 and not unmask_change:
            still_masked = x_t == self.mask_token
            mask = (k > 0) & still_masked
        else:
            mask = k > 0

        if mask.any():
            probs = off_diag / λ.unsqueeze(-1).clamp_min(1e-8)
            if controlled_unmasking:
                B, L = x_t.size()
                for b in range(B):
                    tokens_to_unmask = mask[b].nonzero(as_tuple=True)[0]
                    num_tokens = tokens_to_unmask.size(0)

                    # Calculate allowed tokens to unmask based on step size (h)
                    allowed_unmasks = max(1, int(num_tokens * h[b].item()))

                    if allowed_unmasks < num_tokens:
                        # Select top tokens based on confidence
                        token_confidences, _ = probs[b, tokens_to_unmask].max(dim=-1)
                        top_confidence_indices = torch.topk(
                            token_confidences, allowed_unmasks
                        ).indices
                        selected_tokens = tokens_to_unmask[top_confidence_indices]
                    else:
                        selected_tokens = tokens_to_unmask

                    # Sample destinations only for selected tokens
                    dest_probs = probs[b, selected_tokens]
                    dest = torch.multinomial(dest_probs.to(dtype), 1).squeeze(-1)
                    x_next[b, selected_tokens] = dest

            else:
                # Original behavior: unmask all eligible tokens
                masked_probs = probs[mask]
                dest = torch.multinomial(masked_probs.to(dtype), 1).squeeze(-1)
                x_next[mask] = dest

        return x_next

    def _step_0(self, x_t, u, h, strategy="", dtype=torch.float32):
        intensity = u.sum(dim=-1)
        p_jump = 1.0 - torch.exp(-h * intensity)
        mask = torch.rand_like(intensity) < p_jump
        x_next = x_t.clone()
        if mask.sum() > 0:
            x_next[mask] = categorical(u[mask].to(dtype=dtype))
        return x_next


# remove rest
SOLVER_REGISTRY: dict[str, Type[Solver]] = {
    "mixture_euler": MixtureDiscreteEulerSolver,
    "mixture_euler_with_cumulative_scalar": MixtureDiscreteEleurSolverWithCumulativeScalar,
}


def get_solver_by_name(name: str) -> Type[Solver]:
    if name not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver name '{name}'. Available: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[name]
