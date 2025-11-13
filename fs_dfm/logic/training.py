#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import math
from contextlib import nullcontext
from typing import Optional, Sequence
from omegaconf.dictconfig import DictConfig

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from flow_matching.loss import MixturePathGeneralizedKL, ForwardKLDistillationLoss
from flow_matching.path import ProbPath
from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical

from utils.logging import TrainLogger

from .flow import SourceDistribution
from .adaptive_ce_kl_loss import AdaptiveCEKLLoss
from .state import TrainState
from utils.metrics import log_metrics


def _get_lr(
    lr: float,
    step: int,
    warmup: int,
    n_iters: int,
    eta_min_ratio: float,
    constant_lr=False,
):
    if constant_lr:
        return lr

    if step < warmup:
        return lr * (step / warmup)
    else:
        eta_min = eta_min_ratio * lr
        cosine_decay = 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / (n_iters - warmup))
        )
        return eta_min + (lr - eta_min) * cosine_decay


def optimization_step(
    state: TrainState,
    scaler_teacher: GradScaler,
    scaler_student: GradScaler,
    loss_teacher: Optional[Tensor],
    loss_student: Optional[Tensor],
    optim_params: DictConfig,
    logger: TrainLogger,
    train_teacher: bool = True,
    train_student: bool = True,
) -> None:
    if loss_teacher is None:
        train_teacher = False

    # === Backward passes ===
    if train_teacher:
        scaler_teacher.scale(loss_teacher).backward()
        scaler_teacher.unscale_(state.optimizer)

    if train_student:
        scaler_student.scale(loss_student).backward()
        scaler_student.unscale_(state.student_optimizer)

    # === Learning rate schedule ===
    lr = _get_lr(
        lr=optim_params.lr,
        step=state.step,
        warmup=optim_params.warmup,
        n_iters=optim_params.n_iters,
        eta_min_ratio=optim_params.eta_min_ratio,
        constant_lr=optim_params.constant_lr,
    )

    teacher_lr = _get_lr(
        lr=optim_params.teacher_lr,
        step=state.step,
        warmup=optim_params.warmup,
        n_iters=optim_params.n_iters,
        eta_min_ratio=optim_params.eta_min_ratio,
        constant_lr=optim_params.constant_lr,
    )

    if train_teacher:
        for g in state.optimizer.param_groups:
            g["lr"] = teacher_lr
    if train_student:
        for g in state.student_optimizer.param_groups:
            g["lr"] = lr

    if state.step % optim_params.log_lr_every == 0:
        logger.log_lr(value=lr, step=state.step)

    if state.step % optim_params.log_lr_every == 0:
        logger.log_lr_teacher(value=teacher_lr, step=state.step)

    # === Gradient clipping ===
    if optim_params.grad_clip and optim_params.grad_clip > 0:
        if train_teacher:
            torch.nn.utils.clip_grad_norm_(
                state.model.parameters(), max_norm=optim_params.grad_clip
            )
        if train_student:
            torch.nn.utils.clip_grad_norm_(
                state.student_model.parameters(), max_norm=optim_params.grad_clip
            )

    # === Optimizer step ===
    if train_teacher and loss_teacher is not None:
        scaler_teacher.step(state.optimizer)
        scaler_teacher.update()
    if train_student:
        scaler_student.step(state.student_optimizer)
        scaler_student.update()

    # === Zero gradients ===
    if train_teacher:
        state.optimizer.zero_grad(set_to_none=True)
    state.student_optimizer.zero_grad(set_to_none=True)


def sample_weighted_dt_uniform_t(
    step_sizes,  # list or 1D tensor of dt options
    dt_weights,  # same length as step_sizes, manual weights (>=0)
    sampling_steps: int,
    batch_size: int,
    time_epsilon: float,
    device=None,
):
    device = device or "cpu"
    step_sizes_t = torch.as_tensor(step_sizes, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(dt_weights, dtype=torch.float32, device=device)

    valid_mask = step_sizes_t <= (1.0 - time_epsilon)
    if not torch.any(valid_mask):
        raise ValueError("No valid dt: all violate t + dt ≤ 1 - time_epsilon.")

    step_sizes_v = step_sizes_t[valid_mask]
    weights_v = torch.clamp(weights_t[valid_mask], min=0)

    if torch.all(weights_v == 0):
        # Fallback to uniform if all provided weights are zero on the valid set
        probs = torch.full_like(weights_v, 1.0 / weights_v.numel())
    else:
        probs = weights_v / weights_v.sum()

    idx = torch.multinomial(probs, num_samples=batch_size, replacement=True)
    dt = step_sizes_v[idx]

    # Compute the max valid t-index for each sampled dt
    max_indices = torch.floor((1.0 - time_epsilon - dt) * sampling_steps).to(torch.long)

    # Sample t-index uniformly from [0, max_index] (inclusive). No clamping needed.
    t_indices = torch.floor(
        torch.rand(batch_size, device=device) * (max_indices.to(torch.float32) + 1.0)
    ).to(torch.long)

    # Map to [0,1) grid
    t = t_indices.to(torch.float32) / float(sampling_steps)
    return t, dt


def step(
    state: TrainState,
    teacher_loss_fn: nn.Module,
    student_loss_fn: nn.Module,
    path: ProbPath,
    scaler_teacher: GradScaler,
    scaler_student: GradScaler,
    iterator: DataLoader,
    device: torch.device,
    source_distribution: SourceDistribution,
    logger: TrainLogger,
    train_teacher: bool = True,
    train_student: bool = True,
    optim_params: Optional[DictConfig] = None,
    time_epsilon: float = 0.0,
    step_sizes: Optional[Sequence[float]] = None,
    sampling_steps: int = 1024,
    vocab_size: int = 100000,
    distill_th: float = 1 / 513.0,
    solver: Solver = None,
    teacher_type: str = "RK_4",
    unmask_change=False,
    controlled_unmasking=False,
    blend_logits=False,
    can_apply_dt=True,
    dt_weights: Optional[Sequence[float]] = None,
    use_generator_not_logic: bool = False,
    ema_freq: int = 1000,
    just_student: bool = False,
    dt_weights_2: bool = False,
    dt_weights_2_freq: int = 10000,
) -> Tensor:
    state.train_student()
    state.train()

    x_1 = next(iterator)["input_ids"].to(device)
    x_0 = source_distribution.sample_like(x_1)

    # Sample from path
    with torch.no_grad():
        if step_sizes is not None:
            if dt_weights is not None:
                if dt_weights_2:
                    temp = state.step // dt_weights_2_freq
                    dt_weights = [min(v * (2**temp), 1024) for v in dt_weights]
                t, dt = sample_weighted_dt_uniform_t(
                    step_sizes=step_sizes,
                    dt_weights=dt_weights,
                    sampling_steps=sampling_steps,
                    batch_size=x_1.shape[0],
                    time_epsilon=time_epsilon,
                    device=device,
                )
            else:
                idx = torch.randint(0, len(step_sizes), (x_1.shape[0],), device=device)
                dt = torch.tensor(step_sizes, device=device)[idx]
                max_indices = (
                    ((1.0 - time_epsilon - dt) * sampling_steps)
                    .floor()
                    .clamp(min=0, max=sampling_steps - 1)
                    .long()
                )
                t_indices = torch.floor(
                    torch.rand(x_1.shape[0], device=device)
                    * (max_indices.to(torch.float32) + 1.0)
                ).to(torch.long)
                t_indices = torch.minimum(t_indices, max_indices)
                t = t_indices.float() / float(sampling_steps)
        else:
            t = torch.rand(x_1.shape[0], device=device) * (1.0 - time_epsilon)
            max_dt = 1.0 - time_epsilon - t
            dt = torch.rand_like(t) * max_dt

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    # Forward and compute loss
    ctx_teacher = nullcontext() if train_teacher else torch.no_grad()
    ctx_student = nullcontext() if train_student else torch.no_grad()

    loss_teacher = None
    loss_student = None
    dict_loss = None

    if not just_student:
        with ctx_teacher:
            loss_teacher = teacher_trainer(state, path_sample, teacher_loss_fn, x_1)

        with ctx_student:
            loss_student, dict_loss = student_trainer(
                state=state,
                path_sample=path_sample,
                dt=dt,
                loss_fn=student_loss_fn,
                vocab_size=vocab_size,
                distill_th=distill_th,
                solver=solver,
                teacher_type=teacher_type,
                train_student=train_student,
                x_1=x_1,
                unmask_change=unmask_change,
                controlled_unmasking=controlled_unmasking,
                blend_logits=blend_logits,
                can_apply_dt=can_apply_dt,
                path=path,
                use_generator_not_logic=use_generator_not_logic,
            )
    else:
        with ctx_student:
            loss_student, dict_loss = just_student_trainer(
                state=state,
                path_sample=path_sample,
                dt=dt,
                loss_fn=student_loss_fn,
                vocab_size=vocab_size,
                distill_th=distill_th,
                solver=solver,
                teacher_type=teacher_type,
                train_student=train_student,
                x_1=x_1,
                unmask_change=unmask_change,
                controlled_unmasking=controlled_unmasking,
                blend_logits=blend_logits,
                can_apply_dt=can_apply_dt,
                path=path,
                use_generator_not_logic=use_generator_not_logic,
            )

    # Optimization step (only if training=true)
    if train_teacher or train_student:
        optimization_step(
            state=state,
            scaler_teacher=scaler_teacher,
            scaler_student=scaler_student,
            loss_teacher=loss_teacher,
            loss_student=loss_student,
            optim_params=optim_params,
            logger=logger,
            train_teacher=train_teacher,
            train_student=train_student,
        )

    if state.use_ema and state.step % ema_freq == 0:
        if state.step % 2001 == 0:
            logger.info(f"EMA is started for step {state.step}")
        state.update_student_ema()
        if state.step % 2001 == 0:
            logger.info(f"EMA is finished for step {state.step}")

    loss_teacher = loss_teacher.detach() if loss_teacher is not None else None
    loss_student = loss_student.detach() if loss_student is not None else None

    return loss_teacher, loss_student, dict_loss


def teacher_trainer(state, path_sample, loss_fn, x_1):
    logits = state.model(x_t=path_sample.x_t, time=path_sample.t)

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        loss = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
    elif isinstance(loss_fn, MixturePathGeneralizedKL):
        loss = loss_fn(
            logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t
        ).mean()
    else:
        raise ValueError("Invalid loss function")

    return loss


def student_trainer(
    state,
    path_sample,
    dt,
    loss_fn,
    vocab_size,
    distill_th,
    solver,
    teacher_type="RK_4",
    train_student=True,
    x_1=None,
    unmask_change=False,
    controlled_unmasking=False,
    blend_logits=False,
    can_apply_dt=True,
    path=None,
    use_generator_not_logic=False,
):
    x_t = path_sample.x_t
    t = path_sample.t

    logits = state.student_model(x_t, t, dt=dt)
    probs = torch.softmax(logits, dim=-1)
    u_raw = solver.finite_probs_to_generator_differentiable(
        probs, x_t, dt, t=t, can_apply_dt=can_apply_dt
    )

    if not train_student:
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
        elif isinstance(loss_fn, MixturePathGeneralizedKL):
            loss = loss_fn(logits=logits, x_1=x_1, x_t=x_t, t=t).mean()
        elif isinstance(loss_fn, ForwardKLDistillationLoss):
            loss_temp_fn = nn.CrossEntropyLoss()
            loss = loss_temp_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
        else:
            raise ValueError("Invalid loss function")

        return loss, None

    if teacher_type == "RK_4":
        teacher_logits, x_next, teacher_u = get_RK_4_estimate(
            state,
            t,
            dt,
            x_t,
            solver,
            vocab_size,
            distill_th,
            unmask_change=unmask_change,
            controlled_unmasking=controlled_unmasking,
            blend_logits=blend_logits,
            can_apply_dt=can_apply_dt,
            path=path,
            use_generator_not_logic=use_generator_not_logic,
        )
    elif teacher_type == "heun_average":
        teacher_logits, x_next, teacher_u = heun_average(
            state,
            t,
            dt,
            x_t,
            solver,
            vocab_size,
            distill_th,
            unmask_change=unmask_change,
            controlled_unmasking=controlled_unmasking,
            blend_logits=blend_logits,
            can_apply_dt=can_apply_dt,
            path=path,
            use_generator_not_logic=use_generator_not_logic,
        )
    else:
        raise ValueError("Invalid Teacher Type!!")

    dict_loss = None
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        loss = loss_fn(logits.flatten(0, 1), x_next.flatten(0, 1)).mean()
    elif isinstance(loss_fn, MixturePathGeneralizedKL):
        loss = loss_fn(
            logits=logits, x_1=x_next, x_t=path_sample.x_t, t=path_sample.t
        ).mean()
    elif isinstance(loss_fn, AdaptiveCEKLLoss):
        loss, dict_loss = loss_fn(
            student_logits=logits, x_next=x_next, x_t=path_sample.x_t, dt=dt
        )
    elif isinstance(loss_fn, ForwardKLDistillationLoss):
        if not use_generator_not_logic:
            loss = loss_fn(teacher_logits, logits, do_not_apply_softmax=False)
        else:
            student_log_probs = F.log_softmax(u_raw, dim=-1)
            teacher_probs = F.softmax(teacher_u, dim=-1)  # if teacher_u are logits
            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    else:
        raise ValueError("Invalid loss function")

    return loss, dict_loss


def just_student_trainer(
    state,
    path_sample,
    dt,
    loss_fn,
    vocab_size,
    distill_th,
    solver,
    teacher_type="RK_4",
    train_student=True,
    x_1=None,
    unmask_change=False,
    controlled_unmasking=False,
    blend_logits=False,
    can_apply_dt=True,
    path=None,
    use_generator_not_logic=False,
):
    x_t = path_sample.x_t
    t = path_sample.t

    logits = state.student_model(x_t, t, dt=dt)
    probs = torch.softmax(logits, dim=-1)
    u_raw = solver.finite_probs_to_generator_differentiable(
        probs, x_t, dt, t=t, can_apply_dt=can_apply_dt
    )

    if not train_student:
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
        elif isinstance(loss_fn, MixturePathGeneralizedKL):
            loss = loss_fn(logits=logits, x_1=x_1, x_t=x_t, t=t).mean()
        elif isinstance(loss_fn, ForwardKLDistillationLoss):
            loss_temp_fn = nn.CrossEntropyLoss()
            loss = loss_temp_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
        else:
            raise ValueError("Invalid loss function")

        return loss, None

    if teacher_type == "RK_4":
        teacher_logits, x_next, teacher_u = get_RK_4_estimate(
            state,
            t,
            dt,
            x_t,
            solver,
            vocab_size,
            distill_th,
            unmask_change=unmask_change,
            controlled_unmasking=controlled_unmasking,
            blend_logits=blend_logits,
            can_apply_dt=can_apply_dt,
            path=path,
            use_generator_not_logic=use_generator_not_logic,
        )
    elif teacher_type == "heun_average":
        teacher_logits, x_next, teacher_u = heun_average(
            state,
            t,
            dt,
            x_t,
            solver,
            vocab_size,
            distill_th,
            unmask_change=unmask_change,
            controlled_unmasking=controlled_unmasking,
            blend_logits=blend_logits,
            can_apply_dt=can_apply_dt,
            path=path,
            use_generator_not_logic=use_generator_not_logic,
        )
    else:
        raise ValueError("Invalid Teacher Type!!")

    shortcut_mask = dt < distill_th

    dict_loss = None
    loss_fn_1 = MixturePathGeneralizedKL(path=path, reduction="none")
    loss_1 = loss_fn_1(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)

    if not use_generator_not_logic:
        if isinstance(loss_fn, ForwardKLDistillationLoss):
            loss_fn_2 = ForwardKLDistillationLoss(reduction="none")
            loss_2 = loss_fn_2(teacher_logits, logits, do_not_apply_softmax=False)
        elif isinstance(loss_fn, nn.CrossEntropyLoss):
            loss_fn_2 = nn.CrossEntropyLoss(reduction="none")
            loss_2 = loss_fn_2(
                teacher_logits.flatten(0, 1), x_next.flatten(0, 1)
            ).mean()  # maybe need to change x_next to x_1
    else:
        student_log_probs = F.log_softmax(u_raw, dim=-1)
        teacher_probs = F.softmax(teacher_u, dim=-1)
        loss_2 = F.kl_div(student_log_probs, teacher_probs, reduction="none")

    while loss_1.ndim > 1:
        loss_1 = loss_1.mean(dim=-1)
    while loss_2.ndim > 1:
        loss_2 = loss_2.mean(dim=-1)

    per_sample = torch.where(shortcut_mask, loss_1, loss_2)
    loss = per_sample.mean()

    return loss, dict_loss


def get_next_step_teacher(teacher_logits, path, vocab_size, x_t, dt, t):
    probs_teacher = torch.softmax(teacher_logits, dim=-1)
    x_1 = categorical(probs_teacher.to(dtype=teacher_logits.dtype))

    scheduler_output = path.scheduler(t=t)
    k_t = scheduler_output.alpha_t
    d_k_t = scheduler_output.d_alpha_t

    delta_1 = F.one_hot(x_1, num_classes=vocab_size).to(k_t.dtype)
    scale = (d_k_t / (1 - k_t)).view(-1, 1, 1)  # [B, 1, 1]
    u_teacher = scale * delta_1  # Now shape [B, L, vocab_size]

    delta_t = F.one_hot(x_t, num_classes=vocab_size)
    u_teacher = torch.where(
        delta_t.to(dtype=torch.bool), torch.zeros_like(u_teacher), u_teacher
    )

    intensity = u_teacher.sum(dim=-1)
    mask_jump = torch.rand(size=x_t.shape, device=x_t.device) < 1 - torch.exp(
        -dt[:, None] * intensity
    )

    x_t_teacher = x_t.clone()
    if mask_jump.sum() > 0:
        x_t_teacher[mask_jump] = categorical(u_teacher[mask_jump])

    return x_t_teacher, u_teacher


@torch.no_grad()
def get_RK_4_estimate(
    state,
    t,
    dt,
    x_t,
    solver,
    vocab_size,
    distill_th=1 / 513.0,
    unmask_change=False,
    controlled_unmasking=False,
    blend_logits=False,
    can_apply_dt=True,
    path=None,
    use_generator_not_logic=False,
):
    dt_half = dt / 2.0
    t_mid = (t + dt_half).clamp_(0.0, 1.0)
    t_next = (t + dt).clamp_(0.0, 1.0)
    shortcut_mask = dt < distill_th

    semi_teacher_model = (
        state.student_model if state.use_ema is False else state.student_ema_model
    )

    with torch.no_grad():
        k1_logits = semi_teacher_model(x_t, t, dt=dt_half)
        k1_probs = torch.softmax(k1_logits, dim=-1)
        u1 = solver.finite_probs_to_generator(
            k1_probs, x_t, dt_half, t=t, can_apply_dt=can_apply_dt
        )
        x_mid_1 = solver._step(
            x_t,
            u1,
            dt_half,
            dtype=torch.float32,
            unmask_change=unmask_change,
            controlled_unmasking=controlled_unmasking,
            p_1t=k1_probs,
            t=t,
        )

        k2_logits = semi_teacher_model(x_mid_1, t_mid, dt=dt_half)
        k2_probs = torch.softmax(k2_logits, dim=-1)
        u2 = solver.finite_probs_to_generator(
            k2_probs, x_mid_1, dt_half, t=t_mid, can_apply_dt=can_apply_dt
        )
        x_mid_2 = solver._step(
            x_mid_1,
            u2,
            dt_half,
            dtype=torch.float32,
            unmask_change=unmask_change,
            controlled_unmasking=controlled_unmasking,
            p_1t=k2_probs,
            t=t_mid,
        )

        k3_logits = semi_teacher_model(x_mid_2, t_mid, dt=dt_half)
        k3_probs = torch.softmax(k3_logits, dim=-1)
        u3 = solver.finite_probs_to_generator(
            k3_probs, x_mid_2, dt_half, t=t_mid, can_apply_dt=can_apply_dt
        )
        x_mid_3 = solver._step(
            x_mid_2,
            u3,
            dt_half,
            dtype=torch.float32,
            unmask_change=unmask_change,
            controlled_unmasking=controlled_unmasking,
            p_1t=k3_probs,
            t=t_mid,
        )

        k4_logits = semi_teacher_model(x_mid_3, t_next, dt=dt_half)
        k4_probs = torch.softmax(k4_logits, dim=-1)
        u4 = solver.finite_probs_to_generator(
            k4_probs, x_mid_3, dt_half, t=t_next, can_apply_dt=can_apply_dt
        )

        artificial_teacher_logits = (
            k1_logits + 2 * k2_logits + 2 * k3_logits + k4_logits
        ) / 6.0
        artificial_teacher_u = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0

        teacher_logit = state.model(x_t, t)
        x_t_teacher, u_teacher = get_next_step_teacher(
            teacher_logit, path, vocab_size, x_t, dt, t
        )

        if blend_logits == True:
            artificial_teacher_logits = torch.where(
                shortcut_mask[:, None, None], teacher_logit, artificial_teacher_logits
            )
            artificial_teacher_u = torch.where(
                shortcut_mask[:, None, None], u_teacher, artificial_teacher_u
            )

        artificial_teacher_probs = torch.softmax(artificial_teacher_logits, dim=-1)
        u_raw = solver.finite_probs_to_generator(
            artificial_teacher_probs, x_t, dt, t=t, can_apply_dt=can_apply_dt
        )

        if not use_generator_not_logic:
            x_next = solver._step(
                x_t,
                u_raw,
                dt,
                dtype=torch.float32,
                unmask_change=unmask_change,
                controlled_unmasking=controlled_unmasking,
                p_1t=artificial_teacher_probs,
                t=t,
            )
        else:
            x_next = solver._step(
                x_t,
                artificial_teacher_u,
                dt,
                dtype=torch.float32,
                unmask_change=unmask_change,
                controlled_unmasking=controlled_unmasking,
                p_1t=artificial_teacher_probs,
                t=t,
            )

        if blend_logits == False:
            x_next = torch.where(shortcut_mask[:, None], x_t_teacher, x_next)
            artificial_teacher_logits = torch.where(
                shortcut_mask[:, None, None], teacher_logit, artificial_teacher_logits
            )
            artificial_teacher_u = torch.where(
                shortcut_mask[:, None, None], u_teacher, artificial_teacher_u
            )

    log_metrics(
        {
            "x_t != x_mid_1": (x_t != x_mid_1).sum().item(),
            "x_t != x_mid_2": (x_t != x_mid_2).sum().item(),
            "x_t != x_mid_3": (x_t != x_mid_3).sum().item(),
            "x_t != x_next": (x_t != x_next).sum().item(),
            "shortcut_mask": shortcut_mask.sum().item(),
        }
    )

    return artificial_teacher_logits, x_next, artificial_teacher_u


@torch.no_grad()
def heun_average(
    state,
    t: torch.Tensor,
    dt: torch.Tensor,
    x_t: torch.Tensor,
    solver,
    vocab_size: int,
    distill_th: float = 1 / 513.0,
    unmask_change=False,
    controlled_unmasking=False,
    blend_logits=False,
    can_apply_dt=True,
    path=None,
    use_generator_not_logic=False,
):
    dt_half = dt / 2.0
    t_mid = (t + dt_half).clamp_(0.0, 1.0)
    shortcut_mask = dt < distill_th
    semi_teacher_model = (
        state.student_model if state.use_ema is False else state.student_ema_model
    )

    with torch.no_grad():
        k1_logits = semi_teacher_model(x_t, t, dt=dt_half)
        k1_probs = torch.softmax(k1_logits, dim=-1)
        u1 = solver.finite_probs_to_generator(
            k1_probs, x_t, dt_half, t=t, can_apply_dt=can_apply_dt
        )
        x_pred = solver._step(
            x_t,
            u1,
            dt_half,
            dtype=torch.float32,
            unmask_change=unmask_change,
            controlled_unmasking=controlled_unmasking,
            p_1t=k1_probs,
            t=t,
        )

        k2_logits = semi_teacher_model(x_pred, t_mid, dt=dt_half)
        k2_probs = torch.softmax(k2_logits, dim=-1)
        u2 = solver.finite_probs_to_generator(
            k2_probs, x_pred, dt_half, t=t_mid, can_apply_dt=can_apply_dt
        )

        artificial_teacher_logits = 0.5 * (k1_logits + k2_logits)
        artificial_teacher_u = 0.5 * (u1 + u2)

        teacher_logit = state.model(x_t, t)
        x_t_teacher, u_teacher = get_next_step_teacher(
            teacher_logit, path, vocab_size, x_t, dt, t
        )

        if blend_logits == True:
            artificial_teacher_logits = torch.where(
                shortcut_mask[:, None, None], teacher_logit, artificial_teacher_logits
            )
            artificial_teacher_u = torch.where(
                shortcut_mask[:, None, None], u_teacher, artificial_teacher_u
            )

        artificial_teacher_probs = torch.softmax(artificial_teacher_logits, dim=-1)
        u_raw = solver.finite_probs_to_generator(
            artificial_teacher_probs, x_t, dt, t=t, can_apply_dt=can_apply_dt
        )

        if not use_generator_not_logic:
            x_next = solver._step(
                x_t,
                u_raw,
                dt,
                dtype=torch.float32,
                unmask_change=unmask_change,
                controlled_unmasking=controlled_unmasking,
                p_1t=artificial_teacher_probs,
                t=t,
            )
        else:
            x_next = solver._step(
                x_t,
                artificial_teacher_u,
                dt,
                dtype=torch.float32,
                unmask_change=unmask_change,
                controlled_unmasking=controlled_unmasking,
                p_1t=artificial_teacher_probs,
                t=t,
            )

        if blend_logits == False:
            x_next = torch.where(shortcut_mask[:, None], x_t_teacher, x_next)
            artificial_teacher_logits = torch.where(
                shortcut_mask[:, None, None], teacher_logit, artificial_teacher_logits
            )
            artificial_teacher_u = torch.where(
                shortcut_mask[:, None, None], u_teacher, artificial_teacher_u
            )

    log_metrics(
        {
            "x_t != x_pred": (x_t != x_pred).sum().item(),
            "x_t != x_next": (x_t != x_next).sum().item(),
            "shortcut_mask": shortcut_mask.sum().item(),
        }
    )

    return artificial_teacher_logits, x_next, artificial_teacher_u
