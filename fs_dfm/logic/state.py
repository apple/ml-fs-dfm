#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import copy
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from data import DataState
from flow_matching.utils import ModelWrapper


class WrappedModel(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        # Returns probabilities (your code expects this)
        return torch.softmax(self.model(x_t=x, time=t, **extras).float(), dim=-1)


def _unwrap(m: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped in DDP; else return as-is."""
    return m.module if hasattr(m, "module") else m


class TrainState:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        step: int,
        data_state: DataState,
        student_model: Optional[nn.Module] = None,
        student_optimizer: Optional[Optimizer] = None,
        # --- EMA options ---
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_copy_buffers: bool = True,  # copy BN running stats, etc., instead of EMA-ing them
    ):
        self._model = model
        self._student_model = student_model
        self._optimizer = optimizer
        self._student_optimizer = student_optimizer
        self._step = step
        self._data_state = data_state

        # EMA settings
        self._use_ema = bool(use_ema and (student_model is not None))
        self._ema_decay = float(ema_decay)
        self._ema_copy_buffers = bool(ema_copy_buffers)

        # Build EMA model if requested
        self._student_ema_model: Optional[nn.Module] = None
        if self._use_ema:
            base_student = _unwrap(self._student_model)
            # deep-copy structure & weights (on same device/dtype)
            self._student_ema_model = copy.deepcopy(base_student)
            # freeze & eval
            self._student_ema_model.requires_grad_(False)
            self._student_ema_model.eval()

    # ------------------ flags & accessors ------------------

    def is_distillation(self) -> bool:
        return self._student_model is not None and self._student_optimizer is not None

    @property
    def step(self) -> int:
        return self._step

    @property
    def use_ema(self) -> bool:
        return self._use_ema

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def student_optimizer(self) -> Optional[Optimizer]:
        return self._student_optimizer

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def wrapped_model(self) -> ModelWrapper:
        return WrappedModel(model=self._model)

    @property
    def student_model(self) -> Optional[nn.Module]:
        return self._student_model

    @property
    def student_ema_model(self) -> Optional[nn.Module]:
        """The EMA copy of the student (plain nn.Module, never DDP)."""
        return self._student_ema_model

    @property
    def wrapped_student_model(self) -> ModelWrapper:
        return WrappedModel(model=self._student_model)

    @property
    def wrapped_student_ema_model(self) -> Optional[ModelWrapper]:
        return (
            WrappedModel(model=self._student_ema_model)
            if self._student_ema_model is not None
            else None
        )

    @property
    def data_state(self) -> DataState:
        return self._data_state

    # ------------------ modes & compile ------------------

    def model_freezing(self) -> None:
        for p in self._model.parameters():
            p.requires_grad_(False)

    def compile_model(self) -> None:
        self._model = torch.compile(self._model)

    def compile_student_model(self) -> None:
        self._student_model = torch.compile(self._student_model)

    def eval(self) -> None:
        self.train(training=False)

    def train(self, training: bool = True) -> None:
        self._model.train(mode=training)

    def eval_student(self) -> None:
        self.train_student(training=False)

    def train_student(self, training: bool = True) -> None:
        self._student_model.train(mode=training)

    def eval_student_ema(self) -> None:
        if self._student_ema_model is not None:
            self._student_ema_model.eval()

    # ------------------ EMA update ------------------

    @torch.no_grad()
    def update_student_ema(self, decay: Optional[float] = None) -> None:
        """
        EMA update: theta_ema <- decay * theta_ema + (1 - decay) * theta
        Must be called AFTER student optimizer step.
        """
        if (
            not self._use_ema
            or self._student_ema_model is None
            or self._student_model is None
        ):
            return
        d = float(self._ema_decay if decay is None else decay)

        ema_m = self._student_ema_model
        student_m = _unwrap(self._student_model)

        # keep on same device/dtype
        ema_m.to(next(student_m.parameters()).device)

        for p_ema, p in zip(ema_m.parameters(), student_m.parameters()):
            # in-place EMA: p_ema = d*p_ema + (1-d)*p
            p_ema.data.lerp_(p.data, 1.0 - d)

        # Buffers (e.g., BatchNorm running_mean/var): usually COPY, not EMA
        if self._ema_copy_buffers:
            for b_ema, b in zip(ema_m.buffers(), student_m.buffers()):
                b_ema.copy_(b)

    # ------------------ checkpointing ------------------

    def restore_checkpoint(
        self, ckpt_dir: Path, device: torch.device, rank: int
    ) -> None:
        if ckpt_dir.is_file() and ckpt_dir.stat().st_size > 0:
            loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=True)

            # Restore teacher
            if self._optimizer and "teacher_optimizer" in loaded_state:
                self._optimizer.load_state_dict(loaded_state["teacher_optimizer"])
            if self._model and "teacher_model" in loaded_state:
                _unwrap(self._model).load_state_dict(loaded_state["teacher_model"])

            # Restore student if available
            if self._student_model and "student_model" in loaded_state:
                _unwrap(self._student_model).load_state_dict(
                    loaded_state["student_model"]
                )
            if self._student_optimizer and "student_optimizer" in loaded_state:
                self._student_optimizer.load_state_dict(
                    loaded_state["student_optimizer"]
                )

            # Restore EMA (create if missing)
            if ("student_ema_model" in loaded_state) and (
                self._student_model is not None
            ):
                if self._student_ema_model is None:
                    base_student = _unwrap(self._student_model)
                    self._student_ema_model = copy.deepcopy(base_student)
                    self._student_ema_model.requires_grad_(False)
                    self._student_ema_model.eval()
                self._student_ema_model.load_state_dict(
                    loaded_state["student_ema_model"]
                )

            self._step = loaded_state["step"]
            self._data_state.test.sampler.load_state_dict(loaded_state["test_sampler"])
            self._data_state.train.sampler.load_state_dict(
                loaded_state["train_sampler"]
            )
        else:
            ckpt_dir.parent.mkdir(exist_ok=True, parents=True)
            if rank == 0:
                logging.warning(
                    f"No checkpoint found at {ckpt_dir}. Returned the same state as input"
                )

    def save_checkpoint(
        self,
        ckpt_dir: str,
        rank: int,
        train_student: bool = True,
        train_teacher: bool = True,
        step: int = 0,
    ) -> None:
        saved_state = {
            "step": self._step,
            "train_sampler": self._data_state.train.sampler.state_dict(),
            "test_sampler": self._data_state.test.sampler.state_dict(),
        }

        if self._model and train_teacher:
            saved_state["teacher_model"] = _unwrap(self._model).state_dict()
        if self._optimizer and train_teacher:
            saved_state["teacher_optimizer"] = self._optimizer.state_dict()

        if self._student_model and train_student:
            saved_state["student_model"] = _unwrap(self._student_model).state_dict()
        if self._student_optimizer and train_student:
            saved_state["student_optimizer"] = self._student_optimizer.state_dict()

        # Save EMA (even if not training right now)
        if self._student_ema_model is not None:
            saved_state["student_ema_model"] = self._student_ema_model.state_dict()

        if rank == 0:
            ckpt_path = Path(ckpt_dir)
            torch.save(saved_state, ckpt_path)
            versioned_path = ckpt_path.with_name(
                f"{ckpt_path.stem}_step_{step}{ckpt_path.suffix}"
            )
            torch.save(saved_state, versioned_path)
            out_path = Path("/mnt/task_wrapper/user_output/artifacts/checkpoint.pth")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(saved_state, out_path)
