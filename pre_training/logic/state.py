#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import logging
from pathlib import Path

import torch
from data import DataState

from torch import nn
from torch.optim import Optimizer


class TrainState:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        step: int,
        data_state: DataState,
    ):
        self._model = model
        self._optimizer = optimizer
        self._step = step
        self._data_state = data_state

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def data_state(self) -> DataState:
        return self._data_state

    def compile_model(self) -> None:
        self._model = torch.compile(self._model)

    def restore_checkpoint(
        self, ckpt_dir: Path, device: torch.device, rank: int
    ) -> None:
        if ckpt_dir.exists():
            loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=True)

            self.optimizer.load_state_dict(loaded_state["optimizer"])
            self.model.module.load_state_dict(loaded_state["model"])
            self.step = loaded_state["step"]
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

    def save_checkpoint(self, ckpt_dir: str, rank: int, step: int = 0) -> None:
        saved_state = {
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.module.state_dict(),
            "step": self.step,
            "train_sampler": self._data_state.train.sampler.state_dict(),
            "test_sampler": self._data_state.test.sampler.state_dict(),
        }

        if rank == 0:
            ckpt_path = Path(ckpt_dir)
            torch.save(saved_state, ckpt_path)
            versioned_path = ckpt_path.with_name(
                f"{ckpt_path.stem}_step_{step}{ckpt_path.suffix}"
            )
            torch.save(saved_state, versioned_path)
            out_path = Path(
                "/mnt/task_wrapper/user_output/cache_dir/last_model/checkpoint.pth"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(saved_state, out_path)

    def eval(self) -> None:
        self.train(training=False)

    def train(self, training: bool = True) -> None:
        self._model.train(mode=training)
