#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from dataclasses import dataclass, field
from pathlib import Path

import torch
from logic.flow import SourceDistribution
from model import Transformer
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def load_cfg_from_path(work_dir: str) -> OmegaConf:
    work_dir = Path(work_dir)

    root_dir = work_dir if work_dir.is_dir() else work_dir.parents[1]

    cfg_path = root_dir / ".hydra/config.yaml"

    return OmegaConf.load(cfg_path)


def load_model_from_path(
    work_dir: str,
    source_distribution: SourceDistribution,
    device: torch.device,
    vocab_size: int,
    cfg: OmegaConf,
    teacher_model: bool = True,
) -> nn.Module:
    work_dir = Path(work_dir)

    if work_dir.is_dir():
        ckpt_dir = work_dir / "checkpoints" / "checkpoint.pth"
    else:
        ckpt_dir = work_dir

    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=True)

    if teacher_model:
        model = Transformer(
            config=cfg.model,
            vocab_size=vocab_size,
            masked=source_distribution.masked,
            dt_conditioned=False,
        ).to(device)
        model = DDP(model, device_ids=[device])
        missing_keys, unexpected_keys = model.module.load_state_dict(
            loaded_state["teacher_model"]
        )
        print("teacher_model is loaded!!!")
    else:
        model = Transformer(
            config=cfg.model,
            vocab_size=vocab_size,
            masked=source_distribution.masked,
            dt_conditioned=True,
        ).to(device)
        model = DDP(model, device_ids=[device])
        missing_keys, unexpected_keys = model.module.load_state_dict(
            loaded_state["student_model"]
        )
        print("student_model is loaded!!!")

    return model, missing_keys, unexpected_keys


@dataclass
class WorkDirectory:
    root: Path = field(metadata={"help": "Root work directory"})
    checkpoint: Path = field(metadata={"help": "Checkpoint directory"})
    samples: Path = field(metadata={"help": "Samples directory"})


def get_work_dirs(work_dir: str, rank: int) -> WorkDirectory:
    work_dir = Path(work_dir)

    sample_dir = work_dir / "samples"
    checkpoint_dir = work_dir / "checkpoints" / "checkpoint.pth"

    if rank == 0:
        sample_dir.mkdir(exist_ok=True)
        checkpoint_dir.parents[0].mkdir(exist_ok=True)

    return WorkDirectory(root=work_dir, checkpoint=checkpoint_dir, samples=sample_dir)
