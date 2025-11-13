#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


"""distributed_launcher.py
================================
Multi-node, multi-GPU entry-point that works on  Slurm, or plain
`torchrun`.  It relies only on the standard environment variables recognised by
**PyTorch DDP** (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`,
`LOCAL_RANK`).
"""

from __future__ import annotations

import os
import socket
import hydra
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, open_dict

from train import run_mp_training

###########################################################################
# Helper functions
###########################################################################


def _first_defined(keys: list[str], default: str | None = None) -> str | None:
    """Return the first existing environment variable from *keys* or *default*."""
    for k in keys:
        if (val := os.environ.get(k)) is not None:
            return val
    return default


def _int_env(keys: list[str], default: int | None = None) -> int | None:
    val = _first_defined(keys)
    return int(val) if val is not None else default


def init_distributed(local_rank: int, world_size: int, port: int) -> None:
    """Initialise *torch.distributed*.

    Works under torchrun, Slurm, or single-node fallback.  We respect any
    existing env-vars so external launchers (torchrun, etc.) can take
    full control.  If they are **absent**, we synthesise a minimal set so at
    least single-machine multi-GPU runs succeed.
    """

    # ── Canonical env variable names expected by PyTorch ────────────────
    rank = _int_env(["RANK", "SLURM_PROCID"], default=0)
    local_rank_env = _int_env(["LOCAL_RANK", "SLURM_LOCALID"])
    world_size_env = _int_env(["WORLD_SIZE", "SLURM_NTASKS"])

    # Fallbacks when a custom launcher (e.g. mp.spawn) starts each node.
    if local_rank_env is None:
        local_rank_env = local_rank
        os.environ["LOCAL_RANK"] = str(local_rank_env)
    if world_size_env is None:
        world_size_env = world_size
        os.environ["WORLD_SIZE"] = str(world_size_env)
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)

    # Master address/port: trust launcher-set ones, else choose sensible defaults.
    master_addr = _first_defined(["MASTER_ADDR"], default="127.0.0.1")
    master_port = _first_defined(["MASTER_PORT"], default=str(port))
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    # Pick correct GPU and bring up the process group.
    torch.cuda.set_device(local_rank_env)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=world_size_env,
        rank=int(os.environ["RANK"]),
    )


###########################################################################
# Hydra main
###########################################################################


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(
    cfg: DictConfig,
) -> None:  # noqa: C901  (# too-complex – readability > 12 OK here)
    """Launch training on *nodes × gpus_per_node* GPUs.

    The same script can be executed
    • locally on one machine (mp.spawn mode),
    • under Slurm / Submitit (which sets `SLURM_*`).
    """

    hydra_cfg = HydraConfig.get()
    work_dir = (
        hydra_cfg.run.dir
        if hydra_cfg.mode == RunMode.RUN
        else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    )
    os.makedirs(work_dir, exist_ok=True)
    with open_dict(cfg):
        cfg.work_dir = work_dir

    # ── Cluster topology from config ────────────────────────────────────
    gpus_per_node: int = cfg.compute.ngpus
    nodes: int = cfg.compute.nodes
    world_size: int = gpus_per_node * nodes

    # Choose a default port that is unlikely to be in use; launchers may override it.
    default_port = 12346

    # ── Single-process debug mode ───────────────────────────────────────
    if world_size == 1:
        run_mp_training(rank=0, world_size=1, cfg=cfg, port=default_port)
        return

    # ── Worker wrapper passed to mp.spawn ───────────────────────────────
    def _worker(
        local_rank: int, world_size_: int, cfg_: DictConfig, port_: int
    ) -> None:
        init_distributed(local_rank, world_size_, port_)
        run_mp_training(
            rank=int(os.environ["RANK"]), world_size=world_size_, cfg=cfg_, port=port_
        )

    # ── Multiprocessing launch (1 process per GPU) ──────────────────────
    mp.set_start_method("forkserver", force=True)
    mp.spawn(
        _worker,
        args=(world_size, cfg, default_port),
        nprocs=gpus_per_node,
        join=True,
    )


if __name__ == "__main__":  # pragma: no cover – direct execution
    main()
