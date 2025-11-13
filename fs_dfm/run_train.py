#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import os

import hydra
import torch.multiprocessing as mp

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from train import run_mp_training


# @hydra.main(version_base=None, config_path="configs", config_name="config_multiplication")
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    print(f"hydra_cfg is {hydra_cfg}.")
    work_dir = (
        hydra_cfg.run.dir
        if hydra_cfg.mode == RunMode.RUN
        else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    )
    os.makedirs(work_dir, exist_ok=True)

    with open_dict(cfg):
        cfg.work_dir = work_dir

    print(f"work_dir is {cfg.work_dir}")
    port = 12346

    if cfg.compute.ngpus == 1:
        run_mp_training(rank=0, world_size=1, cfg=cfg, port=port)
    else:
        mp.set_start_method("forkserver")
        mp.spawn(
            run_mp_training,
            args=(cfg.compute.ngpus, cfg, port),
            nprocs=cfg.compute.ngpus,
            join=True,
        )


if __name__ == "__main__":
    main()
