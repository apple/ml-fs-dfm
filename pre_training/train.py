#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import datetime
import os
import shutil
import math
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from data import data
from flow_matching.loss import MixturePathGeneralizedKL

from logic import evaluate, flow, generate, training
from logic.state import TrainState
from model import Transformer
from omegaconf import OmegaConf
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from utils import checkpointing, logging


def _to_number(x):
    # Accept Tensor, int, float, bool
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.detach().item()
        else:
            # fallback: average for multi-element tensors
            x = x.detach().float().mean().item()
    elif isinstance(x, bool):
        x = float(x)
    return float(x)


def run_train(rank: int, cfg: OmegaConf) -> None:
    torch.manual_seed(cfg.training.seed + rank)

    # Logging and configuration
    work_dirs = checkpointing.get_work_dirs(work_dir=cfg.work_dir, rank=rank)
    logger = logging.TrainLogger(log_dir=work_dirs.root, rank=rank, cfg=cfg)
    logger.info(work_dirs)
    logger.info(cfg)
    files = os.listdir(".")  # List files in the current directory
    logger.info(f"ls is : {files}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.log_devices(device=device, logger=logger)

    # Data
    save_path = os.path.join(cfg.data.cache_dir, "processed_data", "tokenizer_dir")
    if rank == 0 and not os.path.exists(save_path) and cfg.data.hf_dataset:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.save_pretrained(save_path)

    dist.barrier()

    if not cfg.data.hf_dataset:
        save_path = os.path.join(save_path, "tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=save_path)
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "[SEP]"})  # Use [SEP] as EOS
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(save_path)

    vocab_size = tokenizer.vocab_size
    logger.info(f"vocab_size is {vocab_size}")

    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size
    )
    add_token = 1 if source_distribution.masked else 0

    # Model initialization
    model = Transformer(
        config=cfg.model, vocab_size=vocab_size, masked=source_distribution.masked
    ).to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in the model: {num_parameters}")
    logger.info(f"Number of parameters in the model: {num_parameters/1000000.} M")

    model = DDP(model, device_ids=[rank], static_graph=True)
    logger.info(model)

    save_pre_trained_path = os.path.join(
        cfg.data.cache_dir, cfg.training.pre_trained_model
    )
    if os.path.exists(save_pre_trained_path):
        loaded_state = torch.load(
            save_pre_trained_path, map_location=device, weights_only=True
        )
        key = "model"
        if not (key in loaded_state) and "teacher_model" in loaded_state:
            key = "teacher_model"
        missing_keys, unexpected_keys = model.module.load_state_dict(loaded_state[key])
        logger.info("****************")
        logger.info(f"⚠️  missing_keys is: {missing_keys}")
        logger.info(f"⚠️  unexpected_keys is: {unexpected_keys}")
        logger.info("****************")

    dist.barrier()

    # Optimizer initialization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
        fused=cfg.optim.fused,
    )
    logger.info(f"Optimizer: {optimizer}")
    scaler = torch.amp.GradScaler("cuda")
    logger.info(f"Scaler: {scaler}")

    data_state = data.get_data_state(config=cfg, tokenizer=tokenizer)

    # Train state
    logger.info(f"Startig point of training state!!!")
    state = TrainState(model=model, optimizer=optimizer, step=1, data_state=data_state)
    logger.info(f"Training state is created!!!")
    out_path = Path("/mnt/task_wrapper/user_output/cache_dir/last_model/checkpoint.pth")
    if out_path.exists():
        logger.info(f"The model will restore from {out_path}!!!")
        state.restore_checkpoint(ckpt_dir=out_path, device=device, rank=rank)
    else:
        logger.info(f"The model maybe restore from {work_dirs.checkpoint}!!!")
        state.restore_checkpoint(
            ckpt_dir=work_dirs.checkpoint, device=device, rank=rank
        )

    train_iter, eval_iter = data.get_data_loaders(config=cfg, data_state=data_state)

    if cfg.model.compile:
        state.compile_model()
        torch.set_float32_matmul_precision("high")

    # Flow matching
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type, exponent=cfg.flow.exponent
    )
    loss_fn = flow.get_loss_function(loss_function=cfg.flow.loss_function, path=path)
    # Elbo may have singularity at 1
    time_epsilon = 1e-3 if isinstance(loss_fn, MixturePathGeneralizedKL) else 0.0

    num_train_steps = cfg.optim.n_iters
    logger.info(f"Starting training loop at step {state.step}.")

    train_loss_values = []

    while state.step <= num_train_steps:
        loss = training.step(
            loss_fn=loss_fn,
            path=path,
            state=state,
            scaler=scaler,
            iterator=train_iter,
            optim_params=cfg.optim,
            device=device,
            source_distribution=source_distribution,
            logger=logger,
            training=True,
            time_epsilon=time_epsilon,
        )

        train_loss_values.append(loss)

        # Train logging
        if state.step % cfg.logging.log_freq == 0:
            agg_train_loss_values = torch.tensor(
                train_loss_values, device=device
            ).mean()
            dist.all_reduce(agg_train_loss_values, dist.ReduceOp.AVG)
            logger.log_metric(
                value=agg_train_loss_values.item(),
                name="Loss",
                stage="Train",
                step=state.step,
            )

            train_loss_values = []

        # Checkpoint
        if state.step % cfg.training.snapshot == 0:
            logger.info("Saving checkpoint...", step=state.step)

            state.save_checkpoint(
                ckpt_dir=work_dirs.checkpoint, rank=rank, step=state.step
            )
            logger.info(
                f"Saving checkpoint is finished {work_dirs.checkpoint}", step=state.step
            )

        # Evaluation loss
        if state.step % cfg.training.eval_freq == 0:
            logger.info("Evaluating loss...", step=state.step)

            eval_loss = training.step(
                state=state,
                loss_fn=loss_fn,
                path=path,
                scaler=scaler,
                iterator=eval_iter,
                device=device,
                source_distribution=source_distribution,
                logger=logger,
                training=False,
                time_epsilon=time_epsilon,
            )

            dist.all_reduce(eval_loss, dist.ReduceOp.AVG)
            logger.log_metric(
                value=eval_loss.item(),
                name="Eval Loss",
                stage="Evaluation",
                step=state.step,
            )

        samples = None
        # Generation
        if state.step % cfg.training.perplexity_freq == 0:
            state.eval()

            logger.info("Generating text...", step=state.step)

            samples = generate.generate_samples(
                model=state.model,
                step=state.step,
                sample_dir=work_dirs.samples,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                rank=rank,
                device=device,
                path=path,
                source_distribution=source_distribution,
                sample_batch_size=cfg.eval.sample_batch_size,
                sequence_length=cfg.model.length,
                sampling_steps=cfg.flow.sampling_steps,
                time_epsilon=time_epsilon,
            )

            perplexity = evaluate.compute_perplexity(
                samples=samples,
                perplexity_batch_size=cfg.eval.perplexity_batch_size,
            )
            dist.all_reduce(perplexity, dist.ReduceOp.AVG)
            logger.log_metric(
                value=perplexity.item(),
                name="Perplexity",
                stage="Evaluation",
                step=state.step,
            )

            entropy = evaluate.compute_entropy(samples=samples)
            dist.all_reduce(entropy, dist.ReduceOp.AVG)

            logger.log_metric(
                value=entropy.item(),
                name="Entropy",
                stage="Evaluation",
                step=state.step,
            )

        dist.barrier()

        state.step = state.step + 1

    if (state.step == num_train_steps) and (rank == 0):
        logger.info("Saving checkpoint...", step=state.step)

        state.save_checkpoint(ckpt_dir=work_dirs.checkpoint, rank=rank, step=state.step)

    logger.finish()


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    torch.cuda.set_device(rank)

    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)


def cleanup() -> None:
    dist.destroy_process_group()


def run_mp_training(rank: int, world_size: int, cfg: OmegaConf, port: int) -> None:
    try:
        setup(rank=rank, world_size=world_size, port=port)
        run_train(rank=rank, cfg=cfg)
    finally:
        cleanup()
