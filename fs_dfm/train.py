#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import datetime
import os
import shutil
import traceback
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import OmegaConf
from transformers import GPT2TokenizerFast, AutoTokenizer, PreTrainedTokenizerFast

from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import get_solver_by_name
from logic import evaluate, flow, generate, training
from logic.state import TrainState
from data import data
from utils import checkpointing, logging
from model import Transformer


def run_train(rank: int, cfg: OmegaConf) -> None:
    torch.manual_seed(cfg.training.seed + rank)

    # Logging and configuration
    work_dirs = checkpointing.get_work_dirs(work_dir=cfg.work_dir, rank=rank)
    logger = logging.TrainLogger(log_dir=work_dirs.root, rank=rank, cfg=cfg)
    logger.info(work_dirs)
    logger.info(cfg)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.log_devices(device=device, logger=logger)

    # Data
    save_path = os.path.join(cfg.data.cache_dir, "processed_data", "tokenizer_dir")
    if rank == 0 and not os.path.exists(save_path):
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
    teacher_model = Transformer(
        config=cfg.model,
        vocab_size=vocab_size,
        masked=source_distribution.masked,
        dt_conditioned=False,
    ).to(device)

    student_model = Transformer(
        config=cfg.model,
        vocab_size=vocab_size,
        masked=source_distribution.masked,
        dt_conditioned=True,
    ).to(device)

    teacher_num_parameters = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"Number of parameters in the teacher model: {teacher_num_parameters}")
    logger.info(
        f"Number of parameters in the teacher model: {teacher_num_parameters/1000000.} M"
    )

    student_num_parameters = sum(p.numel() for p in student_model.parameters())
    logger.info(f"Number of parameters in the student model: {student_num_parameters}")
    logger.info(
        f"Number of parameters in the student model: {student_num_parameters/1000000.} M"
    )

    if not cfg.training.just_student:
        teacher_model = DDP(teacher_model, device_ids=[rank], static_graph=True)
        logger.info("*** Teacher Model is:")
        logger.info(teacher_model)

    student_model = DDP(student_model, device_ids=[rank], static_graph=True)
    logger.info("*** Student Model is:")
    logger.info(student_model)

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
        if not cfg.training.just_student:
            missing_keys_teacher, unexpected_keys_teacher = (
                teacher_model.module.load_state_dict(loaded_state[key])
            )
        else:
            missing_keys_teacher, unexpected_keys_teacher = (
                teacher_model.load_state_dict(loaded_state[key])
            )
        logger.info("****************")
        logger.info(f"⚠️  missing_keys_teacher is: {missing_keys_teacher}")
        logger.info(f"⚠️  unexpected_keys_teacher is: {unexpected_keys_teacher}")
        logger.info("****************")
        missing_keys_student, unexpected_keys_student = (
            student_model.module.load_state_dict(loaded_state[key], strict=False)
        )
        logger.info(f"⚠️  missing_keys_student is: {missing_keys_student}")
        logger.info(f"⚠️  unexpected_keys_student is: {unexpected_keys_student}")
        logger.info("****************")

    dist.barrier()

    # Optimizer initialization
    teacher_optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=cfg.optim.teacher_lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
        fused=cfg.optim.fused,
    )
    if cfg.training.just_student:
        teacher_optimizer = None
    logger.info(f"Teacher Optimizer: {teacher_optimizer}")

    student_optimizer = optim.AdamW(
        student_model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
        fused=cfg.optim.fused,
    )
    logger.info(f"Student Optimizer: {student_optimizer}")

    scaler_teacher = torch.amp.GradScaler("cuda")
    scaler_student = torch.amp.GradScaler("cuda")

    data_state = data.get_data_state(config=cfg, tokenizer=tokenizer)

    # Train state
    logger.info(f"Startig point of training state!!!")
    state = TrainState(
        model=teacher_model,
        optimizer=teacher_optimizer,
        step=1,
        data_state=data_state,
        student_model=student_model,
        student_optimizer=student_optimizer,
        use_ema=cfg.training.use_ema,
        ema_decay=cfg.training.ema_decay,
        ema_copy_buffers=cfg.training.ema_copy_buffers,
    )

    logger.info(f"Training state is created!!!")
    out_path = Path("/mnt/task_wrapper/user_output/artifacts/checkpoint.pth")
    if out_path.exists():
        logger.info(f"The model will restore from {out_path}!!!")
        state.restore_checkpoint(ckpt_dir=out_path, device=device, rank=rank)
        logger.info(f"The model restored from {out_path}!!!")
    else:
        logger.info(f"The model maybe restore from {work_dirs.checkpoint}!!!")
        state.restore_checkpoint(
            ckpt_dir=work_dirs.checkpoint, device=device, rank=rank
        )

    train_iter, eval_iter = data.get_data_loaders(config=cfg, data_state=data_state)

    if cfg.model.compile:
        state.compile_model()
        state.compile_student_model()
        torch.set_float32_matmul_precision("high")

    # Flow matching
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type, exponent=cfg.flow.exponent
    )
    teacher_loss_fn = flow.get_loss_function(
        loss_function=cfg.flow.teacher_loss_function, path=path
    )
    student_loss_fn = flow.get_loss_function(
        loss_function=cfg.flow.student_loss_function,
        path=path,
        temperature=cfg.flow.temperature,
    )
    # Elbo may have singularity at 1
    time_epsilon = (
        1e-5
        if isinstance(teacher_loss_fn, MixturePathGeneralizedKL)
        or isinstance(student_loss_fn, MixturePathGeneralizedKL)
        else 0.0
    )

    num_train_steps = cfg.optim.n_iters
    logger.info(f"Starting training loop at step {state.step}.")

    solver_class = get_solver_by_name(cfg.flow.student_solver)
    mask_token = vocab_size if source_distribution.masked else -1
    solver = solver_class(
        model=state.wrapped_student_model,
        path=path,
        vocabulary_size=vocab_size,
        # source_distribution_p=source_distribution,
        mask_token=mask_token,
    )
    teacher_train_loss_values = []
    student_train_loss_values = []

    while state.step <= num_train_steps:
        train_teacher = cfg.training.train_teacher
        if (
            cfg.training.teacher_cut_off
            and state.step > cfg.training.teacher_cut_off_it
        ):
            train_teacher = False
            state.model_freezing()
            if state.step % cfg.training.teacher_cut_off_it == 1:
                logger.info(
                    f"⚠️⚠️⚠️  The teacher is no longer trained after iteration {cfg.training.teacher_cut_off_it}!!"
                )

        controlled_unmasking = cfg.training.controlled_unmasking
        if controlled_unmasking and cfg.training.controlled_unmasking_type == "Testing":
            controlled_unmasking = False
        loss = training.step(
            state=state,
            teacher_loss_fn=teacher_loss_fn,
            student_loss_fn=student_loss_fn,
            path=path,
            scaler_teacher=scaler_teacher,
            scaler_student=scaler_student,
            iterator=train_iter,
            device=device,
            source_distribution=source_distribution,
            logger=logger,
            train_teacher=train_teacher,
            train_student=cfg.training.train_student,
            optim_params=cfg.optim,
            time_epsilon=time_epsilon,
            step_sizes=cfg.flow.step_sizes,
            sampling_steps=cfg.flow.sampling_steps,
            vocab_size=vocab_size + add_token,
            distill_th=(1 / (cfg.flow.sampling_steps / 2.0 + 1.0)),
            solver=solver,
            teacher_type=cfg.training.teacher_type,
            unmask_change=cfg.training.unmask_change,
            controlled_unmasking=controlled_unmasking,
            blend_logits=cfg.training.blend_logits,
            can_apply_dt=cfg.training.can_apply_dt,
            dt_weights=cfg.flow.dt_weights,
            use_generator_not_logic=cfg.training.use_generator_not_logic,
            ema_freq=cfg.training.ema_freq,
            just_student=cfg.training.just_student,
            dt_weights_2=cfg.flow.dt_weights_2,
            dt_weights_2_freq=cfg.flow.dt_weights_2_freq,
        )

        loss_teacher, loss_student, dict_loss = loss

        if loss_teacher is not None:
            teacher_train_loss_values.append(loss_teacher)
        if loss_student is not None:
            student_train_loss_values.append(loss_student)

        # Train logging
        if state.step % cfg.logging.log_freq == 0:
            if not cfg.training.just_student:
                agg_teacher_train_loss_values = torch.tensor(
                    teacher_train_loss_values, device=device
                ).mean()
                dist.all_reduce(agg_teacher_train_loss_values, dist.ReduceOp.AVG)
                logger.log_metric(
                    value=agg_teacher_train_loss_values.item(),
                    name="Teacher Loss",
                    stage="Train",
                    step=state.step,
                )

                teacher_train_loss_values = []

            agg_student_train_loss_values = torch.tensor(
                student_train_loss_values, device=device
            ).mean()
            dist.all_reduce(agg_student_train_loss_values, dist.ReduceOp.AVG)
            logger.log_metric(
                value=agg_student_train_loss_values.item(),
                name="Student Loss",
                stage="Train",
                step=state.step,
            )

            student_train_loss_values = []

        # Checkpoint
        if state.step % cfg.training.snapshot == 0:
            logger.info("Saving checkpoint...", step=state.step)

            state.save_checkpoint(
                ckpt_dir=work_dirs.checkpoint,
                rank=rank,
                train_student=cfg.training.train_student,
                train_teacher=cfg.training.train_teacher,
                step=state.step,
            )
            logger.info(
                f"Saving checkpoint is finished {work_dirs.checkpoint}", step=state.step
            )

        # Evaluation loss
        if state.step % cfg.training.eval_freq == 0:
            logger.info("Evaluating loss...", step=state.step)

            i = 0
            while i < len(cfg.flow.step_sizes):
                controlled_unmasking = cfg.training.controlled_unmasking
                if (
                    controlled_unmasking
                    and cfg.training.controlled_unmasking_type == "Training"
                ):
                    controlled_unmasking = False
                eval_loss = training.step(
                    state=state,
                    teacher_loss_fn=teacher_loss_fn,
                    student_loss_fn=student_loss_fn,
                    path=path,
                    scaler_teacher=scaler_teacher,
                    scaler_student=scaler_student,
                    iterator=eval_iter,
                    device=device,
                    source_distribution=source_distribution,
                    logger=logger,
                    train_teacher=False,
                    train_student=False,
                    time_epsilon=time_epsilon,
                    step_sizes=cfg.flow.step_sizes,
                    sampling_steps=2**i,
                    vocab_size=vocab_size + add_token,
                    distill_th=(1 / (cfg.flow.sampling_steps / 2.0 + 1.0)),
                    solver=solver,
                    teacher_type=cfg.training.teacher_type,
                    unmask_change=cfg.training.unmask_change,
                    controlled_unmasking=controlled_unmasking,
                    blend_logits=cfg.training.blend_logits,
                    can_apply_dt=cfg.training.can_apply_dt,
                    dt_weights=cfg.flow.dt_weights,
                    use_generator_not_logic=cfg.training.use_generator_not_logic,
                    ema_freq=cfg.training.ema_freq,
                    just_student=cfg.training.just_student,
                    dt_weights_2=cfg.flow.dt_weights_2,
                    dt_weights_2_freq=cfg.flow.dt_weights_2_freq,
                )

                loss_teacher, loss_student, dict_loss = eval_loss

                if loss_teacher is not None:
                    dist.all_reduce(loss_teacher, dist.ReduceOp.AVG)
                    logger.log_metric(
                        value=loss_teacher.item(),
                        name=f"Eval Teacher Loss - step size = {2 ** i}",
                        stage="Evaluation",
                        step=state.step,
                    )

                dist.all_reduce(loss_student, dist.ReduceOp.AVG)
                logger.log_metric(
                    value=loss_student.item(),
                    name=f"Eval Student Loss - step size = {2 ** i}",
                    stage="Evaluation",
                    step=state.step,
                )
                i += 1

        # Generation
        if state.step % cfg.training.perplexity_freq == 0:
            state.eval()
            state.eval_student()

            logger.info("Generating text...", step=state.step)
            logger.info("Teacher model is started!!!", step=state.step)

            samples = generate.generate_samples(
                model=state.wrapped_model,
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
                name=f"Perplexity",
                stage="Evaluation",
                step=state.step,
            )

            entropy = evaluate.compute_entropy(samples=samples)
            dist.all_reduce(entropy, dist.ReduceOp.AVG)
            logger.log_metric(
                value=entropy.item(),
                name=f"Entropy",
                stage="Evaluation",
                step=state.step,
            )

            logger.info("Student model is started!!!", step=state.step)
            i = 0
            while i < len(cfg.flow.step_sizes):
                controlled_unmasking = cfg.training.controlled_unmasking
                if (
                    controlled_unmasking
                    and cfg.training.controlled_unmasking_type == "Training"
                ):
                    controlled_unmasking = False
                wrapped_student_model = (
                    state.wrapped_student_model
                    if state.use_ema is False
                    else state.wrapped_student_ema_model
                )
                samples = generate.generate_few_steps_samples(
                    model=wrapped_student_model,
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
                    sampling_steps=2**i,
                    time_epsilon=time_epsilon,
                    student_solver=cfg.flow.student_solver,
                    unmask_change=cfg.training.unmask_change,
                    controlled_unmasking=controlled_unmasking,
                    can_apply_dt=cfg.training.can_apply_dt,
                )

                perplexity = evaluate.compute_perplexity(
                    samples=samples,
                    perplexity_batch_size=cfg.eval.perplexity_batch_size,
                )
                dist.all_reduce(perplexity, dist.ReduceOp.AVG)
                logger.log_metric(
                    value=perplexity.item(),
                    name=f"Perplexity Student - step size = {2 ** i}",
                    stage="Evaluation",
                    step=state.step,
                )

                entropy = evaluate.compute_entropy(samples=samples)
                dist.all_reduce(entropy, dist.ReduceOp.AVG)
                logger.log_metric(
                    value=entropy.item(),
                    name=f"Entropy Student - step size = {2 ** i}",
                    stage="Evaluation",
                    step=state.step,
                )
                i += 1

            dist.barrier()

        state.step = state.step + 1

    if (state.step == num_train_steps) and (rank == 0):
        logger.info("Saving checkpoint...", step=state.step)

        state.save_checkpoint(
            ckpt_dir=work_dirs.checkpoint,
            rank=rank,
            train_student=cfg.training.train_student,
            train_teacher=cfg.training.train_teacher,
            step=state.step,
        )

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
