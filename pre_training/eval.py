import datetime
import os
from pathlib import Path
import math
import traceback

import torch
import torch.distributed as dist

from data import data
from flow_matching.loss import MixturePathGeneralizedKL
from logic.eval_multiplication import evaluate_generated_math
from logic import evaluate, flow, generate

from torch.utils.data import DataLoader
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


def log_multiplication_metrics(
    metrics,
    logger,
    step,
    log_traceback: bool = True,
    log_non_numeric: bool = True,
    prefix: str = "eval",
):
    for k in sorted(metrics.keys()):
        v = metrics[k]
        try:
            val = _to_number(v)
            if val is None:
                if log_non_numeric:
                    msg = f"[metrics] skip non-numeric key={k} type={type(v).__name__} value={_safe_repr(v)}"
                    # prefer .info, fall back to print-y levels
                    if hasattr(logger, "info"):
                        logger.info(msg, step=step)
                    elif hasattr(logger, "warning"):
                        logger.warning(msg)
                continue
            if math.isnan(val) or math.isinf(val):
                if hasattr(logger, "info"):
                    logger.info(
                        f"[metrics] skip NaN/Inf key={k} value={val}", step=step
                    )
                continue

            name = f"{prefix}_{k}"
            logger.log_metric(value=val, name=name, stage="Evaluation", step=step)

        except Exception as e:
            # Build a rich error message
            err_hdr = f"[metrics] exception key={k} type={type(v).__name__}: {e.__class__.__name__}: {e}"
            tb_txt = traceback.format_exc() if log_traceback else ""
            full_msg = f"{err_hdr}\n{tb_txt}" if tb_txt else err_hdr

            # Use logger.exception if available (auto-includes traceback)
            if log_traceback and hasattr(logger, "exception"):
                logger.exception(err_hdr, step=step)
            else:
                if hasattr(logger, "error"):
                    logger.error(full_msg, step=step)
                elif hasattr(logger, "info"):
                    logger.info(full_msg, step=step)

            if raise_on_error:
                raise


def ddp_sum_scalar(x, device, dtype=torch.long):
    x = torch.as_tensor(x, device=device, dtype=dtype).sum()  # <-- collapse to 0-dim
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


def run_eval(
    rank: int,
    seed: int,
    work_dir: str,
    pre_trained_model_path: str,
    batch_size: int,
    perplexity_n_samples: int,
    sampling_steps: int,
    eval_perplexity: bool,
    eval_elbo: bool,
    elbo_data: str,
    world_size: int,
    n_discretization: float = 1024,
) -> None:
    torch.manual_seed(seed + rank)

    # Logging and configuration
    work_dirs = checkpointing.get_work_dirs(work_dir=work_dir, rank=rank)
    work_dirs.checkpoint = Path(pre_trained_model_path)
    print(f" work_dirs.checkpoint is { work_dirs.checkpoint}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    cfg = checkpointing.load_cfg_from_path(work_dir=work_dirs.checkpoint)
    logger = logging.TrainLogger(log_dir=work_dirs.root, rank=rank, cfg=cfg)

    # Data
    save_path = os.path.join(cfg.data.cache_dir, "processed_data", "tokenizer_dir")
    if not cfg.data.hf_dataset:
        save_path = os.path.join(save_path, "tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=save_path)
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "[SEP]"})  # Use [SEP] as EOS
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    logger.info(f"vocab_size is {vocab_size}")

    # Flow matching
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type, exponent=cfg.flow.exponent
    )
    loss_fn = flow.get_loss_function(loss_function=cfg.flow.loss_function, path=path)
    # Elbo may have singularity at 1
    time_epsilon = 1e-3 if isinstance(loss_fn, MixturePathGeneralizedKL) else 0.0

    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size
    )

    model, missing_keys, unexpected_keys = checkpointing.load_model_from_path(
        work_dir=work_dirs.checkpoint,
        device=device,
        source_distribution=source_distribution,
        cfg=cfg,
        vocab_size=vocab_size,
    )
    if not cfg.data.hf_dataset:
        data_state = data._get_dataset(
            name=elbo_data,
            mode="test",
            cache_dir=cfg.data.cache_dir,
            block_size=cfg.model.length,
            num_proc=cfg.data.num_workers,
            batch_size=batch_size,
            ngpus=world_size,
            hf_dataset=False,
            tokenizer=tokenizer,
        )
    else:
        data_state = data._get_dataset(
            name=elbo_data,
            mode="validation",
            cache_dir=cfg.data.cache_dir,
            block_size=cfg.model.length,
            num_proc=cfg.data.num_workers,
            batch_size=batch_size,
            ngpus=world_size,
            hf_dataset=True,
            tokenizer=tokenizer,
        )

    dataloader = DataLoader(
        data_state.dataset,
        batch_size=batch_size,
        sampler=data_state.sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        shuffle=(data_state.sampler is None),
    )

    model.eval()
    logger.info(model)
    logger.info("****************")
    logger.info(f"⚠️  missing_keys is: {missing_keys}")
    logger.info(f"⚠️  unexpected_keys is: {unexpected_keys}")
    logger.info("****************")

    if cfg.model.compile:
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")

    if eval_perplexity:
        assert perplexity_n_samples // batch_size > 0

        i = 1
        while 2**i <= sampling_steps:

            samples = []
            samples_left_10 = []
            samples_left_4 = []
            samples_per_25 = []
            samples_per_50 = []

            num_predicted_tokens_10 = []
            num_correct_predicted_10 = []

            num_predicted_tokens_4 = []
            num_correct_predicted_4 = []

            num_predicted_tokens_25 = []
            num_correct_predicted_25 = []

            num_predicted_tokens_50 = []
            num_correct_predicted_50 = []

            for _ in range(perplexity_n_samples // batch_size):
                samples.append(
                    generate.generate_samples(
                        model=model,
                        step=2**i,
                        sample_dir=work_dirs.samples,
                        vocab_size=vocab_size,
                        tokenizer=tokenizer,
                        rank=rank,
                        device=device,
                        path=path,
                        source_distribution=source_distribution,
                        sample_batch_size=batch_size,
                        sequence_length=cfg.model.length,
                        sampling_steps=2**i,
                        time_epsilon=time_epsilon,
                    )
                )

                samples_10, metrics = generate.generate_samples_with_dataset(
                    model=model,
                    step=2**i,
                    sample_dir=work_dirs.samples,
                    vocab_size=vocab_size,
                    tokenizer=tokenizer,
                    rank=rank,
                    device=device,
                    path=path,
                    source_distribution=source_distribution,
                    sample_batch_size=batch_size,
                    sequence_length=cfg.model.length,
                    sampling_steps=2**i,
                    time_epsilon=time_epsilon,
                    dataloader=dataloader,
                    controller_mode="left_k",
                    controller_left_k=(cfg.model.length // 2 - 1),
                    return_metrics=True,
                )
                num_predicted_tokens_10.append(metrics["num_predicted_tokens"])
                num_correct_predicted_10.append(metrics["num_correct_predicted"])
                samples_left_10.append(samples_10)

                samples_4, metrics = generate.generate_samples_with_dataset(
                    model=model,
                    step=2**i,
                    sample_dir=work_dirs.samples,
                    vocab_size=vocab_size,
                    tokenizer=tokenizer,
                    rank=rank,
                    device=device,
                    path=path,
                    source_distribution=source_distribution,
                    sample_batch_size=batch_size,
                    sequence_length=cfg.model.length,
                    sampling_steps=2**i,
                    time_epsilon=time_epsilon,
                    dataloader=dataloader,
                    controller_mode="left_k",
                    controller_left_k=(cfg.model.length // 4),
                    return_metrics=True,
                )
                num_predicted_tokens_4.append(metrics["num_predicted_tokens"])
                num_correct_predicted_4.append(metrics["num_correct_predicted"])
                samples_left_4.append(samples_4)

                samples_25, metrics = generate.generate_samples_with_dataset(
                    model=model,
                    step=2**i,
                    sample_dir=work_dirs.samples,
                    vocab_size=vocab_size,
                    tokenizer=tokenizer,
                    rank=rank,
                    device=device,
                    path=path,
                    source_distribution=source_distribution,
                    sample_batch_size=batch_size,
                    sequence_length=cfg.model.length,
                    sampling_steps=2**i,
                    time_epsilon=time_epsilon,
                    dataloader=dataloader,
                    controller_mode="percentage",
                    controller_pct=0.25,
                    return_metrics=True,
                )
                num_predicted_tokens_25.append(metrics["num_predicted_tokens"])
                num_correct_predicted_25.append(metrics["num_correct_predicted"])
                samples_per_25.append(samples_25)

                samples_50, metrics = generate.generate_samples_with_dataset(
                    model=model,
                    step=2**i,
                    sample_dir=work_dirs.samples,
                    vocab_size=vocab_size,
                    tokenizer=tokenizer,
                    rank=rank,
                    device=device,
                    path=path,
                    source_distribution=source_distribution,
                    sample_batch_size=batch_size,
                    sequence_length=cfg.model.length,
                    sampling_steps=2**i,
                    time_epsilon=time_epsilon,
                    dataloader=dataloader,
                    controller_mode="percentage",
                    controller_pct=0.5,
                    return_metrics=True,
                )
                num_predicted_tokens_50.append(metrics["num_predicted_tokens"])
                num_correct_predicted_50.append(metrics["num_correct_predicted"])
                samples_per_50.append(samples_50)

                dist.barrier()

            samples = torch.cat(samples, dim=0)
            samples_left_10 = torch.cat(samples_left_10, dim=0)
            samples_left_4 = torch.cat(samples_left_4, dim=0)
            samples_per_25 = torch.cat(samples_per_25, dim=0)
            samples_per_50 = torch.cat(samples_per_50, dim=0)

            num_correct_predicted_10 = ddp_sum_scalar(num_correct_predicted_10, device)
            num_predicted_tokens_10 = ddp_sum_scalar(num_predicted_tokens_10, device)

            num_correct_predicted_4 = ddp_sum_scalar(num_correct_predicted_4, device)
            num_predicted_tokens_4 = ddp_sum_scalar(num_predicted_tokens_4, device)

            num_correct_predicted_25 = ddp_sum_scalar(num_correct_predicted_25, device)
            num_predicted_tokens_25 = ddp_sum_scalar(num_predicted_tokens_25, device)

            num_correct_predicted_50 = ddp_sum_scalar(num_correct_predicted_50, device)
            num_predicted_tokens_50 = ddp_sum_scalar(num_predicted_tokens_50, device)

            logger.log_metric(
                value=num_correct_predicted_10.item() / num_predicted_tokens_10.item(),
                name=f"accuracy_left_{(cfg.model.length // 2 - 1)}",
                stage="Evaluation",
                step=2**i,
            )

            logger.log_metric(
                value=num_correct_predicted_4.item() / num_predicted_tokens_4.item(),
                name=f"accuracy_left_{(cfg.model.length // 4)}",
                stage="Evaluation",
                step=2**i,
            )

            logger.log_metric(
                value=num_correct_predicted_25.item() / num_predicted_tokens_25.item(),
                name=f"accuracy_per_25",
                stage="Evaluation",
                step=2**i,
            )

            logger.log_metric(
                value=num_correct_predicted_50.item() / num_predicted_tokens_50.item(),
                name=f"accuracy_per_50",
                stage="Evaluation",
                step=2**i,
            )

            perplexity = evaluate.compute_perplexity(
                samples=samples,
                perplexity_batch_size=cfg.eval.perplexity_batch_size,
            )
            dist.all_reduce(perplexity, dist.ReduceOp.AVG)

            entropy = evaluate.compute_entropy(samples=samples)
            dist.all_reduce(entropy, dist.ReduceOp.AVG)

            perplexity_left_10 = evaluate.compute_perplexity(
                samples=samples_left_10[:, (cfg.model.length // 2 - 1) :],
                perplexity_batch_size=cfg.eval.perplexity_batch_size,
            )
            dist.all_reduce(perplexity_left_10, dist.ReduceOp.AVG)
            entropy_left_10 = evaluate.compute_entropy(samples=samples_left_10)
            dist.all_reduce(entropy_left_10, dist.ReduceOp.AVG)
            logger.log_metric(
                value=perplexity_left_10.item(),
                name=f"perplexity_left_{(cfg.model.length // 2 - 1)}",
                stage="Evaluation",
                step=2**i,
            )
            logger.log_metric(
                value=entropy_left_10.item(),
                name=f"entropy_left_{(cfg.model.length // 2 - 1)}",
                stage="Evaluation",
                step=2**i,
            )

            perplexity_left_4 = evaluate.compute_perplexity(
                samples=samples_left_4[:, (cfg.model.length // 4) :],
                perplexity_batch_size=cfg.eval.perplexity_batch_size,
            )
            dist.all_reduce(perplexity_left_4, dist.ReduceOp.AVG)
            entropy_left_4 = evaluate.compute_entropy(samples=samples_left_4)
            dist.all_reduce(entropy_left_4, dist.ReduceOp.AVG)
            logger.log_metric(
                value=perplexity_left_4.item(),
                name=f"perplexity_left_{(cfg.model.length // 4)}",
                stage="Evaluation",
                step=2**i,
            )
            logger.log_metric(
                value=entropy_left_4.item(),
                name=f"entropy_left_{(cfg.model.length // 4)}",
                stage="Evaluation",
                step=2**i,
            )

            perplexity_per_25 = evaluate.compute_perplexity(
                samples=samples_per_25,
                perplexity_batch_size=cfg.eval.perplexity_batch_size,
            )
            dist.all_reduce(perplexity_per_25, dist.ReduceOp.AVG)
            entropy_per_25 = evaluate.compute_entropy(samples=samples_per_25)
            dist.all_reduce(entropy_per_25, dist.ReduceOp.AVG)
            logger.log_metric(
                value=perplexity_per_25.item(),
                name=f"perplexity_per_25",
                stage="Evaluation",
                step=2**i,
            )
            logger.log_metric(
                value=entropy_per_25.item(),
                name=f"entropy_per_25",
                stage="Evaluation",
                step=2**i,
            )

            perplexity_per_50 = evaluate.compute_perplexity(
                samples=samples_per_50,
                perplexity_batch_size=cfg.eval.perplexity_batch_size,
            )
            dist.all_reduce(perplexity_per_50, dist.ReduceOp.AVG)
            entropy_per_50 = evaluate.compute_entropy(samples=samples_per_50)
            dist.all_reduce(entropy_per_50, dist.ReduceOp.AVG)
            logger.log_metric(
                value=perplexity_per_50.item(),
                name=f"perplexity_per_50",
                stage="Evaluation",
                step=2**i,
            )
            logger.log_metric(
                value=entropy_per_50.item(),
                name=f"entropy_per_50",
                stage="Evaluation",
                step=2**i,
            )

            logger.log_metric(
                value=perplexity.item(),
                name=f"Perplexity",
                stage="Evaluation",
                step=2**i,
            )
            logger.log_metric(
                value=entropy.item(), name=f"Entropy", stage="Evaluation", step=2**i
            )

            if rank == 0:
                print(
                    f"Step {2 ** i} -> Perplexity: {perplexity:.2f}, Entropy: {entropy:.2f}"
                )

            i = i + 1

    if eval_elbo:
        data_state = data._get_dataset(
            name=elbo_data,
            mode="validation",
            cache_dir=cfg.data.cache_dir,
            block_size=cfg.model.length,
            num_proc=cfg.data.num_workers,
            batch_size=batch_size,
            ngpus=world_size,
        )

        dataloader = DataLoader(
            data_state.dataset,
            batch_size=batch_size,
            sampler=data_state.sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.sampler is None),
        )

        elbo, num_elements = evaluate.estimate_likelihood(
            model=model,
            dataloader=dataloader,
            source_distribution=source_distribution,
            n_discretization=n_discretization,
            device=device,
            batch_size=batch_size,
            path=path,
        )
        dist.barrier()

        dist.all_reduce(elbo, dist.ReduceOp.SUM)
        dist.all_reduce(num_elements, dist.ReduceOp.SUM)

        if rank == 0:
            print(f"ELBO: {torch.exp(elbo / num_elements).item():.2f}")


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    torch.cuda.set_device(rank)

    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)


def cleanup() -> None:
    dist.destroy_process_group()


def run_mp_eval(
    rank: int,
    world_size: int,
    seed: int,
    work_dir: str,
    pre_trained_model_path: str,
    batch_size: int,
    sampling_steps: int,
    eval_elbo: bool,
    eval_perplexity: bool,
    elbo_data: str,
    perplexity_n_samples: int,
    port: int,
) -> None:
    try:
        setup(rank=rank, world_size=world_size, port=port)
        run_eval(
            rank=rank,
            seed=seed,
            work_dir=work_dir,
            pre_trained_model_path=pre_trained_model_path,
            batch_size=batch_size,
            sampling_steps=sampling_steps,
            eval_elbo=eval_elbo,
            eval_perplexity=eval_perplexity,
            elbo_data=elbo_data,
            world_size=world_size,
            perplexity_n_samples=perplexity_n_samples,
        )
    finally:
        cleanup()
