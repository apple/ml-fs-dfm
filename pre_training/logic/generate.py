#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from pathlib import Path
from typing import Optional, List

import torch
from flow_matching.path import ProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm

from .flow import SourceDistribution


class WrappedModel(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        # Note: logit's precision is important.
        return torch.softmax(self.model(x_t=x, time=t).float(), -1)


def generate_samples(
    model: nn.Module,
    step: int,
    vocab_size: int,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    device: torch.device,
    path: ProbPath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    time_epsilon: float = 0.0,
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
) -> Tensor:
    wrapped_probability_denoiser = WrappedModel(model=model)

    add_token = 1 if source_distribution.masked else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    x_init = source_distribution.sample(
        tensor_size=(sample_batch_size, sequence_length), device=device
    )

    sample = solver.sample(
        x_init=x_init,
        step_size=1 / sampling_steps,
        verbose=False,
        dtype_categorical=dtype_categorical,
        time_grid=torch.tensor([0.0, 1.0 - time_epsilon]),
    )

    sentences = tokenizer.batch_decode(sample)

    if sample_dir is not None:
        file_name = sample_dir / f"iter_{step}" / f"sample_{rank}.txt"
        file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with open(file_name, "w") as file:
            for sentence in sentences:
                file.write(f"{sentence}\n{'=' * 20} New sample {'=' * 20}\n")

    return sample


def decode_with_edit_mask(
    batch_ids: torch.Tensor,  # [B, L] token ids
    edit_mask: torch.Tensor,  # [B, L] bool/0-1; True => show placeholder
    tokenizer,
    placeholder: str = "<MASK>",
    skip_special_tokens: bool = True,
) -> List[str]:
    if batch_ids.shape != edit_mask.shape:
        raise ValueError(
            f"Shape mismatch: batch_ids {batch_ids.shape} vs edit_mask {edit_mask.shape}"
        )

    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    ids_cpu = batch_ids.detach().cpu().tolist()  # List[List[int]]
    mask_cpu = edit_mask.to(torch.bool).detach().cpu().tolist()

    out_texts: List[str] = []
    for seq_ids, seq_mask in zip(ids_cpu, mask_cpu):
        toks = tokenizer.convert_ids_to_tokens(
            seq_ids, skip_special_tokens=False
        )  # one-to-one with ids
        out_toks = []
        for tid, tok, m in zip(seq_ids, toks, seq_mask):
            if m:
                out_toks.append(placeholder)
            else:
                if skip_special_tokens and tid in special_ids:
                    continue
                out_toks.append(tok)
        out_texts.append(tokenizer.convert_tokens_to_string(out_toks))
    return out_texts


def decode_with_mask(
    batch_ids: torch.Tensor, tokenizer, mask_id: int, placeholder: str = "<MASK>"
) -> list[str]:
    out = []
    for seq in batch_ids.detach().cpu().tolist():  # List[int]
        toks = tokenizer.convert_ids_to_tokens(
            seq, skip_special_tokens=False  # List[str]
        )
        # swap mask token id for a readable string
        for i, tid in enumerate(seq):
            if tid == mask_id:
                toks[i] = placeholder
        text = tokenizer.convert_tokens_to_string(toks)
        out.append(text)
    return out


def build_controller_prefix(
    controller_mode: str,
    controller_pct: float,
    controller_left_k: int,
    controller_merge_prob: float = 1.0,
    *,
    include_merge_prob: bool = False,
    suffix: str = "eval",
) -> str:
    # normalize pct: allow 0.3 or 30 -> both become 30
    pct_val = int(
        round(controller_pct if controller_pct > 1.0 else controller_pct * 100)
    )
    if controller_mode == "percentage":
        base = f"per_{pct_val}"
    elif controller_mode == "left_k":
        base = f"left_{int(controller_left_k)}"
    else:
        base = "unknown"

    # (optional) annotate merge prob, clamped to [0,1]
    if include_merge_prob:
        mp = max(0.0, min(1.0, float(controller_merge_prob)))
        base += f"_mp{int(round(mp * 100))}"

    return f"{base}_{suffix}" if suffix else base


def generate_samples_with_dataset(
    model: nn.Module,
    step: int,
    vocab_size: int,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    device: torch.device,
    path: ProbPath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    time_epsilon: float = 0.0,
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
    dataloader: DataLoader = None,
    controller_mode: str = "percentage",  # "percentage" or "left_k"
    controller_merge_prob: float = 1.5,
    controller_pct: float = 0.3,
    controller_left_k: int = 10,
    return_metrics: bool = False,
) -> Tensor:
    wrapped_probability_denoiser = WrappedModel(model=model)

    add_token = 1 if source_distribution.masked else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    all_sentences = []  # store everything here
    all_samples = []
    all_merged_texts = []
    mask_id = vocab_size

    total_pred_tokens = torch.tensor(0).to(device)
    total_correct_pred = torch.tensor(0).to(device)

    for x_1 in tqdm(dataloader, total=len(dataloader)):
        x_1 = x_1["input_ids"].to(device)

        x_init = source_distribution.sample(
            tensor_size=(x_1.shape[0], sequence_length), device=device
        )

        merged_init = _apply_controller_policy(
            x_1,
            x_init,
            mode=controller_mode,
            merge_prob=controller_merge_prob,
            pct=controller_pct,
            left_k=controller_left_k,
        )
        edit_mask = (merged_init == x_init).bool()  # [B, L]

        sample = solver.sample_masked(
            x_init=merged_init,
            step_size=1 / sampling_steps,
            verbose=False,
            dtype_categorical=dtype_categorical,
            time_grid=torch.tensor([0.0, 1.0 - time_epsilon]),
            edit_mask=edit_mask,
        )

        matches = (sample == x_1) & edit_mask
        total_correct_pred += matches.sum()
        total_pred_tokens += edit_mask.sum()

        placeholder = "<MASK>"
        if not source_distribution.masked:
            placeholder = "<RANDOM>"
        merged_texts = decode_with_edit_mask(
            merged_init, edit_mask, tokenizer, placeholder=placeholder
        )
        # merged_texts = decode_with_mask(merged_init, tokenizer, mask_id, placeholder="<MASK>")
        sentences = tokenizer.batch_decode(sample)
        all_sentences.extend(sentences)
        all_merged_texts.extend(merged_texts)
        all_samples.extend(sample)

    if sample_dir is not None:
        prefix = build_controller_prefix(
            controller_mode,
            controller_pct,
            controller_left_k,
            controller_merge_prob,
            include_merge_prob=False,
            suffix="eval",
        )
        file_name = sample_dir / f"iter_{step}" / f"{prefix}_{rank}.txt"
        file_name.parent.mkdir(exist_ok=True, parents=True)
        sep = "=" * 20 + " New sample " + "=" * 20
        with open(file_name, "w", encoding="utf-8") as f:
            for mi, s in zip(all_merged_texts, all_sentences):
                f.write(f"MERGED_INIT: {mi}\nGENERATED: {s}\n{sep}\n")

    all_samples = torch.stack(all_samples, dim=0)

    pred_token_acc = (
        (total_correct_pred / total_pred_tokens)
        if total_pred_tokens > 0
        else float("nan")
    )
    print(
        f"[Eval] Predicted-token accuracy - controller_mode: {controller_mode}"
        f"- controller_pct: {controller_pct} - controller_left_k: {controller_left_k} ---- {pred_token_acc:.4%} "
        f"({total_correct_pred}/{total_pred_tokens})"
    )

    if return_metrics:
        return all_samples, {
            "predicted_token_accuracy": pred_token_acc,
            "num_predicted_tokens": total_pred_tokens,
            "num_correct_predicted": total_correct_pred,
        }
    return all_samples


def _apply_controller_policy(
    x_1: torch.Tensor,
    x_init: torch.Tensor,
    *,
    mode: str = "percentage",  # "percentage" or "left_k"
    merge_prob: float = 1.5,  # with this probability we merge (else keep x_init)
    pct: float = 0.3,  # for "percentage" mode: fraction to take from x_1
    left_k: int = 10,  # for "left_k" mode: keep first K from x_1
) -> torch.Tensor:
    """Return the merged init state used as x_init for sampling."""
    # sometimes skip merging
    if torch.rand(()) > merge_prob:
        return x_init

    if mode == "percentage":
        # Bernoulli mask per token: True => take from x_1, False => keep x_init
        bern = torch.rand_like(x_1, dtype=torch.float32) < pct
        return torch.where(bern, x_1, x_init)

    elif mode == "left_k":
        k = min(left_k, x_1.shape[-1])
        merged = x_init.clone()
        merged[..., :k] = x_1[..., :k]
        return merged

    else:
        # fallback: no merge
        return x_init
