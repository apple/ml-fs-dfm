#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterable, Tuple
import os

from datasets import DatasetDict, load_dataset, load_from_disk
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, AutoTokenizer

from .tokenizer import wt_detokenizer
from .utils import cycle_loader, StatefulDistributedSampler

datasets_to_process = {
    "wikitext103": "wikitext-103-raw-v1",
    "fineweb-edu": "CC-MAIN-2024-10",
}

dataset_to_hf_name = {
    "wikitext103": "Salesforce/wikitext",
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
}


def _get_custom_dataset(
    name: str,
    mode: str,
    block_size: int = 1024,
    num_proc: int = 8,
    tokenizer=None,
    small_data: bool = False,
    cache_dir: str = "",
    use_cls_sep: bool = False,  # False => EOS-only (19). True => CLS+SEP (20).
) -> "Dataset":
    assert tokenizer is not None, "tokenizer must be provided"

    # Ensure EOS and PAD are defined
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "[SEP]"})
    EOS = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    if tokenizer.pad_token is None:
        # Prefer explicit PAD if your vocab has it; else fall back to EOS
        try:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        except Exception:
            pass
    PAD = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else EOS

    # <cache_dir>/processed_data/<name>/<mode>.json
    dataset_path = os.path.join(cache_dir, "processed_data", name, f"{mode}.json")
    data = load_dataset("json", data_files={mode: dataset_path})[mode]

    if small_data:
        data = data.select(range(min(len(data), 1000)))
    print(f"[INFO] Loaded {len(data)} samples for '{mode}' from {dataset_path}")

    # Tokenize + append EOS + pad/truncate per sample
    def preprocess_and_tokenize(batch: Dict):
        # enforce leading space so the first token matches " d" / " *" / " ="
        texts = [(t if t.startswith(" ") else (" " + t)) for t in batch["text"]]

        if use_cls_sep:
            # CLS+SEP path: tokenizer adds specials, we do NOT add EOS manually
            enc = tokenizer(
                texts,
                return_attention_mask=False,
                add_special_tokens=True,  # adds [CLS] ... [SEP]
            )
            processed = []
            for ids in enc["input_ids"]:
                # truncate/pad to block_size
                if len(ids) > block_size:
                    ids = ids[:block_size]
                if len(ids) < block_size:
                    ids = ids + [PAD] * (block_size - len(ids))
                processed.append(ids)
            return {"input_ids": processed}

        else:
            # EOS-only path: no specials; we append EOS ourselves
            enc = tokenizer(
                texts,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            processed = []
            for ids in enc["input_ids"]:
                # reserve space for EOS
                if len(ids) + 1 > block_size:
                    ids = ids[: max(0, block_size - 1)]
                ids = ids + [EOS]
                if len(ids) < block_size:
                    ids = ids + [PAD] * (block_size - len(ids))
                processed.append(ids)
            return {"input_ids": processed}

    tokenized = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc=f"Tokenizing+padding ({mode}) to block={block_size} ({'CLS+SEP' if use_cls_sep else 'EOS-only'})",
    )

    # Keep only input_ids; make it torch-friendly
    if "text" in tokenized.column_names:
        tokenized = tokenized.remove_columns("text")
    tokenized = tokenized.with_format("torch")

    return tokenized


def _get_custom_dataset_cached(
    name: str,
    mode: str,
    cache_dir: str,
    block_size: int,
    num_proc: int,
    tokenizer=None,
    small_data: bool = False,
    force_process: bool = False,
) -> DatasetDict:
    """
    Cache the processed dataset to disk to avoid re-tokenizing every time.
    """
    dataset_name = name.replace("_dataset", "")
    save_path = os.path.join(
        cache_dir, "processed_data", f"{dataset_name}_{mode}_block_{block_size}"
    )

    if os.path.exists(save_path) and not force_process:
        print(f"[INFO] Loading cached dataset from {save_path}")
        return load_from_disk(save_path)

    print("[INFO] Processing dataset from scratch...")
    dataset = _get_custom_dataset(
        name=name,
        mode=mode,
        tokenizer=tokenizer,
        block_size=block_size,
        num_proc=num_proc,
        small_data=small_data,
        cache_dir=cache_dir,
    )
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    print(f"[INFO] Cached dataset saved at {save_path}")
    return dataset


def _get_hf_dataset(
    name: str,
    subset: str,
    mode: str,
    cache_dir: str = None,
    block_size: int = 1024,
    num_proc: int = 8,
    small_data: bool = False,
    tokenizer=None,
) -> DatasetDict:
    detokenizer = None

    data = load_dataset(dataset_to_hf_name[name], name=subset, cache_dir=cache_dir)[
        mode
    ]
    if small_data:
        data = data.select(range(min(len(data), 1_000)))
    print(f"data len is {len(data)}")
    if name == "wikitext103":
        detokenizer = wt_detokenizer

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text

        return detok

    if tokenizer == None:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example: Dict):
        text = example["text"]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens["input_ids"]:
            token.append(EOS)

        return tokens

    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    if name == "fineweb-edu":
        features = tokenized_dataset.features.keys()
        for k in features:
            if k != "input_ids":
                tokenized_dataset = tokenized_dataset.remove_columns(k)
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    def group_texts(examples: Dict):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    chunked_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True
    )
    chunked_dataset = chunked_dataset.with_format("torch")

    return chunked_dataset


def _get_hf_dataset_cached(
    name: str,
    mode: str,
    cache_dir: str,
    block_size: int,
    num_proc: int,
    small_data: bool = False,
    tokenizer=None,
    force_process: bool = False,
) -> DatasetDict:
    save_path = os.path.join(
        cache_dir, "processed_data", f"{name}_{mode}_block_{block_size}"
    )
    if os.path.exists(save_path) and not force_process:
        print(f"Loading preprocessed dataset from {save_path}")
        return load_from_disk(save_path)

    print("Preprocessing dataset from scratch...")
    dataset = _get_hf_dataset(
        name=name,
        mode=mode,
        cache_dir=cache_dir,
        block_size=block_size,
        num_proc=num_proc,
        small_data=small_data,
        tokenizer=tokenizer,
        subset=datasets_to_process[name],
    )
    dataset.save_to_disk(save_path)
    return dataset


@dataclass
class Dataset:
    dataset: DatasetDict = field(metadata={"help": "Huggingface dataset"})
    sampler: StatefulDistributedSampler = field(
        metadata={"help": "Stateful sampler for `dataset`"}
    )


@dataclass
class DataState:
    train: Dataset = field(metadata={"help": "Train dataset"})
    test: Dataset = field(metadata={"help": "Test dataset"})


def _get_dataset(
    name: str,
    mode: str,
    cache_dir: str,
    block_size: int,
    num_proc: int,
    batch_size: int,
    ngpus: int,
    small_data: bool = False,
    tokenizer=None,
    force_process: bool = True,
    hf_dataset: bool = True,
    file_path: str = "",
    tokenizer_path: str = "",
) -> Dataset:
    assert (
        batch_size % ngpus == 0
    ), f"{mode} batch size must be divisible by number of gpus."

    if hf_dataset:
        dataset = _get_hf_dataset_cached(
            name=name,
            mode=mode,
            cache_dir=cache_dir,
            block_size=block_size,
            num_proc=num_proc,
            small_data=small_data,
            tokenizer=tokenizer,
            force_process=force_process,
        )

        sampler = StatefulDistributedSampler(dataset=dataset)

        return Dataset(dataset=dataset, sampler=sampler)

    dataset = _get_custom_dataset_cached(
        name=name,
        mode=mode,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        block_size=block_size,
        num_proc=num_proc,
        small_data=small_data,
        force_process=force_process,
    )

    sampler = StatefulDistributedSampler(dataset=dataset)

    return Dataset(dataset=dataset, sampler=sampler)


def get_data_state(config: OmegaConf, tokenizer=None) -> DataState:
    train = _get_dataset(
        name=config.data.train,
        mode="train",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
        small_data=config.data.small_data,
        tokenizer=tokenizer,
        force_process=config.data.force_process,
        hf_dataset=config.data.hf_dataset,
        file_path=os.path.join(config.data.train, "train.json"),
    )
    if config.data.hf_dataset:
        mode = "validation"
    else:
        mode = "test"

    test = _get_dataset(
        name=config.data.valid,
        mode=mode,
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.eval.batch_size,
        ngpus=config.compute.ngpus,
        small_data=config.data.small_data,
        tokenizer=tokenizer,
        force_process=config.data.force_process,
        hf_dataset=config.data.hf_dataset,
        file_path=os.path.join(config.data.train, "train.json"),
    )

    return DataState(train=train, test=test)


def get_data_loaders(
    config: OmegaConf,
    data_state: DataState,
) -> Tuple[Iterable, Iterable]:
    train_loader = cycle_loader(
        DataLoader(
            data_state.train.dataset,
            batch_size=config.training.batch_size // config.compute.ngpus,
            sampler=data_state.train.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.train.sampler is None),
            persistent_workers=True,
        )
    )

    valid_loader = cycle_loader(
        DataLoader(
            data_state.test.dataset,
            batch_size=config.eval.batch_size // config.compute.ngpus,
            sampler=data_state.test.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.test.sampler is None),
        )
    )

    return iter(train_loader), iter(valid_loader)
