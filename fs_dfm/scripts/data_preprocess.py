#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from datasets import DatasetDict
from fs_dfm.data.data import _get_hf_dataset
import os
from transformers import GPT2TokenizerFast, AutoTokenizer
from fs_dfm.data.data import datasets_to_process


cache_dir = "/mnt/task_wrapper/user_output/artifacts/processed_data"


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
save_path = os.path.join(cache_dir, "tokenizer_dir")
tokenizer.save_pretrained(save_path)

print("************************************************")
print(save_path)
print("************************************************")


for name in datasets_to_process:
    for mode in ["train", "validation"]:
        block_size = 1024 * 2
        save_path = os.path.join(cache_dir, f"{name}_{mode}_block_{block_size}")

        # fineweb edu only has train
        if name == "fineweb-edu" and mode != "train":
            continue

        dataset = _get_hf_dataset(
            name=name,
            subset=datasets_to_process[name],
            mode=mode,
            cache_dir="hf_cache",
            block_size=block_size,
            num_proc=18,
        )
        dataset.save_to_disk(save_path)

        print("************************************************")
        print(save_path)
        print("************************************************")


print("************************************************")
print("Done!!!")
