#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import argparse

import torch.multiprocessing as mp

from eval import run_mp_eval


def main(args: argparse.Namespace):
    port = 12346

    assert args.perplexity_n_samples % args.ngpus == 0
    assert args.batch_size % args.ngpus == 0

    if args.ngpus == 1:
        run_mp_eval(
            rank=0,
            world_size=1,
            seed=args.seed,
            work_dir=args.work_dir,
            pre_trained_model_path=args.pre_trained_model_path,
            batch_size=args.batch_size // args.ngpus,
            sampling_steps=args.sampling_steps,
            eval_elbo=args.eval_elbo,
            eval_perplexity=args.eval_perplexity,
            elbo_data=args.elbo_data,
            perplexity_n_samples=args.perplexity_n_samples // args.ngpus,
            port=port,
            teacher_model=args.teacher_model,
            do_dynamic_step=args.do_dynamic_step,
        )
    else:
        mp.set_start_method("forkserver")

        mp.spawn(
            run_mp_eval,
            args=(
                args.ngpus,
                args.seed,
                args.work_dir,
                args.pre_trained_model_path,
                args.batch_size // args.ngpus,
                args.sampling_steps,
                args.eval_elbo,
                args.eval_perplexity,
                args.elbo_data,
                args.perplexity_n_samples // args.ngpus,
                port,
                args.teacher_model,
                args.do_dynamic_step,
            ),
            nprocs=args.ngpus,
            join=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--pre_trained_model_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--teacher_model", action="store_true")

    parser.add_argument("--eval_elbo", action="store_true")
    parser.add_argument("--eval_perplexity", action="store_true")
    parser.add_argument("--do_dynamic_step", action="store_true")

    # Perplexity parameters
    parser.add_argument("--sampling_steps", type=int, default=1024)
    parser.add_argument("--perplexity_n_samples", type=int, default=1024)

    # ELBO parameters
    parser.add_argument("--elbo_data", type=str, default="wikitext103")

    args = parser.parse_args()
    main(args)
