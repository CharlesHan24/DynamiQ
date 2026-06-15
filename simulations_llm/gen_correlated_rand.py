#!/usr/bin/env python
"""Generate deterministic sampling orders and correlated random tensors."""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_WORKERS = (2, 4, 8)
DEFAULT_RANDOM_DIMENSION = 1 << 28
DEFAULT_CORRELATED_CHUNK_VALUES = 1_000_000


def parse_workers(raw_workers):
    workers = []
    for raw_worker in raw_workers.split(","):
        raw_worker = raw_worker.strip()
        if raw_worker:
            workers.append(int(raw_worker))
    if not workers:
        raise ValueError("--workers must contain at least one worker count")
    return workers


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_correlated_random_vectors(num_values, nranks, output_dir, chunk_values):
    output_dir.mkdir(parents=True, exist_ok=True)
    random_values = np.empty((nranks, num_values), dtype=np.float32)

    for start in range(0, num_values, chunk_values):
        end = min(start + chunk_values, num_values)
        width = end - start

        permutations = np.argsort(np.random.random((nranks, width)), axis=0).astype(np.float32)
        uniforms = np.random.uniform(0, 1, (nranks, width)).astype(np.float32)
        random_values[:, start:end] = (permutations + uniforms) / nranks

        if start == 0 or end == num_values or (start // chunk_values) % 100 == 0:
            print(f"correlated random: workers={nranks} generated {end}/{num_values} values per rank")

    for rank in range(nranks):
        obj = torch.tensor(random_values[rank], dtype=torch.bfloat16)
        out_path = output_dir / f"obj_{nranks}_{rank}.pt"
        torch.save(obj, out_path)
        print(f"wrote {out_path}")


def generate_random_sign_vec(num_values, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    signs = np.random.randint(0, 2, num_values)
    signs = 2 * signs - 1
    out_path = output_dir / "obj_hadamard.pt"
    torch.save(torch.tensor(signs, dtype=torch.bfloat16), out_path)
    print(f"wrote {out_path}")


def generate_random_sampling_order(nsamples, nranks, nepochs, output_path, repeats=1):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_orders = [[] for _ in range(nranks)]

    for _ in range(nepochs):
        orders = list(range(nsamples)) * repeats
        random.shuffle(orders)
        usable_len = len(orders) // nranks * nranks
        orders = orders[:usable_len]

        for rank in range(nranks):
            all_orders[rank].append(orders[rank::nranks])

    with output_path.open("wb") as fout:
        pickle.dump(all_orders, fout)
    print(f"wrote {output_path}")


def generate_sampling_orders(models_dir, workers):
    for nranks in workers:
        generate_random_sampling_order(
            nsamples=82971,
            nranks=nranks,
            nepochs=3,
            output_path=models_dir / f"indices_{nranks}.pkl",
        )
        generate_random_sampling_order(
            nsamples=82211,
            nranks=nranks,
            nepochs=3,
            output_path=models_dir / f"indices_gemma_{nranks}.pkl",
        )
        generate_random_sampling_order(
            nsamples=22408,
            nranks=nranks,
            nepochs=4,
            output_path=models_dir / f"indices_mmlu_new_{nranks}.pkl",
            repeats=4,
        )
        generate_random_sampling_order(
            nsamples=22460,
            nranks=nranks,
            nepochs=4,
            output_path=models_dir / f"indices_mmlu_new_gemma_{nranks}.pkl",
            repeats=4,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deterministic sampler orders and correlated random tensors for simulations."
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--workers", type=str, default=",".join(str(worker) for worker in DEFAULT_WORKERS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-orders", action="store_true", help="Do not generate fixed sampler order files.")
    parser.add_argument("--skip-correlated-rand", action="store_true", help="Do not generate correlated random tensors.")
    parser.add_argument("--skip-hadamard", action="store_true", help="Do not generate the THC Hadamard sign vector.")
    parser.add_argument(
        "--random-dimension",
        type=int,
        default=DEFAULT_RANDOM_DIMENSION,
        help="Base random dimension used to size correlated and Hadamard random tensors.",
    )
    parser.add_argument(
        "--correlated-chunk-values",
        type=int,
        default=DEFAULT_CORRELATED_CHUNK_VALUES,
        help="Number of random values to generate per chunk while building correlated tensors.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    workers = parse_workers(args.workers)
    models_dir = args.models_dir.resolve()
    correlated_rand_dir = models_dir / "correlated_rand"

    seed_everything(args.seed)

    if not args.skip_orders:
        generate_sampling_orders(models_dir, workers)

    if not args.skip_correlated_rand:
        for nranks in workers:
            num_values = args.random_dimension // nranks * 3 + 512
            generate_correlated_random_vectors(
                num_values=num_values,
                nranks=nranks,
                output_dir=correlated_rand_dir,
                chunk_values=args.correlated_chunk_values,
            )

    if not args.skip_hadamard:
        generate_random_sign_vec(args.random_dimension, correlated_rand_dir)


if __name__ == "__main__":
    main()
