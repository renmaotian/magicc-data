"""Explicit, recorded seeding for the WS1.6 holdout retrain.

Motivation: the released V5 model was trained with no seed set anywhere in
magicc/trainer.py or scripts/053_train_v5_run3.py, so its weights are not
bit-reproducible. Reviewer 1 explicitly asks for random seeds in the repository.
The holdout retrain is the flagship new experiment of the revision, so it seeds
every RNG that affects it and records exactly what remains nondeterministic.

`SeededTrainer` subclasses the frozen V5 MAGICCTrainer and ONLY replaces the two
DataLoaders with seeded equivalents (identical batch size / shuffle / workers /
pin_memory / drop_last / persistent_workers). No part of the optimisation,
schedule, loss or augmentation policy is altered, so the holdout run remains
directly comparable with production V5.

Two RNGs matter for the DataLoader:
  * torch generator  -> the shuffling permutation of the training sampler
  * numpy global RNG -> HDF5Dataset.__getitem__ uses np.random for the 2% k-mer
    masking and the sigma=0.01 Gaussian noise. PyTorch reseeds each worker's
    torch RNG but NOT numpy's, so without worker_init_fn the augmentation stream
    is nondeterministic across runs. `_worker_init` fixes that deterministically
    as a function of (base seed, worker id).
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .trainer import MAGICCTrainer

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic_algorithms: bool = False):
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices).

    Returns a dict describing what was seeded and what remains nondeterministic.
    `deterministic_algorithms=False` keeps cuDNN autotuning enabled: on the
    Quadro P2200 disabling it measurably slows training, and the protocol prefers
    documenting residual nondeterminism over paying that cost.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        'seed': seed,
        'seeded': ['PYTHONHASHSEED', 'random.seed', 'numpy.random.seed',
                   'torch.manual_seed', 'torch.cuda.manual_seed',
                   'torch.cuda.manual_seed_all',
                   'DataLoader generator (torch.Generator, seed)',
                   'DataLoader worker_init_fn (numpy + random + torch per worker)'],
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'deterministic_algorithms': bool(deterministic_algorithms),
        'residual_nondeterminism': [
            f'torch.backends.cudnn.deterministic={torch.backends.cudnn.deterministic} '
            f'and torch.use_deterministic_algorithms='
            f'{deterministic_algorithms}: cuDNN/cuBLAS may pick non-deterministic '
            f'kernels, so results are not guaranteed bit-identical '
            f'(cudnn.benchmark={torch.backends.cudnn.benchmark})',
            'non-deterministic CUDA atomics in some backward kernels',
            'FP16 AMP GradScaler loss-scale trajectory depends on observed '
            'overflow timing',
            'DataLoader completion order with num_workers>0 does not affect batch '
            'contents (a seeded sampler fixes the permutation) but can affect '
            'floating-point reduction order on the host',
        ],
        'note': ('Given identical data, identical hardware and this seed, runs are '
                 'statistically reproducible (val metrics within run-to-run noise) '
                 'but not guaranteed bit-identical. Production V5 was trained with '
                 'no seed at all, so it is not reproducible even to this degree.'),
    }


def _worker_init(worker_id: int):
    base = torch.initial_seed() % 2 ** 31
    s = (base + worker_id) % 2 ** 31
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


class SeededTrainer(MAGICCTrainer):
    """Frozen V5 MAGICCTrainer with seeded, reproducible DataLoaders."""

    def __init__(self, *args, seed: int = DEFAULT_SEED, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        g = torch.Generator()
        g.manual_seed(seed)
        nw = self.train_loader.num_workers
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=nw, pin_memory=True, drop_last=True,
            persistent_workers=nw > 0, generator=g,
            worker_init_fn=_worker_init if nw > 0 else None)
        gv = torch.Generator()
        gv.manual_seed(seed + 1)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
            generator=gv, worker_init_fn=_worker_init if nw > 0 else None)
