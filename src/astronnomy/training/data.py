import torch
import numpy as np
import random
import os
from torchvision.transforms import v2
from ..pre_processing_for_ml import FitsDataset


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return (
            len(self.sampler)
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler)
        )

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

    def __hash__(self):
        return hash(self.dataset) + 10000


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):

        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id)
    random.seed(worker_seed)


def get_dataloaders(dataset_root, batch_size):
    num_workers = min(18, len(os.sched_getaffinity(0)))

    prefetch_factor, persistent_workers = (
        (2, True) if num_workers > 0 else (None, False)
    )
    generators = {}
    for mode in ("val", "train"):
        generators[mode] = torch.Generator()
        seed = os.getenv("TRAIN_SEED", None)
        if seed is not None:
            generators[mode].manual_seed(int(seed))

    loaders = tuple(
        MultiEpochsDataLoader(
            dataset=FitsDataset(dataset_root, mode=mode),
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker,
            generator=generators[mode],
            pin_memory=True,
            shuffle=True if mode == "train" else False,
            drop_last=True if mode == "train" else False,
        )
        for mode in ("train", "val")
    )

    return loaders


@torch.no_grad()
def prepare_data(
    data: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    transform=v2.Compose([v2.Identity()]),
):

    data, labels = (
        data.to(device, non_blocking=True, memory_format=torch.channels_last),
        labels.to(device, non_blocking=True, dtype=data.dtype),
    )

    data = transform(data)

    return data, labels
