import math

import torch
from torch.utils.data import Sampler
import torch.distributed as dist

from .resample import class_level_weight


class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class RepeatAugSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU). Heavily based on torch.utils.data.DistributedSampler

    This sampler was taken from https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
    Used in
    Copyright (c) 2015-present, Facebook, Inc.
    """

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            num_repeats=3,
            selected_round=256,
            selected_ratio=0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # Determine the number of samples to select per epoch for each rank.
        # num_selected logic defaults to be the same as original RASampler impl, but this one can be tweaked
        # via selected_ratio and selected_round args.
        selected_ratio = selected_ratio or num_replicas  # ratio to reduce selected samples by, num_replicas if 0
        if selected_round:
            self.num_selected_samples = int(math.floor(
                 len(self.dataset) // selected_round * selected_round / selected_ratio))
        else:
            self.num_selected_samples = int(math.ceil(len(self.dataset) / selected_ratio))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        if isinstance(self.num_repeats, float) and not self.num_repeats.is_integer():
            # resample for repeats w/ non-integer ratio
            repeat_size = math.ceil(self.num_repeats * len(self.dataset))
            indices = indices[torch.tensor([int(i // self.num_repeats) for i in range(repeat_size)])]
        else:
            indices = torch.repeat_interleave(indices, repeats=int(self.num_repeats), dim=0)
        indices = indices.tolist()  # leaving as tensor thrashes dataloader memory
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample per rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # return up to num selected samples
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class ReSampler(Sampler):
    """Resampling the dataset for by class frequency
    """
    def __init__(self, dataset, resample):
        self.gen = torch.Generator().manual_seed(0)
        self.dataset = dataset

        self.len = len(dataset)
        P = torch.tensor(dataset.target_freq())
        sample_cnt = self.len * class_level_weight(P, resample)
        margin = sample_cnt - sample_cnt.floor()
        self.sample_cnt = sample_cnt.floor().long()
        self.sample_cnt[torch.argsort(-margin)[:self.len - self.sample_cnt.sum()]] += 1
        self.sample_cnt = self.sample_cnt.numpy().tolist()

        cls = [[] for _ in P]
        for i in range(self.len):
            cls[dataset.get_target(i)].append(i)
        self.num_cls = list(map(len, cls))
        self.index = torch.tensor([J for I in cls for J in I])

    def __iter__(self):
        base_index = 0
        group = []
        for cls in range(len(self.num_cls)):
            Q = self.sample_cnt[cls] // self.num_cls[cls]
            R = self.sample_cnt[cls] % self.num_cls[cls]
            if Q:
                group.extend([base_index + torch.arange(self.num_cls[cls])] * Q)
            if R:
                group.append(base_index + torch.randperm(self.num_cls[cls], generator=self.gen)[:R])
            base_index += self.num_cls[cls]
        S = self.index[torch.cat(group)][torch.randperm(self.len, generator=self.gen)]
        # count = collections.Counter(map(self.dataset.get_target, S))
        # assert all(I == count[i] for i, I in enumerate(self.sample_cnt))
        return iter(S)

    def __len__(self) -> int:
        return self.len
    
    def set_epoch(self, epoch):
        self.gen = torch.Generator().manual_seed(epoch)
