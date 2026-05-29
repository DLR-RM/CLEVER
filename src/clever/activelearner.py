#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2024. Jongseok Lee                                             #
# All rights reserved.                                                         #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-01-2024                                                             #
# Author: Jongseok Lee                                                         #
# E-mail: jongseok.lee@dlr.de                                                  #
# Website: https://rmc.dlr.de/rm/en/staff/jongseok.lee/                        #
################################################################################

"""
BatchBald implementation for clever system.

Adapted based on: https://github.com/BlackHC/batchbald_redux
"""

import torch
import math
import collections
import random
import numpy as np
import torch.utils.data as data

from dataclasses import dataclass
from toma import toma
from tqdm.auto import tqdm
from typing import Dict, List


class ActiveLearningData:
    """  Splits `dataset` into an active dataset and an available dataset.
    dataset: data.Dataset
    training_dataset: data.Dataset
    pool_dataset: data.Dataset
    training_mask: np.ndarray
    pool_mask: np.ndarray
    """

    def __init__(self, dataset: data.Dataset):
        super().__init__()
        self.dataset = dataset
        self.training_mask = np.full((len(dataset),), False)
        self.pool_mask = np.full((len(dataset),), True)
        self.training_dataset = data.Subset(self.dataset, None)
        self.pool_dataset = data.Subset(self.dataset, None)
        self._update_indices()

    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]

    def get_dataset_indices(self, pool_indices: List[int]) -> List[int]:
        """Transform indices (in `pool_dataset`) to indices in the original `dataset`."""
        indices = self.pool_dataset.indices[pool_indices]
        return indices

    def acquire(self, pool_indices):
        """Acquire elements from the pool dataset into the training dataset.
        Add them to training dataset & remove them from the pool dataset."""
        indices = self.get_dataset_indices(pool_indices)
        self.training_mask[indices] = True
        self.pool_mask[indices] = False
        self._update_indices()

    def remove_from_pool(self, pool_indices):
        indices = self.get_dataset_indices(pool_indices)
        self.pool_mask[indices] = False
        self._update_indices()

    def get_random_pool_indices(self, size) -> torch.LongTensor:
        assert 0 <= size <= len(self.pool_dataset)
        pool_indices = torch.randperm(len(self.pool_dataset))[:size]
        return pool_indices

    def extract_dataset_from_pool(self, size) -> data.Dataset:
        """Extract a dataset randomly from the pool dataset and make those indices unavailable.
        Useful for extracting a validation set."""
        return self.extract_dataset_from_pool_from_indices(self.get_random_pool_indices(size))

    def extract_dataset_from_pool_from_indices(self, pool_indices) -> data.Dataset:
        """Extract a dataset from the pool dataset and make those indices unavailable.
        Useful for extracting a validation set."""
        dataset_indices = self.get_dataset_indices(pool_indices)
        self.remove_from_pool(pool_indices)
        return data.Subset(self.dataset, dataset_indices)

def get_balanced_sample_indices(target_classes: List, num_classes, n_per_digit=2) -> List[int]:
    """Given `target_classes` randomly sample `n_per_digit` for each of the `num_classes` classes."""
    permed_indices = torch.randperm(len(target_classes))
    if n_per_digit == 0:
        return []
    num_samples_by_class = collections.defaultdict(int)
    initial_samples = []
    for i in range(len(permed_indices)):
        permed_index = int(permed_indices[i])
        index, target = permed_index, int(target_classes[permed_index])
        num_target_samples = num_samples_by_class[target]
        if num_target_samples == n_per_digit:
            continue
        initial_samples.append(index)
        num_samples_by_class[target] += 1
        if len(initial_samples) == num_classes * n_per_digit:
            break
    return initial_samples

def get_subset_base_indices(dataset: data.Subset, indices: List[int]):
    return [int(dataset.indices[index]) for index in indices]

def get_base_indices(dataset: data.Dataset, indices: List[int]):
    if isinstance(dataset, data.Subset):
        return get_base_indices(dataset.dataset, get_subset_base_indices(dataset, indices))
    return indices


class RandomFixedLengthSampler(data.Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.
    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).

    dataset: data.Dataset
    target_length: int
    """

    def __init__(self, dataset: data.Dataset, target_length: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(torch.randperm(len(self.dataset)).tolist())
        # Sample slightly more indices to avoid biasing towards start of dataset.
        # Have the same number of duplicates for each sample.
        indices = torch.randperm(
            self.target_length + (-self.target_length % len(self.dataset))
        )
        return iter((indices[:self.target_length] % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length
    

class JointEntropy:
    """Random variables (all with the same # of categories $C$) can 
    be added via `JointEntropy.add_variables`.
    `JointEntropy.compute` computes the joint entropy.
    `JointEntropy.compute_batch` computes the joint entropy of the 
    added variables with each of the variables in the provided batch probabilities in turn."""
    def compute(self) -> torch.Tensor:
        """Computes the entropy of this joint entropy."""
        raise NotImplementedError()

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "JointEntropy":
        """Expands the joint entropy to include more terms."""
        raise NotImplementedError()

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None) -> torch.Tensor:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        raise NotImplementedError()


class ExactJointEntropy(JointEntropy):
    joint_probs_M_K: torch.Tensor

    def __init__(self, joint_probs_M_K: torch.Tensor):
        self.joint_probs_M_K = joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> "ExactJointEntropy":
        return ExactJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    def compute(self) -> torch.Tensor:
        probs_M = torch.mean(self.joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(probs_M) * probs_M
        entropy = torch.sum(nats_M)
        return entropy

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "ExactJointEntropy":
        assert self.joint_probs_M_K.shape[1] == log_probs_N_K_C.shape[1]
        N, K, C = log_probs_N_K_C.shape
        joint_probs_K_M_1 = self.joint_probs_M_K.t()[:, :, None]
        probs_N_K_C = log_probs_N_K_C.exp()
        # TODO: Using lots of memory. Remove for loop.
        for i in range(N):
            probs_i__K_1_C = probs_N_K_C[i][:, None, :].to(joint_probs_K_M_1, non_blocking=True)
            joint_probs_K_M_C = joint_probs_K_M_1 * probs_i__K_1_C
            joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))
        self.joint_probs_M_K = joint_probs_K_M_1.squeeze(2).t()
        return self

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None):
        assert self.joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]
        B, K, C = log_probs_B_K_C.shape
        M = self.joint_probs_M_K.shape[0]
        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)
        pbar = tqdm(total=B, desc="ExactJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int):
            chunked_probs_b_K_C = chunked_log_probs_b_K_C.exp()
            b = chunked_probs_b_K_C.shape[0]
            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.joint_probs_M_K.dtype,
                device=self.joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.joint_probs_M_K,
                    chunked_probs_b_K_C[i].to(self.joint_probs_M_K, non_blocking=True),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K
            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C, dim=(1, 2)),
                non_blocking=True,
            )
            pbar.update(end - start)
        pbar.close()
        return output_entropies_B

def batch_multi_choices(probs_b_C, M: int):
    """
    probs_b_C: Ni... x C

    Returns:
        choices: Ni... x M
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))
    # samples: Ni... x draw_per_xx
    choices = torch.multinomial(probs_B_C, num_samples=M, replacement=True)
    choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [M])
    return choices_b_M

def gather_expand(data, dim, index):
    if gather_expand.DEBUG_CHECKS:
        assert len(data.shape) == len(index.shape)
        assert all(dr == ir or 1 in (dr, ir) for dr, ir in zip(data.shape, index.shape))
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]
    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]
    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)
    return torch.gather(data, dim, index)

gather_expand.DEBUG_CHECKS = False

class SampledJointEntropy(JointEntropy):
    """Random variables (all with the same # of categories $C$) can be 
    added via `SampledJointEntropy.add_variables`.
    `SampledJointEntropy.compute` computes the joint entropy.
    `SampledJointEntropy.compute_batch` computes the joint entropy of
    the added variables with each of the variables in the provided batch probabilities in turn."""
    sampled_joint_probs_M_K: torch.Tensor

    def __init__(self, sampled_joint_probs_M_K: torch.Tensor):
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> "SampledJointEntropy":
        return SampledJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    @staticmethod
    def sample(probs_N_K_C: torch.Tensor, M: int) -> "SampledJointEntropy":
        K = probs_N_K_C.shape[1]
        # S: num of samples per w
        S = M // K
        choices_N_K_S = batch_multi_choices(probs_N_K_C, S).long()
        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]
        probs_N_K_K_S = gather_expand(expanded_probs_N_K_1_C, dim=-1, index=expanded_choices_N_1_K_S)
        # exp sum log seems necessary to avoid 0s?
        probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
        samples_K_M = probs_K_K_S.reshape((K, -1))
        samples_M_K = samples_K_M.t()
        return SampledJointEntropy(samples_M_K)

    def compute(self) -> torch.Tensor:
        sampled_joint_probs_M = torch.mean(self.sampled_joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(sampled_joint_probs_M)
        entropy = torch.mean(nats_M)
        return entropy

    def add_variables(self, log_probs_N_K_C: torch.Tensor, M2: int) -> "SampledJointEntropy":
        K = self.sampled_joint_probs_M_K.shape[1]
        assert K == log_probs_N_K_C.shape[1]
        sample_K_M1_1 = self.sampled_joint_probs_M_K.t()[:, :, None]
        new_sample_M2_K = self.sample(log_probs_N_K_C.exp(), M2).sampled_joint_probs_M_K
        new_sample_K_1_M2 = new_sample_M2_K.t()[:, None, :]
        merged_sample_K_M1_M2 = sample_K_M1_1 * new_sample_K_1_M2
        merged_sample_K_M = merged_sample_K_M1_M2.reshape((K, -1))
        self.sampled_joint_probs_M_K = merged_sample_K_M.t()
        return self

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None):
        assert self.sampled_joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]
        B, K, C = log_probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]
        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)
        pbar = tqdm(total=B, desc="SampledJointEntropy.compute_batch", leave=False)
        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int):
            b = chunked_log_probs_b_K_C.shape[0]
            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.sampled_joint_probs_M_K.dtype,
                device=self.sampled_joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.sampled_joint_probs_M_K,
                    chunked_log_probs_b_K_C[i].to(self.sampled_joint_probs_M_K, non_blocking=True).exp(),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K
            q_1_M_1 = self.sampled_joint_probs_M_K.mean(dim=1, keepdim=True)[None]
            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C / q_1_M_1, dim=(1, 2)) / M,
                non_blocking=True,
            )
            pbar.update(end - start)
        pbar.close()
        return output_entropies_B


class DynamicJointEntropy(JointEntropy):
    """ 
    Joint entroopy formulation

    inner: JointEntropy
    log_probs_max_N_K_C: torch.Tensor
    N: int
    M: int
    """

    def __init__(self, M: int, max_N: int, K: int, C: int, dtype=None, device=None):
        self.M = M
        self.N = 0
        self.max_N = max_N
        self.inner = ExactJointEntropy.empty(K, dtype=dtype, device=device)
        self.log_probs_max_N_K_C = torch.empty((max_N, K, C), dtype=dtype, device=device)

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "DynamicJointEntropy":
        C = self.log_probs_max_N_K_C.shape[2]
        add_N = log_probs_N_K_C.shape[0]
        assert self.log_probs_max_N_K_C.shape[0] >= self.N + add_N
        assert self.log_probs_max_N_K_C.shape[2] == C
        self.log_probs_max_N_K_C[self.N : self.N + add_N] = log_probs_N_K_C
        self.N += add_N
        num_exact_samples = C ** self.N
        if num_exact_samples > self.M:
            self.inner = SampledJointEntropy.sample(self.log_probs_max_N_K_C[: self.N].exp(), self.M)
        else:
            self.inner.add_variables(log_probs_N_K_C)
        return self

    def compute(self) -> torch.Tensor:
        return self.inner.compute()

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None) -> torch.Tensor:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        return self.inner.compute_batch(log_probs_B_K_C, output_entropies_B)
    

def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)
    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)
    pbar.close()
    return entropies_N

def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    pbar = tqdm(total=N, desc="Entropy", leave=False)
    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)
    pbar.close()
    return entropies_N

@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]

def get_batchbald_batch(log_probs_N_K_C: torch.Tensor, 
                        batch_size: int, 
                        num_samples: int, 
                        dtype=None, 
                        device=None
                        ) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape
    batch_size = min(batch_size, N)
    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)
    batch_joint_entropy = DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])
        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()
        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)
        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")
        candidate_score, candidate_index = scores_N.max(dim=0)
        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return CandidateBatch(candidate_scores, candidate_indices)

def get_bald_batch(log_probs_N_K_C: torch.Tensor, 
                   batch_size: int, 
                   dtype=None, 
                   device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape
    batch_size = min(batch_size, N)
    candidate_indices = []
    candidate_scores = []
    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)
    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)
    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())

def get_random_batch(log_probs_N_K_C: torch.Tensor, 
                     batch_size: int, 
                     dtype=None, 
                     device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape
    batch_size = min(batch_size, N)
    candidate_indices = random.sample(range(0, N), batch_size)
    candidate_scores = [1/float(batch_size) for i in range(len(candidate_indices))]
    candidate=CandidateBatch([],[])
    candidate.scores = candidate_scores
    candidate.indices = candidate_indices
    return candidate


class TransformedDataset(data.Dataset):
    """
    Transforms a dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        transformer (LambdaType): (idx, sample) -> transformed_sample
    """

    def __init__(self, dataset, *, transformer=None, vision_transformer=None):
        self.dataset = dataset
        assert not transformer or not vision_transformer
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = lambda _, data_label: (vision_transformer(data_label[0]), data_label[1])

    def __getitem__(self, idx):
        return self.transformer(idx, self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

def get_targets(dataset):
    """Get the targets of a dataset without any target transforms.
    This supports subsets and other derivative datasets."""
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])
    return torch.as_tensor(dataset.targets)
