import os
from pathlib import Path
from typing import Union

import lightning as L
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


def load_tokens(filename: Union[str, Path]) -> torch.Tensor:
    """Load token data from a numpy file and convert to PyTorch tensor.

    Args:
        filename: Path to the numpy file containing token data

    Returns:
        PyTorch tensor of token data with dtype torch.long
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class FineWebDataset(IterableDataset):
    """
    An iterable dataset for the FineWeb dataset that yields pre-batched
    tensors of (X, Y). It streams data from a list of shard files.
    """

    def __init__(self, shard_paths: list[str], batch_size: int, seq_length: int):
        super().__init__()
        self.shard_paths = shard_paths
        self.B = batch_size
        self.T = seq_length

    def __iter__(self):
        """
        The iterator method called by the DataLoader.
        It loops through assigned shards and yields batches.
        """
        for shard_path in self.shard_paths:
            tokens = load_tokens(shard_path)

            current_pos = 0

            while current_pos + (self.B * self.T) + 1 <= len(tokens):
                chunk = tokens[current_pos : current_pos + self.B * self.T + 1]

                X = chunk[:-1].view(self.B, self.T)
                Y = chunk[1:].view(self.B, self.T)

                yield X, Y

                current_pos += self.B * self.T


class FineWebDataModule(L.LightningDataModule):
    """
    A LightningDataModule for the FineWeb dataset.

    It handles the DDP-aware splitting of data shards and provides
    train and validation DataLoaders.
    """

    def __init__(self, data_dir: str, batch_size: int, seq_length: int, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.B = batch_size
        self.T = seq_length
        self.num_workers = num_workers

    def setup(self, stage: str):
        """Called on every DDP process to set up the datasets."""
        shard_paths = sorted([os.path.join(self.data_dir, s) for s in os.listdir(self.data_dir)])
        logger.info(f"Found {len(shard_paths)} shards")

        if stage == "fit" or stage is None:
            rank = self.trainer.global_rank
            world_size = self.trainer.world_size

            # Find and distribute shards for each split
            train_shards = [p for p in shard_paths if "train" in p]
            val_shards = [p for p in shard_paths if "val" in p]
            logger.info(f"Shards split: {len(train_shards)} train, {len(val_shards)} val")

            # Each rank gets a unique slice of the shards
            train_shards_for_rank = train_shards[rank::world_size]
            val_shards_for_rank = val_shards[rank::world_size]

            self.train_dataset = FineWebDataset(
                shard_paths=train_shards_for_rank, batch_size=self.B, seq_length=self.T
            )
            self.val_dataset = FineWebDataset(
                shard_paths=val_shards_for_rank, batch_size=self.B, seq_length=self.T
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=None, num_workers=self.num_workers
        )  # Batch size is None because the dataset already returns batches

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=None, num_workers=self.num_workers)
