from typing import Callable, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from mvreid.core.data.base import MultiViewReidDataset
from mvreid.core.data.dummy import DummyDataset

# from mvreid.core.data.veri776 import VeRi776Dataset


class ReidDataModule(L.LightningDataModule):
    """
    LightningDataModule that accepts a partially initialized Hydra dataset factory,
    and dynamically instantiates the training and validation splits.
    """

    def __init__(
        self,
        dataset_factory: Callable[..., torch.utils.data.Dataset],
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_factory = dataset_factory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Dynamically pass the split to the partially-initialized Hydra dataset factory
        self.train_dataset = self.dataset_factory(split="train")
        self.val_dataset = self.dataset_factory(split="val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            # ReidSample inherits from TensorClass, so torch.stack is required to collate
            collate_fn=torch.stack,
            drop_last=True,  # Keeps input shapes constant to prevent torch.compile re-compilations
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=torch.stack,
            drop_last=False,
        )


__all__ = [
    "MultiViewReidDataset",
    "ReidDataModule",
    "DummyDataset",
    # "VeRi776Dataset",
]
