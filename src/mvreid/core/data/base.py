from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from mvreid.core._typing import ReidSample


class MultiViewReidDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_views: int,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self._n_views: int = n_views
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.persistent_workers: bool = persistent_workers

        self.train_dataset: Optional[Dataset[ReidSample]] = None
        self.val_dataset: Optional[Dataset[ReidSample]] = None

    def _dataloader(
        self, dataset: Dataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.num_workers > 0,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=torch.stack,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False, drop_last=False)
