import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataModule


class _DummyDataset(Dataset[ReidSample]):
    def __init__(self, n_views: int, length: int) -> None:
        self._n_views = n_views
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index) -> ReidSample:
        return ReidSample(
            images=torch.randn(self._n_views, 3, 224, 224),
            camera_ids=[random.randint(0, 10) for _ in range(self._n_views)],
            entity_id=random.randint(0, 100),
            batch_size=[],
        )


class DummyDataModule(MultiViewReidDataModule):
    def __init__(
        self,
        n_views: int,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_length: int = 10,
        val_length: int = 3,
    ) -> None:
        super().__init__(
            n_views=n_views,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.train_length = train_length
        self.val_length = val_length

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = _DummyDataset(self._n_views, self.train_length)
        self.val_dataset = _DummyDataset(self._n_views, self.val_length)
