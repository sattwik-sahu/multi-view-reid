import random

import torch

from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataset


class DummyDataset(MultiViewReidDataset[int, int]):
    def __init__(self, n_views: int) -> None:
        super().__init__(n_views=n_views)

    def __len__(self) -> int:
        return 100

    def __getitem__(self, index) -> "ReidSample[int, int]":
        return ReidSample(
            images=torch.randn(self._n_views, 3, 224, 224),
            camera_ids=[random.randint(0, 10) for _ in range(self._n_views)],
            entity_id=random.randint(0, 100),
            batch_size=[],
        )
