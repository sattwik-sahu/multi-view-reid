import random

import torch

from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataset


class DummyDataset(MultiViewReidDataset):
    """A dummy dataset class.

    TODO Remove after creating other dataset classes. This class is
    added here only for demonstrating the basic structure of the
    dataset classes.
    """

    def __init__(self, n_views: int) -> None:
        super().__init__(n_views=n_views)

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index) -> ReidSample:
        return ReidSample(
            images=torch.randn(self._n_views, 3, 224, 224),
            camera_ids=[random.randint(0, 10) for _ in range(self._n_views)],
            entity_id=random.randint(0, 100),
            batch_size=[],
        )
