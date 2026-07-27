import random
from typing import Literal

import torch

from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataset


class DummyDataset(MultiViewReidDataset):
    """A dummy dataset class.

    TODO Remove after creating other dataset classes. This class is
    added here only for demonstrating the basic structure of the
    dataset classes.
    """

    def __init__(self, n_views: int, split: Literal["train", "val", "test"]) -> None:
        super().__init__(n_views=n_views, split=split)

    def __len__(self) -> int:
        match self._split:
            case "train":
                return 10
            case "val":
                return 3
            case "test":
                return 5
            case _:
                return 0

    def __getitem__(self, index) -> ReidSample:
        return ReidSample(
            images=torch.randn(self._n_views, 3, 128, 128),
            camera_ids=[random.randint(0, 10) for _ in range(self._n_views)],
            entity_id=random.randint(0, 100),
            batch_size=[],
        )
