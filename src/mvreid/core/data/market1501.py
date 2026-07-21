import torch

from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataset

EntityId = int
CameraId = int


class Market1501Dataset(MultiViewReidDataset[EntityId, CameraId]):
    """The Market-1501 dataset class."""

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index) -> ReidSample[EntityId, CameraId]:
        pass
