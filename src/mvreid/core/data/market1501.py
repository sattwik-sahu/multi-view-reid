from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataset

type EntityId = int
"""The type of the entity ID in the Market-1501 dataset. You may change according to the actual data."""

type CameraId = int
"""The type of the camera ID in the Market-1501 dataset. You may change according to the actual data."""


class Market1501Dataset(MultiViewReidDataset[EntityId, CameraId]):
    """The Market-1501 dataset class."""

    def __init__(self, n_views: int) -> None:
        super().__init__(n_views=n_views)

    def __len__(self) -> int:
        pass

    def __getitem__(self, index) -> ReidSample[EntityId, CameraId]:
        pass
