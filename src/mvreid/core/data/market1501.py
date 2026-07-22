from pathlib import Path

from mvreid.core.data.base import MultiViewReidDataset


class Market1501Dataset(MultiViewReidDataset):
    """The Market-1501 dataset class."""

    def __init__(self, data_root: Path, n_views: int) -> None:
        super().__init__(n_views=n_views)

        self._data_root: Path = data_root

    # TODO Implement these methods
    # def __len__(self) -> int:
    #     pass
    #
    # def __getitem__(self, index) -> ReidSample:
    #     pass
