import torch

from mvreid.core._typing import ReidSample

# from stable_pretraining.backbone.utils import vit_hf


class MultiViewReidDataset(torch.utils.data.Dataset[ReidSample]):
    """The base class for a multi-view reidentification dataset."""

    def __init__(self, n_views: int, split: str) -> None:
        """
        Creates a multi-view reidentification dataset object.

        Args:
            n_views (int): The number of views in each sample.
        """
        super().__init__()

        self._n_views: int = n_views
        self._split: str = split
