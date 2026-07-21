from pathlib import Path
from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataset
from typing import Literal
from collections import defaultdict
import torchvision.transforms as transforms
import torch


class Veri776Dataet(MultiViewReidDataset[int, int]):
    """
    Veri776 dataset implementation for multiview reid
    Class inherits from the MultiViewReidDataset class
    Args:
        TODO: Write the deatiled arguments here
    """

    def __init__(
        self,
        root: str | Path,
        n_views: int,
        split: Literal["train", "test", "query"] = "train",
        transform: transforms.Transform | None = None,
        relabel: bool = True,
    ) -> None:
        super().__init__(n_views=n_views)
        self.root: Path = Path(root)
        self.relabel: bool = relabel

        self.data_path = self.root / split
        self.transform: transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        self._samples_by_id: dict[int, list[tuple[Path, int]]] = defaultdict(list)
        self._parse_dataset()

        self.entity_ids = sorted(list(self._samples_by_id.keys()))

        if self.relabel:
            self.pid_map = {old: new for new, old in enumerate(self.entity_ids)}
        self.pid_map = {old: old for old in self.entity_ids}

    def _parse_dataset(self) -> None:
        pass

    def __len__():
        pass

    def __getitem__():
        pass
