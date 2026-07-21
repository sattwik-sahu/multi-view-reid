from pathlib import Path
from typing import Literal
from collections import defaultdict

from torchvision.transforms import v2 as transforms
import torch
from PIL import Image

from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataset


class Veri776Dataset(MultiViewReidDataset[int, int]):
    """
    Veri776 dataset implementation for multiview reid
    Class inherits from the MultiViewReidDataset class
    Args:
        root (str | Path): Root directory containing the dataset.
        n_views (int): Number of images to return per vehicle in each sample.
        split (Literal["train", "test", "query"]): Which data split to load.
        transform (transforms.Transform | None): Image augmentation/processing pipeline.
        relabel (bool): If True, maps original PIDs to a continuous range [0, N-1].
            Usually True for 'train' and False for 'test/query'.

    Returns:
        ReidSample: A TensorClass containing images of shape (n_views, 3, H, W),
            camera IDs, and the vehicle entity ID.
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
        self.transform = transforms.Compose(
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
        else:
            self.pid_map = {old: old for old in self.entity_ids}

    def _parse_dataset(self) -> None:
        """Parses file name and groups them by PID and CamID"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Directory not found:,{self.data_path}")
        image_paths: list = list(self.data_path.glob("*.jpg"))

        for path in image_paths:
            filename = path.stem
            parts = filename.split("_")

            if len(parts) < 2:
                continue
            pid = int(parts[0])

            if pid <= 0:
                continue

            cam_str = parts[1]
            camid = int(cam_str[1:]) if cam_str.startswith("c") else int(cam_str)

            self._samples_by_id[pid].append((path, camid))

    def __len__(self) -> int:
        """Returns the number of unique vehicle entities"""
        return len(self.entity_ids)

    def __getitem__(self, index: int) -> ReidSample[int, int]:
        """
        feteches n_views for a specific vehicle
        Args:
            index(int): Index of the vehicle in the entity_ids list
        Returns:
            ReidSample: TensorClass with images (n_views,3,H,W) and metadata

        """

        original_pid = self.entity_ids[index]
        mapped_pid = self.pid_map[original_pid]

        available_images = self._samples_by_id[original_pid]

        indices = torch.randint(0, len(available_images), (self._n_views)).tolist()

        selected_images = []
        selected_cams = []
        for i in indices:
            path, cam_id = available_images[i]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)

            selected_images.append(img)
            selected_cams.append(cam_id)

        # images shape: (n_views, 3, H, W)
        images_tensor = torch.stack(selected_images)

        return ReidSample(
            images=images_tensor,
            camera_ids=selected_cams,
            entity_id=mapped_pid,
            batch_size=[],
        )
