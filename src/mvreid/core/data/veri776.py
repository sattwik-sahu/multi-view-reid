from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms

from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataModule


class _Veri776SplitDataset(Dataset[ReidSample]):
    def __init__(
        self,
        root: str | Path,
        n_views: int,
        split: str,
        transform: transforms.Transform | None = None,
        relabel: bool = True,
    ) -> None:
        self._n_views = n_views
        self._relabel = relabel

        split_map = {
            "train": "image_train",
            "test": "image_test",
            "query": "image_query",
        }
        data_path = Path(root) / split_map[split]

        self._transform = transform or transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        self._samples_by_id: dict[int, list[tuple[Path, int]]] = defaultdict(list)
        self._parse_dataset(data_path)

        self._entity_ids: list[int] = sorted(self._samples_by_id.keys())

        if self._relabel:
            self._pid_map: dict[int, int] = {
                old: new for new, old in enumerate(self._entity_ids)
            }
        else:
            self._pid_map: dict[int, int] = {old: old for old in self._entity_ids}

    def _parse_dataset(self, data_path: Path) -> None:
        if not data_path.exists():
            raise FileNotFoundError(f"Directory not found: {data_path}")

        image_paths = list(data_path.glob("*.jpg"))

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
        return len(self._entity_ids)

    def __getitem__(self, index: int) -> ReidSample:
        original_pid = self._entity_ids[index]
        mapped_pid = self._pid_map[original_pid]

        available_images = self._samples_by_id[original_pid]
        indices = torch.randint(0, len(available_images), (self._n_views,)).tolist()

        selected_images: list[torch.Tensor] = []
        selected_cams = []

        for i in indices:
            path, cam_id = available_images[i]
            img = read_image(path=path)

            if self._transform:
                img = self._transform(img)

            selected_images.append(img)
            selected_cams.append(cam_id)

        return ReidSample(
            images=torch.stack(selected_images),
            camera_ids=selected_cams,
            entity_id=mapped_pid,
        )


class Veri776DataModule(MultiViewReidDataModule):
    def __init__(
        self,
        root: str | Path,
        n_views: int,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        relabel: bool = True,
    ) -> None:
        super().__init__(
            n_views=n_views,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.root = root
        self.relabel = relabel

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = _Veri776SplitDataset(
            root=self.root, n_views=self._n_views, split="train", relabel=self.relabel
        )
        self.val_dataset = _Veri776SplitDataset(
            root=self.root, n_views=self._n_views, split="test", relabel=self.relabel
        )
