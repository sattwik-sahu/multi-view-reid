from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torchvision.transforms import v2 as transforms

from mvreid.core._typing import ReidSample
from mvreid.core.data.base import MultiViewReidDataset


class Veri776Dataset(MultiViewReidDataset):
    """
    Veri776 dataset implementation for multiview reid.
    This class groups images by vehicle identity to provide multi-view samples.
    """

    def __init__(
        self,
        root: str | Path,
        n_views: int,
        split: Literal["train", "test", "query"] = "train",
        transform: transforms.Transform | None = None,
        relabel: bool = True,
    ) -> None:
        # Initialize the base class which stores n_views in self._n_views
        super().__init__(n_views=n_views)

        self.root: Path = Path(root)
        self.relabel: bool = relabel
        self.split: str = split

        split_map = {
            "train": "image_train",
            "test": "image_test",
            "query": "image_query",
        }

        self.data_path: Path = self.root / split_map[split]

        self.transform: transforms.Transform = transform or transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        # Dictionary format: { vehicle_id: [(img_path, cam_id), ...] }
        self._samples_by_id: dict[int, list[tuple[Path, int]]] = defaultdict(list)
        self._parse_dataset()

        # Unique vehicle identities for entity-based indexing
        self.entity_ids: list[int] = sorted(list(self._samples_by_id.keys()))

        # Mapping for continuous PID labels [0, N-1]
        if self.relabel:
            self.pid_map: dict[int, int] = {
                old: new for new, old in enumerate(self.entity_ids)
            }
        else:
            self.pid_map: dict[int, int] = {old: old for old in self.entity_ids}

    def _parse_dataset(self) -> None:
        """Parses filename and groups them by PID and CamID."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_path}")

        # List all JPG images in the directory
        image_paths = list(self.data_path.glob("*.jpg"))

        for path in image_paths:
            # Filename structure: [pid]_c[camid]_[frameid].jpg
            filename = path.stem
            parts = filename.split("_")

            if len(parts) < 2:
                continue

            # Extract PID (Vehicle ID)
            pid = int(parts[0])
            if pid <= 0:  # Skip background or junk
                continue

            # Extract Camera ID (e.g., 'c001' -> 1)
            cam_str = parts[1]
            camid = int(cam_str[1:]) if cam_str.startswith("c") else int(cam_str)

            # Group image paths by their identity
            self._samples_by_id[pid].append((path, camid))

    def __len__(self) -> int:
        """Returns the number of unique vehicle entities."""
        return len(self.entity_ids)

    def __getitem__(self, index: int) -> ReidSample:
        """Fetches a multi-view sample for a specific vehicle."""

        # 1. Select the specific vehicle identity based on the index
        original_pid = self.entity_ids[index]
        mapped_pid = self.pid_map[original_pid]

        # 2. Get all available images for this specific vehicle identity
        available_images = self._samples_by_id[original_pid]

        # 3. Randomly select exactly n_views indices.
        # If n_views > images for this PID, replacement sampling occurs automatically.
        indices = torch.randint(0, len(available_images), (self._n_views,)).tolist()

        selected_images = []
        selected_cams = []

        # 4. Load and transform each selected view
        for i in indices:
            path, cam_id = available_images[i]
            img = Image.open(path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            selected_images.append(img)
            selected_cams.append(cam_id)

        # 5. Return the sample. images tensor shape: (n_views, 3, 128, 128)
        return ReidSample(
            images=torch.stack(selected_images),
            camera_ids=selected_cams,
            entity_id=mapped_pid,
        )
